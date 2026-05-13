"""
Cheirality breakdown test across different frame windows.

Reproduces the SfM pipeline up to triangulation for three consecutive frames,
then categorises every track edge into four buckets using GT epipolar geometry:

  GOOD          — triangulation succeeded  AND  epipolar check passes
  CHEIRALITY    — CheiralityException  AND  epipolar OK
  BAD_MATCH     — epipolar check fails  AND  triangulation succeeded
  BOTH          — CheiralityException  AND  epipolar fails

Poses are chained from frame 0 through the target window so that accumulated
drift is realistic (same behaviour as the real SfM pipeline).

Output images are saved per-window to /tmp/cheirality_breakdown/<window>/
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

import cv2
import gtsam
import numpy as np
import pytest

from autocal.engine.features import (
    build_tracks,
    cal_to_K,
    chain_essential_matrix,
    detect_sift,
    filter_matches_ransac,
    match_sift,
    undistort_keypoints,
)
from autocal.gtsam_bridge.conversions import (
    calibration_from_mcap_msg,
    pose3_from_frame_transform,
)
from autocal.io.mcap_reader import iter_messages

MCAP_PATH = Path("data/eth3d_exhibition_hall.mcap")
OUT_ROOT  = Path("/tmp/cheirality_breakdown")

pytestmark = pytest.mark.skipif(
    not MCAP_PATH.exists(),
    reason=f"dataset not found: {MCAP_PATH}",
)

# Each window is (label, [ts_a, ts_b, ts_c]) — must be consecutive
WINDOWS = [
    ("frames_00-02", [0, 1_000_000_000, 2_000_000_000]),
    ("frames_10-12", [10_000_000_000, 11_000_000_000, 12_000_000_000]),
    ("frames_20-22", [20_000_000_000, 21_000_000_000, 22_000_000_000]),
    ("frames_61-64", [61_000_000_000, 62_000_000_000, 63_000_000_000, 64_000_000_000]),
]


# ---------------------------------------------------------------------------
# Category labels
# ---------------------------------------------------------------------------

class Cat(Enum):
    GOOD       = auto()
    CHEIRALITY = auto()
    BAD_MATCH  = auto()
    BOTH       = auto()


COLOURS = {
    Cat.GOOD:       (0, 200,   0),
    Cat.CHEIRALITY: (0, 140, 255),
    Cat.BAD_MATCH:  (255,  0, 255),
    Cat.BOTH:       (0,   0, 255),
}

LABELS = {
    Cat.GOOD:       "good",
    Cat.CHEIRALITY: "cheirality (valid match, pose noise broke DLT)",
    Cat.BAD_MATCH:  "bad match (RANSAC false positive)",
    Cat.BOTH:       "both (bad match + cheirality)",
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _fundamental_matrix(
    pose_a: gtsam.Pose3,
    pose_b: gtsam.Pose3,
    K: np.ndarray,
) -> np.ndarray:
    R_cw_b = pose_b.rotation().matrix().T
    t_a    = pose_a.translation()
    t_b    = pose_b.translation()
    R_AB   = R_cw_b @ pose_a.rotation().matrix()
    t_AB   = R_cw_b @ (t_a - t_b)
    skew   = np.array([
        [       0, -t_AB[2],  t_AB[1]],
        [ t_AB[2],        0, -t_AB[0]],
        [-t_AB[1],  t_AB[0],        0],
    ])
    Ki = np.linalg.inv(K)
    return Ki.T @ (skew @ R_AB) @ Ki


def _sampson(F: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> float:
    a = np.array([p_a[0], p_a[1], 1.0])
    b = np.array([p_b[0], p_b[1], 1.0])
    Fa  = F @ a
    FTb = F.T @ b
    num = (b @ Fa) ** 2
    den = Fa[0]**2 + Fa[1]**2 + FTb[0]**2 + FTb[1]**2
    return num / den if den > 1e-12 else float("inf")


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

SCALE = 0.15


def _draw_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
    colour: tuple[int, int, int],
    title: str,
) -> np.ndarray:
    h, w = img_a.shape[:2]
    sw, sh = int(w * SCALE), int(h * SCALE)
    canvas = np.hstack([cv2.resize(img_a, (sw, sh)),
                        cv2.resize(img_b, (sw, sh))])
    for (px_a, py_a), (px_b, py_b) in pairs:
        pa = (int(px_a * SCALE), int(py_a * SCALE))
        pb = (int(px_b * SCALE) + sw, int(py_b * SCALE))
        cv2.line(canvas, pa, pb, colour, 1, cv2.LINE_AA)
        cv2.circle(canvas, pa, 4, colour, -1)
        cv2.circle(canvas, pb, 4, colour, -1)
    cv2.putText(canvas, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _decode(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_window(
    window_ts: list[int],
    all_images: dict[int, bytes],
    all_gt_poses: dict[int, gtsam.Pose3],
    all_keypoints: dict[int, np.ndarray],
    all_keypoints_u: dict[int, np.ndarray],
    all_descriptors: dict[int, np.ndarray],
    cal: gtsam.Cal3DS2 | gtsam.Cal3Fisheye,
    all_ts: list[int],
) -> dict:
    """
    Chain poses from ts=0 up through the window, build tracks for the two
    consecutive pairs in the window, triangulate, and return categorised results.
    """
    K = cal_to_K(cal)

    # Build matches for ALL pairs from 0 up to end of window (needed for chaining)
    max_ts = max(window_ts)
    chain_ts = [ts for ts in all_ts if ts <= max_ts]

    matches_all: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for k in range(len(chain_ts) - 1):
        ts_a, ts_b = chain_ts[k], chain_ts[k + 1]
        if ts_a not in all_keypoints_u or ts_b not in all_keypoints_u:
            continue
        raw = match_sift(all_descriptors[ts_a], all_descriptors[ts_b], ratio=0.75)
        inliers = filter_matches_ransac(
            all_keypoints_u[ts_a], all_keypoints_u[ts_b], raw,
            ransac_threshold=2.0, min_inliers=8,
        )
        if inliers:
            matches_all[(ts_a, ts_b)] = inliers

    # Chain poses from frame 0 through the window
    init_poses = chain_essential_matrix(chain_ts, all_keypoints_u, matches_all, K)

    # Build and triangulate tracks for the window pairs only
    window_pairs = [(window_ts[k], window_ts[k + 1]) for k in range(len(window_ts) - 1)]
    window_matches = {p: matches_all[p] for p in window_pairs if p in matches_all}
    tracks = build_tracks(window_matches)

    for track in tracks:
        visible = [(ts, ki) for ts, ki in track.observations.items()
                   if ts in init_poses]
        if len(visible) < 2:
            track.point3d = None
            continue
        pose_vec = gtsam.Pose3Vector([init_poses[ts] for ts, _ in visible])
        meas_vec = gtsam.Point2Vector([
            gtsam.Point2(float(all_keypoints[ts][ki][0]),
                         float(all_keypoints[ts][ki][1]))
            for ts, ki in visible
        ])
        try:
            pt = gtsam.triangulatePoint3(pose_vec, cal, meas_vec, 1e-9, True)
            track.point3d = np.array([pt[0], pt[1], pt[2]])
            track._exc = None
        except Exception as exc:
            track.point3d = None
            track._exc = exc

    # Categorise each track edge
    SAMPSON_THR = 1.5
    pair_cats: dict[tuple[int, int], dict[Cat, list]] = {
        p: {c: [] for c in Cat} for p in window_pairs
    }

    for track in tracks:
        obs = track.observations
        for ts_a, ts_b in window_pairs:
            if ts_a not in obs or ts_b not in obs:
                continue
            ki_a, ki_b = obs[ts_a], obs[ts_b]
            pu_a = all_keypoints_u[ts_a][ki_a]
            pu_b = all_keypoints_u[ts_b][ki_b]
            F  = _fundamental_matrix(all_gt_poses[ts_a], all_gt_poses[ts_b], K)
            sd = _sampson(F, pu_a, pu_b)

            epipolar_ok = sd < SAMPSON_THR ** 2
            cheirality  = (
                track._exc is not None and
                ("Cheirality" in type(track._exc).__name__ or
                 "Cheirality" in str(track._exc))
            )

            if not cheirality and epipolar_ok:
                cat = Cat.GOOD
            elif cheirality and epipolar_ok:
                cat = Cat.CHEIRALITY
            elif not cheirality and not epipolar_ok:
                cat = Cat.BAD_MATCH
            else:
                cat = Cat.BOTH

            raw_a = all_keypoints[ts_a][ki_a]
            raw_b = all_keypoints[ts_b][ki_b]
            pair_cats[(ts_a, ts_b)][cat].append(
                ((float(raw_a[0]), float(raw_a[1])),
                 (float(raw_b[0]), float(raw_b[1])))
            )

    return pair_cats


# ---------------------------------------------------------------------------
# Module-level fixture: load data once for all windows
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_dataset():
    """Load all frames from 0 to ts=22s, with GT poses and features."""
    max_ts = 64_000_000_000
    all_ts_wanted = set(range(0, max_ts + 1_000_000_000, 1_000_000_000))

    images:      dict[int, bytes]       = {}
    gt_poses:    dict[int, gtsam.Pose3] = {}
    cal = None

    for topic, t_ns, msg in iter_messages(str(MCAP_PATH)):
        if t_ns > max_ts and topic == "/camera/image":
            break
        if topic == "/camera/image"        and t_ns in all_ts_wanted: images[t_ns]   = bytes(msg.data)
        elif topic == "/tf"                and t_ns in all_ts_wanted: gt_poses[t_ns] = pose3_from_frame_transform(msg)
        elif topic == "/camera/calibration" and cal is None:           cal            = calibration_from_mcap_msg(msg)

    print(f"\nLoaded {len(images)} images, {len(gt_poses)} GT poses")

    # Detect features once for all frames
    keypoints:   dict[int, np.ndarray] = {}
    descriptors: dict[int, np.ndarray] = {}
    for ts in sorted(images):
        kps, descs = detect_sift(images[ts], n_features=0)
        keypoints[ts]   = kps
        descriptors[ts] = descs

    keypoints_u = undistort_keypoints(keypoints, cal)
    all_ts = sorted(images.keys())

    return dict(
        images=images, gt_poses=gt_poses, cal=cal,
        keypoints=keypoints, keypoints_u=keypoints_u,
        descriptors=descriptors, all_ts=all_ts,
    )


# ---------------------------------------------------------------------------
# Tests — one per window
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,window_ts", WINDOWS)
def test_cheirality_breakdown(label, window_ts, full_dataset):
    ds = full_dataset
    pair_cats = _run_window(
        window_ts,
        ds["images"], ds["gt_poses"],
        ds["keypoints"], ds["keypoints_u"], ds["descriptors"],
        ds["cal"], ds["all_ts"],
    )

    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = {c: 0 for c in Cat}

    for k in range(len(window_ts) - 1):
        ts_a, ts_b = window_ts[k], window_ts[k + 1]
        cats = pair_cats.get((ts_a, ts_b), {c: [] for c in Cat})
        img_a = _decode(ds["images"][ts_a])
        img_b = _decode(ds["images"][ts_b])
        n_total = sum(len(v) for v in cats.values())

        for cat in Cat:
            pairs = cats[cat]
            totals[cat] += len(pairs)
            title = (f"{ts_a//1_000_000_000}s→{ts_b//1_000_000_000}s  "
                     f"{LABELS[cat]}  ({len(pairs)}/{n_total})")
            canvas = _draw_pair(img_a, img_b, pairs, COLOURS[cat], title)
            fname = out_dir / f"pair{k+1}_{cat.name.lower()}.jpg"
            cv2.imwrite(str(fname), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])

    total = sum(totals.values())
    print(f"\n--- {label} ---")
    for cat in Cat:
        pct = totals[cat] / total * 100 if total else 0
        print(f"  {cat.name:<12} {totals[cat]:4d}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<12} {total:4d}")
    print(f"  Images saved to {out_dir}/")

    assert total > 50, f"{label}: too few tracks ({total})"
