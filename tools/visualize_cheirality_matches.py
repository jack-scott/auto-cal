"""
Visualize the specific frame-to-frame matches that cause TriangulationCheiralityException.

Loads three consecutive frames from the MCAP, runs SIFT+matching, then draws
side-by-side match images.  Cheirality-failing matches (from the test cases) are
highlighted in red; all other matches are drawn in green.

Usage:
    pixi run python tools/visualize_cheirality_matches.py data/eth3d_exhibition_hall.mcap
    pixi run python tools/visualize_cheirality_matches.py data/eth3d_exhibition_hall.mcap --output-dir /tmp/matches
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gtsam
import numpy as np

from autocal.engine.features import (
    detect_sift,
    filter_matches_ransac,
    match_sift,
    undistort_keypoints,
    cal_to_K,
)
from autocal.gtsam_bridge.conversions import calibration_from_mcap_msg
from autocal.io.mcap_reader import iter_messages

IMAGE_TOPIC = "/camera/image"
CAL_TOPIC   = "/camera/calibration"

# Exact pixel pairs from the cheirality test cases (img_id: timestamp in ns)
# Format: (img_id_a, img_id_b, [(px_a, py_a, px_b, py_b), ...])
CHEIRALITY_EXAMPLES = [
    (0, 1_000_000_000, [
        (3084.00, 1525.64, 4349.70, 1566.80),
        ( 506.75, 1500.19, 4454.26, 1462.74),
    ]),
    (1_000_000_000, 2_000_000_000, [
        ( 265.42, 1510.16, 2243.52, 1570.03),
    ]),
]

SCALE = 0.15  # downscale factor for output images (full res is 6048×4032)


def _load_frames(mcap_path: str, wanted_ts: set[int]) -> dict[int, bytes]:
    frames: dict[int, bytes] = {}
    for topic, t_ns, msg in iter_messages(mcap_path, topics=[IMAGE_TOPIC]):
        if t_ns in wanted_ts:
            frames[t_ns] = bytes(msg.data)
        if len(frames) == len(wanted_ts):
            break
    return frames


def _load_calibration(mcap_path: str) -> gtsam.Cal3DS2 | gtsam.Cal3Fisheye:
    for _, _, msg in iter_messages(mcap_path, topics=[CAL_TOPIC]):
        return calibration_from_mcap_msg(msg)
    raise RuntimeError("No calibration found in MCAP")


def _draw_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    kps_a: np.ndarray,
    kps_b: np.ndarray,
    all_matches: list[tuple[int, int]],
    bad_pixels_a: list[tuple[float, float]],
    bad_pixels_b: list[tuple[float, float]],
    scale: float = SCALE,
    title: str = "",
) -> np.ndarray:
    """Draw side-by-side match image with cheirality matches highlighted in red."""
    h, w = img_a.shape[:2]
    sw, sh = int(w * scale), int(h * scale)
    small_a = cv2.resize(img_a, (sw, sh))
    small_b = cv2.resize(img_b, (sw, sh))

    canvas = np.hstack([small_a, small_b])

    def s(px, py):
        return int(px * scale), int(py * scale)

    # Draw all matches in green (thin)
    for i, j in all_matches:
        pa = s(kps_a[i, 0], kps_a[i, 1])
        pb = s(kps_b[j, 0] + w, kps_b[j, 1])  # offset x for right panel
        cv2.line(canvas, pa, pb, (0, 180, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, pa, 3, (0, 200, 0), -1)
        cv2.circle(canvas, pb, 3, (0, 200, 0), -1)

    # Draw cheirality examples in red (thick)
    for (px_a, py_a), (px_b, py_b) in zip(bad_pixels_a, bad_pixels_b):
        pa = s(px_a, py_a)
        pb = s(px_b + w, py_b)
        cv2.line(canvas, pa, pb, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, pa, 6, (0, 0, 255), 2)
        cv2.circle(canvas, pb, 6, (0, 0, 255), 2)

    if title:
        cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

    n_green = len(all_matches)
    n_red   = len(bad_pixels_a)
    info = f"matches: {n_green} green (good)  {n_red} red (cheirality exception)"
    cv2.putText(canvas, info, (10, sh - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize cheirality-failing matches between consecutive frames."
    )
    parser.add_argument("mcap", help="Input MCAP path")
    parser.add_argument("--output-dir", default=".", help="Directory to save images (default: .)")
    parser.add_argument("--scale", type=float, default=SCALE,
                        help=f"Downscale factor for output (default: {SCALE})")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio for matching")
    parser.add_argument("--ransac", type=float, default=2.0,
                        help="RANSAC threshold (0 = skip)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted_ts = {ts for pair in CHEIRALITY_EXAMPLES for ts in (pair[0], pair[1])}
    print(f"Loading frames {sorted(wanted_ts)} from {args.mcap} ...")
    frames = _load_frames(args.mcap, wanted_ts)
    print(f"  loaded {len(frames)} frames")

    cal = _load_calibration(args.mcap)
    print(f"  calibration: {cal}")

    for pair_idx, (ts_a, ts_b, bad_pairs) in enumerate(CHEIRALITY_EXAMPLES):
        img_bytes_a = frames.get(ts_a)
        img_bytes_b = frames.get(ts_b)
        if img_bytes_a is None or img_bytes_b is None:
            print(f"  skipping pair {ts_a}/{ts_b} — frames not found")
            continue

        print(f"\nPair {pair_idx+1}: frame {ts_a} ns  ↔  frame {ts_b} ns")

        kps_a, descs_a = detect_sift(img_bytes_a, n_features=0)
        kps_b, descs_b = detect_sift(img_bytes_b, n_features=0)
        print(f"  SIFT: {len(kps_a)} / {len(kps_b)} features")

        matches = match_sift(descs_a, descs_b, ratio=args.ratio)
        print(f"  {len(matches)} matches after ratio test")

        if args.ransac > 0:
            kps_for_geo = undistort_keypoints(
                {ts_a: kps_a, ts_b: kps_b}, cal
            )
            inliers = filter_matches_ransac(
                kps_for_geo[ts_a], kps_for_geo[ts_b], matches,
                ransac_threshold=args.ransac, min_inliers=8,
            )
            print(f"  {len(inliers)} matches after RANSAC")
            draw_matches = inliers
        else:
            draw_matches = matches

        # Find which match indices correspond to the cheirality pixels
        bad_a = [(px, py) for (px, py, _, _) in bad_pairs]
        bad_b = [(px, py) for (_, _, px, py) in bad_pairs]

        arr_a = np.frombuffer(img_bytes_a, dtype=np.uint8)
        arr_b = np.frombuffer(img_bytes_b, dtype=np.uint8)
        img_a = cv2.imdecode(arr_a, cv2.IMREAD_COLOR)
        img_b = cv2.imdecode(arr_b, cv2.IMREAD_COLOR)

        title = f"frames {ts_a//1_000_000_000}s  ↔  {ts_b//1_000_000_000}s"
        canvas = _draw_pair(
            img_a, img_b, kps_a, kps_b, draw_matches,
            bad_a, bad_b,
            scale=args.scale, title=title,
        )

        # Also draw a version with ONLY the cheirality-failing matches
        canvas_bad_only = _draw_pair(
            img_a, img_b, kps_a, kps_b, [],
            bad_a, bad_b,
            scale=args.scale,
            title=f"{title}  (cheirality examples only)",
        )

        out_all  = out_dir / f"matches_pair{pair_idx+1}_all.jpg"
        out_bad  = out_dir / f"matches_pair{pair_idx+1}_cheirality.jpg"
        cv2.imwrite(str(out_all),  canvas,      [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(out_bad),  canvas_bad_only, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  saved {out_all}")
        print(f"  saved {out_bad}")


if __name__ == "__main__":
    main()
