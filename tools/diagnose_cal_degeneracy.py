"""Diagnose the focal-length/distortion degeneracy in calibration optimization.

Run: pixi run python tools/diagnose_cal_degeneracy.py
"""

from __future__ import annotations
from pathlib import Path
import sys

import cv2
import gtsam
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autocal.io.colmap import parse_cameras, parse_images
from autocal.engine.calibration import CalibrationOptions, optimize_sfm, _all_positive_depth
from autocal.engine.features import detect_sift, match_sift, build_tracks, triangulate_tracks

DATA_ROOT = Path(__file__).parent.parent / "data/eth3d_pipes/pipes"
RAW_IMAGES = DATA_ROOT / "images/dslr_images"
RAW_CAL = DATA_ROOT / "dslr_calibration_jpg"
MAX_IMAGES = 6
SIFT_FEATURES = 800


def load_dataset():
    gt_cal, _w, _h = parse_cameras(RAW_CAL / "cameras.txt")
    gt_poses_named = parse_images(RAW_CAL / "images.txt")
    sorted_names = sorted(gt_poses_named.keys())[:MAX_IMAGES]

    images = []
    gt_poses = {}
    for idx, name in enumerate(sorted_names):
        img_path = RAW_IMAGES / Path(name).name
        images.append((idx, img_path.read_bytes()))
        gt_poses[idx] = gt_poses_named[name]

    return gt_cal, images, gt_poses


def check_reprojection_error(
    images, gt_poses, cal, triangulation_cal, max_tracks=500, pixel_noise_px=1.5
):
    """Compute initial GTSAM error for a given calibration and triangulation_cal."""
    img_ids = [i for i, _ in images]

    keypoints = {}
    descriptors = {}
    for img_id, img_bytes in images:
        kps, descs = detect_sift(img_bytes, n_features=SIFT_FEATURES)
        keypoints[img_id] = kps
        descriptors[img_id] = descs

    matches_per_pair = {}
    for k in range(len(img_ids) - 1):
        id_a, id_b = img_ids[k], img_ids[k + 1]
        m = match_sift(descriptors[id_a], descriptors[id_b], ratio=0.75)
        if len(m) >= 8:
            matches_per_pair[(id_a, id_b)] = m

    tracks = build_tracks(matches_per_pair)

    tri_cal = triangulation_cal if triangulation_cal is not None else cal
    K = np.array([
        [tri_cal.fx(), tri_cal.skew(), tri_cal.px()],
        [0,            tri_cal.fy(),   tri_cal.py()],
        [0,            0,              1            ],
    ], dtype=np.float64)

    dist_k = np.array(tri_cal.k(), dtype=np.float64)
    keypoints_for_tri = {}
    if np.any(dist_k != 0.0):
        _tri_fisheye = isinstance(tri_cal, gtsam.Cal3Fisheye)
        for img_id, kps in keypoints.items():
            pts = kps.reshape(-1, 1, 2).astype(np.float64)
            if _tri_fisheye:
                D = dist_k[:4].reshape(4, 1)
                undist = cv2.fisheye.undistortPoints(pts, K, D, P=K)
            else:
                undist = cv2.undistortPoints(pts, K, dist_k[:4], P=K)
            keypoints_for_tri[img_id] = undist.reshape(-1, 2).astype(np.float32)
    else:
        keypoints_for_tri = keypoints

    triangulate_tracks(tracks, keypoints_for_tri, K, gt_poses)
    good = [
        t for t in tracks
        if t.point3d is not None
        and _all_positive_depth(t.point3d, t.observations, gt_poses)
    ]
    good.sort(key=lambda t: len(t.observations), reverse=True)
    triangulated = good[:max_tracks]

    print(f"  Tracks: {len(triangulated)} (from {len(good)} cheirality-ok, {len(tracks)} total)")
    print(f"  Track length distribution:")
    lengths = [len(t.observations) for t in triangulated]
    for n in sorted(set(lengths)):
        count = lengths.count(n)
        print(f"    {n} obs: {count} tracks")

    # Build GTSAM graph
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    X = gtsam.symbol_shorthand.X
    P = gtsam.symbol_shorthand.P
    L = gtsam.symbol_shorthand.L
    id_to_idx = {img_id: i for i, img_id in enumerate(img_ids)}

    for img_id, pose in gt_poses.items():
        initial_values.insert(X(id_to_idx[img_id]), pose)
    initial_values.insert(L(0), cal)

    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0005, 0.0005, 0.0005, 0.005, 0.005, 0.005]))
    for img_id, pose in gt_poses.items():
        graph.add(gtsam.PriorFactorPose3(X(id_to_idx[img_id]), pose, pose_noise))

    pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_noise_px)
    _fisheye = isinstance(cal, gtsam.Cal3Fisheye)
    for j, track in enumerate(triangulated):
        initial_values.insert(P(j), gtsam.Point3(*track.point3d))
        for img_id, kp_idx in track.observations.items():
            if img_id not in id_to_idx:
                continue
            kp = keypoints[img_id][kp_idx]
            measured = gtsam.Point2(float(kp[0]), float(kp[1]))
            if _fisheye:
                graph.add(gtsam.GeneralSFMFactor2Cal3Fisheye(
                    measured, pixel_noise, X(id_to_idx[img_id]), P(j), L(0)))
            else:
                graph.add(gtsam.GeneralSFMFactor2Cal3DS2(
                    measured, pixel_noise, X(id_to_idx[img_id]), P(j), L(0)))

    n_obs = sum(len(t.observations) for t in triangulated)
    total_err = graph.error(initial_values)
    rms_px = (total_err / n_obs * 2 * pixel_noise_px**2) ** 0.5
    print(f"  Total GTSAM error: {total_err:.3e}  ({n_obs} obs, RMS≈{rms_px:.1f}px)")
    return total_err, triangulated, keypoints


def main():
    print("Loading dataset...")
    gt_cal, images, gt_poses = load_dataset()
    gt_k = gt_cal.k()
    print(f"GT cal: fx={gt_cal.fx():.2f} fy={gt_cal.fy():.2f} k1={gt_k[0]:.4f} k2={gt_k[1]:.4f}")

    initial_cal = gtsam.Cal3Fisheye(
        gt_cal.fx() * 1.1, gt_cal.fy() * 1.1,
        0.0, gt_cal.px(), gt_cal.py(),
        gt_k[0], gt_k[1], gt_k[2], gt_k[3],
    )

    print("\n=== Case 1: GT cal + triangulate with GT cal (expect ~0 error) ===")
    err_gt, _, _ = check_reprojection_error(images, gt_poses, gt_cal, triangulation_cal=gt_cal)

    print("\n=== Case 2: initial_cal (10% wrong fx) + triangulate with GT cal ===")
    err_ini_gt, _, _ = check_reprojection_error(images, gt_poses, initial_cal, triangulation_cal=gt_cal)

    print("\n=== Case 3: initial_cal (10% wrong fx) + triangulate with initial_cal ===")
    err_ini_ini, _, _ = check_reprojection_error(images, gt_poses, initial_cal, triangulation_cal=None)

    print(f"\nSummary:")
    print(f"  GT cal / GT tri:      {err_gt:.3e}")
    print(f"  initial cal / GT tri: {err_ini_gt:.3e}  (ratio: {err_ini_gt/err_gt:.1f}x)")
    print(f"  initial cal / ini tri:{err_ini_ini:.3e}  (ratio: {err_ini_ini/err_gt:.1f}x)")

    print("\n=== Running optimize_sfm with GT tri (current approach) ===")
    opts = CalibrationOptions(
        sift_features=SIFT_FEATURES,
        match_ratio=0.75,
        min_matches=8,
        lm_iterations=200,
        pose_noise_m=0.005,
        pose_noise_rad=0.0005,
        cal_noise_frac=0.15,
        cal_cx_noise_frac=0.005,
        k1_sigma=0.1,
        k2_sigma=0.1,
        k3_sigma=0.05,
        k4_sigma=0.1,
        pixel_noise_px=1.5,
        max_tracks=1500,
    )
    result_gt_tri = optimize_sfm(images, gt_poses, initial_cal, opts, triangulation_cal=gt_cal)
    opt = result_gt_tri["calibration"]
    print(f"  fx: {opt.fx():.2f} (GT {gt_cal.fx():.2f}, err {opt.fx()-gt_cal.fx():+.2f})")
    print(f"  k1: {opt.k()[0]:.4f} (GT {gt_k[0]:.4f})")

    print("\n=== Running optimize_sfm WITHOUT tri_cal (initial_cal triangulation) ===")
    result_ini_tri = optimize_sfm(images, gt_poses, initial_cal, opts, triangulation_cal=None)
    opt2 = result_ini_tri["calibration"]
    print(f"  fx: {opt2.fx():.2f} (GT {gt_cal.fx():.2f}, err {opt2.fx()-gt_cal.fx():+.2f})")
    print(f"  k1: {opt2.k()[0]:.4f} (GT {gt_k[0]:.4f})")


if __name__ == "__main__":
    main()
