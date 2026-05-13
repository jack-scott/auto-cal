"""Diagnose the undistorted test failure."""

from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import gtsam
import numpy as np

from autocal.io.colmap import parse_cameras, parse_images
from autocal.engine.calib_solver import CalibrationOptions, optimize_sfm
from autocal.engine.features import (
    all_positive_depth, filter_by_reproj, max_reproj_error,
    detect_sift, match_sift, build_tracks, triangulate_tracks,
)

DATA_ROOT = Path(__file__).parent.parent / "data/eth3d_pipes/pipes"
UNDIST_IMAGES = DATA_ROOT / "images/dslr_images_undistorted"
UNDIST_CAL = DATA_ROOT / "dslr_calibration_undistorted"
MAX_IMAGES = 6
SIFT_FEATURES = 800


def main():
    gt_cal, _w, _h = parse_cameras(UNDIST_CAL / "cameras.txt")
    gt_poses_named = parse_images(UNDIST_CAL / "images.txt")
    names = sorted(gt_poses_named.keys())[:MAX_IMAGES]

    images = []
    gt_poses = {}
    for idx, name in enumerate(names):
        img_path = UNDIST_IMAGES / Path(name).name
        images.append((idx, img_path.read_bytes()))
        gt_poses[idx] = gt_poses_named[name]

    img_ids = [i for i, _ in images]
    id_to_idx = {img_id: i for i, img_id in enumerate(img_ids)}

    print(f"GT cal: fx={gt_cal.fx():.2f} fy={gt_cal.fy():.2f} cx={gt_cal.px():.2f} cy={gt_cal.py():.2f}")

    # Initial cal: 10% wrong fx/fy, k=0
    initial_cal = gtsam.Cal3DS2(
        gt_cal.fx() * 1.1, gt_cal.fy() * 1.1,
        0.0, gt_cal.px(), gt_cal.py(),
        0.0, 0.0, 0.0, 0.0,
    )

    print("\n--- Error analysis ---")
    # Detect and match features
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

    # Triangulate with GT cal
    K = np.array([[gt_cal.fx(), 0, gt_cal.px()], [0, gt_cal.fy(), gt_cal.py()], [0, 0, 1]], dtype=np.float64)
    tri_k = np.array(gt_cal.k(), dtype=np.float64)
    # No distortion for undistorted images, so keypoints_for_tri = keypoints
    if not np.any(tri_k != 0):
        keypoints_for_tri = keypoints
    else:
        keypoints_for_tri = {}
        for img_id, kps in keypoints.items():
            pts = kps.reshape(-1, 1, 2).astype(np.float64)
            undist = cv2.undistortPoints(pts, K, tri_k[:4], P=K)
            keypoints_for_tri[img_id] = undist.reshape(-1, 2).astype(np.float32)

    triangulate_tracks(tracks, keypoints_for_tri, K, gt_poses)
    good = [t for t in tracks if t.point3d is not None
            and all_positive_depth(t.point3d, t.observations, gt_poses)]

    # Compute reprojection errors with GT cal for all tracks
    errors = sorted(max_reproj_error(t, keypoints, gt_poses, gt_cal) for t in good)
    print(f"Track count: {len(good)}")
    print(f"Max reprojection error (GT cal) percentiles:")
    for pct in [50, 75, 90, 95, 99, 100]:
        idx = int(pct * len(errors) / 100)
        print(f"  p{pct}: {errors[min(idx, len(errors)-1)]:.2f}px")

    # Also compute error with initial_cal
    errors_ini = sorted(max_reproj_error(t, keypoints, gt_poses, initial_cal) for t in good)
    print(f"\nMax reprojection error (initial_cal, 10% wrong fx) percentiles:")
    for pct in [50, 75, 90, 95, 99, 100]:
        idx = int(pct * len(errors_ini) / 100)
        print(f"  p{pct}: {errors_ini[min(idx, len(errors_ini)-1)]:.2f}px")

    # After applying 20px filter on GT cal
    filtered = filter_by_reproj(good, keypoints, gt_poses, gt_cal, 20.0)
    print(f"\nAfter 20px filter: {len(filtered)} tracks")

    # Now run optimize_sfm
    print("\n--- Running optimize_sfm with reprojection filter ---")
    opts = CalibrationOptions(
        sift_features=SIFT_FEATURES,
        match_ratio=0.75,
        min_matches=8,
        lm_iterations=200,
        pose_noise_m=0.005,
        pose_noise_rad=0.0005,
        cal_noise_frac=0.15,
        cal_cx_noise_frac=0.005,
        k1_sigma=0.05,
        k2_sigma=0.02,
        p1_sigma=0.005,
        p2_sigma=0.005,
        pixel_noise_px=1.5,
        max_tracks=1500,
    )
    result = optimize_sfm(images, gt_poses, initial_cal, opts, triangulation_cal=gt_cal)
    opt = result["calibration"]
    print(f"  fx: {opt.fx():.2f} (GT {gt_cal.fx():.2f}, err {opt.fx()-gt_cal.fx():+.2f})")
    print(f"  fy: {opt.fy():.2f} (GT {gt_cal.fy():.2f}, err {opt.fy()-gt_cal.fy():+.2f})")
    print(f"  k:  {list(opt.k())}")

    print("\n--- Running WITH tight k priors (k1_sigma=0.005) ---")
    opts2 = CalibrationOptions(
        sift_features=SIFT_FEATURES,
        match_ratio=0.75,
        min_matches=8,
        lm_iterations=200,
        pose_noise_m=0.005,
        pose_noise_rad=0.0005,
        cal_noise_frac=0.15,
        cal_cx_noise_frac=0.005,
        k1_sigma=0.005,
        k2_sigma=0.005,
        p1_sigma=0.001,
        p2_sigma=0.001,
        pixel_noise_px=1.5,
        max_tracks=1500,
    )
    result2 = optimize_sfm(images, gt_poses, initial_cal, opts2, triangulation_cal=gt_cal)
    opt2 = result2["calibration"]
    print(f"  fx: {opt2.fx():.2f} (GT {gt_cal.fx():.2f}, err {opt2.fx()-gt_cal.fx():+.2f})")
    print(f"  fy: {opt2.fy():.2f} (GT {gt_cal.fy():.2f}, err {opt2.fy()-gt_cal.fy():+.2f})")
    print(f"  k:  {list(opt2.k())}")


if __name__ == "__main__":
    main()
