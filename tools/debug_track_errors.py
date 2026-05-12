"""Debug individual track reprojection errors for the fisheye case."""

from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import gtsam
import numpy as np

from autocal.io.colmap import parse_cameras, parse_images
from autocal.engine.calibration import _all_positive_depth
from autocal.engine.features import detect_sift, match_sift, build_tracks, triangulate_tracks

DATA_ROOT = Path(__file__).parent.parent / "data/eth3d_pipes/pipes"
RAW_IMAGES = DATA_ROOT / "images/dslr_images"
RAW_CAL = DATA_ROOT / "dslr_calibration_jpg"
MAX_IMAGES = 6
SIFT_FEATURES = 800


def main():
    gt_cal, _w, _h = parse_cameras(RAW_CAL / "cameras.txt")
    gt_poses_named = parse_images(RAW_CAL / "images.txt")
    names = sorted(gt_poses_named.keys())[:MAX_IMAGES]

    images = []
    gt_poses = {}
    for idx, name in enumerate(names):
        img_path = RAW_IMAGES / Path(name).name
        images.append((idx, img_path.read_bytes()))
        gt_poses[idx] = gt_poses_named[name]

    img_ids = [i for i, _ in images]
    id_to_idx = {img_id: i for i, img_id in enumerate(img_ids)}

    # Detect SIFT
    keypoints = {}
    descriptors = {}
    for img_id, img_bytes in images:
        kps, descs = detect_sift(img_bytes, n_features=SIFT_FEATURES)
        keypoints[img_id] = kps
        descriptors[img_id] = descs

    # Match sequential
    matches_per_pair = {}
    for k in range(len(img_ids) - 1):
        id_a, id_b = img_ids[k], img_ids[k + 1]
        m = match_sift(descriptors[id_a], descriptors[id_b], ratio=0.75)
        if len(m) >= 8:
            matches_per_pair[(id_a, id_b)] = m

    tracks = build_tracks(matches_per_pair)

    # Undistort and triangulate with GT cal
    K = np.array([[gt_cal.fx(), 0., gt_cal.px()], [0., gt_cal.fy(), gt_cal.py()], [0., 0., 1.]], dtype=np.float64)
    dist_k = np.array(gt_cal.k(), dtype=np.float64)
    D = dist_k[:4].reshape(4, 1)
    keypoints_for_tri = {}
    for img_id, kps in keypoints.items():
        pts = kps.reshape(-1, 1, 2).astype(np.float64)
        undist = cv2.fisheye.undistortPoints(pts, K, D, P=K)
        keypoints_for_tri[img_id] = undist.reshape(-1, 2).astype(np.float32)

    triangulate_tracks(tracks, keypoints_for_tri, K, gt_poses)
    good = [t for t in tracks if t.point3d is not None
            and _all_positive_depth(t.point3d, t.observations, gt_poses)]

    print(f"Total good tracks: {len(good)}")
    print(f"Observation count distribution:")
    for n_obs in sorted(set(len(t.observations) for t in good)):
        count = sum(1 for t in good if len(t.observations) == n_obs)
        print(f"  {n_obs} obs: {count}")

    # For each observation in a 3-view track, compute manual reprojection error
    print("\n--- Manual reprojection errors for first 5 three-view tracks ---")
    three_view = [t for t in good if len(t.observations) == 3]

    pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.5)
    X = gtsam.symbol_shorthand.X
    P = gtsam.symbol_shorthand.P
    L = gtsam.symbol_shorthand.L

    # Build base values
    base_values = gtsam.Values()
    for img_id, pose in gt_poses.items():
        base_values.insert(X(id_to_idx[img_id]), pose)
    base_values.insert(L(0), gt_cal)

    for t_idx, track in enumerate(three_view[:5]):
        pt3d = track.point3d
        print(f"\nTrack {t_idx}: P3D={pt3d}, obs={list(track.observations.keys())}")

        for img_id, kp_idx in track.observations.items():
            kp_dist = keypoints[img_id][kp_idx]  # original distorted
            kp_undist = keypoints_for_tri[img_id][kp_idx]  # undistorted

            pose = gt_poses[img_id]
            R_wc = pose.rotation().matrix()
            R_cw = R_wc.T
            t_world = pose.translation()
            pt_cam = R_cw @ (pt3d - t_world)
            x, y, z = pt_cam

            # Manual KB projection
            r = np.sqrt(x**2 + y**2)
            if r > 1e-10 and z > 0:
                theta = np.arctan2(r, z)
                t2 = theta**2
                k = gt_cal.k()
                rd = theta * (1 + k[0]*t2 + k[1]*t2**2 + k[2]*t2**3 + k[3]*t2**4)
                scale = rd / r
                proj_u = gt_cal.fx() * scale * x + gt_cal.px()
                proj_v = gt_cal.fy() * scale * y + gt_cal.py()
            else:
                proj_u, proj_v = gt_cal.px(), gt_cal.py()

            eu = proj_u - kp_dist[0]
            ev = proj_v - kp_dist[1]

            print(f"  cam{img_id}: kp_dist=({kp_dist[0]:.1f},{kp_dist[1]:.1f}) "
                  f"kp_undist=({kp_undist[0]:.1f},{kp_undist[1]:.1f}) "
                  f"proj=({proj_u:.1f},{proj_v:.1f}) "
                  f"err=({eu:.1f},{ev:.1f}) |e|={np.hypot(eu,ev):.1f}px "
                  f"z_cam={z:.3f}")

            # GTSAM factor
            values_t = gtsam.Values(base_values)
            values_t.insert(P(0), gtsam.Point3(*pt3d))
            g = gtsam.NonlinearFactorGraph()
            g.add(gtsam.GeneralSFMFactor2Cal3Fisheye(
                gtsam.Point2(float(kp_dist[0]), float(kp_dist[1])),
                pixel_noise, X(id_to_idx[img_id]), P(0), L(0)
            ))
            gtsam_err = g.error(values_t)
            exp_err = (eu**2 + ev**2) / (2 * 1.5**2)
            print(f"    GTSAM err={gtsam_err:.4f} (expected {exp_err:.4f})")

    # Summary: compute total error manually and compare to graph.error()
    print("\n--- Full error comparison ---")
    graph = gtsam.NonlinearFactorGraph()
    all_values = gtsam.Values()
    for img_id, pose in gt_poses.items():
        all_values.insert(X(id_to_idx[img_id]), pose)
    all_values.insert(L(0), gt_cal)

    total_manual = 0.0
    n_obs = 0
    for j, track in enumerate(good[:500]):
        all_values.insert(P(j), gtsam.Point3(*track.point3d))
        for img_id, kp_idx in track.observations.items():
            kp = keypoints[img_id][kp_idx]
            graph.add(gtsam.GeneralSFMFactor2Cal3Fisheye(
                gtsam.Point2(float(kp[0]), float(kp[1])),
                pixel_noise, X(id_to_idx[img_id]), P(j), L(0)
            ))
            pose = gt_poses[img_id]
            R_cw = pose.rotation().matrix().T
            t = pose.translation()
            pt_cam = R_cw @ (track.point3d - t)
            x, y, z = pt_cam
            r = np.sqrt(x**2 + y**2)
            if r > 1e-10 and z > 0:
                k = gt_cal.k()
                theta = np.arctan2(r, z)
                t2 = theta**2
                rd = theta * (1 + k[0]*t2 + k[1]*t2**2 + k[2]*t2**3 + k[3]*t2**4)
                scale = rd / r
                proj_u = gt_cal.fx() * scale * x + gt_cal.px()
                proj_v = gt_cal.fy() * scale * y + gt_cal.py()
                eu = proj_u - kp[0]
                ev = proj_v - kp[1]
                total_manual += eu**2 + ev**2
                n_obs += 1

    gtsam_total = graph.error(all_values)
    manual_rms = np.sqrt(total_manual / (2 * n_obs)) if n_obs > 0 else 0
    gtsam_rms = 1.5 * np.sqrt(2 * gtsam_total / n_obs) if n_obs > 0 else 0
    print(f"Manual RMS: {manual_rms:.2f}px  ({n_obs} obs, total |e|^2={total_manual:.3e})")
    print(f"GTSAM error: {gtsam_total:.3e}  RMS: {gtsam_rms:.2f}px")


if __name__ == "__main__":
    main()
