"""Test whether GeneralSFMFactor2Cal3Fisheye computes correct reprojection."""

from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import gtsam
from autocal.io.colmap import parse_cameras, parse_images

DATA_ROOT = Path(__file__).parent.parent / "data/eth3d_pipes/pipes"
RAW_CAL = DATA_ROOT / "dslr_calibration_jpg"
RAW_IMAGES = DATA_ROOT / "images/dslr_images"


def main():
    gt_cal, w, h = parse_cameras(RAW_CAL / "cameras.txt")
    gt_poses_named = parse_images(RAW_CAL / "images.txt")
    names = sorted(gt_poses_named.keys())[:2]
    poses = {i: gt_poses_named[n] for i, n in enumerate(names)}

    print(f"GT cal: {gt_cal}")
    print(f"Pose 0: t={poses[0].translation()}")
    print(f"Pose 1: t={poses[1].translation()}")

    # Test Cal3Fisheye.project() directly with a simple 3D point
    # Create a point in front of camera 0
    pose0 = poses[0]
    R_wc = pose0.rotation().matrix()
    t = pose0.translation()
    # Point 3m in front of camera (along camera Z = R_wc[:,2])
    forward = R_wc[:, 2]  # camera Z axis in world
    pt_world = t + 3.0 * forward

    print(f"\nForward direction: {forward}")
    print(f"3D point (3m ahead of cam 0): {pt_world}")

    # Project using Cal3Fisheye.project() directly
    # First transform to camera frame
    R_cw = R_wc.T
    pt_cam = R_cw @ (pt_world - t)
    print(f"Point in cam frame: {pt_cam}")
    print(f"Expected: near optical axis, z>0")

    # Cal3Fisheye.project() takes a Point2 in normalized coords?
    # Actually Cal3Fisheye.project takes a Unit3 or Point3?
    # Let's test:
    fx, fy = gt_cal.fx(), gt_cal.fy()
    cx, cy = gt_cal.px(), gt_cal.py()
    k = gt_cal.k()
    print(f"\nGT cal params: fx={fx:.2f} cx={cx:.2f} k={list(k)}")

    # Manual Kannala-Brandt projection
    x, y, z = pt_cam
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(r, z)
    theta2 = theta * theta
    rd = theta * (1 + k[0]*theta2 + k[1]*theta2**2 + k[2]*theta2**3 + k[3]*theta2**4)
    if r > 1e-10:
        scale = rd / r
    else:
        scale = 1.0
    u_manual = fx * scale * x + cx
    v_manual = fy * scale * y + cy
    print(f"Manual Kannala-Brandt projection: ({u_manual:.2f}, {v_manual:.2f})")
    print(f"  (should be near image center for point straight ahead)")

    # GTSAM Cal3Fisheye projection via gtsam.Cal3Fisheye:
    try:
        # Cal3Fisheye.project takes a Point2 (normalized coords?)
        pt2d = gtsam.Point2(x/z, y/z)  # normalized
        proj = gt_cal.uncalibrate(pt2d)
        print(f"GTSAM Cal3Fisheye.uncalibrate: ({proj[0]:.2f}, {proj[1]:.2f})")
    except Exception as e:
        print(f"Cal3Fisheye.uncalibrate failed: {e}")

    # Test GeneralSFMFactor2Cal3Fisheye with a single observation
    X = gtsam.symbol_shorthand.X
    P = gtsam.symbol_shorthand.P
    L = gtsam.symbol_shorthand.L

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    values.insert(X(0), poses[0])
    values.insert(L(0), gt_cal)
    values.insert(P(0), gtsam.Point3(*pt_world))

    pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.5)
    # Use manual projection as "measured"
    measured = gtsam.Point2(u_manual, v_manual)
    factor = gtsam.GeneralSFMFactor2Cal3Fisheye(measured, pixel_noise, X(0), P(0), L(0))
    graph.add(factor)

    err = graph.error(values)
    print(f"\nGTSAM factor error with GT cal + manual-projected 'measurement': {err:.6f}")
    print(f"  (should be ~0 if factor uses same Kannala-Brandt formula)")

    # Now test with wrong measurement to verify error is non-zero
    wrong_measured = gtsam.Point2(u_manual + 100, v_manual + 100)
    graph2 = gtsam.NonlinearFactorGraph()
    graph2.add(gtsam.GeneralSFMFactor2Cal3Fisheye(wrong_measured, pixel_noise, X(0), P(0), L(0)))
    err2 = graph2.error(values)
    print(f"GTSAM factor error with 100px off measurement: {err2:.6f}")
    print(f"  expected: (100^2 + 100^2) / (2 * 1.5^2) = {200/(2*2.25):.2f}")

    # Now simulate what the diagnostic does: load real image, detect SIFT, use GT 3D point
    # Just do one observation as a sanity check
    import cv2
    img_path = RAW_IMAGES / Path(names[0]).name
    img_bytes = img_path.read_bytes()
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create(nfeatures=100)
    kps, _ = sift.detectAndCompute(img, None)
    if kps:
        kp = kps[0]
        u_sift, v_sift = kp.pt
        print(f"\nFirst SIFT keypoint: ({u_sift:.2f}, {v_sift:.2f})")

        # Build 3D point from this keypoint using GT cal and poses[0] + poses[1]
        # First, undistort the keypoint
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.array(k[:4], dtype=np.float64).reshape(4, 1)
        pt_dist = np.array([[[u_sift, v_sift]]], dtype=np.float64)
        pt_undist = cv2.fisheye.undistortPoints(pt_dist, K, D, P=K)
        u_undist, v_undist = pt_undist[0, 0]
        print(f"Undistorted: ({u_undist:.2f}, {v_undist:.2f})")

        # Pick a second keypoint in image 1
        img1_path = RAW_IMAGES / Path(names[1]).name
        img1_bytes = img1_path.read_bytes()
        arr1 = np.frombuffer(img1_bytes, dtype=np.uint8)
        img1 = cv2.imdecode(arr1, cv2.IMREAD_GRAYSCALE)
        kps1, _ = sift.detectAndCompute(img1, None)
        if kps1:
            kp1 = kps1[0]
            u_sift1, v_sift1 = kp1.pt
            pt_dist1 = np.array([[[u_sift1, v_sift1]]], dtype=np.float64)
            pt_undist1 = cv2.fisheye.undistortPoints(pt_dist1, K, D, P=K)
            u_undist1, v_undist1 = pt_undist1[0, 0]

            # Triangulate with undistorted points
            R_wc0 = poses[0].rotation().matrix()
            R_cw0 = R_wc0.T
            t0 = poses[0].translation()
            R_wc1 = poses[1].rotation().matrix()
            R_cw1 = R_wc1.T
            t1 = poses[1].translation()
            t_cam0 = -R_cw0 @ t0
            t_cam1 = -R_cw1 @ t1
            P0 = K @ np.hstack([R_cw0, t_cam0[:, None]])
            P1 = K @ np.hstack([R_cw1, t_cam1[:, None]])
            pts0 = np.array([[u_undist], [v_undist]], dtype=np.float64)
            pts1 = np.array([[u_undist1], [v_undist1]], dtype=np.float64)
            pts4d = cv2.triangulatePoints(P0, P1, pts0, pts1)
            w_val = pts4d[3, 0]
            pt3d = (pts4d[:3, 0] / w_val).astype(np.float64)
            print(f"Triangulated 3D point: {pt3d}")

            # Check reprojection into image 0 using GTSAM factor with GT cal
            values3 = gtsam.Values()
            values3.insert(X(0), poses[0])
            values3.insert(L(0), gt_cal)
            values3.insert(P(0), gtsam.Point3(*pt3d))
            graph3 = gtsam.NonlinearFactorGraph()
            # Use ORIGINAL (distorted) keypoint as measurement
            measured_real = gtsam.Point2(u_sift, v_sift)
            graph3.add(gtsam.GeneralSFMFactor2Cal3Fisheye(measured_real, pixel_noise, X(0), P(0), L(0)))
            err3 = graph3.error(values3)
            print(f"Factor error with GT cal, GT triangulated 3D pt, distorted kp: {err3:.4f}")
            # Expected: small (< few pixels squared)

            # Also check what the factor projects to vs measured
            # Manually: project pt3d with gt_cal and pose0
            pt_cam0 = R_cw0 @ (pt3d - t0)
            x0, y0, z0 = pt_cam0
            r0 = np.sqrt(x0**2 + y0**2)
            if r0 > 1e-10:
                theta0 = np.arctan2(r0, z0)
                theta02 = theta0**2
                rd0 = theta0 * (1 + k[0]*theta02 + k[1]*theta02**2 + k[2]*theta02**3 + k[3]*theta02**4)
                scale0 = rd0 / r0
                proj_u0 = fx * scale0 * x0 + cx
                proj_v0 = fy * scale0 * y0 + cy
            else:
                proj_u0, proj_v0 = cx, cy
            print(f"Manual projection of triangulated point: ({proj_u0:.2f}, {proj_v0:.2f})")
            print(f"Original SIFT keypoint:                  ({u_sift:.2f}, {v_sift:.2f})")
            print(f"Pixel error: ({proj_u0-u_sift:.2f}, {proj_v0-v_sift:.2f})")


if __name__ == "__main__":
    main()
