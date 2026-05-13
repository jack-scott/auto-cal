"""
Degenerate frame-pair classification.

A pair is classified based on the relative translation and rotation between two
cameras.  The classification drives whether triangulation is attempted and which
factor types are added to the GTSAM graph.

Two entry points:
  classify_pair_with_poses    — uses pose priors (fast, accurate)
  classify_pair_without_poses — H/E inlier ratio test (image-only)
"""

from __future__ import annotations

from enum import Enum

import cv2
import gtsam
import numpy as np


class PairClass(Enum):
    """Geometric relationship between a camera pair."""
    STATIC           = "STATIC"           # no motion — skip entirely
    PURE_ROTATION    = "PURE_ROTATION"    # rotation only — reprojection factors only
    PURE_TRANSLATION = "PURE_TRANSLATION" # translation only — good for triangulation
    GOOD             = "GOOD"             # mixed — full pipeline


def classify_pair_with_poses(
    pose_i: gtsam.Pose3,
    pose_j: gtsam.Pose3,
    translation_threshold_m: float = 0.02,
    rotation_threshold_deg: float = 2.0,
) -> PairClass:
    """Classify the geometric relationship between two cameras using pose priors.

    Args:
        pose_i: Pose3 of the first camera (GTSAM convention: R_wc, t_world).
        pose_j: Pose3 of the second camera.
        translation_threshold_m: Baseline below this is treated as zero.
        rotation_threshold_deg:  Rotation below this is treated as zero.

    Returns:
        PairClass member describing the pair geometry.
    """
    rel = pose_i.between(pose_j)
    t_norm = np.linalg.norm(rel.translation())
    _, angle = rel.rotation().axisAngle()
    r_deg = np.degrees(abs(angle))

    small_t = t_norm < translation_threshold_m
    small_r = r_deg  < rotation_threshold_deg

    if small_t and small_r:
        return PairClass.STATIC
    if small_t and not small_r:
        return PairClass.PURE_ROTATION
    if not small_t and small_r:
        return PairClass.PURE_TRANSLATION
    return PairClass.GOOD


def classify_pair_without_poses(
    kps_a: np.ndarray,
    kps_b: np.ndarray,
    matches: list[tuple[int, int]],
    K: np.ndarray,
    degenerate_ratio: float = 0.8,
) -> PairClass:
    """Classify pair geometry using the H/E inlier ratio test (no pose priors).

    Fits both a homography (H) and an essential matrix (E) to the matched points.
    A homography perfectly explains pure rotation and planar scenes, so if H
    accounts for as many inliers as E, there is no useful translational baseline
    and triangulation should not be attempted.

    Args:
        kps_a:            Undistorted keypoints for image A, shape (N, 2).
        kps_b:            Undistorted keypoints for image B, shape (N, 2).
        matches:          Matched index pairs [(idx_a, idx_b), ...].
        K:                3×3 camera intrinsic matrix.
        degenerate_ratio: H/E inlier ratio above which the pair is degenerate.
                          Default 0.8 (i.e. H explains ≥80% as many inliers as E).

    Returns:
        PURE_ROTATION if H/E ratio > degenerate_ratio (degenerate — do not triangulate).
        GOOD          if H/E ratio ≤ degenerate_ratio (translational baseline present).

    Note:
        Without pose information this test cannot distinguish STATIC from PURE_ROTATION,
        or PURE_TRANSLATION from GOOD.  Use classify_pair_with_poses when poses are
        available — it is faster and more accurate.
    """
    if len(matches) < 8:
        return PairClass.PURE_ROTATION

    pts_a = np.array([[float(kps_a[i][0]), float(kps_a[i][1])] for i, _ in matches])
    pts_b = np.array([[float(kps_b[j][0]), float(kps_b[j][1])] for _, j in matches])

    _, mask_H = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 3.0)
    _, mask_E = cv2.findEssentialMat(pts_a, pts_b, K, method=cv2.USAC_MAGSAC,
                                     prob=0.999, threshold=1.0)

    n_H = int(mask_H.sum()) if mask_H is not None else 0
    n_E = int(mask_E.sum()) if mask_E is not None else 0

    ratio = n_H / (n_E + 1e-6)
    if ratio > degenerate_ratio:
        return PairClass.PURE_ROTATION
    return PairClass.GOOD
