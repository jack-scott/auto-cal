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
from typing import Any

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


def filter_pairs_by_geometry(
    matches_per_pair: dict[tuple[Any, Any], list[tuple[int, int]]],
    poses: dict[Any, gtsam.Pose3] | None,
    keypoints_u: dict[Any, np.ndarray] | None = None,
    K: np.ndarray | None = None,
    he_secondary_check: bool = True,
    he_secondary_ratio: float = 0.99,
) -> tuple[
    dict[tuple[Any, Any], list[tuple[int, int]]],
    dict[tuple[Any, Any], list[tuple[int, int]]],
    dict[PairClass, int],
    int,
]:
    """Filter pair matches by geometric classification.

    GOOD and PURE_TRANSLATION pairs are returned in the first dict (triangulate normally).
    PURE_ROTATION pairs are returned in the second dict (reprojection-only path).
    STATIC pairs are dropped entirely.

    Uses classify_pair_with_poses when poses are provided (fast, accurate).
    Falls back to classify_pair_without_poses (H/E ratio test) otherwise.

    When poses are provided and he_secondary_check=True, also runs the H/E ratio
    test on pairs the pose classifier accepts.  This catches near-duplicate pairs
    whose true baseline is smaller than the pose noise (e.g. σ_t=100mm > 0.2mm
    true baseline), where the pose classifier is effectively blind.

    Args:
        matches_per_pair:   {(id_a, id_b): [(i, j), ...]} as returned after RANSAC.
        poses:              Pose priors {img_id: Pose3}, or None for image-only path.
        keypoints_u:        Undistorted keypoints {img_id: array}.  Required if poses
                            is None, or for he_secondary_check.
        K:                  3×3 intrinsic matrix.  Required if poses is None, or for
                            he_secondary_check.
        he_secondary_check: When True and poses are provided, run the H/E ratio test
                            as a secondary check on pairs that pass the pose classifier.
                            Requires keypoints_u and K.  Default True.
        he_secondary_ratio: H/E inlier ratio threshold used for the secondary check.
                            Should be stricter than the standalone default (0.8) because
                            the secondary check targets only true near-duplicates (ratio ≈ 1.0)
                            where pose noise hides the zero baseline.  Default 0.99.

    Returns:
        (good_pairs, pure_rotation_pairs, counts, n_he_override) where:
          good_pairs           — pairs to triangulate normally
          pure_rotation_pairs  — PURE_ROTATION pairs for reprojection-only factors
          counts               — {PairClass: count} for all input pairs
          n_he_override        — pairs reclassified by the H/E secondary check
    """
    if poses is None and (keypoints_u is None or K is None):
        raise ValueError(
            "filter_pairs_by_geometry: provide either poses or both keypoints_u and K"
        )

    _he_available = keypoints_u is not None and K is not None
    counts: dict[PairClass, int] = {c: 0 for c in PairClass}
    filtered: dict[tuple[Any, Any], list[tuple[int, int]]] = {}
    pure_rotation: dict[tuple[Any, Any], list[tuple[int, int]]] = {}
    n_he_override = 0

    for (id_a, id_b), m in matches_per_pair.items():
        if poses is not None:
            cls = classify_pair_with_poses(poses[id_a], poses[id_b])
            if (he_secondary_check
                    and _he_available
                    and cls not in (PairClass.STATIC, PairClass.PURE_ROTATION)):
                he_cls = classify_pair_without_poses(
                    keypoints_u[id_a], keypoints_u[id_b], m, K,
                    degenerate_ratio=he_secondary_ratio,
                )
                if he_cls in (PairClass.STATIC, PairClass.PURE_ROTATION):
                    cls = he_cls
                    n_he_override += 1
        else:
            cls = classify_pair_without_poses(keypoints_u[id_a], keypoints_u[id_b], m, K)
        counts[cls] += 1
        if cls == PairClass.STATIC:
            continue
        if cls == PairClass.PURE_ROTATION:
            pure_rotation[(id_a, id_b)] = m
            continue
        filtered[(id_a, id_b)] = m

    return filtered, pure_rotation, counts, n_he_override


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
