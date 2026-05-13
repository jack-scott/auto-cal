"""
Absolute Pose Error (APE) using the evo library.

Wraps evo's SE(3) trajectory alignment and APE metric computation.
Poses are matched by timestamp key (must be equal — no interpolation).

Usage::

    from autocal.metrics.ape import compute_ape

    result = compute_ape(gt_poses, est_poses)
    # result["by_timestamp"] → {t_ns: (trans_err_m, rot_err_deg)}
    # result["stats"]["translation"] → {mean, median, max, rmse}
    # result["stats"]["rotation"]    → {mean, median, max, rmse}
"""

from __future__ import annotations

import dataclasses
from typing import Any

import gtsam
import numpy as np

from evo.core import metrics
from evo.core.trajectory import PosePath3D

from autocal.io.mcap_reader import register_message


# ---------------------------------------------------------------------------
# MCAP message dataclasses
# ---------------------------------------------------------------------------

@register_message
@dataclasses.dataclass
class ApePoseMsg:
    """Per-frame APE written to the /ape topic."""
    translation_m: float
    rotation_deg: float


@register_message
@dataclasses.dataclass
class ApeSummaryMsg:
    """Trajectory-level APE statistics written to the /ape/summary topic."""
    trans_mean_m:   float
    trans_median_m: float
    trans_max_m:    float
    trans_rmse_m:   float
    rot_mean_deg:   float
    rot_median_deg: float
    rot_max_deg:    float
    rot_rmse_deg:   float
    n_poses:        int


def compute_ape(
    gt_poses: dict[Any, gtsam.Pose3],
    est_poses: dict[Any, gtsam.Pose3],
) -> dict:
    """Compute APE between two sets of poses using the evo library.

    Poses are matched by key (timestamp or image id).  Only keys present in
    both dicts are evaluated.  The estimated trajectory is SE(3)-aligned to
    the ground truth (Umeyama, no scale correction) before computing errors,
    following the evo convention.

    Args:
        gt_poses:  {key: Pose3} ground-truth poses (R_wc, t in world).
        est_poses: {key: Pose3} estimated poses (same convention).

    Returns:
        dict with:
          "by_key"  — {key: (translation_m, rotation_deg)} per-pose errors
                      after alignment.
          "stats"   — {"translation": {...}, "rotation": {...}} each with
                      keys: mean, median, max, rmse, std.
          "n_poses" — number of matched poses evaluated.
    """
    # Match by key, preserve insertion order
    keys = [k for k in gt_poses if k in est_poses]
    if len(keys) < 2:
        return {"by_key": {}, "stats": {}, "n_poses": 0}

    # Build 4×4 SE(3) matrices — gtsam Pose3.matrix() = [R_wc | t; 0 | 1]
    # Re-orthogonalize R via SVD to fix floating-point drift before evo's SO(3) check.
    def _to_se3(pose: gtsam.Pose3) -> np.ndarray:
        mat = pose.matrix().copy()
        U, _, Vt = np.linalg.svd(mat[:3, :3])
        mat[:3, :3] = U @ Vt
        return mat

    gt_mats  = [_to_se3(gt_poses[k])  for k in keys]
    est_mats = [_to_se3(est_poses[k]) for k in keys]

    traj_ref = PosePath3D(poses_se3=gt_mats)
    traj_est = PosePath3D(poses_se3=est_mats)

    # SE(3) Umeyama alignment: fit est → gt (no scale correction)
    traj_est.align(traj_ref, correct_scale=False)

    # Translation APE
    ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
    ape_trans.process_data((traj_ref, traj_est))

    # Rotation APE (geodesic angle in degrees)
    ape_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    ape_rot.process_data((traj_ref, traj_est))

    trans_errs: np.ndarray = np.array(ape_trans.error)
    rot_errs:   np.ndarray = np.array(ape_rot.error)

    by_key = {k: (float(te), float(re))
              for k, te, re in zip(keys, trans_errs, rot_errs)}

    def _stats(arr: np.ndarray) -> dict:
        return {
            "mean":   float(arr.mean()),
            "median": float(np.median(arr)),
            "max":    float(arr.max()),
            "rmse":   float(np.sqrt((arr ** 2).mean())),
            "std":    float(arr.std()),
        }

    return {
        "by_key":  by_key,
        "stats":   {
            "translation": _stats(trans_errs),
            "rotation":    _stats(rot_errs),
        },
        "n_poses": len(keys),
    }
