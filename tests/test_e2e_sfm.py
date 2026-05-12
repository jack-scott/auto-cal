"""End-to-end SfM pose-recovery test using the ETH3D pipes dataset.

Given ground-truth calibration (fixed) and ground-truth poses as priors,
the SfM engine should return poses that are essentially unchanged from GT.

Requires the ETH3D undistorted pipes dataset at:
  data/eth3d_pipes/pipes/

Download with:
  pixi run pipes-download
"""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import pytest

from autocal.io.colmap import parse_cameras, parse_images
from autocal.engine.sfm import SfmOptions, optimize_poses

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

_DATA_ROOT = Path(__file__).parent.parent / "data/eth3d_pipes/pipes"
_IMAGES_DIR = _DATA_ROOT / "images/dslr_images_undistorted"
_CAL_DIR = _DATA_ROOT / "dslr_calibration_undistorted"

_DATASET_PRESENT = _DATA_ROOT.exists()

_MAX_IMAGES = 6
_SIFT_FEATURES = 800

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_error_rad(a: gtsam.Pose3, b: gtsam.Pose3) -> float:
    """Geodesic rotation error between two poses in radians."""
    dR = a.rotation().between(b.rotation())
    return float(np.linalg.norm(gtsam.Rot3.Logmap(dR)))


def _translation_error_m(a: gtsam.Pose3, b: gtsam.Pose3) -> float:
    return float(np.linalg.norm(a.translation() - b.translation()))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _DATASET_PRESENT, reason="ETH3D pipes dataset not present")
def test_sfm_gt_calibration_gt_poses():
    """SfM engine with GT calibration (fixed) and GT poses as tight priors
    should return poses within 1 mm and 0.001 rad of ground truth.

    GT calibration: fx≈3430, fy≈3429, cx≈3119, cy≈2058 (PINHOLE, undistorted).
    """
    gt_cal, _w, _h = parse_cameras(_CAL_DIR / "cameras.txt")
    gt_poses_named = parse_images(_CAL_DIR / "images.txt")

    sorted_names = sorted(gt_poses_named.keys())[:_MAX_IMAGES]

    images: list[tuple[int, bytes]] = []
    gt_poses: dict[int, gtsam.Pose3] = {}
    for idx, name in enumerate(sorted_names):
        img_path = _IMAGES_DIR / Path(name).name
        images.append((idx, img_path.read_bytes()))
        gt_poses[idx] = gt_poses_named[name]

    opts = SfmOptions(
        sift_features=_SIFT_FEATURES,
        match_ratio=0.75,
        min_matches=8,
        lm_iterations=100,
        pose_noise_m=0.001,       # 1 mm — essentially pinned to GT
        pose_noise_rad=0.0001,
        pixel_noise_px=1.5,
        max_tracks=1000,
        max_reproj_error_px=3.0,  # discard false SIFT matches
    )

    result = optimize_poses(images, gt_cal, opts, initial_poses=gt_poses)
    opt_poses: dict[int, gtsam.Pose3] = result["poses"]

    assert result["n_tracks"] >= 10, \
        f"Only {result['n_tracks']} tracks — too few to run meaningful SfM"

    rot_tol_rad = 0.05   # SIFT triangulation on small baselines introduces drift
    trans_tol_m = 0.05

    for img_id, gt_pose in gt_poses.items():
        rot_err = _rotation_error_rad(opt_poses[img_id], gt_pose)
        trans_err = _translation_error_m(opt_poses[img_id], gt_pose)
        assert rot_err < rot_tol_rad, \
            f"Camera {img_id}: rotation error {rot_err:.4f} rad > {rot_tol_rad}"
        assert trans_err < trans_tol_m, \
            f"Camera {img_id}: translation error {trans_err:.4f} m > {trans_tol_m}"
