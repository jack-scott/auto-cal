"""End-to-end calibration tests using the ETH3D pipes dataset.

Tests that the SfM optimizer recovers known ground-truth camera intrinsics
when given noisy initial pose estimates (simulating GPS uncertainty).

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
from autocal.engine.calibration import CalibrationOptions, optimize_sfm

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

_DATA_ROOT = Path(__file__).parent.parent / "data/eth3d_pipes/pipes"
_IMAGES_DIR = _DATA_ROOT / "images/dslr_images_undistorted"
_CAL_DIR = _DATA_ROOT / "dslr_calibration_undistorted"

_DATASET_PRESENT = _DATA_ROOT.exists()

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_MAX_IMAGES = 6       # first N images — keeps test time manageable
_SIFT_FEATURES = 800  # cap per image


@pytest.fixture(scope="module")
def eth3d_data():
    """Load GT calibration, poses, and image bytes once per module."""
    gt_cal, width, height = parse_cameras(_CAL_DIR / "cameras.txt")
    gt_poses_named = parse_images(_CAL_DIR / "images.txt")

    sorted_names = sorted(gt_poses_named.keys())[:_MAX_IMAGES]

    images = []
    gt_poses: dict[int, gtsam.Pose3] = {}
    for idx, name in enumerate(sorted_names):
        img_path = _IMAGES_DIR / Path(name).name
        images.append((idx, img_path.read_bytes()))
        gt_poses[idx] = gt_poses_named[name]

    return {
        "gt_cal": gt_cal,
        "width": width,
        "height": height,
        "images": images,
        "gt_poses": gt_poses,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_translation_noise(
    poses: dict[int, gtsam.Pose3],
    sigma_m: float,
    rng: np.random.Generator,
) -> dict[int, gtsam.Pose3]:
    noisy = {}
    for k, pose in poses.items():
        noise = rng.normal(0.0, sigma_m, 3)
        noisy[k] = gtsam.Pose3(pose.rotation(), gtsam.Point3(*(pose.translation() + noise)))
    return noisy


def _make_opts(noise_m: float) -> CalibrationOptions:
    return CalibrationOptions(
        sift_features=_SIFT_FEATURES,
        match_ratio=0.75,
        min_matches=8,
        lm_iterations=200,
        pose_noise_m=noise_m,
        pose_noise_rad=0.05,
        cal_noise_frac=0.2,
        cal_cx_noise_frac=0.01,
        # Undistorted images: pin distortion near zero to prevent fx/k degeneracy
        k1_sigma=0.05,
        k2_sigma=0.02,
        p1_sigma=0.005,
        p2_sigma=0.005,
        pixel_noise_px=1.5,
        max_tracks=1500,
    )


def _print_comparison(label: str, got: gtsam.Cal3DS2, expected: gtsam.Cal3DS2) -> None:
    print(f"\n{label}")
    print(f"  fx: {got.fx():.2f}  (GT {expected.fx():.2f}, Δ {got.fx()-expected.fx():+.2f})")
    print(f"  fy: {got.fy():.2f}  (GT {expected.fy():.2f}, Δ {got.fy()-expected.fy():+.2f})")
    print(f"  cx: {got.px():.2f}  (GT {expected.px():.2f}, Δ {got.px()-expected.px():+.2f})")
    print(f"  cy: {got.py():.2f}  (GT {expected.py():.2f}, Δ {got.py()-expected.py():+.2f})")
    k = got.k()
    print(f"  k1: {k[0]:.6f}  k2: {k[1]:.6f}  p1: {k[2]:.6f}  p2: {k[3]:.6f}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _DATASET_PRESENT, reason="ETH3D pipes dataset not present")
@pytest.mark.parametrize("noise_m", [0.05, 0.5, 1.0, 2.0])
def test_calibration_noise_recovery(eth3d_data, noise_m):
    """Optimizer should recover PINHOLE intrinsics given noisy pose priors.

    Ground truth: fx≈3430, fy≈3429, cx≈3119, cy≈2058, k=0 (undistorted images).

    Strategy: triangulate with GT calibration (correct 3D seed points), then
    optimise with a 10%-off calibration as the starting point.  This ensures
    the optimiser sees non-zero initial reprojection error and has a gradient
    to follow toward the correct calibration.
    """
    gt_cal: gtsam.Cal3DS2 = eth3d_data["gt_cal"]
    images: list[tuple[int, bytes]] = eth3d_data["images"]
    gt_poses: dict[int, gtsam.Pose3] = eth3d_data["gt_poses"]

    rng = np.random.default_rng(42)
    noisy_poses = _add_translation_noise(gt_poses, sigma_m=noise_m, rng=rng)

    # Optimisation starts 10% off; triangulation uses GT (correct seed points)
    initial_cal = gtsam.Cal3DS2(
        gt_cal.fx() * 1.1,
        gt_cal.fy() * 1.1,
        0.0,
        gt_cal.px(),
        gt_cal.py(),
        0.0, 0.0, 0.0, 0.0,
    )

    opts = _make_opts(noise_m)
    result = optimize_sfm(images, noisy_poses, initial_cal, opts, triangulation_cal=gt_cal)
    opt_cal: gtsam.Cal3DS2 = result["calibration"]

    _print_comparison(f"noise={noise_m}m", opt_cal, gt_cal)

    # Ground truth focal lengths are ~3430px; tolerances are intentionally
    # generous — tighten once the optimizer is confirmed to converge well.
    fx_tol = 300.0   # ~9% of 3430
    cx_tol = 80.0    # ~2.6% of 3119

    assert result["n_tracks"] >= 10, \
        f"Only {result['n_tracks']} triangulated tracks — too few to constrain calibration"

    assert abs(opt_cal.fx() - gt_cal.fx()) < fx_tol, \
        f"fx error {opt_cal.fx() - gt_cal.fx():.1f}px exceeds {fx_tol}px (noise={noise_m}m)"
    assert abs(opt_cal.fy() - gt_cal.fy()) < fx_tol, \
        f"fy error {opt_cal.fy() - gt_cal.fy():.1f}px exceeds {fx_tol}px (noise={noise_m}m)"
    assert abs(opt_cal.px() - gt_cal.px()) < cx_tol, \
        f"cx error {opt_cal.px() - gt_cal.px():.1f}px exceeds {cx_tol}px (noise={noise_m}m)"
    assert abs(opt_cal.py() - gt_cal.py()) < cx_tol, \
        f"cy error {opt_cal.py() - gt_cal.py():.1f}px exceeds {cx_tol}px (noise={noise_m}m)"


@pytest.mark.skipif(not _DATASET_PRESENT, reason="ETH3D pipes dataset not present")
def test_exact_poses_recovers_calibration(eth3d_data):
    """With exact GT poses and GT seed points, the optimizer must recover
    calibration starting from a 15%-off initial guess.

    This is the tightest possible sanity check: poses are pinned to ground
    truth, 3D seeds are correct, only calibration needs to be recovered.
    """
    gt_cal: gtsam.Cal3DS2 = eth3d_data["gt_cal"]
    images = eth3d_data["images"]
    gt_poses = eth3d_data["gt_poses"]

    initial_cal = gtsam.Cal3DS2(
        gt_cal.fx() * 1.15,   # 15% off
        gt_cal.fy() * 1.15,
        0.0,
        gt_cal.px(),
        gt_cal.py(),
        0.0, 0.0, 0.0, 0.0,
    )

    opts = CalibrationOptions(
        sift_features=_SIFT_FEATURES,
        match_ratio=0.75,
        min_matches=8,
        lm_iterations=300,
        pose_noise_m=0.005,   # nearly fixed
        pose_noise_rad=0.001,
        cal_noise_frac=0.2,   # allow ±20% on fx
        cal_cx_noise_frac=0.005,
        # Undistorted: tight distortion prior prevents fx/k degeneracy
        k1_sigma=0.03,
        k2_sigma=0.01,
        p1_sigma=0.003,
        p2_sigma=0.003,
        pixel_noise_px=1.0,
        max_tracks=2000,
    )

    result = optimize_sfm(images, gt_poses, initial_cal, opts, triangulation_cal=gt_cal)
    opt_cal = result["calibration"]

    _print_comparison("exact poses", opt_cal, gt_cal)

    assert result["n_tracks"] >= 10
    assert abs(opt_cal.fx() - gt_cal.fx()) < 100.0, \
        f"fx error {opt_cal.fx() - gt_cal.fx():.1f}px (exact poses should be tight)"
    assert abs(opt_cal.fy() - gt_cal.fy()) < 100.0
    assert abs(opt_cal.px() - gt_cal.px()) < 30.0
    assert abs(opt_cal.py() - gt_cal.py()) < 30.0
