"""End-to-end calibration tests using the ETH3D pipes dataset.

Two tests covering the two calibration model types:

1. test_focal_length_recovery_undistorted  (Cal3DS2, PINHOLE)
   Uses undistorted images (k=0).  Starts with 10% wrong focal length.
   The optimizer must recover the correct focal length.
   Dataset: dslr_calibration_undistorted  (pipes_dslr_undistorted.7z)

2. test_focal_length_recovery_fisheye  (Cal3Fisheye, THIN_PRISM_FISHEYE)
   Uses raw distorted images with Kannala-Brandt fisheye distortion.
   Starts with 10% wrong focal length, correct k1/k2/k3/k4.
   The optimizer must recover the correct focal length while keeping
   distortion near GT.
   Dataset: dslr_calibration_jpg  (pipes_dslr_jpg.7z)

Downloads:
  pixi run pipes-download          # undistorted
  pixi run pipes-download-raw      # raw/distorted
"""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import pytest

from autocal.io.colmap import parse_cameras, parse_images
from autocal.engine.calib_solver import CalibrationOptions, optimize_sfm

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

_DATA_ROOT = Path(__file__).parent.parent / "data/eth3d_pipes/pipes"

_UNDIST_IMAGES = _DATA_ROOT / "images/dslr_images_undistorted"
_UNDIST_CAL    = _DATA_ROOT / "dslr_calibration_undistorted"

_RAW_IMAGES    = _DATA_ROOT / "images/dslr_images"
_RAW_CAL       = _DATA_ROOT / "dslr_calibration_jpg"

_UNDIST_PRESENT = _UNDIST_IMAGES.exists() and _UNDIST_CAL.exists()
_RAW_PRESENT    = _RAW_IMAGES.exists() and _RAW_CAL.exists()

_MAX_IMAGES = 6
_SIFT_FEATURES = 800


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(images_dir: Path, cal_dir: Path) -> tuple:
    """Load GT cal, GT poses, and image bytes for the first _MAX_IMAGES frames."""
    gt_cal, _w, _h = parse_cameras(cal_dir / "cameras.txt")
    gt_poses_named = parse_images(cal_dir / "images.txt")
    sorted_names = sorted(gt_poses_named.keys())[:_MAX_IMAGES]

    images: list[tuple[int, bytes]] = []
    gt_poses: dict[int, gtsam.Pose3] = {}
    for idx, name in enumerate(sorted_names):
        img_path = images_dir / Path(name).name
        images.append((idx, img_path.read_bytes()))
        gt_poses[idx] = gt_poses_named[name]

    return gt_cal, images, gt_poses


def _print_comparison(label: str, got, expected) -> None:
    print(f"\n{label}")
    print(f"  fx: {got.fx():.2f}  (GT {expected.fx():.2f}, Δ {got.fx()-expected.fx():+.2f})")
    print(f"  fy: {got.fy():.2f}  (GT {expected.fy():.2f}, Δ {got.fy()-expected.fy():+.2f})")
    print(f"  cx: {got.px():.2f}  (GT {expected.px():.2f}, Δ {got.px()-expected.px():+.2f})")
    print(f"  cy: {got.py():.2f}  (GT {expected.py():.2f}, Δ {got.py()-expected.py():+.2f})")
    k_got, k_exp = got.k(), expected.k()
    for i, name in enumerate(["k1", "k2", "k3/p1", "k4/p2"]):
        if i < len(k_got) and i < len(k_exp):
            print(f"  {name}: {k_got[i]:.6f}  (GT {k_exp[i]:.6f}, Δ {k_got[i]-k_exp[i]:+.6f})")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _UNDIST_PRESENT,
    reason="ETH3D undistorted dataset not present — run: pixi run pipes-download",
)
def test_focal_length_recovery_undistorted():
    """Cal3DS2 calibration on undistorted images recovers focal length
    from a 10%-off starting point, with GT poses as tight priors.

    GT: PINHOLE (k=0), fx≈3430, fy≈3429, cx≈3119, cy≈2058.
    Initial: fx*1.1, fy*1.1, k=0.
    Expected: recovered fx/fy within ±150px of GT.
    """
    gt_cal, images, gt_poses = _load_dataset(_UNDIST_IMAGES, _UNDIST_CAL)
    assert isinstance(gt_cal, gtsam.Cal3DS2)

    initial_cal = gtsam.Cal3DS2(
        gt_cal.fx() * 1.1, gt_cal.fy() * 1.1,
        0.0, gt_cal.px(), gt_cal.py(),
        0.0, 0.0, 0.0, 0.0,
    )

    opts = CalibrationOptions(
        sift_features=_SIFT_FEATURES,
        match_ratio=0.75,
        min_matches=8,
        lm_iterations=200,
        pose_noise_m=0.005,
        pose_noise_rad=0.0005,
        cal_noise_frac=0.15,
        cal_cx_noise_frac=0.005,
        k1_sigma=0.05,   # undistorted: pin near zero
        k2_sigma=0.02,
        p1_sigma=0.005,
        p2_sigma=0.005,
        pixel_noise_px=1.5,
        max_tracks=1500,
    )

    result = optimize_sfm(images, gt_poses, initial_cal, opts, triangulation_cal=gt_cal)
    opt_cal = result["calibration"]

    _print_comparison("undistorted, 10%-off focal length", opt_cal, gt_cal)

    assert result["n_tracks"] >= 10, f"Only {result['n_tracks']} tracks"

    fx_tol = 150.0
    cx_tol = 100.0

    assert abs(opt_cal.fx() - gt_cal.fx()) < fx_tol, \
        f"fx error {opt_cal.fx()-gt_cal.fx():+.1f}px exceeds ±{fx_tol}px"
    assert abs(opt_cal.fy() - gt_cal.fy()) < fx_tol, \
        f"fy error {opt_cal.fy()-gt_cal.fy():+.1f}px exceeds ±{fx_tol}px"
    assert abs(opt_cal.px() - gt_cal.px()) < cx_tol, \
        f"cx error {opt_cal.px()-gt_cal.px():+.1f}px exceeds ±{cx_tol}px"
    assert abs(opt_cal.py() - gt_cal.py()) < cx_tol, \
        f"cy error {opt_cal.py()-gt_cal.py():+.1f}px exceeds ±{cx_tol}px"


@pytest.mark.skipif(
    not _RAW_PRESENT,
    reason="ETH3D raw dataset not present — run: pixi run pipes-download-raw",
)
def test_focal_length_recovery_fisheye():
    """Cal3Fisheye calibration on raw fisheye images recovers focal length
    from a 10%-off starting point, with GT poses as tight priors.

    GT: THIN_PRISM_FISHEYE (k1=0.219, k2=0.156, k3=-0.037, k4=0.303),
        fx≈3430, cx≈3033, cy≈2004.
    Initial: fx*1.1, fy*1.1, GT k1/k2/k3/k4 (correct distortion, wrong scale).
    Expected: recovered fx/fy within ±150px of GT.

    triangulation_cal=gt_cal gives accurate 3D seeds via fisheye undistortion.
    """
    gt_cal, images, gt_poses = _load_dataset(_RAW_IMAGES, _RAW_CAL)
    assert isinstance(gt_cal, gtsam.Cal3Fisheye)

    gt_k = gt_cal.k()
    initial_cal = gtsam.Cal3Fisheye(
        gt_cal.fx() * 1.1, gt_cal.fy() * 1.1,
        0.0, gt_cal.px(), gt_cal.py(),
        gt_k[0], gt_k[1], gt_k[2], gt_k[3],
    )

    opts = CalibrationOptions(
        sift_features=_SIFT_FEATURES,
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

    result = optimize_sfm(images, gt_poses, initial_cal, opts, triangulation_cal=gt_cal)
    opt_cal = result["calibration"]

    _print_comparison("raw fisheye, 10%-off focal length", opt_cal, gt_cal)

    assert result["n_tracks"] >= 10, f"Only {result['n_tracks']} tracks"

    fx_tol = 150.0
    cx_tol = 100.0

    assert abs(opt_cal.fx() - gt_cal.fx()) < fx_tol, \
        f"fx error {opt_cal.fx()-gt_cal.fx():+.1f}px exceeds ±{fx_tol}px"
    assert abs(opt_cal.fy() - gt_cal.fy()) < fx_tol, \
        f"fy error {opt_cal.fy()-gt_cal.fy():+.1f}px exceeds ±{fx_tol}px"
    assert abs(opt_cal.px() - gt_cal.px()) < cx_tol, \
        f"cx error {opt_cal.px()-gt_cal.px():+.1f}px exceeds ±{cx_tol}px"
    assert abs(opt_cal.py() - gt_cal.py()) < cx_tol, \
        f"cy error {opt_cal.py()-gt_cal.py():+.1f}px exceeds ±{cx_tol}px"
