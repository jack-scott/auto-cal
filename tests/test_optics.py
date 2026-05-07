"""Tests for autocal.optics.camera."""

import math

import cv2
import numpy as np
import pytest

from autocal.optics.camera import (
    apply_distortion,
    distort_point,
    fx_from_exif,
    intrinsic_matrix,
    overlay_keypoints,
    overlay_projected_points,
    project,
    undistort_point,
    unproject,
)


# ---------------------------------------------------------------------------
# fx_from_exif
# ---------------------------------------------------------------------------

def test_fx_from_exif_known():
    """Caliterra camera: 4.5mm lens, 6.20mm sensor, 4000px wide."""
    expected_fx = 4.5 / 6.20 * 4000
    assert abs(fx_from_exif(4.5, 6.20, 4000) - expected_fx) < 0.01


def test_fx_from_exif_proportional():
    fx1 = fx_from_exif(50.0, 36.0, 7680)
    fx2 = fx_from_exif(50.0, 36.0, 3840)
    assert abs(fx1 / fx2 - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# intrinsic_matrix
# ---------------------------------------------------------------------------

def test_intrinsic_matrix_shape():
    K = intrinsic_matrix(500.0, 500.0, 320.0, 240.0)
    assert K.shape == (3, 3)


def test_intrinsic_matrix_values(simple_K):
    assert simple_K[0, 0] == 500.0  # fx
    assert simple_K[1, 1] == 500.0  # fy
    assert simple_K[0, 2] == 320.0  # cx
    assert simple_K[1, 2] == 240.0  # cy
    assert simple_K[2, 2] == 1.0


# ---------------------------------------------------------------------------
# apply_distortion
# ---------------------------------------------------------------------------

def test_apply_distortion_no_distortion():
    assert apply_distortion(0.25, 0.0, 0.0) == 1.0


def test_apply_distortion_k1_only():
    r2 = 0.5
    k1 = 0.1
    expected = 1.0 + k1 * r2
    assert abs(apply_distortion(r2, k1, 0.0) - expected) < 1e-12


def test_apply_distortion_both():
    r2 = 0.4
    k1, k2 = 0.05, -0.02
    expected = 1.0 + k1 * r2 + k2 * r2 * r2
    assert abs(apply_distortion(r2, k1, k2) - expected) < 1e-12


# ---------------------------------------------------------------------------
# distort / undistort round-trip
# ---------------------------------------------------------------------------

def test_distort_undistort_round_trip_no_distortion():
    pt = np.array([0.3, -0.2])
    assert np.allclose(undistort_point(distort_point(pt, 0.0, 0.0), 0.0, 0.0), pt, atol=1e-10)


def test_distort_undistort_round_trip_with_k1():
    pt = np.array([0.3, -0.2])
    k1, k2 = 0.1, 0.0
    distorted = distort_point(pt, k1, k2)
    recovered = undistort_point(distorted, k1, k2)
    assert np.allclose(recovered, pt, atol=1e-8)


def test_distort_undistort_round_trip_both():
    pt = np.array([0.4, 0.3])
    k1, k2 = -0.15, 0.05
    distorted = distort_point(pt, k1, k2)
    recovered = undistort_point(distorted, k1, k2)
    assert np.allclose(recovered, pt, atol=1e-8)


# ---------------------------------------------------------------------------
# project / unproject
# ---------------------------------------------------------------------------

def test_project_no_distortion(simple_K):
    # Point at (0, 0, 1) in camera frame should project to principal point
    pt3d = np.array([0.0, 0.0, 1.0])
    px = project(pt3d, simple_K)
    assert np.allclose(px, [320.0, 240.0], atol=1e-9)


def test_project_offset_point(simple_K):
    # Point at (1, 0, 1): x/z = 1 → col = 500*1 + 320 = 820
    pt3d = np.array([1.0, 0.0, 1.0])
    px = project(pt3d, simple_K)
    assert abs(px[0] - 820.0) < 1e-9
    assert abs(px[1] - 240.0) < 1e-9


def test_project_behind_camera_raises(simple_K):
    with pytest.raises(ValueError, match="behind camera"):
        project(np.array([0.0, 0.0, -1.0]), simple_K)


def test_project_unproject_bearing_direction(simple_K):
    """unproject(project(pt)) should give a bearing co-linear with pt."""
    pt3d = np.array([0.5, -0.3, 2.0])
    px = project(pt3d, simple_K)
    bearing = unproject(px, simple_K)
    # bearing and pt3d/|pt3d| should be parallel (same direction)
    expected = pt3d / np.linalg.norm(pt3d)
    assert np.allclose(bearing, expected, atol=1e-6)


def test_project_unproject_with_distortion():
    K = intrinsic_matrix(800.0, 800.0, 400.0, 300.0)
    k1, k2 = 0.05, -0.01
    pt3d = np.array([0.2, -0.1, 1.0])
    px = project(pt3d, K, k1, k2)
    bearing = unproject(px, K, k1, k2)
    expected = pt3d / np.linalg.norm(pt3d)
    assert np.allclose(bearing, expected, atol=1e-6)


def test_unproject_unit_norm(simple_K):
    bearing = unproject(np.array([320.0, 240.0]), simple_K)
    assert abs(np.linalg.norm(bearing) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# overlay helpers (smoke tests — check output is valid JPEG)
# ---------------------------------------------------------------------------

def test_overlay_keypoints_returns_jpeg(synthetic_jpeg):
    kps = [cv2.KeyPoint(32.0, 32.0, 5.0)]
    result = overlay_keypoints(synthetic_jpeg, kps)
    assert isinstance(result, bytes)
    assert len(result) > 100
    # Verify it decodes as a valid image
    arr = np.frombuffer(result, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape[0] == 64 and img.shape[1] == 64


def test_overlay_projected_points_returns_jpeg(synthetic_jpeg):
    K = intrinsic_matrix(50.0, 50.0, 32.0, 32.0)  # small K for 64×64 image
    pts3d = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]])
    R_cw = np.eye(3)
    t_cw = np.array([0.0, 0.0, 0.0])
    result = overlay_projected_points(synthetic_jpeg, pts3d, K, R_cw, t_cw)
    assert isinstance(result, bytes)
    arr = np.frombuffer(result, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None
