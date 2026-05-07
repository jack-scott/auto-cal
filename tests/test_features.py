"""Tests for autocal.engine.features."""

import io
import math

import cv2
import gtsam
import numpy as np
import pytest

from autocal.engine.features import (
    Track,
    build_tracks,
    detect_sift,
    draw_matches,
    match_sift,
    triangulate,
    triangulate_tracks,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _checkerboard_jpeg(size: int = 128, square: int = 16) -> bytes:
    """Generate a checkerboard image and return JPEG bytes."""
    img = np.zeros((size, size), dtype=np.uint8)
    for r in range(0, size, square):
        for c in range(0, size, square):
            if (r // square + c // square) % 2 == 0:
                img[r:r+square, c:c+square] = 255
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return bytes(buf)


def _blank_jpeg(size: int = 64) -> bytes:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return bytes(buf)


def _pose_at(tx: float, ty: float, tz: float) -> gtsam.Pose3:
    """Camera at (tx,ty,tz) looking along +Z with identity orientation (R_cw = I)."""
    return gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(tx, ty, tz))


# ---------------------------------------------------------------------------
# detect_sift
# ---------------------------------------------------------------------------

def test_detect_sift_returns_arrays():
    jpeg = _checkerboard_jpeg()
    kps, descs = detect_sift(jpeg)
    assert kps.ndim == 2 and kps.shape[1] == 2
    assert descs.ndim == 2 and descs.shape[1] == 128
    assert kps.shape[0] == descs.shape[0]


def test_detect_sift_finds_features_on_textured_image():
    jpeg = _checkerboard_jpeg(size=256, square=32)
    kps, descs = detect_sift(jpeg)
    assert kps.shape[0] > 10, "expected many features on checkerboard"


def test_detect_sift_blank_image_returns_empty():
    jpeg = _blank_jpeg()
    kps, descs = detect_sift(jpeg)
    assert kps.shape == (0, 2)
    assert descs.shape == (0, 128)


def test_detect_sift_invalid_bytes_returns_empty():
    kps, descs = detect_sift(b"\x00\x01\x02")
    assert kps.shape == (0, 2)
    assert descs.shape == (0, 128)


def test_detect_sift_n_features_cap():
    # nfeatures is a soft cap in OpenCV; requesting fewer should still reduce results
    jpeg = _checkerboard_jpeg(size=256, square=16)
    kps_all, _ = detect_sift(jpeg)
    kps_cap, _ = detect_sift(jpeg, n_features=5)
    assert kps_cap.shape[0] < kps_all.shape[0]


# ---------------------------------------------------------------------------
# match_sift
# ---------------------------------------------------------------------------

def test_match_sift_same_image_matches_self():
    jpeg = _checkerboard_jpeg()
    kps, descs = detect_sift(jpeg)
    if len(descs) < 2:
        pytest.skip("not enough features")
    matches = match_sift(descs, descs, ratio=0.9)
    assert len(matches) > 0


def test_match_sift_empty_descs_returns_empty():
    empty = np.zeros((0, 128), dtype=np.float32)
    descs = np.random.rand(10, 128).astype(np.float32)
    assert match_sift(empty, descs) == []
    assert match_sift(descs, empty) == []


def test_match_sift_indices_in_range():
    jpeg = _checkerboard_jpeg()
    kps, descs = detect_sift(jpeg)
    if len(descs) < 4:
        pytest.skip("not enough features")
    matches = match_sift(descs, descs)
    for i, j in matches:
        assert 0 <= i < len(descs)
        assert 0 <= j < len(descs)


def test_match_sift_ratio_reduces_matches():
    jpeg = _checkerboard_jpeg()
    _, descs = detect_sift(jpeg)
    if len(descs) < 4:
        pytest.skip("not enough features")
    matches_loose = match_sift(descs, descs, ratio=0.99)
    matches_strict = match_sift(descs, descs, ratio=0.5)
    assert len(matches_strict) <= len(matches_loose)


# ---------------------------------------------------------------------------
# build_tracks
# ---------------------------------------------------------------------------

def test_build_tracks_simple_chain():
    # A-B: (0→0), (1→1)
    # B-C: (0→2), (1→3)
    # Expect two tracks: A:0,B:0,C:2 and A:1,B:1,C:3
    matches = {
        ("A", "B"): [(0, 0), (1, 1)],
        ("B", "C"): [(0, 2), (1, 3)],
    }
    tracks = build_tracks(matches)
    assert len(tracks) == 2
    for t in tracks:
        assert len(t.observations) == 3


def test_build_tracks_disjoint_pairs():
    matches = {
        ("A", "B"): [(0, 0)],
        ("C", "D"): [(5, 7)],
    }
    tracks = build_tracks(matches)
    assert len(tracks) == 2
    for t in tracks:
        assert len(t.observations) == 2


def test_build_tracks_merges_transitive():
    # A:0 matches B:0, B:0 matches C:0 — same track
    matches = {
        ("A", "B"): [(0, 0)],
        ("B", "C"): [(0, 0)],
        ("A", "C"): [(0, 0)],
    }
    tracks = build_tracks(matches)
    assert len(tracks) == 1
    assert len(tracks[0].observations) == 3


def test_build_tracks_empty_returns_empty():
    assert build_tracks({}) == []


def test_build_tracks_no_multi_image_returns_empty():
    # Single-image pair — should still produce tracks since each pair has 2 images
    matches = {("A", "B"): []}
    tracks = build_tracks(matches)
    assert tracks == []


# ---------------------------------------------------------------------------
# triangulate
# ---------------------------------------------------------------------------

def test_triangulate_known_point():
    """Two cameras baseline along X; point at world origin should triangulate."""
    # Intrinsics: fx=fy=500, cx=cy=0 (origin-centred)
    K = np.array([[500, 0, 0], [0, 500, 0], [0, 0, 1]], dtype=np.float64)

    # Point at (0, 0, 10) in world
    pt_world = np.array([0.0, 0.0, 10.0])

    # Camera A at (-1, 0, 0), Camera B at (+1, 0, 0), both looking along +Z (R_cw=I)
    pose_a = _pose_at(-1.0, 0.0, 0.0)
    pose_b = _pose_at(+1.0, 0.0, 0.0)

    # Project pt_world into each camera:
    # cam frame: R_cw @ (pt_world - t_world) = pt_world - t_cam  (R_cw = I)
    def proj(pose: gtsam.Pose3) -> np.ndarray:
        R = pose.rotation().matrix()
        t = pose.translation()
        pc = R @ (pt_world - t)
        return np.array([K[0,0]*pc[0]/pc[2] + K[0,2],
                         K[1,1]*pc[1]/pc[2] + K[1,2]])

    kp_a = proj(pose_a)
    kp_b = proj(pose_b)

    pt3d = triangulate(kp_a, kp_b, K, pose_a, pose_b)
    assert pt3d is not None
    assert np.allclose(pt3d, pt_world, atol=1e-6)


def test_triangulate_offset_point():
    """Point offset from camera baseline triangulates correctly."""
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
    pt_world = np.array([2.0, -1.0, 8.0])

    pose_a = _pose_at(0.0, 0.0, 0.0)
    pose_b = _pose_at(1.0, 0.0, 0.0)

    def proj(pose: gtsam.Pose3) -> np.ndarray:
        R = pose.rotation().matrix()
        t = pose.translation()
        pc = R @ (pt_world - t)
        return np.array([K[0,0]*pc[0]/pc[2] + K[0,2],
                         K[1,1]*pc[1]/pc[2] + K[1,2]])

    kp_a = proj(pose_a)
    kp_b = proj(pose_b)
    pt3d = triangulate(kp_a, kp_b, K, pose_a, pose_b)
    assert pt3d is not None
    assert np.allclose(pt3d, pt_world, atol=1e-5)


# ---------------------------------------------------------------------------
# triangulate_tracks
# ---------------------------------------------------------------------------

def test_triangulate_tracks_fills_point3d():
    K = np.array([[500, 0, 0], [0, 500, 0], [0, 0, 1]], dtype=np.float64)
    pt_world = np.array([0.0, 0.0, 10.0])

    pose_a = _pose_at(-1.0, 0.0, 0.0)
    pose_b = _pose_at(+1.0, 0.0, 0.0)

    def proj(pose: gtsam.Pose3) -> np.ndarray:
        R = pose.rotation().matrix()
        t = pose.translation()
        pc = R @ (pt_world - t)
        return np.array([[K[0,0]*pc[0]/pc[2], K[1,1]*pc[1]/pc[2]]], dtype=np.float32)

    kps = {
        "A": proj(pose_a),
        "B": proj(pose_b),
    }
    track = Track(observations={"A": 0, "B": 0})
    triangulate_tracks([track], kps, K, {"A": pose_a, "B": pose_b})
    assert track.point3d is not None
    assert np.allclose(track.point3d, pt_world, atol=1e-5)


def test_triangulate_tracks_skips_without_poses():
    K = np.eye(3, dtype=np.float64)
    track = Track(observations={"A": 0, "B": 0})
    kps = {"A": np.array([[0.0, 0.0]], dtype=np.float32),
           "B": np.array([[1.0, 0.0]], dtype=np.float32)}
    triangulate_tracks([track], kps, K, {})  # no poses
    assert track.point3d is None


# ---------------------------------------------------------------------------
# draw_matches
# ---------------------------------------------------------------------------

def test_draw_matches_returns_jpeg():
    jpeg = _checkerboard_jpeg()
    kps, descs = detect_sift(jpeg)
    if len(kps) < 2:
        pytest.skip("not enough features")
    matches = [(0, 1)]
    out = draw_matches(jpeg, kps, jpeg, kps, matches)
    assert isinstance(out, bytes)
    assert out[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_draw_matches_empty_matches():
    jpeg = _checkerboard_jpeg()
    kps, _ = detect_sift(jpeg)
    out = draw_matches(jpeg, kps, jpeg, kps, [])
    assert isinstance(out, bytes)
    assert len(out) > 0
