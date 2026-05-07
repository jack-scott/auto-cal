"""Tests for autocal.gtsam_bridge.conversions."""

import math

import gtsam
import numpy as np
import pytest

from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform

from autocal.gtsam_bridge.conversions import (
    cal3bundler_from_camera_calibration,
    camera_calibration_from_cal3bundler,
    frame_transform_from_pose3,
    pose3_from_frame_transform,
)
from autocal.io.mcap_writer import ns_to_timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ft(tx, ty, tz, qx=0.0, qy=0.0, qz=0.0, qw=1.0,
             parent="map", child="camera_link", t_ns=0) -> FrameTransform:
    ft = FrameTransform()
    ft.timestamp.CopyFrom(ns_to_timestamp(t_ns))
    ft.parent_frame_id = parent
    ft.child_frame_id = child
    ft.translation.x = tx
    ft.translation.y = ty
    ft.translation.z = tz
    ft.rotation.x = qx
    ft.rotation.y = qy
    ft.rotation.z = qz
    ft.rotation.w = qw
    return ft


def _rot_z(angle_rad: float) -> gtsam.Rot3:
    """Rot3 representing a rotation of angle_rad around Z."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return gtsam.Rot3(np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ]))


# ---------------------------------------------------------------------------
# pose3_from_frame_transform / frame_transform_from_pose3  round-trips
# ---------------------------------------------------------------------------

def test_pose3_from_ft_identity():
    """Identity FrameTransform → identity Pose3."""
    ft = _make_ft(0, 0, 0)
    pose = pose3_from_frame_transform(ft)
    assert np.allclose(pose.translation(), [0, 0, 0], atol=1e-12)
    assert np.allclose(pose.rotation().matrix(), np.eye(3), atol=1e-12)


def test_pose3_from_ft_translation_only():
    """Pure translation FrameTransform maps to correct Pose3 translation."""
    ft = _make_ft(1.0, 2.0, 3.0)
    pose = pose3_from_frame_transform(ft)
    assert np.allclose(pose.translation(), [1.0, 2.0, 3.0], atol=1e-12)
    assert np.allclose(pose.rotation().matrix(), np.eye(3), atol=1e-12)


def test_pose3_roundtrip_translation():
    """Pose3 → FrameTransform → Pose3 round-trip preserves translation."""
    t_expected = np.array([4.0, -2.5, 1.1])
    pose_in = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(*t_expected))
    ft = frame_transform_from_pose3(pose_in, "map", "camera_link", 0)
    pose_out = pose3_from_frame_transform(ft)
    assert np.allclose(pose_out.translation(), t_expected, atol=1e-12)
    assert np.allclose(pose_out.rotation().matrix(), np.eye(3), atol=1e-12)


def test_pose3_roundtrip_rotation():
    """Pose3 → FrameTransform → Pose3 round-trip preserves rotation."""
    # R_cw = 90° rotation around Z
    R_cw = _rot_z(math.pi / 2)
    pose_in = gtsam.Pose3(R_cw, gtsam.Point3(0, 0, 0))
    ft = frame_transform_from_pose3(pose_in, "map", "camera_link", 0)
    pose_out = pose3_from_frame_transform(ft)
    assert np.allclose(pose_out.rotation().matrix(), R_cw.matrix(), atol=1e-12)


def test_pose3_roundtrip_full():
    """Full Pose3 with translation and rotation survives a round-trip."""
    t_expected = np.array([1.0, -3.0, 2.0])
    R_cw = _rot_z(math.pi / 3)  # 60° around Z
    pose_in = gtsam.Pose3(R_cw, gtsam.Point3(*t_expected))
    ft = frame_transform_from_pose3(pose_in, "map", "camera_link", 0)
    pose_out = pose3_from_frame_transform(ft)
    assert np.allclose(pose_out.translation(), t_expected, atol=1e-12)
    assert np.allclose(pose_out.rotation().matrix(), R_cw.matrix(), atol=1e-12)


def test_frame_transform_from_pose3_metadata():
    """frame_transform_from_pose3 sets parent/child/timestamp correctly."""
    pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0))
    t_ns = 1_413_742_851_000_000_000
    ft = frame_transform_from_pose3(pose, "earth", "base_link", t_ns)
    assert ft.parent_frame_id == "earth"
    assert ft.child_frame_id == "base_link"
    recovered_ns = ft.timestamp.seconds * 1_000_000_000 + ft.timestamp.nanos
    assert recovered_ns == t_ns


def test_ft_rotation_convention():
    """TF rotation = R_wc; a 90° CW-around-Z camera sees X-right in camera = Y in world."""
    # R_cw rotates world +X → camera +Y (camera is rotated 90° CW from world around Z)
    angle = math.pi / 2
    R_cw = _rot_z(angle)  # world X maps to camera Y
    pose = gtsam.Pose3(R_cw, gtsam.Point3(0, 0, 0))
    ft = frame_transform_from_pose3(pose, "map", "camera_link", 0)
    # TF rotation is R_wc = R_cw.inverse() = rotation by -90° around Z
    # Quaternion for -90° around Z: qz = sin(-45°), qw = cos(-45°)
    expected_qz = math.sin(-math.pi / 4)
    expected_qw = math.cos(-math.pi / 4)
    assert abs(ft.rotation.z - expected_qz) < 1e-9


# ---------------------------------------------------------------------------
# cal3bundler_from_camera_calibration
# ---------------------------------------------------------------------------

def _make_cc(fx, cx, cy, k1, k2, model="plumb_bob") -> CameraCalibration:
    cc = CameraCalibration()
    cc.distortion_model = model
    cc.K.extend([fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0])
    cc.D.extend([k1, k2, 0.0, 0.0, 0.0])
    return cc


def test_cal3bundler_from_cc_basic():
    fx, cx, cy, k1, k2 = 800.0, 320.0, 240.0, -0.1, 0.02
    cal = cal3bundler_from_camera_calibration(_make_cc(fx, cx, cy, k1, k2))
    assert abs(cal.fx() - fx) < 1e-9
    assert abs(cal.px() - cx) < 1e-9
    assert abs(cal.py() - cy) < 1e-9
    assert abs(cal.k1() - k1) < 1e-9
    assert abs(cal.k2() - k2) < 1e-9


def test_cal3bundler_from_cc_rejects_non_plumb_bob():
    cc = _make_cc(800, 320, 240, 0, 0, model="equidistant")
    with pytest.raises(ValueError, match="plumb_bob"):
        cal3bundler_from_camera_calibration(cc)


def test_cal3bundler_from_cc_accepts_empty_model():
    """Empty distortion_model string is accepted (treated as plumb_bob)."""
    cc = _make_cc(500.0, 320.0, 240.0, 0.0, 0.0, model="")
    cal = cal3bundler_from_camera_calibration(cc)
    assert abs(cal.fx() - 500.0) < 1e-9


def test_cal3bundler_from_cc_missing_K_raises():
    cc = CameraCalibration()
    cc.distortion_model = "plumb_bob"
    cc.D.extend([0.0, 0.0])
    with pytest.raises(ValueError, match="K must have 9"):
        cal3bundler_from_camera_calibration(cc)


def test_cal3bundler_from_cc_missing_D_raises():
    cc = CameraCalibration()
    cc.distortion_model = "plumb_bob"
    cc.K.extend([500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0])
    cc.D.extend([0.1])  # only 1 element, need ≥2
    with pytest.raises(ValueError, match="D must have"):
        cal3bundler_from_camera_calibration(cc)


# ---------------------------------------------------------------------------
# camera_calibration_from_cal3bundler
# ---------------------------------------------------------------------------

def test_cal3bundler_roundtrip():
    """Cal3Bundler → CameraCalibration → Cal3Bundler preserves all values."""
    fx, k1, k2, cx, cy = 750.0, -0.05, 0.01, 400.0, 300.0
    cal_in = gtsam.Cal3Bundler(fx, k1, k2, cx, cy)
    cc = camera_calibration_from_cal3bundler(cal_in, "camera_link", 0, 800, 600)
    cal_out = cal3bundler_from_camera_calibration(cc)
    assert abs(cal_out.fx() - fx) < 1e-9
    assert abs(cal_out.px() - cx) < 1e-9
    assert abs(cal_out.py() - cy) < 1e-9
    assert abs(cal_out.k1() - k1) < 1e-9
    assert abs(cal_out.k2() - k2) < 1e-9


def test_camera_calibration_from_cal3bundler_structure():
    """CameraCalibration from Cal3Bundler has correct K, D, R, P structure."""
    fx, k1, k2, cx, cy = 600.0, 0.0, 0.0, 320.0, 240.0
    cal = gtsam.Cal3Bundler(fx, k1, k2, cx, cy)
    cc = camera_calibration_from_cal3bundler(cal, "camera_link", 0, 640, 480)

    assert cc.distortion_model == "plumb_bob"
    assert cc.width == 640
    assert cc.height == 480
    assert len(cc.K) == 9
    assert len(cc.D) == 5
    assert len(cc.R) == 9
    assert len(cc.P) == 12

    # K[0]=fx, K[2]=cx, K[5]=cy
    assert abs(cc.K[0] - fx) < 1e-9
    assert abs(cc.K[2] - cx) < 1e-9
    assert abs(cc.K[5] - cy) < 1e-9

    # R is identity
    expected_R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert np.allclose(list(cc.R), expected_R, atol=1e-12)
