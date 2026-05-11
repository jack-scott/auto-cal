"""
GTSAM ↔ Foxglove protobuf conversions.

Coordinate conventions (REP-103 / REP-105)
-------------------------------------------
Camera frame (ROS REP-103): X right, Y down, Z forward (into scene)
World frame (ENU / REP-103): X east, Y north, Z up

GTSAM Pose3(R, t):
  R = rotation()    → R_cw  (world→camera, rotates world vectors into camera frame)
  t = translation() → position of camera in world frame (numpy ndarray shape-(3,))

FrameTransform(parent, child, translation, rotation):
  translation = child origin in parent frame → same as Pose3.translation()
  rotation    = quaternion [x,y,z,w] that rotates child vectors into parent frame
              = R_wc = R_cw.inverse()

Therefore:
  Pose3 → FrameTransform: rotation = pose.rotation().inverse().toQuaternion()
  FrameTransform → Pose3: R_cw = R_wc.inverse(); build from rotation matrix

GTSAM quaternion API:
  Rot3.toQuaternion()  → gtsam.Quaternion  (.w(), .x(), .y(), .z())
  Rot3.inverse()       → Rot3
  Rot3(matrix_3x3)     → Rot3  (construct from numpy 3×3)

Cal3Bundler ↔ CameraCalibration:
  Cal3Bundler(fx, k1, k2, cx, cy) — only 3 DOF optimised: fx, k1, k2
  CameraCalibration.distortion_model must be "plumb_bob"
  D = [k1, k2, 0, 0, 0]
  K = [fx, 0, cx, 0, fx, cy, 0, 0, 1]  (row-major)
  R = identity (rectification)
  P = [fx, 0, cx, 0,  0, fx, cy, 0,  0, 0, 1, 0]  (3×4)
"""

from __future__ import annotations

import math

import gtsam
import numpy as np

from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from google.protobuf.timestamp_pb2 import Timestamp

from autocal.io.mcap_writer import ns_to_timestamp


def pose3_from_frame_transform(ft: FrameTransform) -> gtsam.Pose3:
    """Convert a FrameTransform proto to a GTSAM Pose3.

    The TF rotation (R_wc) is inverted to produce R_cw as expected by Pose3.

    Args:
        ft: FrameTransform message where translation = child origin in parent
            and rotation [x,y,z,w] = R_parent_child (= R_wc for map→camera_link).

    Returns:
        gtsam.Pose3 with rotation=R_cw and translation=camera position in world.
    """
    t = np.array([ft.translation.x, ft.translation.y, ft.translation.z])
    # Build R_wc from quaternion, then invert to get R_cw
    q = ft.rotation
    R_wc = _quat_xyzw_to_matrix(q.x, q.y, q.z, q.w)
    R_cw = gtsam.Rot3(R_wc).inverse()
    return gtsam.Pose3(R_cw, gtsam.Point3(*t))


def frame_transform_from_pose3(
    pose: gtsam.Pose3,
    parent_frame_id: str,
    child_frame_id: str,
    t_ns: int,
) -> FrameTransform:
    """Convert a GTSAM Pose3 to a FrameTransform proto.

    Args:
        pose:            GTSAM Pose3 (R_cw, translation = camera in world).
        parent_frame_id: e.g. "map"
        child_frame_id:  e.g. "camera_link"
        t_ns:            Timestamp in Unix nanoseconds.

    Returns:
        FrameTransform with translation = camera position in map,
        rotation [x,y,z,w] = R_wc (R_cw.inverse()).
    """
    ft = FrameTransform()
    ft.timestamp.CopyFrom(ns_to_timestamp(t_ns))
    ft.parent_frame_id = parent_frame_id
    ft.child_frame_id = child_frame_id

    t = pose.translation()
    ft.translation.x = float(t[0])
    ft.translation.y = float(t[1])
    ft.translation.z = float(t[2])

    # pose.rotation() = R_cw; TF wants R_wc = R_cw.inverse()
    q = pose.rotation().inverse().toQuaternion()
    ft.rotation.w = float(q.w())
    ft.rotation.x = float(q.x())
    ft.rotation.y = float(q.y())
    ft.rotation.z = float(q.z())
    return ft


def cal3ds2_from_camera_calibration(cc: CameraCalibration) -> gtsam.Cal3DS2:
    """Extract a Cal3DS2 from a CameraCalibration proto.

    Args:
        cc: CameraCalibration with distortion_model="plumb_bob",
            K (9 elements row-major), D (at least 2 elements).

    Returns:
        gtsam.Cal3DS2(fx, fy, skew, cx, cy, k1, k2, p1, p2).

    Raises:
        ValueError: If distortion_model is not "plumb_bob" or K is missing.
    """
    if cc.distortion_model and cc.distortion_model != "plumb_bob":
        raise ValueError(
            f"Unsupported distortion model {cc.distortion_model!r}; "
            "only 'plumb_bob' is supported"
        )
    if len(cc.K) < 9:
        raise ValueError(f"CameraCalibration.K must have 9 elements, got {len(cc.K)}")

    fx   = cc.K[0]
    fy   = cc.K[4]
    skew = cc.K[1]
    cx   = cc.K[2]
    cy   = cc.K[5]
    d    = list(cc.D)
    k1   = d[0] if len(d) > 0 else 0.0
    k2   = d[1] if len(d) > 1 else 0.0
    p1   = d[2] if len(d) > 2 else 0.0
    p2   = d[3] if len(d) > 3 else 0.0
    return gtsam.Cal3DS2(fx, fy, skew, cx, cy, k1, k2, p1, p2)


def camera_calibration_from_cal3ds2(
    cal: gtsam.Cal3DS2,
    frame_id: str,
    t_ns: int,
    width: int,
    height: int,
) -> CameraCalibration:
    """Build a CameraCalibration proto from a Cal3DS2.

    Args:
        cal:      GTSAM Cal3DS2 (fx, fy, skew, cx, cy, k1, k2, p1, p2).
        frame_id: e.g. "camera_link"
        t_ns:     Timestamp in Unix nanoseconds.
        width:    Image width in pixels.
        height:   Image height in pixels.

    Returns:
        CameraCalibration with plumb_bob distortion, K, R (identity), P filled.
    """
    fx   = cal.fx()
    fy   = cal.fy()
    cx   = cal.px()
    cy   = cal.py()
    k    = cal.k()   # [k1, k2, p1, p2]
    k1, k2, p1, p2 = float(k[0]), float(k[1]), float(k[2]), float(k[3])

    cc = CameraCalibration()
    cc.timestamp.CopyFrom(ns_to_timestamp(t_ns))
    cc.frame_id = frame_id
    cc.width = width
    cc.height = height
    cc.distortion_model = "plumb_bob"
    cc.D.extend([k1, k2, p1, p2, 0.0])         # k3 = 0
    cc.K.extend([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0])
    cc.R.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    cc.P.extend([fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0])
    return cc


def cal3bundler_from_camera_calibration(cc: CameraCalibration) -> gtsam.Cal3Bundler:
    """Extract a Cal3Bundler from a CameraCalibration proto.

    Args:
        cc: CameraCalibration with distortion_model="plumb_bob",
            K (9 elements row-major), D (at least 2 elements: k1, k2).

    Returns:
        gtsam.Cal3Bundler(fx, k1, k2, cx, cy).

    Raises:
        ValueError: If distortion_model is not "plumb_bob" or K/D are missing.
    """
    if cc.distortion_model and cc.distortion_model != "plumb_bob":
        raise ValueError(
            f"Unsupported distortion model {cc.distortion_model!r}; "
            "only 'plumb_bob' is supported"
        )
    if len(cc.K) < 9:
        raise ValueError(f"CameraCalibration.K must have 9 elements, got {len(cc.K)}")
    if len(cc.D) < 2:
        raise ValueError(f"CameraCalibration.D must have ≥2 elements, got {len(cc.D)}")

    fx = cc.K[0]
    cx = cc.K[2]
    cy = cc.K[5]
    k1 = cc.D[0]
    k2 = cc.D[1]
    return gtsam.Cal3Bundler(fx, k1, k2, cx, cy)


def camera_calibration_from_cal3bundler(
    cal: gtsam.Cal3Bundler,
    frame_id: str,
    t_ns: int,
    width: int,
    height: int,
) -> CameraCalibration:
    """Build a CameraCalibration proto from a Cal3Bundler.

    Args:
        cal:      GTSAM Cal3Bundler (fx, k1, k2, cx, cy).
        frame_id: e.g. "camera_link"
        t_ns:     Timestamp in Unix nanoseconds.
        width:    Image width in pixels.
        height:   Image height in pixels.

    Returns:
        CameraCalibration with plumb_bob distortion, K, R (identity), P filled.
    """
    fx = cal.fx()
    cx = cal.px()
    cy = cal.py()
    k1 = cal.k1()
    k2 = cal.k2()

    cc = CameraCalibration()
    cc.timestamp.CopyFrom(ns_to_timestamp(t_ns))
    cc.frame_id = frame_id
    cc.width = width
    cc.height = height
    cc.distortion_model = "plumb_bob"
    cc.D.extend([k1, k2, 0.0, 0.0, 0.0])
    cc.K.extend([fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0])
    cc.R.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    cc.P.extend([fx, 0.0, cx, 0.0, 0.0, fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0])
    return cc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _quat_xyzw_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert [x, y, z, w] unit quaternion to 3×3 rotation matrix."""
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)    ],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)    ],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])
