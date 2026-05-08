"""
3D scene visualization for Foxglove Studio.

Writes camera-frustum markers and path lines to a McapWriter using the
foxglove.SceneUpdate schema.  Each entity carries lifetime=(0,0) so it
persists indefinitely once published — scrubbing forward builds up the
path, and jumping to the end shows the complete trajectory.

Coordinate frame: all entities are expressed in the "map" frame (ENU /
REP-103: X east, Y north, Z up).

Camera frame (REP-103): X right, Y down, Z forward (into scene).
"""

from __future__ import annotations

import numpy as np
import gtsam

from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive

from autocal.io.mcap_writer import McapWriter, ns_to_timestamp

# Frustum geometry constants
_FRUSTUM_DEPTH = 0.4   # metres — near-plane depth for frustum lines
_SPHERE_RADIUS = 0.08  # metres — camera-origin sphere

# Default frustum half-angle (radians) — used when no calibration is available
_DEFAULT_HALF_ANGLE = 0.35


def write_camera_path(
    writer: McapWriter,
    poses: dict[int, gtsam.Pose3],
    topic: str,
    r: float,
    g: float,
    b: float,
    cal: gtsam.Cal3Bundler | None = None,
    frame_id: str = "map",
) -> None:
    """Write camera frustum markers and path lines to an MCAP writer.

    For each camera in timestamp order a SceneUpdate is written at the matching
    timestamp containing two entities:
      1. A frustum marker (sphere at origin + 8 LINE_LIST segments).
      2. A path segment connecting this camera to the previous one (LINE_STRIP).

    All entities have infinite lifetime (seconds=0, nanos=0) so they accumulate
    as playback advances.

    Args:
        writer:   Open McapWriter to write into.
        poses:    Mapping of Unix-nanosecond timestamp → GTSAM Pose3 (R_cw, t=cam in world).
        topic:    MCAP topic string, e.g. "/scene/cameras/optimized".
        r,g,b:    RGB colour components in [0, 1].
        cal:      Optional Cal3Bundler used to shape the frustum.  If None a
                  default half-angle is used.
        frame_id: Parent frame for all SceneEntity messages.
    """
    if not poses:
        return

    sorted_ts = sorted(poses)

    # Frustum shape in camera frame (REP-103: X right, Y down, Z forward)
    if cal is not None:
        hw = _FRUSTUM_DEPTH * cal.px() / cal.fx()
        hh = _FRUSTUM_DEPTH * cal.py() / cal.fx()
    else:
        hw = hh = _FRUSTUM_DEPTH * np.tan(_DEFAULT_HALF_ANGLE)

    # 4 near-plane corners in camera frame
    corners_cam = np.array([
        [-hw, -hh, _FRUSTUM_DEPTH],
        [ hw, -hh, _FRUSTUM_DEPTH],
        [ hw,  hh, _FRUSTUM_DEPTH],
        [-hw,  hh, _FRUSTUM_DEPTH],
    ])

    prev_pos: np.ndarray | None = None

    for t_ns in sorted_ts:
        pose = poses[t_ns]
        t_cam = pose.translation()          # camera position in world (ENU)
        R_wc = pose.rotation().inverse().matrix()  # rotates camera→world

        # Transform frustum corners to world frame
        corners_world = (R_wc @ corners_cam.T).T + t_cam

        su = SceneUpdate()

        # ------------------------------------------------------------------ #
        # Entity 1: frustum marker
        # ------------------------------------------------------------------ #
        entity_id = f"cam_{t_ns}"
        e_frustum = su.entities.add()
        e_frustum.id = entity_id
        e_frustum.frame_id = frame_id
        e_frustum.timestamp.CopyFrom(ns_to_timestamp(t_ns))
        # lifetime = (0,0) → infinite persistence
        e_frustum.lifetime.seconds = 0
        e_frustum.lifetime.nanos = 0

        # Sphere at camera origin
        sp = e_frustum.spheres.add()
        sp.pose.position.x = float(t_cam[0])
        sp.pose.position.y = float(t_cam[1])
        sp.pose.position.z = float(t_cam[2])
        sp.pose.orientation.w = 1.0
        sp.size.x = _SPHERE_RADIUS * 2
        sp.size.y = _SPHERE_RADIUS * 2
        sp.size.z = _SPHERE_RADIUS * 2
        sp.color.r = r
        sp.color.g = g
        sp.color.b = b
        sp.color.a = 1.0

        # Frustum lines: 4 rays from center to corners + 4 outline segments
        # LINE_LIST: points come in pairs, each pair = one segment
        line = e_frustum.lines.add()
        line.type = LinePrimitive.LINE_LIST
        line.thickness = 0.02
        line.pose.orientation.w = 1.0
        line.color.r = r
        line.color.g = g
        line.color.b = b
        line.color.a = 0.8

        # 4 rays: center→corner
        for i in range(4):
            _add_point(line, t_cam)
            _add_point(line, corners_world[i])

        # 4 outline edges: TL→TR→BR→BL→TL
        for i in range(4):
            _add_point(line, corners_world[i])
            _add_point(line, corners_world[(i + 1) % 4])

        # ------------------------------------------------------------------ #
        # Entity 2: path segment from previous camera to this one
        # ------------------------------------------------------------------ #
        if prev_pos is not None:
            e_path = su.entities.add()
            e_path.id = f"path_{t_ns}"
            e_path.frame_id = frame_id
            e_path.timestamp.CopyFrom(ns_to_timestamp(t_ns))
            e_path.lifetime.seconds = 0
            e_path.lifetime.nanos = 0

            seg = e_path.lines.add()
            seg.type = LinePrimitive.LINE_STRIP
            seg.thickness = 0.03
            seg.pose.orientation.w = 1.0
            seg.color.r = r
            seg.color.g = g
            seg.color.b = b
            seg.color.a = 0.6
            _add_point(seg, prev_pos)
            _add_point(seg, t_cam)

        prev_pos = t_cam.copy()

        writer.write(topic, su, t_ns)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_point(line: LinePrimitive, xyz: np.ndarray) -> None:
    pt = line.points.add()
    pt.x = float(xyz[0])
    pt.y = float(xyz[1])
    pt.z = float(xyz[2])
