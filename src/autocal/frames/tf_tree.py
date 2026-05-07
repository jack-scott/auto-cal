"""
TF tree — store and query FrameTransform messages.

Follows ROS REP-105 conventions:
  FrameTransform(parent, child, translation, rotation) encodes the pose of
  the child frame in the parent frame.  The quaternion is (x, y, z, w).

Usage::

    from autocal.frames.tf_tree import TFTree

    tree = TFTree()
    tree.add(frame_transform_msg, t_ns)

    tf = tree.lookup_transform("map", "camera_link", t_ns)
    # tf.translation  shape-(3,) [x, y, z]
    # tf.rotation     shape-(4,) [x, y, z, w]

    pt_map = tree.transform_point(pt_camera, "camera_link", "map", t_ns)

Interpolation:
  Translation: linear (LERP).
  Rotation: spherical linear (SLERP) — implemented without scipy.
  Static transforms (single entry per edge) are returned as-is at any t_ns.

Errors:
  KeyError   — edge (parent, child) is unknown.
  ValueError — t_ns is outside the buffered time range for that edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Transform:
    """A stamped rigid-body transform."""
    t_ns: int
    translation: np.ndarray   # shape (3,) [x, y, z]
    rotation: np.ndarray      # shape (4,) [x, y, z, w]  (Hamilton/ROS convention)


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions.

    Both quaternions are [x, y, z, w].  Returns a normalised quaternion.
    Handles the double-cover ambiguity (q and -q represent the same rotation).
    """
    # Ensure shortest path
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    theta = math.acos(abs(dot))
    if theta < 1e-10:
        # Quaternions are nearly identical — use LERP
        q = q0 + alpha * (q1 - q0)
    else:
        s = math.sin(theta)
        q = (math.sin((1 - alpha) * theta) / s) * q0 + (math.sin(alpha * theta) / s) * q1
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-12 else q0


class TFTree:
    """Stores and interpolates FrameTransform messages.

    Transforms are keyed by (parent_frame_id, child_frame_id).
    Multiple timestamped entries per edge enable temporal interpolation.
    """

    def __init__(self) -> None:
        # (parent, child) → sorted list of Transform
        self._edges: dict[tuple[str, str], list[Transform]] = {}

    def add(self, msg: Any, t_ns: int) -> None:
        """Ingest one FrameTransform message.

        Args:
            msg:  foxglove.FrameTransform proto instance.
            t_ns: Log timestamp in Unix nanoseconds.
        """
        key = (msg.parent_frame_id, msg.child_frame_id)
        tr = Transform(
            t_ns=t_ns,
            translation=np.array([msg.translation.x, msg.translation.y, msg.translation.z]),
            rotation=np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]),
        )
        if key not in self._edges:
            self._edges[key] = []
        self._edges[key].append(tr)
        # Keep sorted by timestamp for binary-search interpolation
        self._edges[key].sort(key=lambda x: x.t_ns)

    def known_frames(self) -> set[str]:
        """Return all frame IDs that appear in any edge."""
        frames: set[str] = set()
        for parent, child in self._edges:
            frames.add(parent)
            frames.add(child)
        return frames

    def lookup_transform(
        self, parent: str, child: str, t_ns: int
    ) -> Transform:
        """Return the interpolated transform from parent to child at t_ns.

        For a static transform (single entry) the same value is returned
        regardless of t_ns.

        Args:
            parent: Parent frame ID.
            child:  Child frame ID.
            t_ns:   Query time in Unix nanoseconds.

        Returns:
            Interpolated Transform at the requested time.

        Raises:
            KeyError:   Edge (parent, child) has not been added.
            ValueError: t_ns is outside the stored time range (not extrapolating).
        """
        key = (parent, child)
        if key not in self._edges:
            raise KeyError(f"No transform edge ({parent!r} → {child!r}) in TFTree")

        tfs = self._edges[key]

        # Static transform — single entry, return regardless of t_ns
        if len(tfs) == 1:
            return tfs[0]

        t0 = tfs[0].t_ns
        t1 = tfs[-1].t_ns
        if t_ns < t0 or t_ns > t1:
            raise ValueError(
                f"t_ns={t_ns} outside buffered range [{t0}, {t1}] "
                f"for edge ({parent!r} → {child!r})"
            )

        # Binary search for bracket
        lo, hi = 0, len(tfs) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if tfs[mid].t_ns <= t_ns:
                lo = mid
            else:
                hi = mid

        a = tfs[lo]
        b = tfs[hi]
        if a.t_ns == b.t_ns:
            return a

        alpha = (t_ns - a.t_ns) / (b.t_ns - a.t_ns)
        return Transform(
            t_ns=t_ns,
            translation=a.translation + alpha * (b.translation - a.translation),
            rotation=_slerp(a.rotation, b.rotation, alpha),
        )

    def transform_point(
        self, pt: np.ndarray, from_frame: str, to_frame: str, t_ns: int
    ) -> np.ndarray:
        """Transform a 3D point from one frame to another.

        Currently supports only direct edges (no multi-hop chaining).
        For single-hop: if (to_frame → from_frame) edge exists, uses its
        inverse; if (from_frame → to_frame) edge exists, uses it directly.

        Args:
            pt:         shape-(3,) point in from_frame.
            from_frame: Source frame ID.
            to_frame:   Destination frame ID.
            t_ns:       Query time in Unix nanoseconds.

        Returns:
            shape-(3,) point in to_frame.

        Raises:
            KeyError: No edge between the two frames.
        """
        # Direct edge: (to_frame → from_frame) means from_frame is the child.
        # TF convention: translation = child origin in parent, rotation = R_parent_child.
        # So: pt_parent = R @ pt_child + translation
        if (to_frame, from_frame) in self._edges:
            tf = self.lookup_transform(to_frame, from_frame, t_ns)
            R = _quat_to_matrix(tf.rotation)
            return R @ pt + tf.translation
        elif (from_frame, to_frame) in self._edges:
            # Inverse: pt_child = R.T @ (pt_parent - translation)
            tf = self.lookup_transform(from_frame, to_frame, t_ns)
            R = _quat_to_matrix(tf.rotation)
            return R.T @ (pt - tf.translation)
        else:
            raise KeyError(
                f"No edge between {from_frame!r} and {to_frame!r} in TFTree"
            )


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, w] unit quaternion to 3×3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)    ],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)    ],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])
