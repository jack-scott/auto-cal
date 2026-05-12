"""Unit tests for the SfM pipeline components.

Covers the full chain from pose convention → triangulation → GTSAM factor,
using known synthetic geometry so failures point to exactly which step broke.

Coordinate convention tested throughout:
  Pose3(R_wc, t)  — camera-to-world rotation, camera position in world.
"""

from __future__ import annotations

import gtsam
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose_looking_along_x(cam_pos: np.ndarray) -> gtsam.Pose3:
    """Camera at cam_pos with optical axis (cam Z) pointing along world X+.

    Camera axes in world:
      cam X (right) = world Y+
      cam Y (down)  = world -Z
      cam Z (fwd)   = world X+

    R_wc columns = camera axes in world:
      col 0 = [0, 1, 0]   (cam X = world Y)
      col 1 = [0, 0, -1]  (cam Y = world -Z)
      col 2 = [1, 0, 0]   (cam Z = world X)
    """
    R_wc = np.array([
        [0.0,  0.0, 1.0],
        [1.0,  0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    return gtsam.Pose3(gtsam.Rot3(R_wc), gtsam.Point3(*cam_pos))


def _pose_identity(cam_pos: np.ndarray) -> gtsam.Pose3:
    """Camera at cam_pos with identity rotation (cam Z = world Z, looking along world Z+)."""
    return gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(*cam_pos))


def _project_manual(pose: gtsam.Pose3, pt_world: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Manually project pt_world to pixel coords using Pose3(R_wc, t)."""
    R_cw = pose.rotation().matrix().T   # pose stores R_wc; .T = R_cw
    t = pose.translation()
    p_cam = R_cw @ (pt_world - t)
    assert p_cam[2] > 0, "point must be in front of camera"
    norm = p_cam[:2] / p_cam[2]
    return np.array([K[0, 0] * norm[0] + K[0, 2], K[1, 1] * norm[1] + K[1, 2]])


# ---------------------------------------------------------------------------
# 1. GTSAM convention: transform_to with R_wc stored
# ---------------------------------------------------------------------------

class TestGtsamConvention:
    """Verify that Pose3(R_wc, t) gives correct camera-frame coordinates via transformTo."""

    def test_identity_pose_transforms_correctly(self):
        """Camera at origin with identity rotation: point at (0,0,1) → cam Z."""
        pose = _pose_identity(np.array([0.0, 0.0, 0.0]))
        pt = np.array([0.0, 0.0, 1.0])
        p_cam = pose.transformTo(pt)
        assert np.allclose(p_cam, [0.0, 0.0, 1.0], atol=1e-12)

    def test_looking_along_x_transforms_correctly(self):
        """Camera at origin facing X+: point at (1,0,0) → cam Z direction."""
        pose = _pose_looking_along_x(np.array([0.0, 0.0, 0.0]))
        pt = np.array([1.0, 0.0, 0.0])
        p_cam = pose.transformTo(pt)
        # Point is 1m in front, on axis → cam coords = (0, 0, 1)
        assert np.allclose(p_cam, [0.0, 0.0, 1.0], atol=1e-12)

    def test_translated_camera_transforms_correctly(self):
        """Camera at (1,0,0) facing X+: point at (2,0,0) → cam coords (0,0,1)."""
        pose = _pose_looking_along_x(np.array([1.0, 0.0, 0.0]))
        pt = np.array([2.0, 0.0, 0.0])
        p_cam = pose.transformTo(pt)
        assert np.allclose(p_cam, [0.0, 0.0, 1.0], atol=1e-12)

    def test_off_axis_point_correct_sign(self):
        """Point to the right of camera (world Y+) → positive cam X."""
        pose = _pose_looking_along_x(np.array([0.0, 0.0, 0.0]))
        pt = np.array([2.0, 1.0, 0.0])  # 1m to the right in world Y
        p_cam = pose.transformTo(pt)
        # cam X = world Y, so world Y+1 → cam X+1; cam Y = -world Z = 0; cam Z = world X = 2
        assert np.allclose(p_cam, [1.0, 0.0, 2.0], atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Triangulation with non-identity rotation
# ---------------------------------------------------------------------------

class TestTriangulationWithRotation:
    """Verify triangulate() and triangulate_tracks() with non-trivial R_wc."""

    def test_triangulate_cameras_facing_x(self):
        """Two cameras side-by-side facing X+, triangulate a point 3m ahead."""
        from autocal.engine.features import triangulate

        K = np.array([[500.0, 0.0, 0.0],
                      [0.0, 500.0, 0.0],
                      [0.0, 0.0, 1.0]])
        pt_world = np.array([3.0, 0.0, 0.0])  # 3m ahead of both cameras

        pose_a = _pose_looking_along_x(np.array([0.0, -0.5, 0.0]))
        pose_b = _pose_looking_along_x(np.array([0.0,  0.5, 0.0]))

        kp_a = _project_manual(pose_a, pt_world, K)
        kp_b = _project_manual(pose_b, pt_world, K)

        pt3d = triangulate(kp_a, kp_b, K, pose_a, pose_b)
        assert pt3d is not None
        assert np.allclose(pt3d, pt_world, atol=1e-5)

    def test_triangulate_off_axis_point(self):
        """Triangulate a point that is not on the camera axis."""
        from autocal.engine.features import triangulate

        K = np.array([[600.0, 0.0, 0.0],
                      [0.0, 600.0, 0.0],
                      [0.0, 0.0, 1.0]])
        pt_world = np.array([4.0, 1.0, -0.5])

        pose_a = _pose_looking_along_x(np.array([0.0, -1.0, 0.0]))
        pose_b = _pose_looking_along_x(np.array([0.0,  1.0, 0.0]))

        kp_a = _project_manual(pose_a, pt_world, K)
        kp_b = _project_manual(pose_b, pt_world, K)

        pt3d = triangulate(kp_a, kp_b, K, pose_a, pose_b)
        assert pt3d is not None
        assert np.allclose(pt3d, pt_world, atol=1e-5)

    def test_triangulate_tracks_facing_x(self):
        """triangulate_tracks fills point3d correctly for cameras facing X+."""
        from autocal.engine.features import Track, triangulate_tracks

        K = np.array([[500.0, 0.0, 0.0],
                      [0.0, 500.0, 0.0],
                      [0.0, 0.0, 1.0]])
        pt_world = np.array([5.0, 0.5, -0.5])

        pose_a = _pose_looking_along_x(np.array([0.0, -0.5, 0.0]))
        pose_b = _pose_looking_along_x(np.array([0.0,  0.5, 0.0]))

        kp_a = _project_manual(pose_a, pt_world, K).astype(np.float32)
        kp_b = _project_manual(pose_b, pt_world, K).astype(np.float32)

        kps = {"a": np.array([kp_a]), "b": np.array([kp_b])}
        track = Track(observations={"a": 0, "b": 0})
        triangulate_tracks([track], kps, K, {"a": pose_a, "b": pose_b})

        assert track.point3d is not None
        assert np.allclose(track.point3d, pt_world, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. GenericProjectionFactorCal3DS2 with non-identity rotation
# ---------------------------------------------------------------------------

class TestGtsamProjectionFactor:
    """Verify GenericProjectionFactorCal3DS2 gives near-zero error
    when the pose, 3D point, and 2D measurement are mutually consistent."""

    def _make_cal(self, fx: float = 500.0, cx: float = 0.0, cy: float = 0.0) -> gtsam.Cal3DS2:
        return gtsam.Cal3DS2(fx, fx, 0.0, cx, cy, 0.0, 0.0, 0.0, 0.0)

    def test_zero_error_identity_pose(self):
        """Consistent (pose, point, measurement) → near-zero factor error."""
        cal = self._make_cal(fx=500.0, cx=320.0, cy=240.0)
        pt_world = np.array([0.0, 0.0, 5.0])
        pose = _pose_identity(np.array([0.0, 0.0, 0.0]))

        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        measured = _project_manual(pose, pt_world, K)

        X = gtsam.symbol_shorthand.X
        P = gtsam.symbol_shorthand.P
        noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

        graph = gtsam.NonlinearFactorGraph()
        vals = gtsam.Values()
        vals.insert(X(0), pose)
        vals.insert(P(0), gtsam.Point3(*pt_world))
        graph.add(gtsam.GenericProjectionFactorCal3DS2(measured, noise, X(0), P(0), cal))

        assert graph.error(vals) < 1e-8

    def test_zero_error_looking_along_x(self):
        """Same test but with camera facing X+ (non-identity R_wc)."""
        cal = self._make_cal(fx=500.0, cx=0.0, cy=0.0)
        pt_world = np.array([5.0, 0.0, 0.0])  # on the optical axis
        pose = _pose_looking_along_x(np.array([0.0, 0.0, 0.0]))

        K = np.array([[500.0, 0.0, 0.0], [0.0, 500.0, 0.0], [0.0, 0.0, 1.0]])
        measured = _project_manual(pose, pt_world, K)

        X = gtsam.symbol_shorthand.X
        P = gtsam.symbol_shorthand.P
        noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

        graph = gtsam.NonlinearFactorGraph()
        vals = gtsam.Values()
        vals.insert(X(0), pose)
        vals.insert(P(0), gtsam.Point3(*pt_world))
        graph.add(gtsam.GenericProjectionFactorCal3DS2(measured, noise, X(0), P(0), cal))

        assert graph.error(vals) < 1e-8

    def test_nonzero_error_wrong_point(self):
        """Deliberately wrong 3D point → non-zero factor error."""
        cal = self._make_cal(fx=500.0)
        pt_world = np.array([0.0, 0.0, 5.0])
        pt_wrong = np.array([1.0, 1.0, 5.0])  # shifted
        pose = _pose_identity(np.array([0.0, 0.0, 0.0]))

        K = np.array([[500.0, 0.0, 0.0], [0.0, 500.0, 0.0], [0.0, 0.0, 1.0]])
        measured = _project_manual(pose, pt_world, K)

        X = gtsam.symbol_shorthand.X
        P = gtsam.symbol_shorthand.P
        noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

        graph = gtsam.NonlinearFactorGraph()
        vals = gtsam.Values()
        vals.insert(X(0), pose)
        vals.insert(P(0), gtsam.Point3(*pt_wrong))
        graph.add(gtsam.GenericProjectionFactorCal3DS2(measured, noise, X(0), P(0), cal))

        assert graph.error(vals) > 1.0


# ---------------------------------------------------------------------------
# 4. Full round-trip: triangulate then project via GTSAM factor
# ---------------------------------------------------------------------------

class TestTriangulateToGtsamRoundtrip:
    """Triangulate 3D points then verify GTSAM factor error is near-zero."""

    def test_triangulated_point_has_small_gtsam_error(self):
        """Triangulate a point from two cameras, then check GTSAM factor error."""
        from autocal.engine.features import triangulate

        fx, cx, cy = 800.0, 400.0, 300.0
        K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])
        cal = gtsam.Cal3DS2(fx, fx, 0.0, cx, cy, 0.0, 0.0, 0.0, 0.0)

        pt_world = np.array([3.0, 0.5, -0.3])
        pose_a = _pose_looking_along_x(np.array([0.0, -0.5, 0.0]))
        pose_b = _pose_looking_along_x(np.array([0.0,  0.5, 0.0]))

        kp_a = _project_manual(pose_a, pt_world, K)
        kp_b = _project_manual(pose_b, pt_world, K)

        pt3d = triangulate(kp_a, kp_b, K, pose_a, pose_b)
        assert pt3d is not None

        X = gtsam.symbol_shorthand.X
        P = gtsam.symbol_shorthand.P
        noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.5)

        graph = gtsam.NonlinearFactorGraph()
        vals = gtsam.Values()
        vals.insert(X(0), pose_a)
        vals.insert(X(1), pose_b)
        vals.insert(P(0), gtsam.Point3(*pt3d))
        graph.add(gtsam.GenericProjectionFactorCal3DS2(kp_a, noise, X(0), P(0), cal))
        graph.add(gtsam.GenericProjectionFactorCal3DS2(kp_b, noise, X(1), P(0), cal))

        # With exact synthetic data and correct convention the error should be tiny
        assert graph.error(vals) < 1e-6
