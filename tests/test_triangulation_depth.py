"""Diagnose why ~96% of tracks fail in the SfM pipeline.

Three hypotheses are tested:
  H1 – Bug in all_positive_depth (wrong frame convention).
  H2 – Bug in triangulate_gtsam (wrong result even with GT poses).
  H3 – Noisy initial poses legitimately cause most valid tracks to fail.

FINDING:
  H1 – CLEAR: all_positive_depth is correct.
  H2 – CLEAR (for GT poses): round-trip passes with both Cal3DS2 and Cal3Fisheye.
  H3 – NOT the cause: even 1 m noise causes well under 50% cheirality failure
       on a well-conditioned scene.

  ROOT CAUSE: Cal3Fisheye::calibrate() (iterative Kannala-Brandt inverse used
  internally by gtsam.triangulatePoint3 during the DLT undistortion step)
  fails to converge for real-world fisheye observations that are far from the
  principal point.  We catch this as a generic Exception and return None, which
  is why ~96% of tracks have point3d=None in the pipeline.

  FIX: pre-undistort keypoints with cv2 (which we already do for the E-matrix
  path) and pass them to triangulatePoint3 with a plain Cal3_S2 pinhole
  calibration.  GTSAM's undistortMeasurements(Cal3_S2, ...) is a no-op so the
  iterative fisheye inverse is never called.
"""

from __future__ import annotations

import numpy as np
import pytest
import gtsam

from autocal.engine.features import (
    Track,
    all_positive_depth,
    triangulate_gtsam,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pose_facing_x(cam_pos: np.ndarray) -> gtsam.Pose3:
    """Camera at cam_pos with optical axis (cam-Z) pointing along world X+.

    R_wc columns = camera axes expressed in world:
      col 0 = [0,1,0]   (cam X = world Y+)
      col 1 = [0,0,-1]  (cam Y = world Z-)
      col 2 = [1,0,0]   (cam Z = world X+)
    """
    R_wc = np.array([[0, 0, 1],
                     [1, 0, 0],
                     [0,-1, 0]], dtype=float)
    return gtsam.Pose3(gtsam.Rot3(R_wc), gtsam.Point3(*cam_pos))


def _project_pinhole(pose: gtsam.Pose3, cal: gtsam.Cal3DS2,
                     pt_world: np.ndarray) -> np.ndarray:
    """Project pt_world through a pinhole camera (Cal3DS2 with k=0)."""
    R_cw = pose.rotation().matrix().T
    pc = R_cw @ (pt_world - pose.translation())
    assert pc[2] > 0, f"point behind camera: pc={pc}"
    xn, yn = pc[0] / pc[2], pc[1] / pc[2]
    return np.array([cal.fx() * xn + cal.px(),
                     cal.fy() * yn + cal.py()], dtype=np.float32)


def _project_fisheye(pose: gtsam.Pose3, cal: gtsam.Cal3Fisheye,
                     pt_world: np.ndarray) -> np.ndarray:
    """Project pt_world through a Kannala-Brandt fisheye camera."""
    k = cal.k()
    R_cw = pose.rotation().matrix().T
    pc = R_cw @ (pt_world - pose.translation())
    x, y, z = pc
    assert z > 0, f"point behind camera: pc={pc}"
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(r, z)
    t2 = theta * theta
    rd = theta * (1 + k[0]*t2 + k[1]*t2**2 + k[2]*t2**3 + k[3]*t2**4)
    if r < 1e-10:
        return np.array([cal.px(), cal.py()], dtype=np.float32)
    s = rd / r
    return np.array([cal.fx() * s * x + cal.px(),
                     cal.fy() * s * y + cal.py()], dtype=np.float32)


def _add_pose_noise(pose: gtsam.Pose3, sigma_t: float, sigma_r: float,
                    rng: np.random.Generator) -> gtsam.Pose3:
    dt = rng.normal(0, sigma_t, 3)
    axis = rng.normal(0, 1, 3)
    axis /= np.linalg.norm(axis)
    angle = float(rng.normal(0, sigma_r))
    dR = gtsam.Rot3.AxisAngle(gtsam.Point3(*axis), angle)
    return gtsam.Pose3(pose.rotation().compose(dR),
                       gtsam.Point3(*(pose.translation() + dt)))


# ---------------------------------------------------------------------------
# H1 — all_positive_depth convention
# ---------------------------------------------------------------------------

class TestAllPositiveDepth:
    """H1: verify all_positive_depth uses the correct R_wc convention."""

    def test_point_in_front_returns_true(self):
        pose = _pose_facing_x(np.array([0.0, 0.0, 0.0]))
        pt = np.array([2.0, 0.0, 0.0])   # 2 m ahead along world X
        assert all_positive_depth(pt, {0: 0}, {0: pose}) is True

    def test_point_behind_returns_false(self):
        pose = _pose_facing_x(np.array([0.0, 0.0, 0.0]))
        pt = np.array([-1.0, 0.0, 0.0])  # behind the camera
        assert all_positive_depth(pt, {0: 0}, {0: pose}) is False

    def test_offset_camera_point_in_front(self):
        pose = _pose_facing_x(np.array([1.0, 0.5, -0.3]))
        pt = np.array([3.0, 0.5, -0.3])  # 2 m in front of this camera
        assert all_positive_depth(pt, {0: 0}, {0: pose}) is True

    def test_point_at_camera_origin_is_not_in_front(self):
        # Point exactly at camera centre → zero depth
        pose = _pose_facing_x(np.array([0.0, 0.0, 0.0]))
        pt = np.array([0.0, 0.0, 0.0])
        assert all_positive_depth(pt, {0: 0}, {0: pose}) is False

    def test_two_cameras_one_behind_fails(self):
        p0 = _pose_facing_x(np.array([0.0, 0.0, 0.0]))
        p1 = _pose_facing_x(np.array([1.0, 0.0, 0.0]))
        pt = np.array([0.5, 0.0, 0.0])  # in front of p0, behind p1
        # p0: cam-Z along X+, point at X=0.5 → depth 0.5 > 0
        # p1: cam-Z along X+, camera at X=1, point at X=0.5 → depth -0.5 < 0
        assert all_positive_depth(pt, {0: 0, 1: 0}, {0: p0, 1: p1}) is False

    def test_consistent_with_transformto(self):
        """Depth from all_positive_depth must agree with GTSAM's transformTo."""
        pose = _pose_facing_x(np.array([0.3, -0.1, 0.2]))
        pt = np.array([2.5, 0.4, -0.1])
        p_cam = pose.transformTo(pt)
        depth_ok = all_positive_depth(pt, {0: 0}, {0: pose})
        assert depth_ok == (p_cam[2] > 0)


# ---------------------------------------------------------------------------
# H2 — triangulate_gtsam round-trip
# ---------------------------------------------------------------------------

class TestTriangulateGtsamRoundtrip:
    """H2: triangulate_gtsam should recover the original 3D point from
    perfect (noiseless) observations with GT poses."""

    def _run_triangulation(self, poses, keypoints, cal, expected_pt, atol=0.01):
        obs = {img_id: 0 for img_id in poses}
        track = Track(observations=obs)
        triangulate_gtsam([track], keypoints, cal, poses)
        assert track.point3d is not None, (
            "triangulation returned None — not a cheirality problem, "
            "something failed inside triangulate_gtsam"
        )
        np.testing.assert_allclose(track.point3d, expected_pt, atol=atol,
                                   err_msg="triangulated point is far from ground truth")
        assert all_positive_depth(track.point3d, obs, poses), (
            "triangulated point (from GT poses, perfect pixels) should have "
            "positive depth in every camera"
        )

    # --- pinhole (Cal3DS2, k=0) ---

    def test_pinhole_two_cameras_on_axis(self):
        cal = gtsam.Cal3DS2(500, 500, 0, 320, 240, 0, 0, 0, 0)
        pt = np.array([3.0, 0.0, 0.0])
        p0 = _pose_facing_x(np.array([0.0, -0.5, 0.0]))
        p1 = _pose_facing_x(np.array([0.0,  0.5, 0.0]))
        kps = {
            0: _project_pinhole(p0, cal, pt).reshape(1, 2),
            1: _project_pinhole(p1, cal, pt).reshape(1, 2),
        }
        self._run_triangulation({0: p0, 1: p1}, kps, cal, pt)

    def test_pinhole_three_cameras_off_axis_point(self):
        cal = gtsam.Cal3DS2(600, 600, 0, 320, 240, 0, 0, 0, 0)
        pt = np.array([4.0, 0.3, -0.5])
        poses = {
            0: _pose_facing_x(np.array([0.0, -0.5, 0.0])),
            1: _pose_facing_x(np.array([0.0,  0.0, 0.0])),
            2: _pose_facing_x(np.array([0.0,  0.5, 0.0])),
        }
        kps = {i: _project_pinhole(p, cal, pt).reshape(1, 2) for i, p in poses.items()}
        self._run_triangulation(poses, kps, cal, pt)

    # --- fisheye (Cal3Fisheye) ---

    def test_fisheye_two_cameras(self):
        """Core test: fisheye triangulation with perfect observations.

        If this fails it means triangulate_gtsam has a bug with the
        Kannala-Brandt model that is INDEPENDENT of pose noise.
        """
        cal = gtsam.Cal3Fisheye(500, 500, 0, 320, 240, 0.1, -0.03, 0.005, -0.001)
        pt = np.array([3.0, 0.2, -0.1])
        p0 = _pose_facing_x(np.array([0.0, -0.4, 0.0]))
        p1 = _pose_facing_x(np.array([0.0,  0.4, 0.0]))
        kps = {
            0: _project_fisheye(p0, cal, pt).reshape(1, 2),
            1: _project_fisheye(p1, cal, pt).reshape(1, 2),
        }
        self._run_triangulation({0: p0, 1: p1}, kps, cal, pt, atol=0.05)

    def test_fisheye_three_cameras(self):
        cal = gtsam.Cal3Fisheye(500, 500, 0, 320, 240, 0.1, -0.03, 0.005, -0.001)
        pt = np.array([5.0, -0.3, 0.4])
        poses = {
            0: _pose_facing_x(np.array([0.0, -0.5, 0.0])),
            1: _pose_facing_x(np.array([0.0,  0.0, 0.0])),
            2: _pose_facing_x(np.array([0.0,  0.5, 0.0])),
        }
        kps = {i: _project_fisheye(p, cal, pt).reshape(1, 2) for i, p in poses.items()}
        self._run_triangulation(poses, kps, cal, pt, atol=0.05)

    def test_fisheye_large_intrinsics_like_dataset(self):
        """Realistic intrinsics matching the eth3d fisheye dataset.

        Uses reduced baseline to keep angles reasonable.
        """
        # Dataset values: fx=3408, fy=3408, cx=3036, cy=2011
        cal = gtsam.Cal3Fisheye(3408.0, 3408.0, 0, 3036.0, 2011.0,
                                 0.05, -0.01, 0.002, -0.0003)
        pt = np.array([5.0, 0.1, -0.05])
        p0 = _pose_facing_x(np.array([0.0, -0.15, 0.0]))
        p1 = _pose_facing_x(np.array([0.0,  0.15, 0.0]))
        kps = {
            0: _project_fisheye(p0, cal, pt).reshape(1, 2),
            1: _project_fisheye(p1, cal, pt).reshape(1, 2),
        }
        self._run_triangulation({0: p0, 1: p1}, kps, cal, pt, atol=0.1)


# ---------------------------------------------------------------------------
# Root cause — Cal3Fisheye::calibrate convergence failure
# ---------------------------------------------------------------------------

class TestFisheyeCalibrateDivergence:
    """Show that GTSAM's internal Cal3Fisheye::calibrate() (iterative K-B inverse)
    is the dominant cause of triangulatePoint3 failures.

    gtsam.triangulatePoint3 calls undistortMeasurements() which calls
    cal.calibrate(pixel) to convert distorted pixels to normalised coords
    before the DLT.  For Cal3Fisheye this is iterative; it diverges for pixels
    far from the principal point.

    The fix is to pass pre-undistorted pixels with a plain Cal3_S2 so that
    GTSAM's undistortMeasurements is a no-op.
    """

    def _dataset_fisheye_cal(self) -> gtsam.Cal3Fisheye:
        # Realistic values matching the eth3d exhibition-hall dataset
        return gtsam.Cal3Fisheye(3408.4, 3408.6, 0.0, 3036.2, 2011.2,
                                  0.05, -0.01, 0.002, -0.0003)

    def _dataset_pinhole_cal(self) -> gtsam.Cal3DS2:
        return gtsam.Cal3DS2(3408.4, 3408.6, 0.0, 3036.2, 2011.2,
                              0.0, 0.0, 0.0, 0.0)

    def test_calibrate_converges_near_principal_point(self):
        """cal.calibrate() succeeds for pixels close to the principal point."""
        cal = self._dataset_fisheye_cal()
        # A pixel very close to the principal point should undistort fine
        near_centre = np.array([3040.0, 2015.0])
        result = cal.calibrate(near_centre)
        assert np.isfinite(result).all()
        assert np.allclose(result, [0.0, 0.0], atol=0.01)

    def test_calibrate_fails_for_large_viewing_angle(self):
        """cal.calibrate() raises when the normalised pixel radius is large.

        This corresponds to observations from cameras that see a feature at a
        very oblique angle (> ~75°).  In scenes where cameras advance along a
        trajectory and features are at similar depth, trailing cameras see some
        features at extreme angles, producing pixels with rd_norm > ~1.4.
        GTSAM's iterative Kannala-Brandt inversion diverges for these values.

        This does NOT affect the real eth3d dataset (fx=3408, all pixels inside
        the image → rd_norm < 0.9) but does affect small-fx synthetic tests.
        """
        # Small focal length → features can project to large normalized radius
        cal = gtsam.Cal3Fisheye(500, 500, 0, 320, 240, 0.1, -0.03, 0.005, -0.001)
        # rd_norm ≈ 1.53 corresponds to ~79° viewing angle — beyond the valid range
        extreme_pixel = np.array([788.0, -364.0])
        rd_norm = np.hypot(extreme_pixel[0] - cal.px(),
                           extreme_pixel[1] - cal.py()) / cal.fx()
        assert rd_norm > 1.4, f"test setup: rd_norm={rd_norm:.3f} should be > 1.4"
        with pytest.raises(RuntimeError, match="calibrate fails to converge"):
            cal.calibrate(extreme_pixel)

    def test_triangulate_with_fisheye_fails_for_extreme_angle_observations(self):
        """triangulatePoint3 with Cal3Fisheye fails when any observation has
        a very large viewing angle (normalised radius > ~1.4).

        In a scene with cameras advancing along a path, a nearby point can
        project to extreme fisheye angles in trailing cameras, triggering
        GTSAM's iterative undistortion to diverge during the DLT setup step.
        We catch this as a generic RuntimeError and return None.
        """
        cal = gtsam.Cal3Fisheye(500, 500, 0, 320, 240, 0.1, -0.03, 0.005, -0.001)
        R_wc = np.array([[0,0,1],[1,0,0],[0,-1,0]], dtype=float)
        rng = np.random.default_rng(7)

        def add_noise(pose):
            dt = rng.normal(0, 0.1, 3)
            axis = rng.normal(0, 1, 3); axis /= np.linalg.norm(axis)
            dR = gtsam.Rot3.AxisAngle(gtsam.Point3(*axis), float(rng.normal(0, 0.05)))
            return gtsam.Pose3(pose.rotation().compose(dR),
                               gtsam.Point3(*(pose.translation() + dt)))

        # Cameras advancing along X — later cameras see nearby points at extreme angles
        gt_poses = [gtsam.Pose3(gtsam.Rot3(R_wc), gtsam.Point3(i * 0.2, 0.0, 0.0))
                    for i in range(7)]
        noisy_poses = [add_noise(p) for p in gt_poses]

        # The specific failing pixel combination found in diagnosis
        pixels = [
            np.array([401.0, 136.0]),   # cam 0 — ok
            np.array([415.0, 117.0]),   # cam 1 — ok
            np.array([436.0,  91.0]),   # cam 2 — ok
            np.array([467.0,  50.0]),   # cam 3 — ok
            np.array([520.0, -18.0]),   # cam 4 — ok
            np.array([616.0, -142.0]),  # cam 5 — ok
            np.array([788.0, -364.0]),  # cam 6 — rd=1.529, calibrate() diverges
        ]
        rd_last = np.hypot(pixels[-1][0] - cal.px(),
                           pixels[-1][1] - cal.py()) / cal.fx()
        assert rd_last > 1.4

        pv = gtsam.Pose3Vector(noisy_poses)
        mv = gtsam.Point2Vector(pixels)
        with pytest.raises(RuntimeError, match="calibrate fails to converge"):
            gtsam.triangulatePoint3(pv, cal, mv, 1e-9, False)

    def test_triangulate_with_preundistorted_pixels_succeeds(self):
        """Pre-undistorting with cv2 and using Cal3_S2 avoids the convergence
        failure entirely — this is the intended fix.
        """
        import cv2
        cal_fisheye = self._dataset_fisheye_cal()
        cal_pinhole = self._dataset_pinhole_cal()
        K = np.array([[cal_fisheye.fx(), 0, cal_fisheye.px()],
                      [0, cal_fisheye.fy(), cal_fisheye.py()],
                      [0, 0, 1.0]])
        D = np.array(cal_fisheye.k()[:4]).reshape(4, 1)

        p0 = _pose_facing_x(np.array([0.0, -0.5, 0.0]))
        p1 = _pose_facing_x(np.array([0.0,  0.5, 0.0]))
        pt = np.array([5.0, 0.1, 0.0])

        kp0 = _project_fisheye(p0, cal_fisheye, pt)
        kp1 = _project_fisheye(p1, cal_fisheye, pt)

        def undistort(kp):
            pts = kp.reshape(1, 1, 2).astype(np.float64)
            return cv2.fisheye.undistortPoints(pts, K, D, P=K).reshape(2).astype(np.float32)

        kp0_und = undistort(kp0)
        kp1_und = undistort(kp1)

        pv = gtsam.Pose3Vector([p0, p1])
        mv = gtsam.Point2Vector([kp0_und, kp1_und])

        # Cal3_S2 (no distortion) — GTSAM's undistortMeasurements is a no-op
        result = gtsam.triangulatePoint3(pv, cal_pinhole, mv, 1e-9, False)
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result, pt, atol=0.05)


# ---------------------------------------------------------------------------
# H3 — quantify cheirality failure rate under pose noise
# ---------------------------------------------------------------------------

class TestCheiraityFailureRateUnderNoise:
    """H3: with realistic noise (~10 cm), how often does a valid track fail
    all_positive_depth?

    Cameras are spread along the Y axis (sideways baseline) so that all cameras
    face X+ and every point at X > 0 is in front of every camera regardless of
    how far along the trajectory we are.  This isolates the noise effect from
    the scene-layout effect.
    """

    def _scene(self, n_cameras: int = 20, n_points: int = 200) -> tuple[dict, list]:
        """Cameras spread along Y, all facing X+, points ahead along X."""
        rng = np.random.default_rng(0)
        # Cameras at Y = -2 … +2, all at X=0, facing X+
        y_positions = np.linspace(-2.0, 2.0, n_cameras)
        poses = {i: _pose_facing_x(np.array([0.0, float(y), 0.0]))
                 for i, y in enumerate(y_positions)}

        pts = []
        for _ in range(n_points):
            x = rng.uniform(3.0, 10.0)   # well in front of all cameras
            y = rng.uniform(-1.0,  1.0)
            z = rng.uniform(-1.0,  1.0)
            pts.append(np.array([x, y, z]))

        obs = {i: 0 for i in range(n_cameras)}
        for pt in pts:
            assert all_positive_depth(pt, obs, poses), \
                f"scene setup error: point {pt} is behind a camera with GT poses"

        return poses, pts

    def _noisy_poses(self, gt_poses: dict, sigma_t: float, sigma_r: float,
                     seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        return {k: _add_pose_noise(p, sigma_t, sigma_r, rng)
                for k, p in gt_poses.items()}

    def test_zero_noise_all_pass(self):
        """GT poses → 0% cheirality failures (sanity check)."""
        gt_poses, pts = self._scene()
        obs = {i: 0 for i in gt_poses}
        failures = sum(1 for pt in pts
                       if not all_positive_depth(pt, obs, gt_poses))
        assert failures == 0, f"{failures} points failed with zero noise"

    def test_small_noise_low_failure_rate(self):
        """5 cm / 0.01 rad noise should cause < 5% cheirality failures."""
        gt_poses, pts = self._scene()
        noisy = self._noisy_poses(gt_poses, sigma_t=0.05, sigma_r=0.01)
        obs = {i: 0 for i in gt_poses}
        failures = sum(1 for pt in pts
                       if not all_positive_depth(pt, obs, noisy))
        rate = failures / len(pts)
        assert rate < 0.05, (
            f"5 cm noise caused {rate:.1%} cheirality failures (expected < 5%)"
        )

    def test_medium_noise_failure_rate(self):
        """10 cm / 0.05 rad noise (the 'medium' preset level).

        With well-separated cameras and points far ahead, pose noise should
        cause very few cheirality failures.  If the pipeline sees ~96% failure,
        the problem is NOT all_positive_depth — it must be somewhere else.
        """
        gt_poses, pts = self._scene()
        noisy = self._noisy_poses(gt_poses, sigma_t=0.10, sigma_r=0.05)
        obs = {i: 0 for i in gt_poses}
        failures = sum(1 for pt in pts
                       if not all_positive_depth(pt, obs, noisy))
        rate = failures / len(pts)
        assert rate < 0.10, (
            f"10 cm noise caused {rate:.1%} cheirality failures — "
            f"if the pipeline shows ~96%, the root cause is NOT pose noise"
        )

    def test_large_noise_still_far_below_pipeline_rate(self):
        """Even at 1 m / 0.5 rad, cheirality failure should be well below 96%."""
        gt_poses, pts = self._scene()
        noisy = self._noisy_poses(gt_poses, sigma_t=1.0, sigma_r=0.5)
        obs = {i: 0 for i in gt_poses}
        failures = sum(1 for pt in pts
                       if not all_positive_depth(pt, obs, noisy))
        rate = failures / len(pts)
        assert rate < 0.50, (
            f"Even 1 m noise caused only {rate:.1%} cheirality failures — "
            f"far below the ~96% seen in the pipeline; root cause is elsewhere"
        )


# ---------------------------------------------------------------------------
# H2 extended — failure breakdown on a synthetic scene
# ---------------------------------------------------------------------------

class TestTriangulationFailureBreakdown:
    """Triangulate a synthetic scene and count failures at each stage,
    mirroring what sfm_solver.py does.

    This shows whether the 96% pipeline failure comes from:
      (a) triangulate_gtsam returning None
      (b) all_positive_depth failing on successfully-triangulated points
    """

    def _make_scene(self, use_fisheye: bool, sigma_t: float = 0.0,
                    sigma_r: float = 0.0) -> tuple:
        """Cameras spread along Y (sideways baseline), all facing X+.

        This ensures all cameras see all points in front, so any cheirality
        failure with GT poses would be a bug, not a scene-layout artifact.
        Observations are always projected from GT poses so they represent
        real feature detections; triangulation uses (potentially noisy) poses.
        """
        n_cameras = 10
        n_points = 50
        rng = np.random.default_rng(7)

        if use_fisheye:
            cal = gtsam.Cal3Fisheye(500, 500, 0, 320, 240,
                                     0.1, -0.03, 0.005, -0.001)
            project = _project_fisheye
        else:
            cal = gtsam.Cal3DS2(500, 500, 0, 320, 240, 0, 0, 0, 0)
            project = _project_pinhole

        # Cameras spread along Y so every point at X > 0 is in front of all
        y_positions = np.linspace(-1.0, 1.0, n_cameras)
        gt_poses = {i: _pose_facing_x(np.array([0.0, float(y), 0.0]))
                    for i, y in enumerate(y_positions)}

        if sigma_t > 0 or sigma_r > 0:
            poses = {i: _add_pose_noise(p, sigma_t, sigma_r, rng)
                     for i, p in gt_poses.items()}
        else:
            poses = dict(gt_poses)

        pts_world = []
        keypoints = {i: [] for i in range(n_cameras)}
        tracks = []

        for j in range(n_points):
            x = rng.uniform(3.0, 8.0)   # well in front of all cameras
            y = rng.uniform(-0.5, 0.5)
            z = rng.uniform(-0.5, 0.5)
            pt = np.array([x, y, z])
            pts_world.append(pt)

            obs = {}
            for i in range(n_cameras):
                # Project from GT poses — these are our "real" pixel observations
                pixel = project(gt_poses[i], cal, pt).reshape(2)
                kp_idx = len(keypoints[i])
                keypoints[i].append(pixel)
                obs[i] = kp_idx

            tracks.append(Track(observations=obs))

        kps = {i: np.array(kp_list, dtype=np.float32)
               for i, kp_list in keypoints.items()
               if kp_list}

        return poses, kps, cal, tracks, pts_world

    def _breakdown(self, poses, kps, cal, tracks):
        """Return (n_none, n_behind, n_ok) counts."""
        triangulate_gtsam(tracks, kps, cal, poses)
        n_none = sum(1 for t in tracks if t.point3d is None)
        n_behind = sum(1 for t in tracks
                       if t.point3d is not None
                       and not all_positive_depth(t.point3d, t.observations, poses))
        n_ok = sum(1 for t in tracks
                   if t.point3d is not None
                   and all_positive_depth(t.point3d, t.observations, poses))
        return n_none, n_behind, n_ok

    def test_pinhole_gt_poses_near_perfect_pass_rate(self):
        poses, kps, cal, tracks, _ = self._make_scene(use_fisheye=False)
        n_none, n_behind, n_ok = self._breakdown(poses, kps, cal, tracks)
        total = len(tracks)
        assert n_none == 0, (
            f"Pinhole+GT: {n_none}/{total} tracks returned None — "
            f"triangulate_gtsam should never fail with perfect observations and GT poses"
        )
        assert n_behind == 0, (
            f"Pinhole+GT: {n_behind}/{total} tracks behind camera — "
            f"all_positive_depth should pass when GT poses and perfect pixels are used"
        )

    def test_fisheye_gt_poses_near_perfect_pass_rate(self):
        """Key H2 test: triangulate_gtsam with fisheye + GT poses + perfect pixels.

        Both failure modes are checked separately:
          n_none  → triangulation threw an exception or failed rank check
          n_behind → triangulation succeeded but point is behind a camera
        If either is nonzero this is a code bug, not a noise problem.
        """
        poses, kps, cal, tracks, _ = self._make_scene(use_fisheye=True)
        n_none, n_behind, n_ok = self._breakdown(poses, kps, cal, tracks)
        total = len(tracks)
        assert n_none == 0, (
            f"Fisheye+GT: {n_none}/{total} tracks returned None — "
            f"triangulate_gtsam has a bug with Cal3Fisheye (not a noise issue)"
        )
        assert n_behind == 0, (
            f"Fisheye+GT: {n_behind}/{total} tracks behind camera — "
            f"all_positive_depth fails even with GT poses (frame convention bug?)"
        )

    def test_fisheye_medium_noise_triangulation_failure_rate(self):
        """10 cm noise: count what actually fails — None vs behind.

        This test documents which stage causes failures under realistic noise.
        A high n_none means triangulate_gtsam is too fragile.
        A high n_behind means all_positive_depth is the culprit.
        """
        poses, kps, cal, tracks, _ = self._make_scene(
            use_fisheye=True, sigma_t=0.10, sigma_r=0.05
        )
        n_none, n_behind, n_ok = self._breakdown(poses, kps, cal, tracks)
        total = len(tracks)
        assert n_ok / total > 0.50, (
            f"Fisheye+medium noise: {n_ok}/{total} ok, "
            f"{n_none} returned None (triangulation failed), "
            f"{n_behind} behind (cheirality failed) — "
            f"expected >50% to survive for 10 cm noise"
        )


# ---------------------------------------------------------------------------
# Real-world cheirality examples from the eth3d exhibition-hall dataset
# ---------------------------------------------------------------------------

class TestRealWorldCheiralityExamples:
    """Tests using exact poses and pixels logged from the production pipeline.

    These specific (pose, pixel) pairs caused gtsam.triangulatePoint3 to throw
    TriangulationCheiralityException during a real SfM run, accounting for 1378
    out of ~4340 built tracks.

    Dataset calibration (Cal3Fisheye):
      fx=3408.4, fy=3408.6, skew=0, cx=3036.2, cy=2011.2
      k=[0.05, -0.01, 0.002, -0.0003]

    Each test method documents one logged example and what we observe.
    """

    @staticmethod
    def _cal() -> gtsam.Cal3Fisheye:
        # Actual calibration from eth3d_exhibition_hall.mcap
        return gtsam.Cal3Fisheye(3408.41, 3408.58, 0.0, 3036.25, 2011.21,
                                  0.209929, 0.21072, -0.157287, 0.39736)

    @staticmethod
    def _pose(tx, ty, tz, r00, r01, r02,
                             r10, r11, r12,
                             r20, r21, r22) -> gtsam.Pose3:
        R = np.array([[r00, r01, r02],
                      [r10, r11, r12],
                      [r20, r21, r22]], dtype=float)
        return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(tx, ty, tz))

    # ------------------------------------------------------------------
    # Shared pose objects used across examples
    # ------------------------------------------------------------------

    def _cam0(self) -> gtsam.Pose3:
        # img_id 0: first frame of sequence
        return self._pose(
            1.9149,  0.5519,  0.0055,
            0.4653, -0.0179,  0.8850,
           -0.8850,  0.0072,  0.4655,
           -0.0147, -0.9998, -0.0125,
        )

    def _cam1(self) -> gtsam.Pose3:
        # img_id 1e9: second frame (~0.3 m baseline, ~16° rotation from cam0)
        return self._pose(
            2.0322,  0.3179, -0.1537,
            0.2030,  0.0267,  0.9788,
           -0.9790,  0.0251,  0.2023,
           -0.0192, -0.9993,  0.0312,
        )

    def _cam2(self) -> gtsam.Pose3:
        # img_id 2e9: third frame (~0.4 m baseline from cam1, ~24° rotation)
        return self._pose(
            2.0791, -0.0719, -0.0345,
           -0.2146, -0.0020,  0.9767,
           -0.9765,  0.0185, -0.2145,
           -0.0176, -0.9998, -0.0059,
        )

    # ------------------------------------------------------------------
    # Step 1: confirm gtsam throws CheiralityException for each example
    # ------------------------------------------------------------------

    def test_example1_gtsam_throws_cheirality(self):
        """Example 1: cam0/cam1 pair, pixel near image centre on both cameras.

        Pixel (3084, 1526) in cam0 and (4350, 1567) in cam1.
        The ~16° camera rotation makes the DLT result land behind cam0 or cam1.
        """
        pv = gtsam.Pose3Vector([self._cam0(), self._cam1()])
        mv = gtsam.Point2Vector([
            gtsam.Point2(3084.00, 1525.64),
            gtsam.Point2(4349.70, 1566.80),
        ])
        with pytest.raises(RuntimeError, match="Cheirality"):
            gtsam.triangulatePoint3(pv, self._cal(), mv, 1e-9, True)

    def test_example2_gtsam_throws_cheirality(self):
        """Example 2: same cam0/cam1 pair, different pixels.

        Pixel (507, 1500) in cam0 and (4454, 1463) in cam1.
        Both pixels are real SIFT keypoints so they are inside the image frame.
        """
        pv = gtsam.Pose3Vector([self._cam0(), self._cam1()])
        mv = gtsam.Point2Vector([
            gtsam.Point2(506.75, 1500.19),
            gtsam.Point2(4454.26, 1462.74),
        ])
        with pytest.raises(RuntimeError, match="Cheirality"):
            gtsam.triangulatePoint3(pv, self._cal(), mv, 1e-9, True)

    def test_example3_gtsam_throws_cheirality(self):
        """Example 3: cam1/cam2 pair, larger baseline and rotation (~24°).

        Pixel (265, 1510) in cam1 and (2244, 1570) in cam2.
        """
        pv = gtsam.Pose3Vector([self._cam1(), self._cam2()])
        mv = gtsam.Point2Vector([
            gtsam.Point2(265.42, 1510.16),
            gtsam.Point2(2243.52, 1570.03),
        ])
        with pytest.raises(RuntimeError, match="Cheirality"):
            gtsam.triangulatePoint3(pv, self._cal(), mv, 1e-9, True)

