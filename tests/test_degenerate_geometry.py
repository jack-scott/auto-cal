"""
Tests for degenerate camera geometry cases found in frames 61-64 of
eth3d_exhibition_hall.

Two distinct failure modes identified:

  NEAR-DUPLICATE FRAMES (61-62, 63-64)
    Baseline ≈ 0.0002 m, rotation ≈ 0°.
    The camera barely moved — effectively the same image twice.
    Consequences:
      - The essential matrix is completely ill-conditioned (no translation
        to constrain it).
      - RANSAC accepts every match as an inlier because x'^T F x ≈ 0 for
        any F when there is no baseline.
      - Wrong-side matches in a symmetric scene pass trivially.
      - Triangulation is geometrically impossible.

  NEAR-PURE-ROTATION (62-63)
    Baseline ≈ 0.057 m, rotation ≈ 23°.
    Valid SIFT matches exist, but the rotation dominates the baseline by a
    large margin.  The DLT is ill-conditioned and consistently throws
    CheiralityException even for correct matches.

These tests define the expected detection behaviour so we can gate on
baseline and rotation-to-translation ratio before attempting triangulation.
"""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import pytest

from autocal.engine.features import (
    cal_to_K,
    detect_sift,
    filter_matches_ransac,
    match_sift,
    undistort_keypoints,
)
from autocal.gtsam_bridge.conversions import (
    calibration_from_mcap_msg,
    pose3_from_frame_transform,
)
from autocal.io.mcap_reader import iter_messages

MCAP_PATH = Path("data/eth3d_exhibition_hall.mcap")

pytestmark = pytest.mark.skipif(
    not MCAP_PATH.exists(),
    reason=f"dataset not found: {MCAP_PATH}",
)


# ---------------------------------------------------------------------------
# Shared fixture — load poses, calibration, images for frames 61-64
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frames_61_64():
    wanted = {ts * 1_000_000_000 for ts in range(61, 65)}
    images, poses, cal = {}, {}, None
    for topic, t_ns, msg in iter_messages(str(MCAP_PATH)):
        if t_ns > 64_000_000_000 and topic != "/camera/calibration":
            if cal is not None:
                break
        if topic == "/camera/image"         and t_ns in wanted: images[t_ns] = bytes(msg.data)
        elif topic == "/tf"                 and t_ns in wanted: poses[t_ns]  = pose3_from_frame_transform(msg)
        elif topic == "/camera/calibration" and cal is None:    cal          = calibration_from_mcap_msg(msg)
        if len(images) == 4 and len(poses) == 4 and cal is not None:
            break
    return {"images": images, "poses": poses, "cal": cal}


def _baseline(pose_a: gtsam.Pose3, pose_b: gtsam.Pose3) -> float:
    return float(np.linalg.norm(pose_a.translation() - pose_b.translation()))


def _rotation_angle_deg(pose_a: gtsam.Pose3, pose_b: gtsam.Pose3) -> float:
    R_rel = pose_b.rotation().matrix().T @ pose_a.rotation().matrix()
    cos_val = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


# ---------------------------------------------------------------------------
# Near-duplicate frame geometry (61-62, 63-64)
# ---------------------------------------------------------------------------

class TestNearDuplicateFrames:
    """Pairs 61-62 and 63-64: baseline < 0.001 m, rotation ≈ 0°.

    These are effectively the same image captured twice.  Any matching or
    triangulation on these pairs is geometrically meaningless.
    """

    @pytest.mark.parametrize("ts_a,ts_b", [
        (61_000_000_000, 62_000_000_000),
        (63_000_000_000, 64_000_000_000),
    ])
    def test_baseline_is_sub_millimetre(self, ts_a, ts_b, frames_61_64):
        b = _baseline(frames_61_64["poses"][ts_a], frames_61_64["poses"][ts_b])
        assert b < 0.001, (
            f"pair {ts_a//1e9:.0f}-{ts_b//1e9:.0f}: baseline={b:.4f}m — "
            f"expected < 1 mm for a near-duplicate pair"
        )

    @pytest.mark.parametrize("ts_a,ts_b", [
        (61_000_000_000, 62_000_000_000),
        (63_000_000_000, 64_000_000_000),
    ])
    def test_rotation_is_negligible(self, ts_a, ts_b, frames_61_64):
        angle = _rotation_angle_deg(frames_61_64["poses"][ts_a], frames_61_64["poses"][ts_b])
        assert angle < 0.5, (
            f"pair {ts_a//1e9:.0f}-{ts_b//1e9:.0f}: rotation={angle:.2f}° — "
            f"expected < 0.5° for a near-duplicate pair"
        )

    @pytest.mark.parametrize("ts_a,ts_b", [
        (61_000_000_000, 62_000_000_000),
        (63_000_000_000, 64_000_000_000),
    ])
    def test_ransac_accepts_almost_all_matches(self, ts_a, ts_b, frames_61_64):
        """RANSAC cannot filter bad matches when baseline is near zero.

        With no camera motion the epipolar constraint is satisfied by every
        point regardless of whether the match is correct.  RANSAC retention
        rate should be near 100%, confirming it provides no useful filtering.
        """
        d = frames_61_64
        cal = d["cal"]
        kps_a, descs_a = detect_sift(d["images"][ts_a], n_features=2000)
        kps_b, descs_b = detect_sift(d["images"][ts_b], n_features=2000)
        kps_u = undistort_keypoints({ts_a: kps_a, ts_b: kps_b}, cal)

        raw     = match_sift(descs_a, descs_b, ratio=0.75)
        inliers = filter_matches_ransac(
            kps_u[ts_a], kps_u[ts_b], raw, ransac_threshold=2.0, min_inliers=8,
        )
        retention = len(inliers) / len(raw) if raw else 0.0
        print(f"\n  pair {ts_a//1e9:.0f}-{ts_b//1e9:.0f}: "
              f"{len(raw)} raw → {len(inliers)} RANSAC inliers ({retention:.1%} retention)")
        assert retention > 0.90, (
            f"Expected RANSAC to keep >90% of matches for a near-duplicate pair "
            f"(epipolar constraint is degenerate), got {retention:.1%}"
        )

    @pytest.mark.parametrize("ts_a,ts_b", [
        (61_000_000_000, 62_000_000_000),
        (63_000_000_000, 64_000_000_000),
    ])
    def test_triangulation_is_ill_conditioned_at_near_zero_baseline(self, ts_a, ts_b, frames_61_64):
        """Near-zero baseline makes triangulation ill-conditioned: tiny pixel
        noise causes enormous depth variation.

        With 0.0002 m baseline the DLT is degenerate — a 1-pixel perturbation
        in one observation shifts the recovered depth by hundreds of metres.
        This makes these pairs useless for seeding GTSAM initial values.
        """
        d = frames_61_64
        pose_a = d["poses"][ts_a]
        pose_b = d["poses"][ts_b]
        cal    = d["cal"]

        def _project_fisheye(pose, pt):
            k = cal.k()
            R_cw = pose.rotation().matrix().T
            pc = R_cw @ (pt - pose.translation())
            x, y, z = pc
            r = np.sqrt(x*x + y*y)
            theta = np.arctan2(r, z)
            t2 = theta * theta
            rd = theta * (1 + k[0]*t2 + k[1]*t2**2 + k[2]*t2**3 + k[3]*t2**4)
            s = rd / r if r > 1e-10 else 0.0
            return np.array([cal.fx() * s * x + cal.px(), cal.fy() * s * y + cal.py()])

        # True 3D point 5 m in front of camera
        pt_world = pose_a.translation() + pose_a.rotation().matrix()[:, 2] * 5.0
        px_a = _project_fisheye(pose_a, pt_world)
        px_b = _project_fisheye(pose_b, pt_world)

        pv = gtsam.Pose3Vector([pose_a, pose_b])

        # Triangulate with exact pixels — should succeed
        mv_exact = gtsam.Point2Vector([gtsam.Point2(*px_a), gtsam.Point2(*px_b)])
        pt_exact = np.array(gtsam.triangulatePoint3(pv, cal, mv_exact, 1e-9, True))
        depth_exact = np.linalg.norm(pt_exact - pose_a.translation())

        # Add 1 pixel noise to one observation
        mv_noisy = gtsam.Point2Vector([
            gtsam.Point2(px_a[0] + 1.0, px_a[1]),
            gtsam.Point2(*px_b),
        ])
        try:
            pt_noisy = np.array(gtsam.triangulatePoint3(pv, cal, mv_noisy, 1e-9, True))
            depth_noisy = np.linalg.norm(pt_noisy - pose_a.translation())
            depth_change = abs(depth_noisy - depth_exact)
        except RuntimeError:
            depth_change = float("inf")

        print(f"\n  pair {ts_a//1e9:.0f}-{ts_b//1e9:.0f}: "
              f"exact depth={depth_exact:.1f}m, "
              f"1px noise → depth change={depth_change:.1f}m")

        # 1 pixel of noise on a 0.2mm baseline causes ~100% relative depth error —
        # the triangulation is completely ill-conditioned
        assert depth_change > 2.0, (
            f"Expected >2m depth change (~100% relative error) from 1px noise on "
            f"0.2mm baseline, got {depth_change:.1f}m — pair is not as degenerate as expected"
        )


# ---------------------------------------------------------------------------
# Near-pure rotation (62-63)
# ---------------------------------------------------------------------------

class TestNearPureRotation:
    """Pair 62-63: baseline ≈ 0.057 m, rotation ≈ 23°.

    The rotation angle is ~400× the baseline in metres — a rotation-dominated
    motion.  Valid feature matches exist, but DLT triangulation is
    ill-conditioned because the baseline is too small relative to the rotation
    to recover reliable depth.
    """

    TS_A = 62_000_000_000
    TS_B = 63_000_000_000

    def test_baseline_and_rotation(self, frames_61_64):
        pose_a = frames_61_64["poses"][self.TS_A]
        pose_b = frames_61_64["poses"][self.TS_B]
        b     = _baseline(pose_a, pose_b)
        angle = _rotation_angle_deg(pose_a, pose_b)
        print(f"\n  pair 62-63: baseline={b:.4f}m  rotation={angle:.1f}°  ratio={angle/b:.0f}°/m")
        assert b < 0.1,   f"baseline={b:.4f}m — expected a small baseline"
        assert angle > 15, f"rotation={angle:.1f}° — expected large rotation"
        # ratio of rotation to baseline: high value = near-pure rotation
        assert angle / b > 100, (
            f"rotation/baseline ratio {angle/b:.0f} °/m — "
            f"expected > 100 to classify as rotation-dominated"
        )

    def test_matches_exist_and_triangulate_well_with_gt_poses(self, frames_61_64):
        """Valid SIFT matches exist and triangulate fine with GT poses.

        Pair 62-63 has good correspondences — the problem is NOT the matching.
        With correct GT poses, DLT succeeds for >99% of matches.  The
        cheirality failures in the pipeline come from noisy chained initial
        poses (accumulated over 62 frames), not from the geometry itself.
        """
        d = frames_61_64
        cal = d["cal"]
        kps_a, descs_a = detect_sift(d["images"][self.TS_A], n_features=2000)
        kps_b, descs_b = detect_sift(d["images"][self.TS_B], n_features=2000)
        kps_u = undistort_keypoints({self.TS_A: kps_a, self.TS_B: kps_b}, cal)

        raw     = match_sift(descs_a, descs_b, ratio=0.75)
        inliers = filter_matches_ransac(
            kps_u[self.TS_A], kps_u[self.TS_B], raw,
            ransac_threshold=2.0, min_inliers=8,
        )
        print(f"\n  pair 62-63: {len(raw)} raw → {len(inliers)} RANSAC inliers")
        assert len(inliers) > 50

        pose_a = d["poses"][self.TS_A]
        pose_b = d["poses"][self.TS_B]
        pv     = gtsam.Pose3Vector([pose_a, pose_b])
        n_fail = 0
        for i, j in inliers:
            mv = gtsam.Point2Vector([
                gtsam.Point2(float(kps_a[i][0]), float(kps_a[i][1])),
                gtsam.Point2(float(kps_b[j][0]), float(kps_b[j][1])),
            ])
            try:
                gtsam.triangulatePoint3(pv, cal, mv, 1e-9, True)
            except RuntimeError:
                n_fail += 1

        fail_rate = n_fail / len(inliers)
        print(f"  GT-pose triangulation failure rate: {n_fail}/{len(inliers)} = {fail_rate:.1%}")
        assert fail_rate < 0.05, (
            f"Expected <5% failures with GT poses for pair 62-63, got {fail_rate:.1%} — "
            f"matching or calibration may be broken"
        )

    def test_triangulation_fails_with_noisy_poses(self, frames_61_64):
        """Adding pose noise (simulating chained essential-matrix drift) causes
        high cheirality failure rate on the same valid matches.

        Pair 62-63 has 57mm baseline and 23° rotation.  Because the rotation
        dominates, even modest pose noise sends the DLT result behind a camera.
        This is what the real pipeline experiences from accumulated chain drift.
        """
        d = frames_61_64
        cal = d["cal"]
        kps_a, descs_a = detect_sift(d["images"][self.TS_A], n_features=2000)
        kps_b, descs_b = detect_sift(d["images"][self.TS_B], n_features=2000)
        kps_u = undistort_keypoints({self.TS_A: kps_a, self.TS_B: kps_b}, cal)

        raw     = match_sift(descs_a, descs_b, ratio=0.75)
        inliers = filter_matches_ransac(
            kps_u[self.TS_A], kps_u[self.TS_B], raw,
            ransac_threshold=2.0, min_inliers=8,
        )

        gt_a = d["poses"][self.TS_A]
        gt_b = d["poses"][self.TS_B]

        rng = np.random.default_rng(42)
        def _add_noise(pose, sigma_t=0.05, sigma_r=0.05):
            dt   = rng.normal(0, sigma_t, 3)
            axis = rng.normal(0, 1, 3); axis /= np.linalg.norm(axis)
            dR   = gtsam.Rot3.AxisAngle(gtsam.Point3(*axis), float(rng.normal(0, sigma_r)))
            return gtsam.Pose3(pose.rotation().compose(dR),
                               gtsam.Point3(*(pose.translation() + dt)))

        noisy_a = _add_noise(gt_a)
        noisy_b = _add_noise(gt_b)
        pv = gtsam.Pose3Vector([noisy_a, noisy_b])

        n_fail = 0
        for i, j in inliers:
            mv = gtsam.Point2Vector([
                gtsam.Point2(float(kps_a[i][0]), float(kps_a[i][1])),
                gtsam.Point2(float(kps_b[j][0]), float(kps_b[j][1])),
            ])
            try:
                gtsam.triangulatePoint3(pv, cal, mv, 1e-9, True)
            except RuntimeError:
                n_fail += 1

        fail_rate = n_fail / len(inliers)
        print(f"\n  pair 62-63 with 5cm/0.05rad noise: "
              f"{n_fail}/{len(inliers)} failures = {fail_rate:.1%}")
        assert fail_rate > 0.5, (
            f"Expected >50% failures with noisy poses on rotation-dominated pair, "
            f"got {fail_rate:.1%}"
        )

    def test_synthetic_rotation_dominated_sensitive_to_noise(self):
        """Synthetic rotation-dominated pair: modest pose noise causes cheirality.

        With 23° rotation and only 57mm baseline, the rotation/translation
        ratio is ~400°/m.  Exact pixels triangulate fine, but adding 5cm / 0.05rad
        noise (typical chain drift) drives most tracks behind a camera.
        """
        cal = gtsam.Cal3Fisheye(3408.41, 3408.58, 0.0, 3036.25, 2011.21,
                                  0.209929, 0.21072, -0.157287, 0.39736)
        # GT pose pair approximating frames 62-63
        t_a = np.array([0.782, 0.297, -0.065])
        t_b = np.array([0.730, 0.278, -0.077])
        pitch = np.radians(23.0)
        R_b = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [             0, 1,             0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ])
        gt_a = gtsam.Pose3(gtsam.Rot3(np.eye(3)), gtsam.Point3(*t_a))
        gt_b = gtsam.Pose3(gtsam.Rot3(R_b),        gtsam.Point3(*t_b))

        # Generate 100 synthetic scene points in front of both cameras
        rng = np.random.default_rng(0)
        scene_pts = t_a + np.column_stack([
            rng.uniform(3.0, 8.0, 100),
            rng.uniform(-1.0, 1.0, 100),
            rng.uniform(-1.0, 1.0, 100),
        ])

        def _project(pose, pt):
            k = cal.k()
            R_cw = pose.rotation().matrix().T
            pc = R_cw @ (pt - pose.translation())
            x, y, z = pc
            r = np.sqrt(x*x + y*y)
            theta = np.arctan2(r, z)
            t2 = theta * theta
            rd = theta * (1 + k[0]*t2 + k[1]*t2**2 + k[2]*t2**3 + k[3]*t2**4)
            s = rd / r if r > 1e-10 else 0.0
            return gtsam.Point2(cal.fx()*s*x + cal.px(), cal.fy()*s*y + cal.py())

        def _add_noise(pose, sigma_t, sigma_r):
            dt   = rng.normal(0, sigma_t, 3)
            axis = rng.normal(0, 1, 3); axis /= np.linalg.norm(axis)
            dR   = gtsam.Rot3.AxisAngle(gtsam.Point3(*axis), float(rng.normal(0, sigma_r)))
            return gtsam.Pose3(pose.rotation().compose(dR),
                               gtsam.Point3(*(pose.translation() + dt)))

        noisy_a = _add_noise(gt_a, 0.05, 0.05)
        noisy_b = _add_noise(gt_b, 0.05, 0.05)
        pv = gtsam.Pose3Vector([noisy_a, noisy_b])

        n_fail = 0
        for pt in scene_pts:
            mv = gtsam.Point2Vector([_project(gt_a, pt), _project(gt_b, pt)])
            try:
                gtsam.triangulatePoint3(pv, cal, mv, 1e-9, True)
            except RuntimeError:
                n_fail += 1

        fail_rate = n_fail / len(scene_pts)
        print(f"\n  synthetic 23° rotation / 57mm baseline + 5cm noise: "
              f"{n_fail}/100 failures = {fail_rate:.0%}")
        assert fail_rate > 0.5, (
            f"Expected >50% cheirality failures on rotation-dominated pair "
            f"with 5cm pose noise, got {fail_rate:.0%}"
        )
