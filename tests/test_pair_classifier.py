"""
Tests for autocal.engine.pair_classifier.

Uses real GT poses from eth3d_exhibition_hall frames 61-64, whose geometry
was characterised in the degenerate-geometry investigation:

  61-62, 63-64  baseline ≈ 0.2 mm, rotation ≈ 0°  → STATIC
  62-63         baseline ≈ 57 mm,  rotation ≈ 23°  → GOOD (both thresholds exceeded)

classify_pair_without_poses — H/E inlier ratio test:
  pair 0-1   (normal motion)    → GOOD        (H/E ratio well below 0.8)
  pair 61-62 (near-duplicate)   → PURE_ROTATION (H/E ratio near 1.0)
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
from autocal.engine.pair_classifier import (
    PairClass,
    classify_pair_with_poses,
    classify_pair_without_poses,
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

_POSE_TIMESTAMPS  = {61_000_000_000, 62_000_000_000, 63_000_000_000, 64_000_000_000}
_IMAGE_TIMESTAMPS = {0, 1_000_000_000, 61_000_000_000, 62_000_000_000}


@pytest.fixture(scope="module")
def gt_poses() -> dict[int, gtsam.Pose3]:
    poses: dict[int, gtsam.Pose3] = {}
    for topic, t_ns, msg in iter_messages(str(MCAP_PATH)):
        if topic == "/tf" and t_ns in _POSE_TIMESTAMPS:
            poses[t_ns] = pose3_from_frame_transform(msg)
        if len(poses) == len(_POSE_TIMESTAMPS):
            break
    assert len(poses) == 4, f"Expected 4 poses, got {len(poses)}"
    return poses


@pytest.fixture(scope="module")
def image_data():
    """Load frames 0, 1, 61, 62 — images + calibration — for H/E ratio tests."""
    images: dict[int, bytes] = {}
    cal = None
    for topic, t_ns, msg in iter_messages(str(MCAP_PATH)):
        if topic == "/camera/image" and t_ns in _IMAGE_TIMESTAMPS:
            images[t_ns] = bytes(msg.data)
        elif topic == "/camera/calibration" and cal is None:
            cal = calibration_from_mcap_msg(msg)
        if len(images) == len(_IMAGE_TIMESTAMPS) and cal is not None:
            break
    assert len(images) == 4 and cal is not None
    return images, cal


# ---------------------------------------------------------------------------
# Real-data classification tests
# ---------------------------------------------------------------------------

class TestRealPoses:
    def test_near_duplicate_61_62_is_static(self, gt_poses):
        result = classify_pair_with_poses(
            gt_poses[61_000_000_000],
            gt_poses[62_000_000_000],
        )
        assert result == PairClass.STATIC, (
            f"Pair 61-62 (≈0.2mm baseline, ≈0° rotation) should be STATIC, got {result}"
        )

    def test_near_duplicate_63_64_is_static(self, gt_poses):
        result = classify_pair_with_poses(
            gt_poses[63_000_000_000],
            gt_poses[64_000_000_000],
        )
        assert result == PairClass.STATIC, (
            f"Pair 63-64 (≈0.2mm baseline, ≈0° rotation) should be STATIC, got {result}"
        )

    def test_rotation_dominated_62_63_is_good(self, gt_poses):
        # 57mm baseline > 20mm threshold; 23° rotation > 2° threshold → GOOD
        result = classify_pair_with_poses(
            gt_poses[62_000_000_000],
            gt_poses[63_000_000_000],
        )
        assert result == PairClass.GOOD, (
            f"Pair 62-63 (57mm baseline, 23° rotation) should be GOOD with "
            f"default thresholds, got {result}"
        )

    def test_baseline_values_are_as_expected(self, gt_poses):
        def baseline(a, b):
            return np.linalg.norm(gt_poses[a].between(gt_poses[b]).translation())

        assert baseline(61_000_000_000, 62_000_000_000) < 0.001  # < 1mm
        assert baseline(63_000_000_000, 64_000_000_000) < 0.001  # < 1mm
        assert 0.04 < baseline(62_000_000_000, 63_000_000_000) < 0.10  # ~57mm


# ---------------------------------------------------------------------------
# Synthetic classification tests — cover all four branches
# ---------------------------------------------------------------------------

class TestSyntheticPoses:
    """Verify each branch of classify_pair_with_poses with controlled inputs."""

    @staticmethod
    def _pose(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0,
              rz_deg: float = 0.0) -> gtsam.Pose3:
        R = gtsam.Rot3.Rz(np.radians(rz_deg))
        return gtsam.Pose3(R, gtsam.Point3(tx, ty, tz))

    def test_static(self):
        # <1cm translation, <1° rotation
        a = self._pose(0.0, 0.0, 0.0)
        b = self._pose(0.005, 0.0, 0.0, rz_deg=0.5)
        assert classify_pair_with_poses(a, b) == PairClass.STATIC

    def test_pure_rotation(self):
        # No translation, 30° rotation
        a = self._pose(0.0, 0.0, 0.0)
        b = self._pose(0.0, 0.0, 0.0, rz_deg=30.0)
        assert classify_pair_with_poses(a, b) == PairClass.PURE_ROTATION

    def test_pure_translation(self):
        # 1m translation, <1° rotation
        a = self._pose(0.0, 0.0, 0.0)
        b = self._pose(1.0, 0.0, 0.0, rz_deg=0.5)
        assert classify_pair_with_poses(a, b) == PairClass.PURE_TRANSLATION

    def test_good(self):
        # 1m translation, 30° rotation
        a = self._pose(0.0, 0.0, 0.0)
        b = self._pose(1.0, 0.0, 0.0, rz_deg=30.0)
        assert classify_pair_with_poses(a, b) == PairClass.GOOD

    def test_custom_thresholds_promote_to_static(self):
        # 57mm baseline is GOOD at default but STATIC at 0.1m threshold
        a = gtsam.Pose3()
        b = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.057, 0.0, 0.0))
        assert classify_pair_with_poses(a, b) == PairClass.PURE_TRANSLATION
        assert classify_pair_with_poses(a, b, translation_threshold_m=0.1) == PairClass.STATIC

    def test_identity_pair_is_static(self):
        a = gtsam.Pose3()
        assert classify_pair_with_poses(a, a) == PairClass.STATIC

    def test_symmetric_pair(self):
        # classify(a, b) == classify(b, a)
        a = self._pose(0.0, 0.0, 0.0)
        b = self._pose(0.5, 0.2, 0.0, rz_deg=15.0)
        assert classify_pair_with_poses(a, b) == classify_pair_with_poses(b, a)


# ---------------------------------------------------------------------------
# H/E inlier ratio tests (no pose priors)
# ---------------------------------------------------------------------------

class TestClassifyWithoutPoses:
    """Verify the H/E ratio test using real SIFT matches."""

    @staticmethod
    def _get_matches(
        images: dict[int, bytes],
        cal,
        ts_a: int,
        ts_b: int,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        kps_a, descs_a = detect_sift(images[ts_a], n_features=0)
        kps_b, descs_b = detect_sift(images[ts_b], n_features=0)
        kps_u = undistort_keypoints({ts_a: kps_a, ts_b: kps_b}, cal)
        raw = match_sift(descs_a, descs_b, ratio=0.75)
        inliers = filter_matches_ransac(kps_u[ts_a], kps_u[ts_b], raw,
                                        ransac_threshold=2.0, min_inliers=8)
        return kps_u[ts_a], kps_u[ts_b], inliers or raw

    def test_normal_pair_is_good(self, image_data):
        images, cal = image_data
        K = cal_to_K(cal)
        kps_a, kps_b, matches = self._get_matches(
            images, cal, 0, 1_000_000_000,
        )
        result = classify_pair_without_poses(kps_a, kps_b, matches, K)
        assert result == PairClass.GOOD, (
            f"Pair 0-1 (normal motion) should be GOOD, got {result}"
        )

    def test_near_duplicate_pair_is_degenerate(self, image_data):
        images, cal = image_data
        K = cal_to_K(cal)
        kps_a, kps_b, matches = self._get_matches(
            images, cal, 61_000_000_000, 62_000_000_000,
        )
        result = classify_pair_without_poses(kps_a, kps_b, matches, K)
        assert result == PairClass.PURE_ROTATION, (
            f"Pair 61-62 (≈0.2mm baseline) should be PURE_ROTATION (degenerate), "
            f"got {result}"
        )

    def test_too_few_matches_is_degenerate(self):
        # Fewer than 8 matches → can't fit H or E reliably → treat as degenerate
        result = classify_pair_without_poses(
            np.random.rand(5, 2),
            np.random.rand(5, 2),
            [(i, i) for i in range(5)],
            np.eye(3),
        )
        assert result == PairClass.PURE_ROTATION
