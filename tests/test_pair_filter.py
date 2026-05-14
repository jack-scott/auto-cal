"""
Tests for filter_pairs_by_geometry.

Covers:
  - STATIC pairs are dropped and counted correctly (real poses, frames 61-62)
  - GOOD pairs are kept (real poses, frames 0-1)
  - Mixed dict: counts and surviving pairs are correct
  - Without poses: H/E ratio test is used (real images, frame 61-62)
  - H/E secondary check: catches near-duplicates that noisy poses classify as GOOD
  - Empty input returns empty output
  - Missing poses AND keypoints raises ValueError
"""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import pytest

from autocal.engine.features import (
    cal_to_K,
    detect_sift,
    match_sift,
    undistort_keypoints,
)
from autocal.engine.pair_classifier import (
    PairClass,
    filter_pairs_by_geometry,
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

_WANT_TS = {0, 1_000_000_000, 61_000_000_000, 62_000_000_000}


@pytest.fixture(scope="module")
def real_data():
    """Load frames 0, 1, 61, 62 — poses, images, calibration."""
    images: dict[int, bytes]       = {}
    poses:  dict[int, gtsam.Pose3] = {}
    cal = None

    for topic, t_ns, msg in iter_messages(str(MCAP_PATH)):
        if topic == "/camera/image"        and t_ns in _WANT_TS: images[t_ns] = bytes(msg.data)
        elif topic == "/tf"                and t_ns in _WANT_TS: poses[t_ns]  = pose3_from_frame_transform(msg)
        elif topic == "/camera/calibration" and cal is None:      cal           = calibration_from_mcap_msg(msg)
        if len(images) == 4 and len(poses) == 4 and cal is not None:
            break

    assert len(images) == 4 and len(poses) == 4 and cal is not None
    kps_raw: dict[int, np.ndarray] = {}
    descs:   dict[int, np.ndarray] = {}
    for ts in sorted(images):
        kps_raw[ts], descs[ts] = detect_sift(images[ts], n_features=0)

    kps_u = undistort_keypoints(kps_raw, cal)
    K = cal_to_K(cal)
    return dict(poses=poses, kps_u=kps_u, descs=descs, K=K)


# ---------------------------------------------------------------------------
# With pose priors
# ---------------------------------------------------------------------------

class TestWithPoses:
    def test_static_pair_is_dropped(self, real_data):
        d = real_data
        raw = match_sift(d["descs"][61_000_000_000], d["descs"][62_000_000_000])
        pairs = {(61_000_000_000, 62_000_000_000): raw}

        filtered, counts, _ = filter_pairs_by_geometry(pairs, d["poses"])

        assert len(filtered) == 0
        assert counts[PairClass.STATIC] == 1
        assert counts[PairClass.GOOD] == 0

    def test_good_pair_is_kept(self, real_data):
        d = real_data
        raw = match_sift(d["descs"][0], d["descs"][1_000_000_000])
        pairs = {(0, 1_000_000_000): raw}

        filtered, counts, _ = filter_pairs_by_geometry(pairs, d["poses"])

        assert (0, 1_000_000_000) in filtered
        assert counts[PairClass.GOOD] == 1
        assert counts[PairClass.STATIC] == 0

    def test_mixed_pairs_counts_and_survivors(self, real_data):
        d = real_data
        good_matches   = match_sift(d["descs"][0], d["descs"][1_000_000_000])
        static_matches = match_sift(d["descs"][61_000_000_000], d["descs"][62_000_000_000])
        pairs = {
            (0, 1_000_000_000):                    good_matches,
            (61_000_000_000, 62_000_000_000):      static_matches,
        }

        filtered, counts, _ = filter_pairs_by_geometry(pairs, d["poses"])

        assert (0, 1_000_000_000) in filtered
        assert (61_000_000_000, 62_000_000_000) not in filtered
        assert counts[PairClass.STATIC] == 1
        assert counts[PairClass.GOOD] == 1
        total = sum(counts.values())
        assert total == len(pairs)

    def test_all_classes_present_in_counts(self, real_data):
        d = real_data
        pairs = {(0, 1_000_000_000): match_sift(d["descs"][0], d["descs"][1_000_000_000])}
        _, counts, _ = filter_pairs_by_geometry(pairs, d["poses"])
        assert set(counts.keys()) == set(PairClass)


# ---------------------------------------------------------------------------
# Without pose priors (H/E ratio test)
# ---------------------------------------------------------------------------

class TestWithoutPoses:
    def test_near_duplicate_pair_is_dropped(self, real_data):
        d = real_data
        raw = match_sift(d["descs"][61_000_000_000], d["descs"][62_000_000_000])
        pairs = {(61_000_000_000, 62_000_000_000): raw}

        filtered, counts, _ = filter_pairs_by_geometry(
            pairs, poses=None,
            keypoints_u=d["kps_u"], K=d["K"],
        )

        assert len(filtered) == 0
        assert counts[PairClass.PURE_ROTATION] == 1

    def test_normal_pair_is_kept(self, real_data):
        d = real_data
        raw = match_sift(d["descs"][0], d["descs"][1_000_000_000])
        pairs = {(0, 1_000_000_000): raw}

        filtered, counts, _ = filter_pairs_by_geometry(
            pairs, poses=None,
            keypoints_u=d["kps_u"], K=d["K"],
        )

        assert (0, 1_000_000_000) in filtered
        assert counts[PairClass.GOOD] == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty(real_data):
    filtered, counts, n_he = filter_pairs_by_geometry({}, real_data["poses"])
    assert filtered == {}
    assert sum(counts.values()) == 0
    assert n_he == 0

def test_raises_without_poses_or_keypoints():
    with pytest.raises(ValueError, match="poses"):
        filter_pairs_by_geometry(
            {(0, 1): [(0, 0)]},
            poses=None,
            keypoints_u=None,
            K=None,
        )


# ---------------------------------------------------------------------------
# H/E secondary check
# ---------------------------------------------------------------------------

class TestHeSecondaryCheck:
    """H/E secondary check catches near-duplicate pairs that noisy poses classify as GOOD."""

    @staticmethod
    def _shifted_pose(pose: gtsam.Pose3, dx_m: float) -> gtsam.Pose3:
        """Return pose with dx_m added to x — simulates pose noise exceeding true baseline."""
        t = pose.translation()
        return gtsam.Pose3(pose.rotation(), t + np.array([dx_m, 0.0, 0.0]))

    def test_secondary_check_drops_degenerate_with_noisy_poses(self, real_data):
        """Pair 61-62 is near-duplicate (0.2mm true baseline).
        Noisy poses (100mm offset) make classify_pair_with_poses call it GOOD.
        H/E secondary check must override and drop it."""
        d = real_data
        raw = match_sift(d["descs"][61_000_000_000], d["descs"][62_000_000_000])
        pairs = {(61_000_000_000, 62_000_000_000): raw}

        real_61 = d["poses"][61_000_000_000]
        noisy_poses = {
            61_000_000_000: real_61,
            62_000_000_000: self._shifted_pose(real_61, dx_m=0.1),
        }

        # Confirm the pose classifier is fooled (no secondary check)
        filtered_no_he, _, _ = filter_pairs_by_geometry(
            pairs, noisy_poses, keypoints_u=d["kps_u"], K=d["K"],
            he_secondary_check=False,
        )
        assert (61_000_000_000, 62_000_000_000) in filtered_no_he, (
            "Prerequisite: noisy poses should fool the pose classifier"
        )

        # Secondary check must catch it
        filtered_he, _, n_he = filter_pairs_by_geometry(
            pairs, noisy_poses, keypoints_u=d["kps_u"], K=d["K"],
            he_secondary_check=True,
        )
        assert len(filtered_he) == 0, "H/E secondary check should drop the near-duplicate pair"
        assert n_he == 1

    def test_secondary_check_disabled_keeps_noisy_pair(self, real_data):
        """With he_secondary_check=False the near-duplicate pair survives despite being degenerate."""
        d = real_data
        raw = match_sift(d["descs"][61_000_000_000], d["descs"][62_000_000_000])
        pairs = {(61_000_000_000, 62_000_000_000): raw}

        real_61 = d["poses"][61_000_000_000]
        noisy_poses = {
            61_000_000_000: real_61,
            62_000_000_000: self._shifted_pose(real_61, dx_m=0.1),
        }

        filtered, _, n_he = filter_pairs_by_geometry(
            pairs, noisy_poses, keypoints_u=d["kps_u"], K=d["K"],
            he_secondary_check=False,
        )
        assert (61_000_000_000, 62_000_000_000) in filtered
        assert n_he == 0

    def test_secondary_check_does_not_drop_genuinely_good_pair(self, real_data):
        """H/E check must not drop a pair with real translation (frames 0-1)."""
        d = real_data
        raw = match_sift(d["descs"][0], d["descs"][1_000_000_000])
        pairs = {(0, 1_000_000_000): raw}

        filtered, _, n_he = filter_pairs_by_geometry(
            pairs, d["poses"], keypoints_u=d["kps_u"], K=d["K"],
            he_secondary_check=True,
        )
        assert (0, 1_000_000_000) in filtered
        assert n_he == 0
