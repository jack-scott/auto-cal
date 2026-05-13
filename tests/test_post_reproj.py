"""
Tests for post-optimisation outlier rejection in optimize_poses.

Verifies:
  - post_reproj_error_px=0 disables the second pass entirely.
  - A tight threshold removes tracks with high reprojection error.
  - Re-optimisation runs and produces a valid pose result.
  - With noisy poses, post-opt rejection improves or matches APE vs no rejection.
"""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import pytest

from autocal.engine.features import filter_by_reproj
from autocal.engine.sfm_solver import SfmOptions, optimize_poses
from autocal.gtsam_bridge.conversions import (
    calibration_from_mcap_msg,
    pose3_from_frame_transform,
)
from autocal.io.mcap_reader import iter_messages
from autocal.metrics.ape import compute_ape

MCAP_PATH     = Path("data/eth3d_exhibition_hall.mcap")
NOISY_MCAP    = Path("data/eth3d_exhibition_hall_noise_easy.mcap")

pytestmark = pytest.mark.skipif(
    not MCAP_PATH.exists(),
    reason=f"dataset not found: {MCAP_PATH}",
)

_WANT_TS = {0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000}


@pytest.fixture(scope="module")
def five_frames():
    images: dict[int, bytes]       = {}
    poses:  dict[int, gtsam.Pose3] = {}
    cal = None
    for topic, t_ns, msg in iter_messages(str(MCAP_PATH)):
        if topic == "/camera/image"        and t_ns in _WANT_TS: images[t_ns] = bytes(msg.data)
        elif topic == "/tf"                and t_ns in _WANT_TS: poses[t_ns]  = pose3_from_frame_transform(msg)
        elif topic == "/camera/calibration" and cal is None:      cal           = calibration_from_mcap_msg(msg)
        if len(images) == 5 and len(poses) == 5 and cal is not None:
            break
    assert len(images) == 5 and len(poses) == 5
    return sorted(images.items()), poses, cal


class TestPostReproj:
    def _run(self, five_frames, post_reproj: float) -> dict:
        img_list, poses, cal = five_frames
        opts = SfmOptions(
            sift_features=500,
            min_parallax_deg=1.0,
            classify_pairs=False,
            match_window=3,
            min_track_length=2,
            post_reproj_error_px=post_reproj,
        )
        return optimize_poses(img_list, cal, opts, initial_poses=poses)

    def test_disabled_at_zero(self, five_frames):
        # When post_reproj_error_px=0, result should be identical across two runs
        # (no second pass, no randomness).
        r1 = self._run(five_frames, post_reproj=0.0)
        r2 = self._run(five_frames, post_reproj=0.0)
        assert r1["n_tracks"] == r2["n_tracks"]

    def test_tight_threshold_reduces_tracks(self, five_frames):
        r_loose = self._run(five_frames, post_reproj=50.0)  # keeps almost everything
        r_tight = self._run(five_frames, post_reproj=1.0)   # strict
        assert r_tight["n_tracks"] <= r_loose["n_tracks"], (
            f"tight threshold should keep <= tracks: got {r_tight['n_tracks']} vs {r_loose['n_tracks']}"
        )

    def test_returned_tracks_pass_their_own_threshold(self, five_frames):
        # After post-opt rejection, every surviving track should have max reproj ≤ threshold
        # when measured against the returned optimised poses.
        threshold = 2.0
        img_list, poses, cal = five_frames
        opts = SfmOptions(
            sift_features=500, min_parallax_deg=1.0,
            classify_pairs=False, match_window=3,
            min_track_length=2, post_reproj_error_px=threshold,
        )
        result = optimize_poses(img_list, cal, opts, initial_poses=poses)

        # Re-run filter_by_reproj on the returned tracks with returned poses
        still_good = filter_by_reproj(
            result["triangulated"],
            result["keypoints"],
            result["poses"],
            cal,
            threshold,
        )
        assert len(still_good) == len(result["triangulated"]), (
            f"Expected all {len(result['triangulated'])} returned tracks to pass "
            f"the {threshold}px threshold, but {len(result['triangulated']) - len(still_good)} do not"
        )

    def test_poses_are_valid_after_reoptimisation(self, five_frames):
        result = self._run(five_frames, post_reproj=2.0)
        # All 5 cameras should be in the result with finite translations
        assert len(result["poses"]) == 5
        for img_id, pose in result["poses"].items():
            t = pose.translation()
            assert np.all(np.isfinite(t)), f"Camera {img_id} has non-finite translation"


@pytest.mark.skipif(not NOISY_MCAP.exists(), reason=f"noisy dataset not found: {NOISY_MCAP}")
class TestPostReprojImprovesPoseAccuracy:
    """Integration test: post-opt rejection should not make APE worse."""

    _WANT = set(range(0, 10_000_000_000, 1_000_000_000))  # first 10 frames

    @pytest.fixture(scope="class")
    def ten_noisy_frames(self):
        images: dict[int, bytes]       = {}
        noisy_poses:  dict[int, gtsam.Pose3] = {}
        gt_poses:     dict[int, gtsam.Pose3] = {}
        cal = None
        for topic, t_ns, msg in iter_messages(str(NOISY_MCAP)):
            if t_ns > 9_000_000_000 and topic == "/camera/image":
                break
            if topic == "/camera/image"        and t_ns in self._WANT: images[t_ns]      = bytes(msg.data)
            elif topic == "/tf"                and t_ns in self._WANT: noisy_poses[t_ns] = pose3_from_frame_transform(msg)
            elif topic == "/tf_gt"             and t_ns in self._WANT: gt_poses[t_ns]    = pose3_from_frame_transform(msg)
            elif topic == "/camera/calibration" and cal is None:        cal               = calibration_from_mcap_msg(msg)
        assert len(images) >= 5 and cal is not None
        return sorted(images.items()), noisy_poses, gt_poses, cal

    def _ape_mean(self, gt, estimated):
        result = compute_ape(gt, estimated)
        return result["stats"]["translation"]["mean"]

    def test_post_reproj_does_not_worsen_ape(self, ten_noisy_frames):
        img_list, noisy_poses, gt_poses, cal = ten_noisy_frames

        base_opts = dict(
            sift_features=500, min_parallax_deg=1.0,
            classify_pairs=True, match_window=3, min_track_length=3,
        )

        r_no_post  = optimize_poses(img_list, cal,
                                    SfmOptions(**base_opts, post_reproj_error_px=0.0),
                                    initial_poses=noisy_poses)
        r_with_post = optimize_poses(img_list, cal,
                                     SfmOptions(**base_opts, post_reproj_error_px=2.0),
                                     initial_poses=noisy_poses)

        shared = set(gt_poses) & set(r_no_post["poses"]) & set(r_with_post["poses"])
        gt_shared   = {k: gt_poses[k] for k in shared}
        ape_no_post  = self._ape_mean(gt_shared, {k: r_no_post["poses"][k]  for k in shared})
        ape_with_post = self._ape_mean(gt_shared, {k: r_with_post["poses"][k] for k in shared})

        assert ape_with_post <= ape_no_post * 1.05, (
            f"Post-opt rejection should not significantly worsen APE: "
            f"{ape_with_post*1000:.1f}mm vs {ape_no_post*1000:.1f}mm without"
        )
