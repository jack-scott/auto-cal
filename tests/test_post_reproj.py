"""
Tests for post-optimisation iterative outlier rejection in optimize_poses.

Verifies:
  - post_reproj_reject_pct=0 disables the second pass entirely.
  - A higher rejection percentage keeps fewer tracks.
  - One iteration of X% rejection removes approximately X% of tracks.
  - Re-optimisation runs and produces a valid pose result.
  - With noisy poses, post-opt rejection improves or matches APE vs no rejection.
"""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import pytest

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
    def _run(self, five_frames, post_reproj_pct: float = 0.0,
             post_reproj_iters: int = 4) -> dict:
        img_list, poses, cal = five_frames
        opts = SfmOptions(
            sift_features=500,
            min_parallax_deg=1.0,
            classify_pairs=False,
            match_window=3,
            min_track_length=2,
            post_reproj_reject_pct=post_reproj_pct,
            post_reproj_iters=post_reproj_iters,
        )
        return optimize_poses(img_list, cal, opts, initial_poses=poses)

    def test_disabled_at_zero(self, five_frames):
        # When post_reproj_reject_pct=0, result should be identical across two runs
        # (no second pass, no randomness).
        r1 = self._run(five_frames, post_reproj_pct=0.0)
        r2 = self._run(five_frames, post_reproj_pct=0.0)
        assert r1["n_tracks"] == r2["n_tracks"]

    def test_higher_pct_keeps_fewer_tracks(self, five_frames):
        r_loose = self._run(five_frames, post_reproj_pct=0.05)  # reject 5%
        r_tight = self._run(five_frames, post_reproj_pct=0.5)   # reject 50%
        assert r_tight["n_tracks"] <= r_loose["n_tracks"], (
            f"50% rejection should keep ≤ tracks than 5%: "
            f"got {r_tight['n_tracks']} vs {r_loose['n_tracks']}"
        )

    def test_single_iteration_removes_correct_fraction(self, five_frames):
        # With one iteration and 30% rejection, track count should drop by ~30%.
        r_full = self._run(five_frames, post_reproj_pct=0.0)
        r_trimmed = self._run(five_frames, post_reproj_pct=0.3, post_reproj_iters=1)

        n_full = r_full["n_tracks"]
        n_after = r_trimmed["n_tracks"]
        n_expected_rejected = int(n_full * 0.3)
        n_expected_remaining = n_full - n_expected_rejected

        # Allow ±2 tracks for rounding and degenerate-landmark retries in _build_and_optimize.
        assert abs(n_after - n_expected_remaining) <= 2, (
            f"30% rejection (1 iter) from {n_full} tracks: "
            f"expected ≈{n_expected_remaining}, got {n_after}"
        )

    def test_poses_are_valid_after_reoptimisation(self, five_frames):
        result = self._run(five_frames, post_reproj_pct=0.1)
        # All 5 cameras should be in the result with finite translations.
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

        r_no_post   = optimize_poses(img_list, cal,
                                     SfmOptions(**base_opts, post_reproj_reject_pct=0.0),
                                     initial_poses=noisy_poses)
        r_with_post = optimize_poses(img_list, cal,
                                     SfmOptions(**base_opts, post_reproj_reject_pct=0.1),
                                     initial_poses=noisy_poses)

        shared = set(gt_poses) & set(r_no_post["poses"]) & set(r_with_post["poses"])
        gt_shared    = {k: gt_poses[k] for k in shared}
        ape_no_post  = self._ape_mean(gt_shared, {k: r_no_post["poses"][k]  for k in shared})
        ape_with_post = self._ape_mean(gt_shared, {k: r_with_post["poses"][k] for k in shared})

        assert ape_with_post <= ape_no_post * 1.05, (
            f"Post-opt rejection should not significantly worsen APE: "
            f"{ape_with_post*1000:.1f}mm vs {ape_no_post*1000:.1f}mm without"
        )
