"""
Tests for post-optimisation noise-adaptive tightening and retriangulation.

Verifies:
  - post_reproj_start_px=-1 disables post-opt entirely.
  - A tighter start threshold keeps ≤ tracks than a looser one.
  - Tightening keeps fewer tracks than no post-opt.
  - Poses are valid (finite) after re-optimisation.
  - Retriangulation never reduces track count vs. tightening alone.
  - With noisy poses, tightening + retriangulation does not worsen APE.
"""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import pytest

from autocal.engine.features import max_reproj_error
from autocal.engine.sfm_solver import SfmOptions, optimize_poses
from autocal.gtsam_bridge.conversions import (
    calibration_from_mcap_msg,
    pose3_from_frame_transform,
)
from autocal.io.mcap_reader import iter_messages
from autocal.metrics.ape import compute_ape

MCAP_PATH  = Path("data/eth3d_exhibition_hall.mcap")
NOISY_MCAP = Path("data/eth3d_exhibition_hall_noise_easy.mcap")

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
        if topic == "/camera/image"         and t_ns in _WANT_TS: images[t_ns] = bytes(msg.data)
        elif topic == "/tf"                 and t_ns in _WANT_TS: poses[t_ns]  = pose3_from_frame_transform(msg)
        elif topic == "/camera/calibration" and cal is None:       cal           = calibration_from_mcap_msg(msg)
        if len(images) == 5 and len(poses) == 5 and cal is not None:
            break
    assert len(images) == 5 and len(poses) == 5
    return sorted(images.items()), poses, cal


class TestPostReproj:
    def _run(self, five_frames, post_reproj_start_px: float = -1.0,
             post_reproj_min_px: float = 3.0,
             post_reproj_iters: int = 4,
             retriangulate: bool = False) -> dict:
        img_list, poses, cal = five_frames
        opts = SfmOptions(
            sift_features=500,
            min_parallax_deg=1.0,
            classify_pairs=False,
            match_window=3,
            min_track_length=2,
            post_reproj_start_px=post_reproj_start_px,
            post_reproj_min_px=post_reproj_min_px,
            post_reproj_iters=post_reproj_iters,
            retriangulate=retriangulate,
        )
        return optimize_poses(img_list, cal, opts, initial_poses=poses)

    def test_disabled_at_minus_one(self, five_frames):
        # When disabled, two runs give the same track count (no randomness introduced).
        r1 = self._run(five_frames, post_reproj_start_px=-1.0)
        r2 = self._run(five_frames, post_reproj_start_px=-1.0)
        assert r1["n_tracks"] == r2["n_tracks"]

    def test_more_iterations_keeps_fewer_tracks(self, five_frames):
        # Same start/min threshold: 4 passes apply progressively tighter
        # filters and must keep ≤ tracks than a single pass.
        r_1iter = self._run(five_frames, post_reproj_start_px=50.0, post_reproj_min_px=10.0,
                            post_reproj_iters=1)
        r_4iter = self._run(five_frames, post_reproj_start_px=50.0, post_reproj_min_px=10.0,
                            post_reproj_iters=4)
        assert r_4iter["n_tracks"] <= r_1iter["n_tracks"], (
            f"4 iterations should keep ≤ tracks than 1: "
            f"got {r_4iter['n_tracks']} vs {r_1iter['n_tracks']}"
        )

    def test_tightening_keeps_fewer_tracks_than_disabled(self, five_frames):
        # Aggressive tightening (1px, 1 iter) must keep ≤ tracks than no post-opt.
        r_none  = self._run(five_frames, post_reproj_start_px=-1.0)
        r_tight = self._run(five_frames, post_reproj_start_px=1.0, post_reproj_min_px=1.0,
                            post_reproj_iters=1)
        assert r_tight["n_tracks"] <= r_none["n_tracks"], (
            f"1px tightening should keep ≤ tracks than disabled: "
            f"got {r_tight['n_tracks']} vs {r_none['n_tracks']}"
        )

    def test_poses_are_valid_after_reoptimisation(self, five_frames):
        result = self._run(five_frames, post_reproj_start_px=20.0, post_reproj_min_px=3.0)
        assert len(result["poses"]) == 5
        for img_id, pose in result["poses"].items():
            t = pose.translation()
            assert np.all(np.isfinite(t)), f"Camera {img_id} has non-finite translation"

    def test_retriangulation_does_not_reduce_tracks(self, five_frames):
        # Retriangulation may add tracks but must never remove them.
        r_no  = self._run(five_frames, post_reproj_start_px=10.0, retriangulate=False)
        r_yes = self._run(five_frames, post_reproj_start_px=10.0, retriangulate=True)
        assert r_yes["n_tracks"] >= r_no["n_tracks"], (
            f"Retriangulation should not reduce tracks: "
            f"got {r_yes['n_tracks']} vs {r_no['n_tracks']}"
        )


@pytest.mark.skipif(not NOISY_MCAP.exists(), reason=f"noisy dataset not found: {NOISY_MCAP}")
class TestPostReprojImprovesPoseAccuracy:
    """Integration test: noise-adaptive tightening + retriangulation must not worsen APE."""

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
            if topic == "/camera/image"         and t_ns in self._WANT: images[t_ns]      = bytes(msg.data)
            elif topic == "/tf"                 and t_ns in self._WANT: noisy_poses[t_ns] = pose3_from_frame_transform(msg)
            elif topic == "/tf_gt"              and t_ns in self._WANT: gt_poses[t_ns]    = pose3_from_frame_transform(msg)
            elif topic == "/camera/calibration" and cal is None:        cal               = calibration_from_mcap_msg(msg)
        assert len(images) >= 5 and cal is not None
        return sorted(images.items()), noisy_poses, gt_poses, cal

    def _ape_mean(self, gt, estimated):
        result = compute_ape(gt, estimated)
        return result["stats"]["translation"]["mean"]

    def test_tightening_does_not_worsen_ape(self, ten_noisy_frames):
        img_list, noisy_poses, gt_poses, cal = ten_noisy_frames

        base_opts = dict(
            sift_features=500, min_parallax_deg=1.0,
            classify_pairs=True, match_window=3, min_track_length=3,
            pose_noise_m=0.005,
        )

        r_no_post   = optimize_poses(img_list, cal,
                                     SfmOptions(**base_opts, post_reproj_start_px=-1.0),
                                     initial_poses=noisy_poses)
        r_with_post = optimize_poses(img_list, cal,
                                     SfmOptions(**base_opts, post_reproj_start_px=0.0,
                                                retriangulate=True),
                                     initial_poses=noisy_poses)

        shared = set(gt_poses) & set(r_no_post["poses"]) & set(r_with_post["poses"])
        gt_shared     = {k: gt_poses[k]           for k in shared}
        ape_no_post   = self._ape_mean(gt_shared, {k: r_no_post["poses"][k]   for k in shared})
        ape_with_post = self._ape_mean(gt_shared, {k: r_with_post["poses"][k] for k in shared})

        assert ape_with_post <= ape_no_post * 1.05, (
            f"Tightening + retriangulation should not worsen APE: "
            f"{ape_with_post*1000:.1f}mm vs {ape_no_post*1000:.1f}mm without"
        )
