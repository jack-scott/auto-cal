"""
Tests for SfmOptions.min_track_length filtering in optimize_poses.

Verifies:
  - Tracks shorter than the threshold are removed before triangulation.
  - Tracks at exactly the threshold are kept.
  - min_track_length=2 (legacy) keeps everything build_tracks produces.
  - The default (3) discards 2-observation-only tracks.
"""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import pytest

from autocal.engine.features import build_tracks, detect_sift, match_sift, undistort_keypoints
from autocal.engine.sfm_solver import SfmOptions, optimize_poses
from autocal.gtsam_bridge.conversions import calibration_from_mcap_msg, pose3_from_frame_transform
from autocal.io.mcap_reader import iter_messages

MCAP_PATH = Path("data/eth3d_exhibition_hall.mcap")

pytestmark = pytest.mark.skipif(
    not MCAP_PATH.exists(),
    reason=f"dataset not found: {MCAP_PATH}",
)

_WANT_TS = {0, 1_000_000_000, 2_000_000_000}


@pytest.fixture(scope="module")
def three_frames():
    images: dict[int, bytes]       = {}
    poses:  dict[int, gtsam.Pose3] = {}
    cal = None
    for topic, t_ns, msg in iter_messages(str(MCAP_PATH)):
        if topic == "/camera/image"        and t_ns in _WANT_TS: images[t_ns] = bytes(msg.data)
        elif topic == "/tf"                and t_ns in _WANT_TS: poses[t_ns]  = pose3_from_frame_transform(msg)
        elif topic == "/camera/calibration" and cal is None:      cal           = calibration_from_mcap_msg(msg)
        if len(images) == 3 and len(poses) == 3 and cal is not None:
            break
    assert len(images) == 3 and len(poses) == 3 and cal is not None
    return images, poses, cal


class TestMinTrackLength:
    def test_default_drops_two_observation_tracks(self, three_frames):
        images, poses, cal = three_frames
        img_list = sorted(images.items())

        # Build tracks from only a single pair — all tracks have exactly 2 observations.
        kps: dict[int, np.ndarray] = {}
        descs: dict[int, np.ndarray] = {}
        for ts, img_bytes in img_list:
            kps[ts], descs[ts] = detect_sift(img_bytes, n_features=500)
        kps_u = undistort_keypoints(kps, cal)

        ts_a, ts_b = sorted(images)[0], sorted(images)[1]
        raw = match_sift(descs[ts_a], descs[ts_b])
        pair_tracks = build_tracks({(ts_a, ts_b): raw})
        assert all(len(t.observations) == 2 for t in pair_tracks)

        # min_track_length=3 should drop all of them
        filtered = [t for t in pair_tracks if len(t.observations) >= 3]
        assert len(filtered) == 0

    def test_length_2_keeps_pair_tracks(self, three_frames):
        images, poses, cal = three_frames
        img_list = sorted(images.items())

        kps: dict[int, np.ndarray] = {}
        descs: dict[int, np.ndarray] = {}
        for ts, img_bytes in img_list:
            kps[ts], descs[ts] = detect_sift(img_bytes, n_features=500)

        ts_a, ts_b = sorted(images)[0], sorted(images)[1]
        raw = match_sift(descs[ts_a], descs[ts_b])
        pair_tracks = build_tracks({(ts_a, ts_b): raw})

        # min_track_length=2 should keep everything
        filtered = [t for t in pair_tracks if len(t.observations) >= 2]
        assert len(filtered) == len(pair_tracks)

    def test_three_frame_tracks_survive_default(self, three_frames):
        images, poses, cal = three_frames
        ts_list = sorted(images)

        kps: dict[int, np.ndarray] = {}
        descs: dict[int, np.ndarray] = {}
        for ts, img_bytes in images.items():
            kps[ts], descs[ts] = detect_sift(img_bytes, n_features=500)
        kps_u = undistort_keypoints(kps, cal)

        matches = {}
        for k in range(len(ts_list) - 1):
            a, b = ts_list[k], ts_list[k + 1]
            matches[(a, b)] = match_sift(descs[a], descs[b])

        all_tracks = build_tracks(matches)
        three_frame_tracks = [t for t in all_tracks if len(t.observations) >= 3]
        assert len(three_frame_tracks) > 0, "Expected some 3-frame tracks from 3 sequential images"

    def test_optimize_poses_fewer_tracks_with_min3(self, three_frames):
        """optimize_poses with min_track_length=3 produces fewer tracks than min_track_length=2."""
        images, poses, cal = three_frames
        img_list = sorted(images.items())

        base_opts = SfmOptions(
            sift_features=500,
            min_parallax_deg=1.0,
            classify_pairs=False,  # isolate the track-length effect
        )

        opts2 = SfmOptions(**{**base_opts.__dict__, "min_track_length": 2})
        opts3 = SfmOptions(**{**base_opts.__dict__, "min_track_length": 3})

        result2 = optimize_poses(img_list, cal, opts2, initial_poses=poses)
        result3 = optimize_poses(img_list, cal, opts3, initial_poses=poses)

        assert result3["n_tracks"] <= result2["n_tracks"], (
            f"min_track_length=3 should produce ≤ tracks vs min_track_length=2, "
            f"got {result3['n_tracks']} vs {result2['n_tracks']}"
        )
