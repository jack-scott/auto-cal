"""
Match quality tests using ground-truth geometry as the correctness oracle.

For each matched keypoint pair (kp_a, kp_b), a match is labelled CORRECT if
the Sampson distance to the GT epipolar line is below a threshold.

This lets us measure what fraction of our pipeline's matches are genuine
correspondences vs. false positives, and guide tuning of ratio threshold,
RANSAC, etc.

Ground truth source: /tf poses from eth3d_exhibition_hall.mcap
"""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np
import pytest

from autocal.engine.features import (
    detect_sift,
    filter_matches_ransac,
    match_sift,
    undistort_keypoints,
    cal_to_K,
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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    """Load the first 3 frames with GT poses and calibration from the MCAP."""
    wanted_ts = {0, 1_000_000_000, 2_000_000_000}

    images: dict[int, bytes] = {}
    poses:  dict[int, gtsam.Pose3] = {}
    cal = None

    for topic, t_ns, msg in iter_messages(str(MCAP_PATH)):
        if topic == "/camera/image" and t_ns in wanted_ts:
            images[t_ns] = bytes(msg.data)
        elif topic == "/tf" and t_ns in wanted_ts:
            poses[t_ns] = pose3_from_frame_transform(msg)
        elif topic == "/camera/calibration" and cal is None:
            cal = calibration_from_mcap_msg(msg)
        if len(images) == 3 and len(poses) == 3 and cal is not None:
            break

    assert len(images) == 3, f"only got {len(images)} frames"
    assert len(poses)  == 3, f"only got {len(poses)} GT poses"
    return {"images": images, "poses": poses, "cal": cal}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _fundamental_matrix(
    pose_a: gtsam.Pose3,
    pose_b: gtsam.Pose3,
    K: np.ndarray,
) -> np.ndarray:
    """Compute F from GT poses such that x_b^T F x_a = 0 for undistorted pixels.

    Uses GTSAM's R_wc / t_world convention:
      R_AB = R_cw_B @ R_wc_A
      t_AB = R_cw_B @ (t_A - t_B)   (A-origin expressed in B-camera frame)
      E    = skew(t_AB) @ R_AB
      F    = K^{-T} E K^{-1}         (same K for both cameras here)
    """
    R_cw_b = pose_b.rotation().matrix().T
    R_wc_a = pose_a.rotation().matrix()
    t_a    = pose_a.translation()
    t_b    = pose_b.translation()

    R_AB = R_cw_b @ R_wc_a
    t_AB = R_cw_b @ (t_a - t_b)

    skew = np.array([
        [     0, -t_AB[2],  t_AB[1]],
        [ t_AB[2],       0, -t_AB[0]],
        [-t_AB[1],  t_AB[0],       0],
    ])
    E = skew @ R_AB

    K_inv = np.linalg.inv(K)
    return K_inv.T @ E @ K_inv


def _sampson_distances(
    F: np.ndarray,
    kps_a: np.ndarray,
    kps_b: np.ndarray,
    matches: list[tuple[int, int]],
) -> np.ndarray:
    """Return the Sampson distance for each match (vectorised).

    kps_a / kps_b are already undistorted pixel coordinates.
    """
    if not matches:
        return np.array([])

    ia = np.array([m[0] for m in matches])
    ib = np.array([m[1] for m in matches])
    pa = np.column_stack([kps_a[ia], np.ones(len(matches))])   # (N,3)
    pb = np.column_stack([kps_b[ib], np.ones(len(matches))])   # (N,3)

    Fp  = (F  @ pa.T).T   # (N,3)  epipolar lines in B
    FTp = (F.T @ pb.T).T  # (N,3)  epipolar lines in A

    num = (pb * Fp).sum(axis=1) ** 2
    den = Fp[:, 0]**2 + Fp[:, 1]**2 + FTp[:, 0]**2 + FTp[:, 1]**2
    den = np.where(den < 1e-12, 1e-12, den)
    return num / den


def _inlier_rate(
    sampson: np.ndarray,
    threshold_px: float = 1.5,
) -> float:
    """Fraction of matches with Sampson distance below threshold (pixels)."""
    if len(sampson) == 0:
        return 0.0
    return float((sampson < threshold_px**2).mean())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMatchQualityPair01:
    """Measure match quality between frames 0 s and 1 s."""

    TS_A = 0
    TS_B = 1_000_000_000

    @pytest.fixture(autouse=True)
    def _setup(self, dataset):
        self.img_a   = dataset["images"][self.TS_A]
        self.img_b   = dataset["images"][self.TS_B]
        self.pose_a  = dataset["poses"][self.TS_A]
        self.pose_b  = dataset["poses"][self.TS_B]
        self.cal     = dataset["cal"]
        self.K       = cal_to_K(self.cal)

        self.kps_a, self.descs_a = detect_sift(self.img_a, n_features=0)
        self.kps_b, self.descs_b = detect_sift(self.img_b, n_features=0)

        undist = undistort_keypoints(
            {self.TS_A: self.kps_a, self.TS_B: self.kps_b}, self.cal
        )
        self.kps_a_u = undist[self.TS_A]
        self.kps_b_u = undist[self.TS_B]

        self.F = _fundamental_matrix(self.pose_a, self.pose_b, self.K)

    def test_feature_counts(self):
        """Sanity check: both frames produce a reasonable number of features."""
        assert len(self.kps_a) > 500, f"frame 0: only {len(self.kps_a)} features"
        assert len(self.kps_b) > 500, f"frame 1: only {len(self.kps_b)} features"

    def test_ratio_075_inlier_rate(self):
        """Baseline: Lowe ratio=0.75 — what fraction are geometrically correct?"""
        matches = match_sift(self.descs_a, self.descs_b, ratio=0.75)
        sampson = _sampson_distances(self.F, self.kps_a_u, self.kps_b_u, matches)
        rate = _inlier_rate(sampson)
        print(f"\n  ratio=0.75: {len(matches)} matches, {rate:.1%} inliers")
        # Just report — no hard assertion, we're measuring baseline
        assert len(matches) > 50, "too few matches to measure quality"
        assert rate > 0.0, "no inliers at all — F matrix may be wrong"

    def test_ratio_065_inlier_rate(self):
        """Stricter ratio=0.65 — does tightening the ratio improve inlier rate?"""
        matches = match_sift(self.descs_a, self.descs_b, ratio=0.65)
        sampson = _sampson_distances(self.F, self.kps_a_u, self.kps_b_u, matches)
        rate = _inlier_rate(sampson)
        print(f"\n  ratio=0.65: {len(matches)} matches, {rate:.1%} inliers")
        assert len(matches) > 20

    def test_ransac_inlier_rate(self):
        """After RANSAC: how much does it improve vs. ratio-test-only?"""
        matches_raw = match_sift(self.descs_a, self.descs_b, ratio=0.75)
        inliers = filter_matches_ransac(
            self.kps_a_u, self.kps_b_u, matches_raw,
            ransac_threshold=2.0, min_inliers=8,
        )
        sampson_raw    = _sampson_distances(self.F, self.kps_a_u, self.kps_b_u, matches_raw)
        sampson_ransac = _sampson_distances(self.F, self.kps_a_u, self.kps_b_u, inliers)
        rate_raw    = _inlier_rate(sampson_raw)
        rate_ransac = _inlier_rate(sampson_ransac)
        print(f"\n  ratio=0.75, no RANSAC:    {len(matches_raw)} matches, {rate_raw:.1%} inliers")
        print(f"  ratio=0.75, RANSAC t=2px: {len(inliers)} matches,   {rate_ransac:.1%} inliers")
        assert rate_ransac >= rate_raw, \
            f"RANSAC made inlier rate worse: {rate_raw:.1%} → {rate_ransac:.1%}"

    def test_inlier_rate_breakdown_by_threshold(self):
        """Show inlier rate at 0.5, 1.0, 1.5, 2.0, 3.0 px Sampson threshold."""
        matches = match_sift(self.descs_a, self.descs_b, ratio=0.75)
        sampson = _sampson_distances(self.F, self.kps_a_u, self.kps_b_u, matches)
        print(f"\n  {len(matches)} matches (ratio=0.75):")
        for thr in [0.5, 1.0, 1.5, 2.0, 3.0]:
            rate = _inlier_rate(sampson, thr)
            print(f"    Sampson < {thr:.1f} px: {rate:.1%}")
        # Soft assertion: majority should be correct at 3px
        assert _inlier_rate(sampson, 3.0) > 0.3, \
            "less than 30% of matches correct at 3px — matching is very noisy"


class TestMatchQualityPair12:
    """Same measurements for the harder pair: frames 1 s and 2 s (24° rotation)."""

    TS_A = 1_000_000_000
    TS_B = 2_000_000_000

    @pytest.fixture(autouse=True)
    def _setup(self, dataset):
        self.img_a   = dataset["images"][self.TS_A]
        self.img_b   = dataset["images"][self.TS_B]
        self.pose_a  = dataset["poses"][self.TS_A]
        self.pose_b  = dataset["poses"][self.TS_B]
        self.cal     = dataset["cal"]
        self.K       = cal_to_K(self.cal)

        self.kps_a, self.descs_a = detect_sift(self.img_a, n_features=0)
        self.kps_b, self.descs_b = detect_sift(self.img_b, n_features=0)

        undist = undistort_keypoints(
            {self.TS_A: self.kps_a, self.TS_B: self.kps_b}, self.cal
        )
        self.kps_a_u = undist[self.TS_A]
        self.kps_b_u = undist[self.TS_B]

        self.F = _fundamental_matrix(self.pose_a, self.pose_b, self.K)

    def test_ratio_075_inlier_rate(self):
        matches = match_sift(self.descs_a, self.descs_b, ratio=0.75)
        sampson = _sampson_distances(self.F, self.kps_a_u, self.kps_b_u, matches)
        rate = _inlier_rate(sampson)
        print(f"\n  ratio=0.75: {len(matches)} matches, {rate:.1%} inliers")
        assert len(matches) > 50
        assert rate > 0.0

    def test_ransac_inlier_rate(self):
        matches_raw = match_sift(self.descs_a, self.descs_b, ratio=0.75)
        inliers = filter_matches_ransac(
            self.kps_a_u, self.kps_b_u, matches_raw,
            ransac_threshold=2.0, min_inliers=8,
        )
        sampson_raw    = _sampson_distances(self.F, self.kps_a_u, self.kps_b_u, matches_raw)
        sampson_ransac = _sampson_distances(self.F, self.kps_a_u, self.kps_b_u, inliers)
        rate_raw    = _inlier_rate(sampson_raw)
        rate_ransac = _inlier_rate(sampson_ransac)
        print(f"\n  ratio=0.75, no RANSAC:    {len(matches_raw)} matches, {rate_raw:.1%} inliers")
        print(f"  ratio=0.75, RANSAC t=2px: {len(inliers)} matches,   {rate_ransac:.1%} inliers")
        assert rate_ransac >= rate_raw

    def test_inlier_rate_breakdown_by_threshold(self):
        matches = match_sift(self.descs_a, self.descs_b, ratio=0.75)
        sampson = _sampson_distances(self.F, self.kps_a_u, self.kps_b_u, matches)
        print(f"\n  {len(matches)} matches (ratio=0.75):")
        for thr in [0.5, 1.0, 1.5, 2.0, 3.0]:
            rate = _inlier_rate(sampson, thr)
            print(f"    Sampson < {thr:.1f} px: {rate:.1%}")
        assert _inlier_rate(sampson, 3.0) > 0.3
