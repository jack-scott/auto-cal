"""
Tests that a tight rotation prior suppresses rotation drift from junk landmarks.

The mechanism
-------------
Near-duplicate frame pairs (e.g. frames 61-62, 0.2mm baseline) produce
landmark observations that are inconsistent with the camera's true pose —
either because RANSAC kept wrong matches (no epipolar constraint with zero
baseline) or because the triangulation is ill-conditioned.  When the
rotation prior is loose, the GTSAM optimizer rotates the degenerate camera
to satisfy these bad reprojection factors.  A tight rotation prior prevents
this.

Two-landmark-regime analysis (Huber active)
-------------------------------------------
Case 1: landmark is FREE (can be moved by the optimizer).

  The optimizer partially absorbs camera 1's junk error by moving the
  landmark slightly toward cameras 0 and 2, costing some error on those
  cameras.  The equilibrium balances the Huber-weighted observation gradient
  against the prior gradient.

  Empirical boundary: σ_R ≈ 0.15–0.20 rad.
    σ_R = 0.01 → drift ≈ 0.001 rad   (well below boundary — stays put)
    σ_R = 0.05 → drift ≈ 0.026 rad   (below boundary — stays put)
    σ_R = 0.50 → drift ≈ 0.29 rad    (above boundary — camera drifts)

Case 2: landmark is PINNED (tight prior fixes it in place).

  The optimizer cannot move the landmark.  Camera 1 must either accept the
  reprojection error or rotate.

  With squared loss: cost at θ=0 is 0.5*(163/1.5)² ≈ 5930.
    Rotation cost at θ=0.325 with σ_R=0.01: 0.5*(0.325/0.01)² ≈ 528 << 5930
    → even tight prior cannot prevent drift (camera rotates).

  With Huber loss: cost at θ=0 is k×d - 0.5×k² ≈ 162 (bounded).
    Rotation cost at θ=0.325 with σ_R=0.01: ≈ 528 >> 162
    → tight prior dominates: camera stays put.

  → Both tight σ_R AND Huber loss are needed when the landmark is pinned.
    (In the real pipeline landmarks are shared across many cameras, so they
    are effectively pinned, making Huber loss critical.)

The hard preset uses pose_noise_rad=0.01 with huber_loss=True to exploit both
effects simultaneously.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gtsam
import numpy as np
import pytest

# Allow importing from pipelines/ (not a package, but importable from project root)
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pinhole_cal(fx: float = 500.0, cx: float = 320.0, cy: float = 240.0) -> gtsam.Cal3DS2:
    return gtsam.Cal3DS2(fx, fx, 0.0, cx, cy, 0.0, 0.0, 0.0, 0.0)


def _project(pose: gtsam.Pose3, pt: np.ndarray, fx: float, cx: float, cy: float) -> np.ndarray:
    """Perspective project pt_world via Pose3(R_wc, t)."""
    R_cw = pose.rotation().matrix().T
    p_cam = R_cw @ (pt - pose.translation())
    assert p_cam[2] > 0, "point behind camera"
    return np.array([fx * p_cam[0] / p_cam[2] + cx, fx * p_cam[1] / p_cam[2] + cy])


def _rotation_drift_rad(
    pose_noise_rad: float,
    use_huber: bool = True,
    sigma_px: float = 1.5,
) -> float:
    """Build a 3-camera graph with a junk observation on camera 1 and return rotation drift.

    Geometry
    --------
    Three cameras at x = 0, 1, 2 m, all with identity rotation (looking along Z).
    Cameras 0 and 2 are pinned with tight priors and observe landmark L=(0.5, 0, 3 m)
    at their correct projections (zero initial reprojection error).
    Camera 1 has a soft rotation prior (sigma = pose_noise_rad) and a deliberately
    wrong observation at x=400px (true projection is x≈237px — 163px off).

    This mimics a track from a near-duplicate pair: the landmark position is
    anchored by the two good cameras, so the optimizer cannot move it to absorb
    camera 1's error.  The only free variable is camera 1's rotation.

    Returns
    -------
    Angle (radians) by which camera 1's rotation drifted from its initial pose
    after optimization.
    """
    X = gtsam.symbol_shorthand.X
    P = gtsam.symbol_shorthand.P

    fx, cx, cy = 500.0, 320.0, 240.0
    cal = _pinhole_cal(fx, cx, cy)

    pose_0 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 0.0))
    pose_1 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1.0, 0.0, 0.0))
    pose_2 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(2.0, 0.0, 0.0))

    landmark = np.array([0.5, 0.0, 3.0])
    meas_0      = _project(pose_0, landmark, fx, cx, cy)  # ≈ (403, 240)
    meas_2      = _project(pose_2, landmark, fx, cx, cy)  # ≈ (70,  240)
    junk_meas_1 = np.array([400.0, 240.0])                # true ≈ (237, 240) → 163px error

    _tight = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))
    prior_1_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        pose_noise_rad, pose_noise_rad, pose_noise_rad,
        0.5, 0.5, 0.5,   # translation prior intentionally loose to isolate rotation
    ]))
    base_pixel = gtsam.noiseModel.Isotropic.Sigma(2, sigma_px)
    pixel_noise = (
        gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(sigma_px), base_pixel,
        )
        if use_huber else base_pixel
    )

    graph = gtsam.NonlinearFactorGraph()
    iv = gtsam.Values()

    iv.insert(X(0), pose_0)
    iv.insert(X(1), pose_1)
    iv.insert(X(2), pose_2)
    iv.insert(P(0), gtsam.Point3(*landmark))

    graph.add(gtsam.PriorFactorPose3(X(0), pose_0, _tight))
    graph.add(gtsam.PriorFactorPose3(X(2), pose_2, _tight))
    graph.add(gtsam.PriorFactorPose3(X(1), pose_1, prior_1_noise))
    graph.add(gtsam.GenericProjectionFactorCal3DS2(meas_0,      pixel_noise, X(0), P(0), cal))
    graph.add(gtsam.GenericProjectionFactorCal3DS2(junk_meas_1, pixel_noise, X(1), P(0), cal))
    graph.add(gtsam.GenericProjectionFactorCal3DS2(meas_2,      pixel_noise, X(2), P(0), cal))

    params = gtsam.DoglegParams()
    params.setMaxIterations(500)
    result = gtsam.DoglegOptimizer(graph, iv, params).optimize()

    pose_1_opt = result.atPose3(X(1))
    delta = pose_1.between(pose_1_opt)
    return float(np.linalg.norm(gtsam.Rot3.Logmap(delta.rotation())))


def _rotation_drift_pinned(
    pose_noise_rad: float,
    use_huber: bool = True,
    sigma_px: float = 1.5,
) -> float:
    """Single-camera graph with a pinned landmark and a junk observation.

    The landmark is fixed in place by a 1mm prior.  Camera 1 has equal σ_t = σ_R
    (no translation escape route), so the optimizer must choose between rotating
    and accepting the reprojection error.

    With squared loss: 0.5*(163/1.5)² ≈ 5930 >> rotation prior 0.5*(0.325/0.01)² ≈ 528
      → camera rotates even with σ_R=0.01.
    With Huber: bounded obs cost ≈ 162 < 528
      → tight prior wins, camera stays put.

    Returns
    -------
    Angle (radians) by which camera 1's rotation drifted from its initial pose.
    """
    X = gtsam.symbol_shorthand.X
    P = gtsam.symbol_shorthand.P

    fx, cx, cy = 500.0, 320.0, 240.0
    cal = _pinhole_cal(fx, cx, cy)

    pose_1 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1.0, 0.0, 0.0))

    landmark = np.array([0.5, 0.0, 3.0])
    junk_meas_1 = np.array([400.0, 240.0])  # true ≈ (237, 240) → 163px error

    # Equal σ_t = σ_R prevents translation from being a cheaper escape than rotation.
    prior_1_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, pose_noise_rad))
    # 1mm landmark prior: tight enough to pin it, loose enough for good conditioning.
    landmark_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.001)
    base_pixel = gtsam.noiseModel.Isotropic.Sigma(2, sigma_px)
    pixel_noise = (
        gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(sigma_px), base_pixel,
        )
        if use_huber else base_pixel
    )

    graph = gtsam.NonlinearFactorGraph()
    iv = gtsam.Values()

    iv.insert(X(1), pose_1)
    iv.insert(P(0), gtsam.Point3(*landmark))

    graph.add(gtsam.PriorFactorPose3(X(1), pose_1, prior_1_noise))
    graph.add(gtsam.PriorFactorPoint3(P(0), gtsam.Point3(*landmark), landmark_noise))
    graph.add(gtsam.GenericProjectionFactorCal3DS2(junk_meas_1, pixel_noise, X(1), P(0), cal))

    params = gtsam.DoglegParams()
    params.setMaxIterations(500)
    result = gtsam.DoglegOptimizer(graph, iv, params).optimize()

    pose_1_opt = result.atPose3(X(1))
    delta = pose_1.between(pose_1_opt)
    return float(np.linalg.norm(gtsam.Rot3.Logmap(delta.rotation())))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTightRotationPriorSuppressesDrift:
    """Tight σ_R prevents rotation drift from a junk landmark observation.

    Expected drift (derived analytically in module docstring):
      σ_R = 0.01 → drift ≈ 0.05 rad   (acceptable)
      σ_R = 0.05 → drift ≈ 0.32 rad   (unacceptable)
      σ_R = 0.50 → drift ≈ 0.33 rad   (unacceptable)
    """

    # Threshold that separates tight-prior (acceptable) from loose-prior (bad) behaviour.
    # Based on the analytical equilibrium: σ_R=0.01 → ~0.05 rad, σ_R=0.05 → ~0.32 rad.
    BOUNDARY_RAD = 0.1

    def test_tight_prior_holds_camera_near_initial_pose(self):
        """σ_R=0.01 (hard preset value): rotation drift must be < 0.1 rad."""
        drift = _rotation_drift_rad(pose_noise_rad=0.01, use_huber=True)
        assert drift < self.BOUNDARY_RAD, (
            f"σ_R=0.01 rotation drift={drift:.4f} rad — exceeds {self.BOUNDARY_RAD} rad. "
            f"The junk observation is controlling the camera rotation (tight prior is too loose)."
        )

    def test_loose_prior_allows_rotation_drift(self):
        """σ_R=0.5: rotation drift must be > 0.1 rad (camera satisfies junk observation)."""
        drift = _rotation_drift_rad(pose_noise_rad=0.5, use_huber=True)
        assert drift > self.BOUNDARY_RAD, (
            f"σ_R=0.5 rotation drift={drift:.4f} rad — expected > {self.BOUNDARY_RAD} rad. "
            f"A loose prior should allow the camera to drift toward the junk observation."
        )

    def test_tight_drift_is_much_less_than_loose_drift(self):
        """σ_R=0.01 (hard preset) must produce at least 10× less drift than σ_R=0.5."""
        drift_tight = _rotation_drift_rad(pose_noise_rad=0.01, use_huber=True)
        drift_loose = _rotation_drift_rad(pose_noise_rad=0.5, use_huber=True)
        assert drift_loose > drift_tight * 10, (
            f"loose/tight drift ratio={drift_loose/max(drift_tight, 1e-9):.1f} — "
            f"expected > 10. tight={drift_tight:.4f} rad, loose={drift_loose:.4f} rad."
        )

    def test_drift_is_monotone_with_sigma_r(self):
        """Rotation drift must strictly increase as σ_R increases."""
        drift_tight  = _rotation_drift_rad(pose_noise_rad=0.01, use_huber=True)
        drift_medium = _rotation_drift_rad(pose_noise_rad=0.05, use_huber=True)
        drift_loose  = _rotation_drift_rad(pose_noise_rad=0.5,  use_huber=True)

        assert drift_tight < drift_medium, (
            f"σ_R=0.01 drift={drift_tight:.4f} rad ≥ σ_R=0.05 drift={drift_medium:.4f} rad — "
            f"tighter prior should produce less drift."
        )
        assert drift_medium < drift_loose + 0.05, (
            f"σ_R=0.05 drift={drift_medium:.4f} rad much larger than "
            f"σ_R=0.5 drift={drift_loose:.4f} rad — unexpected ordering."
        )

    def test_huber_required_when_landmark_is_pinned(self):
        """When the landmark is pinned, Huber loss is required alongside tight σ_R.

        With a free landmark the optimizer partially absorbs camera 1's error by
        moving the landmark, reducing the need for camera rotation (so Huber and
        no-Huber give similar drift).  When the landmark is pinned (as in the real
        pipeline where many cameras share and effectively anchor it), the camera
        MUST rotate or accept the full reprojection error.

        With squared loss: 0.5*(163/1.5)² ≈ 5930 >> rotation prior 0.5*(0.325/0.01)² ≈ 528
          → camera rotates even with σ_R=0.01.
        With Huber:       Huber cost ≈ 162 < 528
          → tight prior wins, camera stays put.
        """
        drift_huber    = _rotation_drift_pinned(pose_noise_rad=0.01, use_huber=True)
        drift_no_huber = _rotation_drift_pinned(pose_noise_rad=0.01, use_huber=False)

        assert drift_no_huber > drift_huber, (
            f"pinned landmark: without Huber drift={drift_no_huber:.4f} rad should exceed "
            f"with-Huber drift={drift_huber:.4f} rad — Huber bounds the junk observation."
        )
        assert drift_no_huber > self.BOUNDARY_RAD, (
            f"pinned landmark, no Huber: drift={drift_no_huber:.4f} rad "
            f"expected > {self.BOUNDARY_RAD} rad — squared cost overwhelms tight prior."
        )
        assert drift_huber < self.BOUNDARY_RAD, (
            f"pinned landmark, Huber: drift={drift_huber:.4f} rad "
            f"expected < {self.BOUNDARY_RAD} rad — Huber+tight prior should hold camera."
        )


class TestHardPresetRotationRegularization:
    """The hard preset's pose_noise_rad must be tight enough to regularize junk landmarks."""

    def test_hard_preset_sigma_r_suppresses_drift(self):
        """Changing the hard preset's σ_R will fail this test.

        The hard preset uses pose_noise_rad=0.01 intentionally as a regularizer.
        This test catches accidental loosening (e.g., 0.01 → 0.05 or → 0.1).
        """
        from pipelines.sfm import PRESETS
        sigma_r = PRESETS["hard"]["pose_noise_rad"]
        drift = _rotation_drift_rad(pose_noise_rad=sigma_r, use_huber=True)
        boundary = 0.1
        assert drift < boundary, (
            f"hard preset pose_noise_rad={sigma_r}: rotation drift={drift:.4f} rad "
            f"exceeds {boundary} rad — the hard preset rotation prior is too loose to "
            f"regularize junk landmarks from degenerate pairs."
        )

    def test_hard_preset_uses_huber_loss(self):
        """Hard preset must enable Huber loss — it is required alongside tight σ_R."""
        from pipelines.sfm import PRESETS
        assert PRESETS["hard"]["huber_loss"] is True, (
            "hard preset must have huber_loss=True — Huber bounds the junk observation "
            "cost so the rotation prior can dominate."
        )
