# GTSAM SfM Design Patterns

Notes extracted from official GTSAM SfM/visual-SLAM examples
(SFMExample.cpp, SFMExample_bal.cpp, testVisualISAMExample.py, etc.)
and confirmed against our own pipeline experiments.

## Triangulation

- **Never use 2-view DLT** for initializing 3D points. It is numerically poor,
  especially when the two chosen views have low baseline or high distortion.
- Use `gtsam.triangulatePoint3(pose_vec, calibration, measurements, rank_tol=1e-9, optimize=True)`.
  This uses all visible cameras and applies nonlinear refinement.
- **Observations must be distorted pixels** (not undistorted) — GTSAM applies the
  calibration model internally, so passing undistorted coordinates is wrong.
- `SmartProjectionPoseFactor` handles triangulation internally (no explicit point
  variables in the graph). Detects degenerate geometry via `point()` returning optional.
  Best for batch SfM with many tracks.

## Observation convention

- Observations should be **distorted pixel coordinates** when using any calibration
  model that applies distortion (Cal3DS2, Cal3Fisheye).
  `project()` applies the model's distortion internally and returns the distorted pixel.
- Do NOT use undistorted keypoints as factor observations with Cal3Fisheye — produces
  systematic error per unit of k1.

## Max iterations

- **200 iterations is sufficient for pinhole** (Cal3DS2 with small distortion).
- **Fisheye (Cal3Fisheye) needs more** — GTSAM examples use up to 10,000 iterations
  due to higher nonlinearity of the equidistant model.
- Our current default is 200; increase `--lm-iterations` for fisheye datasets if
  the optimizer hasn't converged.

## Optimizer choice

- **`DoglegOptimizer`** is the default in GTSAM examples (trust-region, more robust
  than LM for large initial error). Use it for SfM.
- `LevenbergMarquardtOptimizer` also works but may need more iterations for fisheye.
- Both share `NonlinearOptimizerParams.setMaxIterations()`.

## Factor types

- **`GenericProjectionFactor`** (calibration fixed): most common for SfM when calibration is known.
- **`GeneralSFMFactor2`** (calibration variable): self-calibration — Jacobian w.r.t. calibration params.
- **`SmartProjectionPoseFactor`**: landmark is implicit (no point key), triangulated on demand.
  Best for visual SLAM and batch SfM with many tracks.
- Do NOT mix factor types (e.g., GenericProjectionFactor for some cameras, GeneralSFMFactor2
  for others) — Jacobian shape mismatches cause silent failures.

## Gauge freedom (required priors)

- Two priors are required to fix the gauge:
  1. Tight prior on the **first camera pose** (fixes 6 DoF global position/orientation).
  2. Prior on **one 3D point** (fixes 1 DoF scale) — OR use metric poses from GPS/prior.
- Without both, the system has a valid null space and the optimizer won't converge.
- With SmartFactor, the second prior (scale DoF) is still needed.

## Noise models

- Projection factor: `Isotropic.Sigma(2, 1.5)` = 1.5 px std dev (our default).
- Pose prior: `Diagonal.Sigmas([rot_rad, rot_rad, rot_rad, t_m, t_m, t_m])`.
- Landmark prior: `Isotropic.Sigma(3, 0.1)` = 0.1 unit std dev.
- Zero pixel noise crashes the optimizer (singular information matrix).

## SmartProjectionPoseFactor usage

```python
from gtsam import SmartProjectionParams, SmartProjectionPoseFactor

params = SmartProjectionParams()
params.setRankTolerance(1e-9)
params.setLandmarkDistanceThreshold(10.0)   # reject points > 10m from camera
params.setDynamicOutlierRejectionThreshold(3.0)  # reject if reproj > 3 sigma

# One factor per track; add measurements incrementally
smart = SmartProjectionPoseFactor(pixel_noise, K, params)
for cam_idx, pixel in track.measurements:
    smart.add(pixel, X(cam_idx))
graph.push_back(smart)
```

## Pose perturbation (for testing / initialization)

Use manifold retraction rather than direct Euclidean noise:

```python
pose.retract(0.1 * np.random.standard_normal(6))
```

Direct Euclidean noise can produce invalid SE(3) elements.

## Cal3Fisheye specifics

- Equidistant model: `r_d = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)`
- `Cal3Fisheye.k()` returns `[k1, k2, k3, k4]` — matches `cv2.fisheye` D coefficient order.
- More nonlinear than pinhole → needs more iterations and Dogleg is strongly preferred.
- Foxglove distortion model name: `"kannala_brandt"` (NOT `"equidistant"` — Foxglove rejects it).
