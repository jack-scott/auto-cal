# auto-cal — project notes for Claude

## What this project does

Visual SfM calibration pipeline. All data flows through MCAP files:
input MCAP → load into GTSAM → solve graph → write output MCAP.

## Data model

Three primitive MCAP types the engine understands:

### Image — `foxglove.CompressedImage`
Fields used: `timestamp`, `frame_id` (child frame, e.g. `camera_link`), `data` (JPEG bytes), `format`.

### Calibration — `foxglove.CameraCalibration`
Fields used: `timestamp`, `frame_id`, `width`, `height`, `distortion_model` (`"plumb_bob"`),
`D` ([k1, k2, p1, p2, k3]), `K` (3×3 row-major intrinsic matrix).
Aligned to an image by matching `frame_id` + nearest timestamp.

### Pose — `foxglove.FrameTransform`
Fields used: `timestamp`, `parent_frame_id` (`"map"`), `child_frame_id` (matches image `frame_id`),
`translation` ({x,y,z} = camera position in map), `rotation` ({x,y,z,w} quaternion = R_wc).
Aligned to an image by matching `child_frame_id` + timestamp.
Covariance is stored in a companion JSON topic `/tf_covariance` keyed by timestamp+frame:
  `{"timestamp_ns": int, "child_frame_id": str, "sigma_xyz_m": float, "sigma_rpy_rad": float}`
  (or full 6×6 if available — TBD at dataset ingestion layer; engine reads whatever is there).

A **Camera** = Image + Calibration + Pose, resolved by aligning on `frame_id` + timestamp.
Multiple frame IDs → multiple cameras in the same GTSAM graph.

## Map frame and initialisation

Always work in `map` frame.

**No initial poses in MCAP**
- Chain relative poses from the essential matrix between each successive image pair:
  frame 0 at origin (R_wc = identity, t = 0), frame k pose = compose(frame k-1 pose, relative E).
- Essential matrix → relative R + t via `cv2.findEssentialMat` + `cv2.recoverPose` (pick
  the solution consistent with points in front of both cameras).
- Translation is unit-scale only — that is fine, the GTSAM graph resolves scale from feature tracks.
- These chained poses are initial values only; no prior factors are added since there is no
  external reference.

**Partial poses (some cameras have poses, some don't)**
- No special handling. Cameras with poses in the MCAP get a prior factor; cameras without
  do not. GTSAM resolves everything together from the feature-track constraints.

**With poses in MCAP**
- Each pose is loaded as a GTSAM prior with noise from the covariance companion topic.

**With GPS**
- First camera placed at map origin; derive TF `map → ECEF` from its GPS fix.
- Every subsequent camera expressed in map frame.
- Camera in ECEF = camera → map → ECEF.

**Default orientation (no prior)**
- Camera optical axis (Z in REP-103) points along world X+.
- R_wc = [[0, 0, 1], [1, 0, 0], [0, -1, 0]]  (see coordinate conventions below)

## Pose covariance

Set at the dataset ingestion layer — the engine treats it as opaque and uses it directly
as the GTSAM prior noise model. The engine does not infer or set covariance itself.

## Engine variants (run separately, in sequence)

1. **SfM engine** — solves camera *poses* given a fixed calibration.
   - Variables: Pose3 per camera.
   - Factors: `GenericProjectionFactorCal3DS2` (fixed Cal3DS2 constant, not a key),
              pose priors (if input poses present).
   - Input topics: `/camera/image`, `/camera/calibration`, `/tf` (optional priors).
   - Output topics: `/tf` (optimised poses).

2. **Calibration engine** — solves *calibration* given fixed poses + 3D geometry.
   - Variables: Cal3DS2 (one per camera model, shared if same model).
   - Factors: `GeneralSFMFactor2Cal3DS2` or `GenericProjectionFactorCal3DS2` with
              calibration as variable key, tight pose priors.
   - Input topics: `/camera/image`, `/camera/calibration` (initial guess), `/tf` (fixed poses).
   - Output topics: `/camera/calibration` (optimised intrinsics).

## Coordinate conventions

Camera frame (ROS REP-103): X right, Y down, Z forward (optical axis).
World/map frame: X+ forward (arbitrary heading), Y+ left, Z+ up — or ENU when GPS present.

**GTSAM Pose3 convention** (matches GTSAM's native expectation):
  `Pose3(R, t)` where R = R_wc, t = camera position in world.
  `transform_to(P)` = `R.inverse() × (P − t)` = `R_cw × (P − t)` = point in camera frame ✓

  The Foxglove `FrameTransform.rotation` quaternion already encodes R_wc (child→parent = camera→map),
  so FrameTransform → Pose3 is a direct conversion with no inversion needed.

**"Facing X+" default R_wc**:
  Camera Z (optical axis) = world X+, camera X (right) = world Y+, camera Y (down) = world −Z.
  R_wc columns = camera axes expressed in world = [[0,1,0], [0,0,-1], [1,0,0]]^T
               = [[0, 0, 1], [1, 0, 0], [0, -1, 0]]

**IMPORTANT — historical bug to avoid**: earlier code stored R_cw in Pose3 rotation, requiring
a flip (`_to_gtsam_pose`) at every GTSAM boundary. New code must store R_wc natively to
match GTSAM's expectation and avoid this error.

## Known issues and open problems

See `docs/degenerate_pair_geometry.md` for a full write-up.

**Degenerate frame pairs in eth3d_exhibition_hall (frames 61-64):**
- Pairs 61-62 and 63-64 are near-duplicate frames (0.2 mm baseline). RANSAC accepts
  ~98% of all matches as inliers — the essential matrix is completely ill-conditioned.
  These pairs produce no useful geometry and cause cheirality failures in surrounding tracks.
- Pair 62-63 is rotation-dominated (57 mm baseline, 23° rotation). GT poses triangulate
  fine (99.7% success), but 5 cm / 0.05 rad pose noise causes 98.5% failure.
- The classifier in `src/autocal/engine/pair_classifier.py` detects STATIC pairs by
  baseline and rotation thresholds. `classify_pair_without_poses` is a not-implemented stub.
- With noisy poses, skipping STATIC pairs before triangulation is the correct fix.
  Rotation-dominated pairs under noise are an open question (ratio gate vs. reprojection only).

## Investigation guidelines for Claude

**Before touching any code, understand the failure mode at each pipeline stage.**
The symptoms at the end of the pipeline (e.g. "30% cheirality failures") rarely point
directly to the fix. In this project:

- A high cheirality failure rate is not evidence that matches are bad. It can also mean
  the camera poses fed to the triangulator are wrong or degenerate.
- A high RANSAC retention rate (>90%) is a red flag, not a sign of good matches — it
  means the epipolar constraint is not constraining anything.
- APE getting *worse* after optimisation means some cameras have no valid constraints
  (under-determined), not that the optimizer is broken.

**Investigation workflow:**
1. Add diagnostic prints or a breakdown test to measure the failure at each stage.
2. Isolate the failing cases (which pairs, which frames, which tracks).
3. Use GT geometry as an oracle to classify failures — e.g. Sampson distance against
   GT poses separates bad matches from pose-noise-induced triangulation failures.
4. Look at actual images and matches before drawing conclusions.
5. Only then design a fix, and write a test that would have caught the problem first.

**Do not:**
- Add a cv2 fallback triangulator because GTSAM throws — this masks the real problem.
- Change exception handling or tolerances to make failures disappear silently.
- Assume a fix is correct without a test that fails before the fix and passes after.
