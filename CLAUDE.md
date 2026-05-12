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
