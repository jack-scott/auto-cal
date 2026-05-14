# Pipeline Design and Implementation Plan

## What the pipeline needs to do

Given pose-prior MCAPs (camera images + initial poses from `/tf`), produce a
refined camera calibration and/or refined poses.  The key insight is that pose
priors let us do geometry classification *before* any image processing, which
makes the matching and triangulation stages much cheaper and more reliable.

---

## What already exists

| Component | Location | State |
|-----------|----------|-------|
| SIFT detection + MCAP caching | `features.py: detect_sift, encode/decode_sift_features` | done |
| Lowe ratio matching | `features.py: match_sift` | done |
| RANSAC + essential matrix filtering | `features.py: filter_matches_ransac` | done |
| Track building (union-find) | `features.py: build_tracks` | done, min 2 obs |
| Multi-view triangulation (GTSAM) | `features.py: triangulate_gtsam` | done, parallax + dist filters |
| Pose chain initialisation | `features.py: chain_essential_matrix` | done |
| SfM graph + Dogleg optimiser | `sfm_solver.py: optimize_poses` | done, sequential pairs only |
| Pair classification with poses | `pair_classifier.py: classify_pair_with_poses` | done |
| H/E ratio test without poses | `pair_classifier.py: classify_pair_without_poses` | done |

---

## What is missing

### 1. Keyframe selection

Currently every frame enters the graph.  For calibration we want a diverse
subset — enough translation and rotation excitation, no redundant near-duplicate
frames.

```python
def select_keyframes(
    poses: dict[int, gtsam.Pose3],
    img_ids: list[int],
    min_translation_m: float = 0.05,
    min_rotation_deg: float = 3.0,
) -> list[int]:
    """Return a subset of img_ids such that consecutive keyframes differ by at
    least min_translation_m OR min_rotation_deg from the previous keyframe."""
```

Location: new function in `features.py` or a new `keyframes.py`.

Tuning note: pure-rotation keyframes are valuable for focal-length excitation but
not for triangulation.  Don't filter them out — the pair classifier handles them
downstream.

---

### 2. Covisibility window matching

`sfm_solver.py` currently matches only sequential pairs (k ↔ k+1).  After
keyframe selection the selected frames may be non-sequential (gaps where frames
were dropped), and even sequential keyframes benefit from a small look-ahead
window for track length.

```python
# Replace:
for k in range(len(img_ids) - 1):
    id_a, id_b = img_ids[k], img_ids[k + 1]

# With:
for k in range(len(img_ids)):
    for offset in range(1, window + 1):          # window = 3–5
        if k + offset >= len(img_ids):
            break
        id_a, id_b = img_ids[k], img_ids[k + offset]
```

Location: `sfm_solver.py: optimize_poses`, matching loop.

---

### 3. Pair classifier integration into sfm_solver

The classifier exists but `optimize_poses` does not call it.  Every pair —
including STATIC ones — currently goes through RANSAC and triangulation.

Where to insert (after the RANSAC loop, before track building):

```python
if initial_poses is not None:
    filtered_by_class: dict[tuple, list] = {}
    for (id_a, id_b), m in matches_per_pair.items():
        cls = classify_pair_with_poses(poses[id_a], poses[id_b])
        if cls == PairClass.STATIC:
            continue                        # skip — no useful geometry
        if cls == PairClass.PURE_ROTATION:
            continue                        # skip triangulation (see §4)
        filtered_by_class[(id_a, id_b)] = m
    matches_per_pair = filtered_by_class
```

Without pose priors, fall back to the H/E ratio test per pair.

Location: `sfm_solver.py: optimize_poses`, after RANSAC filtering.

---

### 4. PURE_ROTATION pair handling

Currently PURE_ROTATION pairs are silently dropped.  The correct behaviour is to
reproject observations against *existing* landmarks (not triangulate new ones).
This adds reprojection constraints for those cameras without requiring a baseline.

This directly fixes the camera 61–64 isolation problem at hard noise: those
cameras have no good translating pairs to anchor them, but they do share
landmarks with adjacent cameras.

```python
if pair_class == PairClass.PURE_ROTATION:
    # Use H-based filtering (E is ill-conditioned with no translation)
    matches = filter_matches_homography(kp_a, kp_b, raw_matches)
    for match in matches:
        lm_id = find_existing_landmark(match, track_map)
        if lm_id is not None:
            graph.add(gtsam.GenericProjectionFactorCal3DS2(
                gtsam.Point2(*observed_uv), obs_noise, X(frame_j), L(lm_id), cal
            ))
```

This requires the landmark map to be built before processing PURE_ROTATION pairs,
i.e. the pipeline must do one pass of triangulation first, then a second pass
adding pure-rotation reprojection factors.  Forward-pass dependency: does not
fit the current single-pass batch structure.

For now, dropping PURE_ROTATION pairs is correct and safe.  The reprojection-only
path is the next major architectural change once the pipeline moves toward
incremental or multi-pass structure.

---

### 5. Minimum track length filter (≥3 frames)

`build_tracks` currently keeps any track with ≥2 observations.  Two-frame tracks
add weak constraints to the graph.  Three-frame minimum significantly improves
calibration conditioning.

```python
# After build_tracks:
tracks = [t for t in tracks if len(t.observations) >= 3]
```

This is a one-liner change in `sfm_solver.py` after `build_tracks(matches_per_pair)`.

Trade-off: reduces track count, which may matter on short sequences.  Make it
configurable via `SfmOptions.min_track_length: int = 2`.

---

### 6. Outlier rejection + re-optimise

After the first optimisation pass, landmarks with high reprojection error should
be removed and the graph re-run.  This is already partially handled by
`filter_by_reproj` (called pre-optimisation when `max_reproj_error_px > 0` and
priors are present).  What is missing is a *post-optimisation* pass using the
optimised poses:

```python
result = optimizer.optimize()
opt_poses = extract_poses(result)

# Post-optimisation outlier rejection
good_tracks = filter_by_reproj(triangulated, keypoints, opt_poses,
                               calibration, threshold_px=2.0)
if len(good_tracks) < len(triangulated):
    result = rebuild_and_optimize(good_tracks, opt_poses, ...)
```

Location: `sfm_solver.py: _build_and_optimize` or a wrapper in `optimize_poses`.

**Iterative tightening (better than a fixed threshold):**

A single threshold fails because the appropriate level depends on the noise level and
how well the current poses are converged.  Metashape runs 4–6 BA passes with
decreasing thresholds.  The right approach:

```python
for threshold_px in [20.0, 10.0, 5.0, 3.0]:
    result = optimise(graph, initial_values)
    good_tracks = filter_by_reproj(..., threshold_px=threshold_px)
    graph, initial_values = rebuild(good_tracks, result)
```

Each BA pass improves poses enough that the next rejection round can afford to be
stricter.  The initial loose threshold (20px) avoids discarding tracks that are only
bad because the initial poses are noisy.

**Noise-adaptive initial threshold:**

When a noise level is known (σ_t_m, focal_px, scene_depth_m):

```python
expected_reproj_px = (sigma_t_m / scene_depth_m) * focal_px
start_threshold = max(5.0, 3.0 * expected_reproj_px)
```

This replaces the hard-coded 5px that cuts 180→9 tracks at medium noise.

---

### 7. Noise-adaptive pose prior

The prior noise model should reflect actual pose uncertainty, not a fixed preset.
At hard level (σ_t=100mm, σ_R=2.87°) a tight prior actively misleads the
optimiser: it forces cameras to stay near a wrong initialisation instead of letting
the geometric constraints pull them to the right solution.

```python
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
    sigma_r_rad, sigma_r_rad, sigma_r_rad,
    sigma_t_m,   sigma_t_m,   sigma_t_m,
]))
```

The `pose_noise_m` and `pose_noise_rad` preset parameters already control this.
The issue is that the hard preset currently tightens σ_R to 0.01 rad as a
*regulariser* against junk landmarks, which conflicts with using it as a true
uncertainty model.  These two concerns pull in opposite directions:

- **True uncertainty model**: prior should be loose (100mm / 2.87°) so geometry
  dominates over the noisy prior.
- **Junk-landmark regulariser**: prior should be tight (0.01 rad) so degenerate
  cameras don't drift.

The resolution is to fix junk-landmark connectivity (item §4 reprojection-only
path) so the regulariser role is no longer needed at hard level, and the prior
noise can then be set to the true uncertainty (loose).

---

### 8. Coarse-to-fine initialisation

At hard perturbation (100mm), Dogleg converges to a local minimum because the
initial point is too far from the solution.  Two strategies to widen the basin:

**Sequential pose refinement before full BA:**

```python
# Fix frame 0, solve frame 1 from landmarks visible in {0, 1}
# Fix {0, 1}, solve frame 2 from landmarks visible in {0, 1, 2}
# Use the resulting sequential chain as initial estimate for full BA
```

This is incremental PnP + local BA, similar to how Metashape registers frames.
Each new frame is initialised from a clean geometric estimate rather than a
100mm-noisy prior.

**Coarse-to-fine on landmark noise model:**

```python
# Pass 1: loose landmark noise → poses find good geometry
result1 = optimise_with_landmark_noise(graph, initial, sigma_lm=5.0)

# Pass 2: tighten and reoptimise with warm start
result2 = optimise_with_landmark_noise(graph, result1, sigma_lm=0.1)
```

Loose landmark noise lets poses move freely during early iterations; tightening
then locks the geometry.  This is the dual of the pose-prior loosening above.

---

### 9. Global pair selection / longer tracks

Window=3 matching produces mean track length 3.2, which weakly constrains poses.
Metashape's mean track length is 8–15 because it uses global pair selection (a
visual vocabulary tree finds all overlapping pairs regardless of temporal order)
and a smaller ratio test threshold.

Short-term improvements without vocabulary trees:

- **Increase match window** to 6–8 at hard level.
- **Lower Lowe ratio threshold** from 0.75 → 0.70 to get more raw matches; the
  RANSAC step then filters false positives.  Risk: more outliers reaching the
  optimizer; Huber loss handles this.
- **Verify union-find track merging** isn't creating duplicate track IDs for the
  same physical point — spurious splits shorten tracks.

Track length target for hard level: mean > 5, some tracks spanning 10+ frames.

**Learned features (SuperPoint + LightGlue):**

SuperPoint keypoints + LightGlue matching is a plug-in replacement for
SIFT + Lowe ratio that produces significantly more inliers per pair and better
matches in low-texture / repetitive regions.  This is the single highest-impact
change for track length and track quality, at the cost of GPU dependency and
inference time.  Medium effort; high impact at hard level.

---

## Full pipeline (target state)

```
Images + Pose Priors
        │
        ▼
1. SIFT EXTRACTION (with MCAP cache)
   detect_sift per frame
        │
        ▼
2. KEYFRAME SELECTION                           ← missing
   select_keyframes(poses, min_t, min_r)
        │
        ▼
3. COVISIBILITY MATCHING (window=3–5)           ✓ COMPLETED
   match_sift + filter_matches_ransac per pair
        │
        ▼
4. PAIR CLASSIFICATION                          ✓ COMPLETED
   classify_pair_with_poses (or H/E fallback)
   STATIC       → drop pair
   PURE_ROTATION → keep for reprojection pass (step 7b)  ← §4 pending
   GOOD / PURE_TRANSLATION → proceed
        │
        ▼
5. TRACK BUILDING
   build_tracks → filter min_track_length ≥ 3  ✓ COMPLETED
        │
        ▼
6. TRIANGULATION
   triangulate_gtsam with parallax ≥ 2°, max_dist filter
        │
        ▼
7. GTSAM BATCH OPTIMISATION
   PriorFactorPose3 per camera (noise = actual σ_t, σ_R)  ← §7 noise-adaptive
   GenericProjectionFactor per (camera, landmark, observation)
   Robust Huber loss throughout
   Dogleg optimiser
        │
   7b. PURE_ROTATION reprojection factors       ← §4 pending
       match via H, look up existing landmarks,
       add reprojection-only factors for those cameras
        │
        ▼
8. ITERATIVE OUTLIER REJECTION + RE-OPTIMISE   ← §6 (threshold tightening)
   thresholds [20px, 10px, 5px, 3px], warm-start each pass
   replace single-pass 5px with noise-adaptive start threshold
        │
        ▼
Refined poses and/or calibration
```

---

## Implementation order

These are ordered by impact and dependency:

1. **Pair classifier integration** (§3) — ✓ COMPLETED
   `filter_pairs_by_geometry` in `pair_classifier.py`, called from `sfm_solver.py`.
   `SfmOptions.classify_pairs: bool = True`, `he_secondary_check: bool = True`,
   `he_secondary_ratio: float = 0.99` to control secondary H/E check.
   11 tests in `tests/test_pair_filter.py`.
   H/E secondary check catches near-duplicates (ratio ≥ 0.99) when pose noise exceeds
   true baseline. Disabled for hard noise (`--no-he-secondary-check`) — at σ_t=100mm
   removing near-duplicate pairs isolates camera 61 (worse than leaving junk tracks in).

2. **Min track length = 3** (§5) — ✓ COMPLETED
   `SfmOptions.min_track_length: int = 3`, `--min-track-length` CLI flag.
   4 tests in `tests/test_min_track_length.py`.
   Note: with sequential-only matching, 88% of tracks are 2-frame only, so this
   filter reduces track count too aggressively and hurts APE until covisibility
   window matching (item 4) is added.

3. **Post-optimisation outlier rejection** (§6) — ✓ COMPLETED
   `SfmOptions.post_reproj_error_px: float = 5.0`, `--post-reproj-px` CLI flag.
   5 tests in `tests/test_post_reproj.py`.
   Guard added for empty clean list (skip re-optimisation rather than crash).
   Threshold still needs noise-adaptive tuning (5px cuts too many at medium noise).

4. **Covisibility window matching** (§2) — ✓ COMPLETED
   `SfmOptions.match_window: int = 3`, `--match-window` CLI flag.
   4 tests in `tests/test_match_window.py`.
   OpenCV USAC_MAGSAC assertion on degenerate skip-frame pairs fixed with try-except in
   `filter_matches_ransac`.  Easy: 10% APE improvement (420 tracks). Medium: 7%.
   Pre-opt reproj threshold needs tuning per noise level — 4px cuts all skip-frame tracks,
   0px lets noisy landmarks corrupt the optimizer.  Full benefit from post-opt rejection.

5. **Iterative outlier rejection + noise-adaptive threshold** (§6) — next up.
   Replace single-pass 5px with tightening schedule [20px, 10px, 5px, 3px], warm-starting
   each BA pass.  Add noise-adaptive start threshold (§6).
   Expected impact: medium noise improves from 7% to ≥10%; easy post-opt becomes useful.

6. **PURE_ROTATION reprojection path** (§4) — high priority for hard level.
   Requires two-pass structure: triangulate first, then add reprojection-only factors for
   PURE_ROTATION pairs against existing landmarks.  Fixes camera 61–64 isolation.
   Unblocks turning H/E secondary check back on at hard noise.

7. **Noise-adaptive pose prior** (§7) — pair with §4.
   Once junk-landmark isolation is fixed via reprojection-only path, the prior noise can
   be set to the true uncertainty (σ_t_m, σ_r_rad from the dataset), removing the conflict
   between regulariser and uncertainty model.

8. **Coarse-to-fine initialisation** (§8) — if items 5–7 still don't crack hard level.
   Sequential PnP chain before full BA, or coarse-to-fine on landmark noise model.
   High implementation effort; targets the local-minimum problem at σ_t=100mm.

9. **Wider match window + lower ratio threshold** (§9) — low effort, medium impact.
   Test match_window=6–8 at hard level.  Lower Lowe ratio 0.75→0.70 and verify
   track merging isn't splitting long tracks.

10. **Keyframe selection** (§1) — needed for long trajectories / large datasets.
    Can be skipped if input sequences are already sparse.

11. **Learned features (SuperPoint + LightGlue)** (§9) — highest single impact on
    track length and quality; GPU dependency.  Consider after SIFT pipeline is tuned.

---

## Open questions

- **H/E secondary check at hard noise:** removing near-duplicate pairs at σ_t=100mm
  isolates camera 61 (all its connections are either near-duplicates or rotation-dominated).
  The correct fix is PURE_ROTATION reprojection-only factors (§4) so cameras in the
  degenerate cluster still get observation constraints.  Until then, H/E disabled for hard.

- **Post-opt threshold tuning:** 5px cuts too many tracks at medium noise (180→9).
  Iterative tightening schedule (§6) replaces the fixed threshold; the noise-adaptive
  start threshold avoids cutting at levels proportional to the noise floor.

- **Rotation/translation ratio gate:** pairs like 62-63 (57mm, 23°, ratio 407°/m)
  are classified as GOOD but cause 98% cheirality failure under noisy poses.
  Worth adding a configurable `max_rotation_translation_ratio_deg_per_m` threshold
  to `classify_pair_with_poses`.

- **Homography matching for PURE_ROTATION:** if reprojection-only is implemented,
  pairs with pure rotation should use H-based filtering rather than E, since E is
  ill-conditioned with no translation.

- **Tight σ_R regulariser vs true uncertainty model (§7):** currently the hard preset
  sets σ_R=0.01 as a regulariser, which conflicts with using the prior as a true
  uncertainty model (should be 0.05 rad for 2.87°).  Resolving this requires §4
  (reprojection-only path) to eliminate the need for junk-landmark regularisation.

- **Local minimum at hard level:** at σ_t=100mm, Dogleg may converge to a local
  minimum.  Coarse-to-fine initialisation (§8) or sequential PnP chain addresses this
  if iterative outlier rejection (§6) + PURE_ROTATION path (§4) are not sufficient.

- **Track length (mean 3.2 vs target >5):** primary lever is wider match window and
  lower ratio threshold.  Verify union-find track merging isn't splitting physical
  points into multiple short tracks.

- **Calibration excitation:** for the calibration engine specifically, we want
  pairs with different depths in the scene (varying parallax) not just angular
  diversity.  Keyframe selection may need a depth-diversity criterion.
