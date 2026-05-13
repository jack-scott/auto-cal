# Development Workflow

How to add features to the SfM pipeline, validate them, and record results.

---

## The iteration loop

```
1. Add feature to sfm_solver.py / features.py / pair_classifier.py
2. Write tests (unit + integration where applicable)
3. Run unit tests: pixi run test
4. Run each noise level: pixi run eth3d-exhibition-hall-noise-{easy,medium,hard}-sfm
5. Update docs/e2e_results.md with new numbers
6. Mark the item COMPLETED in docs/pipeline_plan.md
```

---

## Running the pipeline

### Prerequisites

Download the dataset once:
```
pixi run eth3d-exhibition-hall-prepare
```

### Noise levels — run in order (easy → medium → hard)

| Task | σ_t | σ_R | Notes |
|------|-----|-----|-------|
| `eth3d-exhibition-hall-noise-easy-sfm` | 5 mm | 0.11° | pose classifier catches degenerate pairs |
| `eth3d-exhibition-hall-noise-medium-sfm` | 20 mm | 0.57° | σ_t > 0.2mm baseline; classifier partially blind |
| `eth3d-exhibition-hall-noisy-sfm` | 100 mm | 2.87° | classifier completely blind; H/E fallback needed |

Each task auto-runs the perturb step first (via `depends-on`).

For clean ground-truth comparison (no noise):
```
pixi run eth3d-exhibition-hall-sfm
```

---

## Pass criteria

Defined in `docs/e2e_results.md`. The four criteria for each noise level:

| # | Criterion |
|---|-----------|
| 1 | Optimised APE ≤ initial APE (optimizer must not make things worse) |
| 2 | Optimised APE ≤ 90% of initial APE (at least 10% improvement) |
| 3 | Degenerate pairs detected and dropped |
| 4 | No cheirality failures from pairs 61-62 / 63-64 |

Criteria 3 and 4 are blocked for medium/hard until H/E secondary check is integrated
(pose noise > true 0.2mm baseline → pose classifier blind).

---

## Key thresholds and why they are set the way they are

### `--reproj-filter-px` (pre-optimisation filter)

Applied to tracks before the first optimisation pass, using the *noisy initial poses*.
At 3408px focal length and σ_t=5mm:
- Skip-frame pairs (2 frames apart) accumulate ~10px apparent reprojection error
- 4px cuts all skip-frame tracks → defeats covisibility window matching
- 0px lets noisy initial triangulations corrupt the Dogleg optimiser
- **8px is the working value for easy** — keeps skip-frame tracks while rejecting the worst outliers
- Medium uses no pre-filter (empty `--reproj-filter-px 0`) because σ_t=20mm creates ~50px spread

### `post_reproj_error_px` (post-optimisation filter)

Applied after the first optimisation pass, using *optimised poses*. This is the intended
outlier-rejection stage for removing tracks that survived initial filtering but are truly bad.
- 5px default in `SfmOptions`
- Empty clean-list guard: if all tracks rejected, skip re-optimisation (leave first-pass result)
- **Still needs noise-adaptive tuning** — 5px at medium noise cuts too many tracks

### `--min-parallax-deg`

Minimum angle between camera rays for a triangulated point to be kept. 2° is the working
value for this dataset. Lower values let poorly-conditioned landmarks into the graph.

### `--min-track-length` / `SfmOptions.min_track_length`

Minimum number of camera observations a track must have to enter the graph.
- Default 3 (min_track_length=3 filters ~88% of tracks under sequential-only matching)
- Window matching (match_window=3) restores most of these — without window matching,
  this filter actively hurts APE by removing too many constraints
- Always set match_window ≥ 2 when using min_track_length ≥ 3

### `--match-window` / `SfmOptions.match_window`

How many frames ahead each frame matches against (window=3 → frame k matches k+1, k+2, k+3).
- Window=1 is sequential-only; fine for ground-truth poses, poor for noisy poses
- Window=3 is the working default; increases candidate pairs from ~67 to ~192 for 68-frame sequence
- Larger windows add pairs that span larger pose uncertainty, which can create noisy triangulations
  unless the pre-opt reproj filter is tuned accordingly

---

## Adding a new pipeline feature

### 1. Identify where it fits

Use `docs/pipeline_plan.md` to locate the feature in the pipeline and understand dependencies.
All the open items are documented there with the expected location in the code.

### 2. Implement in the right file

| Feature type | File |
|---|---|
| Feature detection / matching / track building | `src/autocal/engine/features.py` |
| Pair classification (degenerate geometry) | `src/autocal/engine/pair_classifier.py` |
| Graph construction / optimisation loop | `src/autocal/engine/sfm_solver.py` |
| CLI and orchestration | `pipelines/sfm.py` |

### 3. Add `SfmOptions` field and CLI flag

Every new tunable goes in `SfmOptions` (dataclass in `sfm_solver.py`) and as a `--kebab-case`
flag in `pipelines/sfm.py`. This keeps the pipeline scriptable and testable with any combination.

### 4. Write tests

Location: `tests/test_<feature_name>.py`

Typical test structure:
- Unit test the new function directly with synthetic inputs
- Integration test calling `optimize_poses` with `SfmOptions` that enables the feature
- One test disabling the feature (verify default/passthrough behaviour)
- One test with a tight/strict setting (verify the feature actually fires)

Run with: `pixi run test`

Test data lives in `data/` — `eth3d_exhibition_hall.mcap` is the primary fixture.

### 5. Check pipeline output format

`optimize_poses` returns a dict with at minimum:
- `poses`: `dict[int, gtsam.Pose3]`
- `n_tracks`: int
- `triangulated`: list of triangulated points
- `keypoints`: `dict[int, list]`

Post-opt rejection tests verify that returned tracks actually pass their own threshold
(see `test_post_reproj.py: test_returned_tracks_pass_their_own_threshold`).

### 6. Record results

After the tests pass, run all three noise tasks and update `docs/e2e_results.md`:
- Copy the terminal output block
- Update the criteria table (✓/✗ PASS/FAIL)
- Add a note if behaviour changed from the previous run

---

## Debugging the pipeline

### Check what the classifier is doing

The pipeline prints a summary line when pairs are dropped:
```
Pair classification: dropped 2 STATIC + 0 PURE_ROTATION, 192 pairs remain (window=3)
```
If this line doesn't appear, `classify_pairs=True` isn't reaching the classifier path
(check `SfmOptions.classify_pairs` and whether `initial_poses` is not None).

### Check track counts at each stage

The pipeline prints:
```
Track length filter (≥3): 3623 dropped, 1354 remain
  → 420 ok after 8px reproj filter
Post-opt: 41 tracks rejected (reproj>5.0px), re-optimising on 379...
```
If tracks drop to 0 before optimisation, the reproj filter is too tight for the noise level.

### Cheirality failures

```
cv2 warning: cheirality failure on pair (61, 62)
```
These come from `filter_matches_ransac` when a near-duplicate pair passes the pose
classifier (because pose noise inflates the baseline). Fix is H/E secondary check.

### Optimizer divergence

If APE *increases* after optimisation, common causes:
1. Noisy initial triangulations (too many bad landmarks) — tighten pre-opt reproj filter
2. Degenerate pairs not being filtered — their cameras end up with almost no constraints
3. Huber loss disabled — for hard noise, `--huber-loss` limits the influence of large residuals

---

## Dataset notes

**eth3d-exhibition-hall** — 68 frames, fisheye (Kannala-Brandt), handheld indoor.
- Frames 61-62 and 63-64 are near-duplicate (0.2mm baseline, <0.01° rotation)
- These are STATIC pairs that the pose classifier catches at easy noise but misses at medium/hard
- Frame rate is 1 Hz (timestamps 0, 1e9, 2e9, ..., 67e9 ns)

**Noise is injected by `tools/perturb_mcap.py`** — the output MCAP contains `/tf` (noisy)
and `/tf_gt` (ground truth) topics, so APE can be computed against GT.
