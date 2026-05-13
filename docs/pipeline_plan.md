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

This requires knowing which landmarks are already in the graph when processing
the pair — it is a forward-pass dependency that does not fit the current batch
structure.  For now, dropping PURE_ROTATION pairs is correct and safe.  The
reprojection-only path is future work once the pipeline moves toward incremental
or multi-pass structure.

Stub to add to `sfm_solver.py` comments when the classifier is integrated:
```python
# TODO: PURE_ROTATION pairs should add reprojection factors against existing
# landmarks rather than being skipped.  Requires incremental landmark tracking.
```

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
3. COVISIBILITY MATCHING (window=3–5)           ← missing (sequential only)
   match_sift + filter_matches_ransac per pair
        │
        ▼
4. PAIR CLASSIFICATION                          ✓ COMPLETED
   classify_pair_with_poses (or H/E fallback)
   STATIC       → drop pair
   PURE_ROTATION → drop triangulation (keep for future reprojection pass)
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
   PriorFactorPose3 per camera (from /tf)
   GenericProjectionFactor per (camera, landmark, observation)
   Dogleg optimiser
        │
        ▼
8. OUTLIER REJECTION + RE-OPTIMISE             ← missing (pre-opt only today)
   filter_by_reproj on optimised poses (2px threshold)
   re-run optimiser on surviving tracks
        │
        ▼
Refined poses and/or calibration
```

---

## Implementation order

These are ordered by impact and dependency:

1. **Pair classifier integration** (§3) — ✓ COMPLETED
   `filter_pairs_by_geometry` in `pair_classifier.py`, called from `sfm_solver.py`.
   `SfmOptions.classify_pairs: bool = True` to enable/disable.
   8 tests in `tests/test_pair_filter.py`.

2. **Min track length = 3** (§5) — ✓ COMPLETED
   `SfmOptions.min_track_length: int = 3`, `--min-track-length` CLI flag.
   4 tests in `tests/test_min_track_length.py`.
   Note: with sequential-only matching, 88% of tracks are 2-frame only, so this
   filter reduces track count too aggressively and hurts APE until covisibility
   window matching (item 4) is added.

3. **Post-optimisation outlier rejection** (§6) — moderate complexity.
   Needs `filter_by_reproj` called after optimisation with optimised poses.

4. **Covisibility window matching** (§2) — replaces sequential loop.
   Longer tracks, better conditioning.  Test that track count and APE improve.

5. **Keyframe selection** (§1) — needed for long trajectories / large datasets.
   Can be skipped if input sequences are already sparse.

6. **PURE_ROTATION reprojection path** (§4) — future work.
   Requires incremental or multi-pass structure.

---

## Open questions

- **Rotation/translation ratio gate:** pairs like 62-63 (57mm, 23°, ratio 407°/m)
  are classified as GOOD but cause 98% cheirality failure under noisy poses.
  Worth adding a configurable `max_rotation_translation_ratio_deg_per_m` threshold
  to `classify_pair_with_poses`.

- **Homography matching for PURE_ROTATION:** if reprojection-only is implemented,
  pairs with pure rotation should use H-based filtering rather than E, since E is
  ill-conditioned with no translation.

- **Calibration excitation:** for the calibration engine specifically, we want
  pairs with different depths in the scene (varying parallax) not just angular
  diversity.  Keyframe selection may need a depth-diversity criterion.
