# Degenerate Frame-Pair Geometry

## Background

During investigation of ~30% triangulation failures in the `eth3d_exhibition_hall`
dataset, two distinct degenerate geometry cases were identified in frames 61-64.
These failure modes are general and will appear in any dataset with similar motion
profiles.

## The Two Failure Modes

### 1. Near-Duplicate Frames (STATIC)

**Observed in:** pairs 61-62 and 63-64

**Geometry:** baseline ≈ 0.2 mm, rotation ≈ 0°

**What happens at each pipeline stage:**

| Stage | Effect |
|-------|--------|
| SIFT matching | Many matches found — the images look nearly identical |
| Essential matrix | Completely ill-conditioned (no baseline to constrain it) |
| RANSAC | Accepts 97-98% of all matches as inliers — `x'^T F x ≈ 0` for any `F` when there is no baseline |
| Triangulation | Geometrically impossible — rays are parallel, no intersection |

The RANSAC acceptance rate being near 100% is a diagnostic signal:
epipolar geometry cannot reject anything when there is no motion.
Wrong-side matches in symmetric scenes (reflections, repeated textures)
pass trivially.

**Correct action:** skip entirely — no useful geometric information.

---

### 2. Rotation-Dominated Pair (PURE_ROTATION)

**Observed in:** pair 62-63

**Geometry:** baseline ≈ 57 mm, rotation ≈ 23°, ratio ≈ 407°/m

**What happens:**

With ground-truth poses, 99.7% of RANSAC inliers triangulate successfully —
the matches are real and the geometry is valid. The problem only appears under
pose noise:

| Pose noise | Triangulation failure rate |
|------------|---------------------------|
| GT poses (0 noise) | 0.3% |
| 5 cm / 0.05 rad noise | 98.5% |

The DLT is sensitive to pose error when rotation dominates translation.
A small angular error in the pose rotates the epipolar plane significantly,
placing reconstructed points behind one of the cameras (CheiralityException).

**Note on default thresholds:** With `translation_threshold_m=0.02` (2 cm),
this pair is classified as `GOOD` because 57 mm > 20 mm. The default thresholds
are sufficient to gate the truly degenerate cases. Whether to add a
rotation/translation ratio check depends on how much pose noise is expected
— this is an open question.

**Correct action:** if pose noise is high, consider reprojection-only factors
(no new landmark triangulation) for rotation-dominated pairs.

---

## Classification

See `src/autocal/engine/pair_classifier.py`.

```
STATIC           t < t_thresh  AND  r < r_thresh  →  skip entirely
PURE_ROTATION    t < t_thresh  AND  r ≥ r_thresh  →  reprojection factors only
PURE_TRANSLATION t ≥ t_thresh  AND  r < r_thresh  →  triangulate, good baseline
GOOD             t ≥ t_thresh  AND  r ≥ r_thresh  →  full pipeline
```

Default thresholds: `translation_threshold_m=0.02`, `rotation_threshold_deg=2.0`.

Without pose priors, `classify_pair_without_poses` uses the **H/E inlier ratio test**:

```python
_, mask_H = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 3.0)
_, mask_E = cv2.findEssentialMat(pts_a, pts_b, K, cv2.USAC_MAGSAC, 0.999, 1.0)
ratio = mask_H.sum() / (mask_E.sum() + 1e-6)
# ratio > 0.8  →  PURE_ROTATION (degenerate — do not triangulate)
# ratio ≤ 0.8  →  GOOD
```

A homography perfectly explains pure rotation and planar scenes, so if H accounts
for as many inliers as E, there is no useful translational baseline.  This collapses
the four-way classification into two cases — the image-only test cannot distinguish
STATIC from PURE_ROTATION, or PURE_TRANSLATION from GOOD.

## Pipeline Impact

In the exhibition hall dataset with 0.1 m / 0.05 rad pose noise, frames 61-64
produced almost no valid tracks. With the SfM optimizer, those cameras were
left under-constrained, and the optimiser made poses *worse* overall:

```
APE (initial, noisy): mean=0.147m  rotation=2.3°
APE (optimised):      mean=0.282m  rotation=4.1°
```

Skipping STATIC pairs before track building and triangulation would eliminate
this source of degradation.

## Diagnostic Tests

`tests/test_degenerate_geometry.py` — characterises the two failure modes:

- `TestNearDuplicateFrames` — verifies baseline, rotation, RANSAC acceptance rate,
  and depth sensitivity to 1-pixel noise on real pairs 61-62 and 63-64.
- `TestNearPureRotation` — verifies GT triangulation succeeds and noisy pose
  triangulation fails on pair 62-63.

`tests/test_pair_classifier.py` — verifies `classify_pair_with_poses` returns
the expected class for real GT poses and synthetic controlled inputs.

`tests/test_cheirality_breakdown.py` — full breakdown of tracks into
GOOD / CHEIRALITY / BAD_MATCH / BOTH categories using GT epipolar geometry as
an oracle. Key finding: frames 00-22 are ~97% GOOD; frames 61-64 are ~85% BOTH
(bad match AND cheirality failure simultaneously).

## Open Questions

1. **Rotation/translation ratio gate:** Should pairs with very high rotation-to-
   translation ratios (e.g. >100°/m) be downgraded to PURE_ROTATION regardless
   of absolute baseline? This would catch the 62-63 case under noisy conditions.

2. **Homography RANSAC for PURE_ROTATION:** For reprojection-only factors on
   rotation-dominated pairs, matches should be filtered with a homography model
   rather than the essential matrix. Not yet implemented.

3. **Image-only accuracy:** `classify_pair_without_poses` (H/E ratio) is implemented
   and catches the near-duplicate case.  It cannot distinguish STATIC from PURE_ROTATION
   or flag rotation-dominated-but-translating pairs like 62-63.  Use with-poses version
   when priors are available.
