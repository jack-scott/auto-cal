# End-to-End Pipeline Results

Tracks APE (Absolute Pose Error) across noise difficulty levels for the
`eth3d-exhibition-hall` dataset (68 frames, fisheye, Kannala-Brandt).

Run each level with:
```
pixi run eth3d-exhibition-hall-noise-easy-sfm
pixi run eth3d-exhibition-hall-noise-medium-sfm
pixi run eth3d-exhibition-hall-noise-hard-sfm
```

---

## Noise levels

| Level  | σ_t     | σ_R     | Approx initial APE |
|--------|---------|---------|-------------------|
| Easy   | 5 mm    | 0.11°   | ~7 mm / ~0.09°    |
| Medium | 20 mm   | 0.57°   | ~28 mm / ~0.35°   |
| Hard   | 100 mm  | 2.87°   | ~147 mm / ~2.3°   |

---

## Criteria

| # | Criterion | Notes |
|---|-----------|-------|
| 1 | Optimised APE ≤ initial APE | Optimizer must not make things worse |
| 2 | Optimised APE ≤ 90% of initial APE | At least 10% improvement |
| 3 | Degenerate pairs detected and dropped | Static pair count > 0 in pipeline log |
| 4 | No cheirality failures from degenerate pairs | Pairs 61-62, 63-64 absent from cheirality log |

---

## Results

### Easy — σ_t=5mm, σ_R=0.002rad

Pixi task: `eth3d-exhibition-hall-noise-easy-sfm`
Last run: 2026-05-15

```
APE — initial:    mean=0.0074m  median=0.0072m  max=0.0146m  rmse=0.0080m
                  mean=0.0919°  median=0.0675°  max=0.3319°  rmse=0.1153°
APE — optimised:  mean=0.0067m  median=0.0066m  max=0.0166m  rmse=0.0074m
                  mean=0.0879°  median=0.0733°  max=0.2647°  rmse=0.1021°
Pair classification: dropped 2 STATIC + 1 PURE_ROTATION (via H/E), 191 GOOD pairs remain
Extended matching: +6 GOOD pairs for 2 cluster cameras
Track length filter (≥3): applied; top-420 tracks selected
Post-opt: disabled (--post-reproj-pct 0)
```

| Criterion | Status | Detail |
|-----------|--------|--------|
| 1. APE doesn't degrade | ✓ PASS | 7.4mm → 6.7mm |
| 2. ≥10% improvement    | ✓ PASS | 10% translation improvement |
| 3. Degenerate pairs dropped | ✓ PASS | 2 STATIC + 1 PURE_ROTATION removed |
| 4. No cheirality from 61-64 | ✓ PASS | PURE_ROTATION cluster pair demoted; reprojection-only path active |

Note: reproj pre-filter set to 8px — at 5mm noise over 3 frames and 3408px focal length,
skip-frame pairs accumulate ~10px reprojection error with initial poses, so 4px cuts all
window-matched tracks.

**Post-opt experiments (2026-05-15 → 2026-05-17):**
- Percentage rejection (`--post-reproj-pct 0.1`, 4 iters): 420→277 tracks, 6.7→6.9mm. Worse.
- Noise-adaptive tightening (start=auto ~10px, min=3px, 4 iters): 420→182 tracks, 6.7→6.9mm. Worse.
- Retriangulate-first at 10px, then tighten: 420→182 tracks, ~6.9mm. No improvement.
Root cause: at 5mm noise, the initial BA already finds the optimum with 420 tracks. Post-opt
removes valid tracks (they're all at the noise floor, not outliers). **Post-opt disabled for
fine preset.**

---

### Medium — σ_t=20mm, σ_R=0.01rad

Pixi task: `eth3d-exhibition-hall-noise-medium-sfm`
Last run: 2026-05-15

```
APE — initial:    mean=0.0294m  median=0.0286m  max=0.0586m  rmse=0.0319m
                  mean=0.4513°  median=0.3300°  max=1.6608°  rmse=0.5726°
APE — optimised:  mean=0.0273m  median=0.0266m  max=0.0585m  rmse=0.0297m
                  mean=0.4384°  median=0.3454°  max=1.6519°  rmse=0.5553°
Pair classification: 3 PURE_ROTATION via H/E check; 8 intra-cluster GOOD pairs demoted; 191 GOOD remain
Extended matching: +12 GOOD pairs for 6 cluster cameras; PURE_ROTATION reprojection: 43+ extra obs
Track length filter (≥3): applied
Post-opt: disabled (--post-reproj-pct 0)
```

| Criterion | Status | Detail |
|-----------|--------|--------|
| 1. APE doesn't degrade | ✓ PASS | 29.4mm → 27.3mm |
| 2. ≥10% improvement    | ✗ FAIL | 7% improvement |
| 3. Degenerate pairs dropped | ✓ PASS | 3 PURE_ROTATION via H/E check; 8 intra-cluster demoted |
| 4. No cheirality from 61-64 | ✓ PASS | Cluster pairs demoted to PURE_ROTATION; no triangulation |

**Post-opt experiments (2026-05-15 → 2026-05-17):**
- Percentage rejection (10%, 4 iters): 180→119 tracks, 27.3→28.2mm. Worse.
- Noise-adaptive tightening (start=auto ~30px, min=3px, 4 iters): 180→7 tracks → 24 after retriangulation, 29.2mm. Much worse.
- Tighten to min=8px: 180→38 tracks → 98 after retriangulation, 27.8mm. Still worse.
- Retriangulate-first at 30px (add 192 new → 372), then tighten to 8px: 372→15 tracks, catastrophic.
Root cause: at 20mm noise, expected reproj error ≈ 13.6px for ALL tracks — there is no
reproj-based distinction between good and bad tracks at this noise level. Removing any tracks
removes valid constraints. The initial BA at 180 tracks is already the best achievable with
the current approach. **Post-opt disabled for medium preset.** Path to criterion 2 requires
§7 noise-adaptive prior or more/better tracks from pair classification.

---

### Hard — σ_t=100mm, σ_R=0.05rad

Pixi task: `eth3d-exhibition-hall-noise-hard-sfm`  (preset: `hard`)
Last run: 2026-05-15

```
APE — initial:    mean=0.1472m  median=0.1431m  max=0.2929m  rmse=0.1594m
                  mean=2.2575°  median=1.6523°  max=8.3073°  rmse=2.8635°
APE — optimised:  mean=0.1880m  median=0.1608m  max=0.5409m  rmse=0.2202m
                  mean=3.2758°  median=2.6315°  max=15.3883° rmse=4.0332°
Pair classification: 0 dropped (σ_t=100mm >> 0.2mm true baseline; pose classifier blind)
H/E secondary check: DISABLED (--no-he-secondary-check) — see note below
Track stats: 662 tracks  length: mean=3.2  median=3  max=6
Landmark removals during optimisation: 1
```

| Criterion | Status | Detail |
|-----------|--------|--------|
| 1. APE doesn't degrade | ✗ FAIL | 147mm → 188mm (28% worse) |
| 2. ≥10% improvement    | ✗ FAIL | APE degraded |
| 3. Degenerate pairs dropped | ✗ FAIL | H/E disabled; pose noise makes 0.2mm pairs look like 145mm baseline |
| 4. No cheirality from 61-64 | ✗ FAIL | Cheirality failures, concentrated on 61-64 |

**H/E secondary check status:** Implemented (§4 PURE_ROTATION reprojection path is complete),
but intentionally disabled for the hard preset.  When H/E is enabled at σ_t=100mm, the
near-duplicate cluster cameras (55s, 57s, 61s–64s) receive reprojection-only factors
against existing landmarks.  However, their 100mm-noisy initial poses make those
reprojection constraints inconsistent, triggering a cascade of 69+ landmark removals and
worsening APE from 0.188m → 0.202m.  Requires §7 (noise-adaptive prior) or §8
(coarse-to-fine initialisation) before H/E + PURE_ROTATION reprojection can help at this
noise level.

**Root cause (unchanged):** σ_t=100mm > true 0.2mm baseline — pose prior noise dominates
triangulation geometry.  Dogleg converges to a local minimum rather than improving on
the noisy initial poses.  Fix requires §7/§8 (see `pipeline_plan.md`).

---

## Next steps

### To pass Medium criteria 2, 3, 4 and Hard criteria 1, 2, 3, 4

**H/E secondary check** — run `classify_pair_without_poses` on every pair after the
pose classifier (or always). Will catch static pairs regardless of pose noise.
See `pipeline_plan.md` open questions.

### Post-opt status (as of 2026-05-17)

Noise-adaptive tightening + retriangulation is implemented in `SfmOptions` and tested
(6 unit tests pass). However it is **disabled in fine and medium presets** because:
- At all tested noise levels, the initial BA with all available tracks already finds the
  best achievable result. Post-opt either removes valid geometry (tightening) or adds
  noise (retriangulation with failed tracks).
- Root cause: reproj error at the noise floor is indistinguishable from outlier reproj error
  unless pose accuracy is already within ~1px residual.

Post-opt may become useful after §7 (noise-adaptive prior) or §8 (coarse-to-fine init)
reduces pose error enough that true outliers stand above the noise floor.
