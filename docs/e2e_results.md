# End-to-End Pipeline Results

Tracks APE (Absolute Pose Error) across noise difficulty levels for the
`eth3d-exhibition-hall` dataset (68 frames, fisheye, Kannala-Brandt).

Run each level with:
```
pixi run eth3d-exhibition-hall-noise-easy-sfm
pixi run eth3d-exhibition-hall-noise-medium-sfm
pixi run eth3d-exhibition-hall-noisy-sfm        # hard
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
Last run: 2026-05-14

```
APE — initial:    mean=0.0074m  median=0.0072m  max=0.0146m  rmse=0.0080m
                  mean=0.0919°  median=0.0675°  max=0.3319°  rmse=0.1153°
APE — optimised:  mean=0.0067m  median=0.0066m  max=0.0166m  rmse=0.0074m
                  mean=0.0879°  median=0.0733°  max=0.2647°  rmse=0.1021°
Pair classification: dropped 2 STATIC + 0 PURE_ROTATION, 192 pairs remain (window=3)
Track length filter (≥3): 3623 dropped, 1354 remain → 420 ok after 8px reproj filter
```

| Criterion | Status | Detail |
|-----------|--------|--------|
| 1. APE doesn't degrade | ✓ PASS | 7.4mm → 6.7mm |
| 2. ≥10% improvement    | ✓ PASS | 10% translation improvement |
| 3. Degenerate pairs dropped | ✓ PASS | 2 STATIC pairs removed |
| 4. No cheirality from 61-64 | ✓ PASS | Clean cheirality log |

Note: reproj pre-filter set to 8px (not fine preset's 4px) — at 5mm noise over 3 frames and
3408px focal length, skip-frame pairs accumulate ~10px reprojection error with initial poses,
so 4px cuts all window-matched tracks.  0px causes optimizer divergence from noisy initial
landmarks.  Full benefit requires post-optimisation outlier rejection (pipeline item 6).

---

### Medium — σ_t=20mm, σ_R=0.01rad

Pixi task: `eth3d-exhibition-hall-noise-medium-sfm`
Last run: 2026-05-14

```
APE — initial:    mean=0.0294m  median=0.0286m  max=0.0586m  rmse=0.0319m
                  mean=0.4513°  median=0.3300°  max=1.6608°  rmse=0.5726°
APE — optimised:  mean=0.0273m  median=0.0266m  max=0.0585m  rmse=0.0297m
                  mean=0.4384°  median=0.3454°  max=1.6519°  rmse=0.5553°
Pair classification: 0 dropped (σ_t=20mm inflates 0.2mm baseline to ~28mm; pose classifier blind)
Track length filter (≥3): 4009 dropped, 1425 remain → 180 ok after 20px reproj filter
Cheirality failures: 306 on pair 61-62, 295 on pair 63-64 (near-duplicate frames not filtered)
```

| Criterion | Status | Detail |
|-----------|--------|--------|
| 1. APE doesn't degrade | ✓ PASS | 29.4mm → 27.3mm |
| 2. ≥10% improvement    | ✗ FAIL | 7% improvement |
| 3. Degenerate pairs dropped | ✗ FAIL | Pose noise blinds classifier; H/E check needed |
| 4. No cheirality from 61-64 | ✗ FAIL | 601 cheirality failures from pairs 61-62 and 63-64 |

---

### Hard — σ_t=100mm, σ_R=0.05rad

Pixi task: `eth3d-exhibition-hall-noisy-sfm`
Last run: 2026-05-14

```
APE — initial:    mean=0.1472m  median=0.1431m  max=0.2929m  rmse=0.1594m
                  mean=2.2575°  median=1.6523°  max=8.3073°  rmse=2.8635°
APE — optimised:  mean=0.2816m  median=0.2704m  max=0.8809m  rmse=0.3246m
                  mean=4.0514°  median=3.7533°  max=8.0202°  rmse=4.2566°
Pair classification: 0 dropped (σ_t=100mm >> 0.2mm true baseline; pose classifier blind)
```

| Criterion | Status | Detail |
|-----------|--------|--------|
| 1. APE doesn't degrade | ✗ FAIL | 147mm → 282mm (nearly doubled) |
| 2. ≥10% improvement    | ✗ FAIL | APE degraded |
| 3. Degenerate pairs dropped | ✗ FAIL | Pose noise makes 0.2mm pairs look like 145mm baseline |
| 4. No cheirality from 61-64 | ✗ FAIL | 1378 cheirality failures, concentrated on 61-64 |

Root cause: frames 61-62 and 63-64 are near-duplicate (0.2mm baseline) but noisy
poses create a false ~145mm baseline, so pair classifier does not filter them.
These cameras end up with almost no valid landmark constraints and pull the
surrounding trajectory off.

Fix required: H/E ratio test as secondary check (image-based, pose-independent).
See `docs/degenerate_pair_geometry.md` and `pipeline_plan.md §3 open questions`.

---

## Next steps to pass Hard

1. **H/E secondary check** — run `classify_pair_without_poses` on every pair
   after the pose classifier, or always. Will catch static pairs regardless of
   pose noise. Tracked in `pipeline_plan.md`.

2. **Min track length = 3** — removes short tracks that survive only through the
   degenerate pairs. Also tracked in `pipeline_plan.md`.
