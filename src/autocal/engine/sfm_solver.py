"""
SfM solver — estimates camera poses given a fixed calibration.

Pose convention
---------------
All Pose3 objects use the GTSAM native convention:
  Pose3(R_wc, t)  where R_wc = camera→world rotation, t = camera position in world.

Pipeline
--------
1. Detect SIFT features in each image.
2. Match sequential pairs (frame k ↔ frame k+1).
3. Build feature tracks via union-find.
4. Initialise poses:
     - If initial_poses provided: use them directly (with prior factors in the graph).
     - If not: chain relative poses from the essential matrix between successive pairs.
       Frame 0 is placed at the origin. Translation is unit-scale only.
5. Triangulate track 3D positions using gtsam.triangulatePoint3 (multi-view, nonlinear
   refinement) with distorted pixel observations.
6. Build GTSAM factor graph:
     - PriorFactorPose3 for each camera that has an initial pose.
     - GenericProjectionFactorCal3DS2 (fixed Cal3DS2 constant, not a key),
       or GenericProjectionFactorCal3Fisheye per (camera, landmark, 2D observation).
7. Optimise with Dogleg (trust-region, more robust than LM for large initial error).
8. Return optimised poses and track data.
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass

import gtsam
import numpy as np

from autocal.engine.features import (
    Track,
    all_positive_depth,
    build_tracks,
    cal_to_K,
    chain_essential_matrix,
    detect_sift,
    filter_by_reproj,
    filter_matches_ransac,
    match_sift,
    triangulate_gtsam,
    undistort_keypoints,
)
from autocal.engine.pair_classifier import PairClass, filter_pairs_by_geometry


@dataclass
class SfmOptions:
    """Tunable parameters for the SfM pose solver."""
    sift_features: int = 0
    match_ratio: float = 0.75
    min_matches: int = 8
    lm_iterations: int = 200
    pose_noise_m: float = 1.0
    pose_noise_rad: float = 0.1
    pixel_noise_px: float = 1.5
    max_tracks: int = 2000
    max_reproj_error_px: float = 0.0
    max_landmark_dist_m: float = 0.0
    min_parallax_deg: float = 1.0
    huber_loss: bool = False
    ransac_threshold: float = 2.0
    classify_pairs: bool = True
    min_track_length: int = 3
    match_window: int = 3
    post_reproj_error_px: float = 5.0


def optimize_poses(
    images: list[tuple[int, bytes]],
    calibration: gtsam.Cal3DS2 | gtsam.Cal3Fisheye,
    opts: SfmOptions,
    initial_poses: dict[int, gtsam.Pose3] | None = None,
    preloaded_features: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """Solve camera poses with calibration held fixed.

    Args:
        images:             Ordered list of (img_id, jpeg_bytes).
        calibration:        Fixed camera intrinsics.
        opts:               Tuning options.
        initial_poses:      If provided, used as initial values and prior factors.
        preloaded_features: If provided, {img_id: (kps, descs)} skips detection.

    Returns:
        Dict with keys: "poses", "initial_poses", "n_tracks",
        "keypoints", "descriptors", "triangulated".
    """
    img_ids = [img_id for img_id, _ in images]
    id_to_idx: dict[int, int] = {img_id: i for i, img_id in enumerate(img_ids)}

    # ------------------------------------------------------------------ #
    # Feature detection (or load from cache)
    # ------------------------------------------------------------------ #
    keypoints: dict[int, np.ndarray] = {}
    descriptors: dict[int, np.ndarray] = {}
    if preloaded_features is not None:
        print(f"Loading cached features for {len(images)} images...", flush=True)
        for img_id, _ in images:
            if img_id not in preloaded_features:
                continue
            kps, descs = preloaded_features[img_id]
            if opts.sift_features > 0 and len(kps) > opts.sift_features:
                kps = kps[:opts.sift_features]
                descs = descs[:opts.sift_features]
            keypoints[img_id] = kps
            descriptors[img_id] = descs
    else:
        print(f"Detecting features in {len(images)} images...", flush=True)
        for i, (img_id, img_bytes) in enumerate(images):
            kps, descs = detect_sift(img_bytes, n_features=opts.sift_features)
            keypoints[img_id] = kps
            descriptors[img_id] = descs
            if (i + 1) % 10 == 0 or i + 1 == len(images):
                print(f"  {i+1}/{len(images)}", flush=True)

    # ------------------------------------------------------------------ #
    # Covisibility window matching
    # ------------------------------------------------------------------ #
    n_candidate = sum(
        min(opts.match_window, len(img_ids) - 1 - k)
        for k in range(len(img_ids) - 1)
    )
    print(f"Matching pairs (window={opts.match_window}, "
          f"{n_candidate} candidate pairs)...", flush=True)
    matches_per_pair: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for k in range(len(img_ids) - 1):
        for offset in range(1, opts.match_window + 1):
            if k + offset >= len(img_ids):
                break
            id_a, id_b = img_ids[k], img_ids[k + offset]
            if (id_a, id_b) in matches_per_pair:
                continue
            m = match_sift(descriptors[id_a], descriptors[id_b], ratio=opts.match_ratio)
            if len(m) >= opts.min_matches:
                matches_per_pair[(id_a, id_b)] = m
    print(
        f"  {len(matches_per_pair)}/{n_candidate} pairs passed ratio test",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # Undistort keypoints for geometry (RANSAC + essential matrix)
    # ------------------------------------------------------------------ #
    _fisheye = isinstance(calibration, gtsam.Cal3Fisheye)
    K = cal_to_K(calibration)
    keypoints_for_geo = undistort_keypoints(keypoints, calibration)

    # ------------------------------------------------------------------ #
    # RANSAC geometric filtering
    # ------------------------------------------------------------------ #
    if opts.ransac_threshold > 0:
        n_before = sum(len(m) for m in matches_per_pair.values())
        filtered: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for (id_a, id_b), m in matches_per_pair.items():
            inliers = filter_matches_ransac(
                keypoints_for_geo[id_a], keypoints_for_geo[id_b], m,
                ransac_threshold=opts.ransac_threshold,
                min_inliers=opts.min_matches,
            )
            if inliers:
                filtered[(id_a, id_b)] = inliers
        matches_per_pair = filtered
        n_after = sum(len(m) for m in matches_per_pair.values())
        print(
            f"  RANSAC: {len(matches_per_pair)}/{n_candidate} pairs kept  "
            f"({n_before} → {n_after} matches)",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # Pair classification — drop degenerate pairs before triangulation
    # ------------------------------------------------------------------ #
    has_priors = initial_poses is not None
    if opts.classify_pairs:
        matches_per_pair, cls_counts = filter_pairs_by_geometry(
            matches_per_pair,
            initial_poses if has_priors else None,
            keypoints_for_geo,
            K,
        )
        n_dropped = cls_counts[PairClass.STATIC] + cls_counts[PairClass.PURE_ROTATION]
        if n_dropped > 0:
            print(
                f"  Pair classification: dropped {cls_counts[PairClass.STATIC]} STATIC "
                f"+ {cls_counts[PairClass.PURE_ROTATION]} PURE_ROTATION, "
                f"{len(matches_per_pair)} pairs remain",
                flush=True,
            )

    # ------------------------------------------------------------------ #
    # Initial poses
    # ------------------------------------------------------------------ #
    if has_priors:
        poses: dict[int, gtsam.Pose3] = dict(initial_poses)
    else:
        poses = chain_essential_matrix(img_ids, keypoints_for_geo, matches_per_pair, K)

    # ------------------------------------------------------------------ #
    # Triangulate — GTSAM multi-view with nonlinear refinement
    # ------------------------------------------------------------------ #
    tracks = build_tracks(matches_per_pair)
    if opts.min_track_length > 2:
        n_before = len(tracks)
        tracks = [t for t in tracks if len(t.observations) >= opts.min_track_length]
        n_dropped = n_before - len(tracks)
        if n_dropped > 0:
            print(f"  Track length filter (≥{opts.min_track_length}): "
                  f"{n_dropped} dropped, {len(tracks)} remain", flush=True)
    triangulate_gtsam(tracks, keypoints, calibration, poses,
                      max_dist=opts.max_landmark_dist_m,
                      min_parallax_deg=opts.min_parallax_deg)

    n_triangulated = sum(1 for t in tracks if t.point3d is not None)
    n_none         = len(tracks) - n_triangulated
    after_cheirality = [
        t for t in tracks
        if t.point3d is not None and all_positive_depth(t.point3d, t.observations, poses)
    ]
    n_behind = n_triangulated - len(after_cheirality)
    good = after_cheirality
    if opts.max_reproj_error_px > 0 and has_priors:
        good = filter_by_reproj(good, keypoints, poses, calibration, opts.max_reproj_error_px)
    n_reproj = len(after_cheirality) - len(good)

    good.sort(key=lambda t: len(t.observations), reverse=True)
    triangulated = good[: opts.max_tracks]
    print(
        f"  Tracks: {len(tracks)} built → "
        f"{n_none} failed triangulation, "
        f"{n_behind} behind camera, "
        f"{n_reproj} reproj>{opts.max_reproj_error_px:.0f}px, "
        f"{len(good)} ok → using top {len(triangulated)}",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # GTSAM factor graph + optimise (with degenerate-landmark retry)
    # ------------------------------------------------------------------ #
    print(
        f"Optimising ({len(triangulated)} tracks, {len(poses)} cameras)...",
        flush=True,
    )
    result, graph, initial_values, n_active = _build_and_optimize(
        triangulated, poses, id_to_idx, img_ids, keypoints,
        calibration, opts, has_priors,
        initial_poses if has_priors else None,
        _fisheye,
    )
    print(
        f"  Error: {graph.error(initial_values):.3e} → {graph.error(result):.3e}",
        flush=True,
    )

    X = gtsam.symbol_shorthand.X
    opt_poses: dict[int, gtsam.Pose3] = {
        img_id: result.atPose3(X(idx))
        for img_id, idx in id_to_idx.items()
    }

    # ------------------------------------------------------------------ #
    # Post-optimisation outlier rejection + re-optimise
    # ------------------------------------------------------------------ #
    if opts.post_reproj_error_px > 0:
        clean = filter_by_reproj(
            triangulated, keypoints, opt_poses, calibration,
            opts.post_reproj_error_px,
        )
        n_rejected = len(triangulated) - len(clean)
        if n_rejected > 0 and len(clean) > 0:
            print(
                f"  Post-opt: {n_rejected} tracks rejected "
                f"(reproj>{opts.post_reproj_error_px:.1f}px), "
                f"re-optimising on {len(clean)}...",
                flush=True,
            )
            result, graph, initial_values, _ = _build_and_optimize(
                clean, opt_poses, id_to_idx, img_ids, keypoints,
                calibration, opts, has_priors,
                initial_poses if has_priors else None,
                _fisheye,
            )
            print(
                f"  Error: {graph.error(initial_values):.3e} → {graph.error(result):.3e}",
                flush=True,
            )
            opt_poses = {
                img_id: result.atPose3(X(idx))
                for img_id, idx in id_to_idx.items()
            }
            triangulated = clean

    return {
        "poses":         opt_poses,
        "initial_poses": poses,
        "n_tracks":      len(triangulated),
        "keypoints":     keypoints,
        "descriptors":   descriptors,
        "triangulated":  triangulated,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_and_optimize(
    triangulated: list,
    poses: dict,
    id_to_idx: dict,
    img_ids: list,
    keypoints: dict,
    calibration,
    opts: SfmOptions,
    has_priors: bool,
    initial_poses: dict | None,
    fisheye: bool,
    max_retries: int = 50,
):
    """Build a GTSAM factor graph and optimize, removing degenerate landmarks on retry.

    When GTSAM throws IndeterminantLinearSystemException it names the offending
    variable in the error message.  p-variables (landmarks) are dropped; x-variables
    (cameras) are pinned with a tight prior.  Up to max_retries removals/pins before
    the exception is re-raised.

    Returns:
        (result_values, graph, initial_values, n_active_tracks)
    """
    X = gtsam.symbol_shorthand.X
    P = gtsam.symbol_shorthand.P

    excluded: set[int] = set()
    pinned: set[int] = set()
    _tight_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    )

    for attempt in range(max_retries + 1):
        active = [(orig_j, t) for orig_j, t in enumerate(triangulated)
                  if orig_j not in excluded]

        graph = gtsam.NonlinearFactorGraph()
        iv = gtsam.Values()

        for img_id, pose in poses.items():
            iv.insert(X(id_to_idx[img_id]), pose)

        if has_priors:
            pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
                opts.pose_noise_rad, opts.pose_noise_rad, opts.pose_noise_rad,
                opts.pose_noise_m,   opts.pose_noise_m,   opts.pose_noise_m,
            ]))
            for img_id, pose in initial_poses.items():
                graph.add(gtsam.PriorFactorPose3(X(id_to_idx[img_id]), pose, pose_noise))
        else:
            graph.add(gtsam.PriorFactorPose3(X(0), poses[img_ids[0]], _tight_noise))

        for x_idx in pinned:
            graph.add(gtsam.PriorFactorPose3(X(x_idx), poses[img_ids[x_idx]], _tight_noise))

        base_noise = gtsam.noiseModel.Isotropic.Sigma(2, opts.pixel_noise_px)
        if opts.huber_loss:
            pixel_noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(opts.pixel_noise_px),
                base_noise,
            )
        else:
            pixel_noise = base_noise

        for new_j, (orig_j, track) in enumerate(active):
            iv.insert(P(new_j), gtsam.Point3(*track.point3d))
            for img_id, kp_idx in track.observations.items():
                if img_id not in id_to_idx:
                    continue
                kp = keypoints[img_id][kp_idx]
                measured = np.array([float(kp[0]), float(kp[1])])
                if fisheye:
                    graph.add(gtsam.GenericProjectionFactorCal3Fisheye(
                        measured, pixel_noise,
                        X(id_to_idx[img_id]), P(new_j),
                        calibration,
                    ))
                else:
                    graph.add(gtsam.GenericProjectionFactorCal3DS2(
                        measured, pixel_noise,
                        X(id_to_idx[img_id]), P(new_j),
                        calibration,
                    ))

        dogleg_params = gtsam.DoglegParams()
        dogleg_params.setMaxIterations(opts.lm_iterations)
        optimizer = gtsam.DoglegOptimizer(graph, iv, dogleg_params)

        try:
            result = optimizer.optimize()
            if excluded or pinned:
                print(f"  Removed {len(excluded)} landmark(s), "
                      f"pinned {len(pinned)} camera(s), "
                      f"{len(active)} tracks remain", flush=True)
            return result, graph, iv, len(active)
        except RuntimeError as exc:
            if attempt == max_retries:
                raise
            err = str(exc)
            match_p = _re.search(r'Symbol: p(\d+)', err)
            match_x = _re.search(r'Symbol: x(\d+)', err)
            if match_p:
                bad_new_j = int(match_p.group(1))
                if bad_new_j < len(active):
                    excluded.add(active[bad_new_j][0])
                    print(f"  Removing degenerate landmark p{bad_new_j} "
                          f"(attempt {attempt+1})", flush=True)
                else:
                    raise
            elif match_x:
                x_idx = int(match_x.group(1))
                pinned.add(x_idx)
                print(f"  Pinning degenerate camera x{x_idx} "
                      f"(attempt {attempt+1})", flush=True)
            else:
                raise

    raise RuntimeError("_build_and_optimize: max retries exceeded")


