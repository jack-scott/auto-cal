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
from autocal.engine.pair_classifier import (
    PairClass,
    classify_pair_with_poses,
    classify_pair_without_poses,
    filter_pairs_by_geometry,
)


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
    he_secondary_check: bool = True
    he_secondary_ratio: float = 0.99
    min_track_length: int = 3
    match_window: int = 3
    degenerate_extra_window: int = 3
    pure_rotation_reprojection: bool = True
    post_reproj_start_px: float = -1.0   # 0 = auto-compute from pose_noise_m; -1 = disabled
    post_reproj_min_px: float = 3.0
    post_reproj_iters: int = 4
    retriangulate: bool = True


def _noise_adaptive_start_px(
    pose_noise_m: float,
    calibration: gtsam.Cal3DS2 | gtsam.Cal3Fisheye,
    triangulated: list,
    poses: dict,
    post_reproj_min_px: float,
) -> float:
    """Compute a noise-adaptive starting reprojection threshold for post-opt tightening.

    Estimates expected reprojection error from pose noise, focal length, and median
    landmark depth.  Returns -1.0 if computation is not possible.
    """
    if pose_noise_m <= 0:
        return -1.0
    focal_px = max(calibration.fx(), calibration.fy())
    depths = []
    for t in triangulated:
        if t.point3d is None:
            continue
        for img_id in t.observations:
            if img_id in poses:
                d = float(np.linalg.norm(t.point3d - poses[img_id].translation()))
                if d > 0:
                    depths.append(d)
                break
    if not depths:
        return -1.0
    median_depth = float(np.median(depths))
    expected_px = pose_noise_m / median_depth * focal_px
    return max(post_reproj_min_px * 2, 3.0 * expected_px)


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
    # Pair classification — separate GOOD pairs from PURE_ROTATION pairs
    # ------------------------------------------------------------------ #
    has_priors = initial_poses is not None
    pure_rotation_pairs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    if opts.classify_pairs:
        matches_per_pair, pure_rotation_pairs, cls_counts, n_he = filter_pairs_by_geometry(
            matches_per_pair,
            initial_poses if has_priors else None,
            keypoints_for_geo,
            K,
            he_secondary_check=opts.he_secondary_check,
            he_secondary_ratio=opts.he_secondary_ratio,
        )
        n_static = cls_counts[PairClass.STATIC]
        n_pr = cls_counts[PairClass.PURE_ROTATION]
        if n_static + n_pr > 0:
            he_note = f" ({n_he} via H/E check)" if n_he > 0 else ""
            print(
                f"  Pair classification: dropped {n_static} STATIC"
                f", {n_pr} PURE_ROTATION→reprojection-only{he_note}"
                f", {len(matches_per_pair)} GOOD pairs remain",
                flush=True,
            )

    # ------------------------------------------------------------------ #
    # Extended matching for cameras in degenerate clusters
    # Cameras that appear in PURE_ROTATION pairs often have limited reach
    # to well-constrained landmarks.  Extend their window so they can
    # observe landmarks triangulated from cameras further along the sequence.
    # ------------------------------------------------------------------ #
    if opts.degenerate_extra_window > 0 and pure_rotation_pairs:
        cluster_cameras = set()
        for id_a, id_b in pure_rotation_pairs:
            cluster_cameras.add(id_a)
            cluster_cameras.add(id_b)

        n_extra_good = 0
        n_extra_pr = 0
        for k in range(len(img_ids)):
            if img_ids[k] not in cluster_cameras:
                continue
            for offset in range(opts.match_window + 1,
                                opts.match_window + opts.degenerate_extra_window + 1):
                if k + offset >= len(img_ids):
                    break
                id_a, id_b = img_ids[k], img_ids[k + offset]
                if (id_a, id_b) in matches_per_pair or (id_a, id_b) in pure_rotation_pairs:
                    continue
                m = match_sift(descriptors[id_a], descriptors[id_b], ratio=opts.match_ratio)
                if len(m) < opts.min_matches:
                    continue
                inliers = filter_matches_ransac(
                    keypoints_for_geo[id_a], keypoints_for_geo[id_b], m,
                    ransac_threshold=opts.ransac_threshold,
                    min_inliers=opts.min_matches,
                )
                if not inliers:
                    continue
                if has_priors:
                    cls = classify_pair_with_poses(initial_poses[id_a], initial_poses[id_b])
                else:
                    cls = classify_pair_without_poses(
                        keypoints_for_geo[id_a], keypoints_for_geo[id_b], inliers, K
                    )
                if cls == PairClass.STATIC:
                    continue
                if cls == PairClass.PURE_ROTATION:
                    pure_rotation_pairs[(id_a, id_b)] = inliers
                    n_extra_pr += 1
                else:
                    matches_per_pair[(id_a, id_b)] = inliers
                    n_extra_good += 1

        if n_extra_good + n_extra_pr > 0:
            print(
                f"  Extended matching for {len(cluster_cameras)} cluster cameras: "
                f"+{n_extra_good} GOOD, +{n_extra_pr} PURE_ROTATION pairs",
                flush=True,
            )

    # ------------------------------------------------------------------ #
    # Cluster-interior demotion
    # Cameras identified as near-duplicate (PURE_ROTATION) form a degenerate
    # cluster.  All GOOD pairs where both endpoints are inside that cluster
    # are demoted to PURE_ROTATION — this prevents triangulation from any
    # intra-cluster geometry.  Run after extended matching so pairs added by
    # the extended window are also caught.
    # ------------------------------------------------------------------ #
    if pure_rotation_pairs:
        cluster_cameras_all: set = set()
        for id_a, id_b in pure_rotation_pairs:
            cluster_cameras_all.add(id_a)
            cluster_cameras_all.add(id_b)
        re_classified: dict = {}
        n_demoted = 0
        for (id_a, id_b), m in matches_per_pair.items():
            if id_a in cluster_cameras_all and id_b in cluster_cameras_all:
                pure_rotation_pairs[(id_a, id_b)] = m
                n_demoted += 1
            else:
                re_classified[(id_a, id_b)] = m
        matches_per_pair = re_classified
        if n_demoted > 0:
            print(f"  Demoted {n_demoted} intra-cluster GOOD pairs to PURE_ROTATION", flush=True)

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
    good_matches_for_retriang = dict(matches_per_pair)  # snapshot for retriangulation
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
    # PURE_ROTATION reprojection path
    # 1. Identify cluster cameras (those appearing in PURE_ROTATION pairs).
    # 2. Drop pure-cluster tracks — tracks whose every observation is within
    #    the cluster have no external anchor and only add bad geometry.
    # 3. Build reverse lookup, then collect extra projection observations
    #    from PURE_ROTATION pair matches that land on existing landmarks.
    # ------------------------------------------------------------------ #
    cluster_cameras: set[int] = set()
    for id_a, id_b in pure_rotation_pairs:
        cluster_cameras.add(id_a)
        cluster_cameras.add(id_b)

    if cluster_cameras:
        n_before = len(triangulated)
        triangulated = [
            t for t in triangulated
            if not all(img_id in cluster_cameras for img_id in t.observations)
        ]
        n_cluster_dropped = n_before - len(triangulated)
        if n_cluster_dropped > 0:
            print(f"  Dropped {n_cluster_dropped} pure-cluster tracks "
                  f"(all observations within cluster cameras)", flush=True)

    obs_to_track_idx: dict[tuple[int, int], int] = {}
    for j, track in enumerate(triangulated):
        for img_id, kp_idx in track.observations.items():
            obs_to_track_idx[(img_id, kp_idx)] = j

    extra_obs: list[tuple[int, int, int]] = []  # (img_id, track_idx, kp_idx)
    if opts.pure_rotation_reprojection:
        for (id_a, id_b), m in pure_rotation_pairs.items():
            for kp_a, kp_b in m:
                t = obs_to_track_idx.get((id_a, kp_a))
                if t is not None and id_b in id_to_idx:
                    extra_obs.append((id_b, t, kp_b))
                t = obs_to_track_idx.get((id_b, kp_b))
                if t is not None and id_a in id_to_idx:
                    extra_obs.append((id_a, t, kp_a))

    if extra_obs:
        n_cameras_with_extra = len({img_id for img_id, _, _ in extra_obs})
        print(
            f"  PURE_ROTATION reprojection: {len(extra_obs)} extra observations "
            f"across {n_cameras_with_extra} cameras",
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
        extra_obs=extra_obs,
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
    # Noise-adaptive post-optimisation: retriangulate first, then tighten
    #
    # Step 1 — Retriangulation: rebuild all candidate tracks from the
    # original good-pair matches using the improved poses from the initial
    # BA.  Filter at start_px (wide, noise-adaptive threshold) and add
    # non-overlapping tracks to the current set.  A combined BA gives
    # better poses from more constraints.
    #
    # Step 2 — Optional tightening: geometrically decay the reprojection
    # threshold from start_px to post_reproj_min_px over post_reproj_iters
    # passes.  Each pass removes tracks above the threshold and re-runs BA
    # warm-started from the previous result.  Starting after retriangulation
    # means poses are already improved; the tightening removes true outliers
    # rather than valid-but-noisy tracks.
    # ------------------------------------------------------------------ #
    start_px = opts.post_reproj_start_px
    if start_px == 0.0:
        start_px = _noise_adaptive_start_px(
            opts.pose_noise_m, calibration, triangulated, opt_poses,
            opts.post_reproj_min_px,
        )

    if start_px > 0:
        current_tracks = triangulated
        current_poses  = opt_poses

        # Step 1: Retriangulate at start_px (wide threshold) to add tracks
        if opts.retriangulate:
            retri_tracks = build_tracks(good_matches_for_retriang)
            retri_tracks = [t for t in retri_tracks
                            if len(t.observations) >= opts.min_track_length]
            triangulate_gtsam(
                retri_tracks, keypoints, calibration, current_poses,
                max_dist=opts.max_landmark_dist_m,
                min_parallax_deg=opts.min_parallax_deg,
            )
            retri_valid = [
                t for t in retri_tracks
                if t.point3d is not None
                and all_positive_depth(t.point3d, t.observations, current_poses)
            ]
            retri_valid = filter_by_reproj(
                retri_valid, keypoints, current_poses, calibration, start_px
            )
            existing_obs: set[tuple] = {
                (img_id, kp_idx)
                for t in current_tracks
                for img_id, kp_idx in t.observations.items()
            }
            new_tracks = [
                t for t in retri_valid
                if not any(
                    (img_id, kp_idx) in existing_obs
                    for img_id, kp_idx in t.observations.items()
                )
            ]
            print(
                f"  Retriangulation: {len(new_tracks)} new tracks "
                f"({len(retri_valid)} clean of {len(retri_tracks)} candidates, "
                f"{start_px:.1f}px threshold)",
                flush=True,
            )
            if new_tracks:
                combined = current_tracks + new_tracks
                combined.sort(key=lambda t: len(t.observations), reverse=True)
                combined = combined[: opts.max_tracks]
                try:
                    result, graph, initial_values, _ = _build_and_optimize(
                        combined, current_poses, id_to_idx, img_ids, keypoints,
                        calibration, opts, has_priors,
                        initial_poses if has_priors else None,
                        _fisheye,
                        extra_obs=extra_obs,
                    )
                    print(
                        f"    Error: {graph.error(initial_values):.3e} → "
                        f"{graph.error(result):.3e}",
                        flush=True,
                    )
                    current_poses = {
                        img_id: result.atPose3(X(idx))
                        for img_id, idx in id_to_idx.items()
                    }
                    current_tracks = combined
                except RuntimeError as exc:
                    print(f"  Retriangulation BA failed ({exc}), keeping original set.",
                          flush=True)

        # Step 2: Iterative tightening from start_px down to post_reproj_min_px
        if start_px > opts.post_reproj_min_px and opts.post_reproj_iters > 0:
            if opts.post_reproj_iters <= 1:
                decay = 1.0
            else:
                decay = (start_px / opts.post_reproj_min_px) ** (
                    1.0 / (opts.post_reproj_iters - 1)
                )
            threshold = start_px

            for iteration in range(opts.post_reproj_iters):
                if not current_tracks:
                    break
                clean = filter_by_reproj(
                    current_tracks, keypoints, current_poses, calibration, threshold
                )
                n_rejected = len(current_tracks) - len(clean)
                if n_rejected > 0 and clean:
                    print(
                        f"  Post-opt iter {iteration + 1}/{opts.post_reproj_iters}: "
                        f"{threshold:.1f}px threshold, rejected {n_rejected}, "
                        f"re-optimising on {len(clean)}...",
                        flush=True,
                    )
                    try:
                        result, graph, initial_values, _ = _build_and_optimize(
                            clean, current_poses, id_to_idx, img_ids, keypoints,
                            calibration, opts, has_priors,
                            initial_poses if has_priors else None,
                            _fisheye,
                            extra_obs=extra_obs,
                        )
                        print(
                            f"    Error: {graph.error(initial_values):.3e} → "
                            f"{graph.error(result):.3e}",
                            flush=True,
                        )
                        current_poses = {
                            img_id: result.atPose3(X(idx))
                            for img_id, idx in id_to_idx.items()
                        }
                        current_tracks = clean
                    except RuntimeError as exc:
                        print(
                            f"  Post-opt iter {iteration + 1} failed ({exc}), stopping early.",
                            flush=True,
                        )
                        break
                threshold = max(opts.post_reproj_min_px, threshold / decay)
                if threshold <= opts.post_reproj_min_px and n_rejected == 0:
                    break

        opt_poses    = current_poses
        triangulated = current_tracks

    track_lengths = [len(t.observations) for t in triangulated]
    return {
        "poses":          opt_poses,
        "initial_poses":  poses,
        "n_tracks":       len(triangulated),
        "track_lengths":  track_lengths,
        "keypoints":      keypoints,
        "descriptors":    descriptors,
        "triangulated":   triangulated,
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
    extra_obs: list[tuple[int, int, int]] | None = None,
    max_retries: int = 300,
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

        orig_to_new_j = {orig_j: new_j for new_j, (orig_j, _) in enumerate(active)}

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

        for img_id, orig_j, kp_idx in (extra_obs or []):
            new_j = orig_to_new_j.get(orig_j)
            if new_j is None or img_id not in id_to_idx:
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
                if x_idx in pinned:
                    # Pinning didn't help — drop every landmark this camera sees
                    cam_id = img_ids[x_idx]
                    to_drop = [
                        orig_j for orig_j, track in enumerate(triangulated)
                        if orig_j not in excluded and cam_id in track.observations
                    ]
                    if not to_drop:
                        raise
                    excluded.update(to_drop)
                    print(f"  Removing {len(to_drop)} landmarks for stubborn camera x{x_idx} "
                          f"(attempt {attempt+1})", flush=True)
                else:
                    pinned.add(x_idx)
                    print(f"  Pinning degenerate camera x{x_idx} "
                          f"(attempt {attempt+1})", flush=True)
            else:
                raise

    raise RuntimeError("_build_and_optimize: max retries exceeded")


