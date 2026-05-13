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

import cv2
import gtsam
import numpy as np

from autocal.engine.features import (
    Track,
    all_positive_depth,
    build_tracks,
    cal_to_K,
    detect_sift,
    filter_by_reproj,
    filter_matches_ransac,
    match_sift,
    triangulate_gtsam,
    undistort_keypoints,
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
    # Sequential matching
    # ------------------------------------------------------------------ #
    print("Matching sequential pairs...", flush=True)
    matches_per_pair: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for k in range(len(img_ids) - 1):
        id_a, id_b = img_ids[k], img_ids[k + 1]
        m = match_sift(descriptors[id_a], descriptors[id_b], ratio=opts.match_ratio)
        if len(m) >= opts.min_matches:
            matches_per_pair[(id_a, id_b)] = m
    print(
        f"  {len(matches_per_pair)}/{len(img_ids)-1} pairs passed ratio test",
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
            f"  RANSAC: {len(matches_per_pair)}/{len(img_ids)-1} pairs kept  "
            f"({n_before} → {n_after} matches)",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # Initial poses
    # ------------------------------------------------------------------ #
    has_priors = initial_poses is not None
    if has_priors:
        poses: dict[int, gtsam.Pose3] = dict(initial_poses)
    else:
        poses = _chain_essential_matrix(img_ids, keypoints_for_geo, matches_per_pair, K)

    # ------------------------------------------------------------------ #
    # Triangulate — GTSAM multi-view with nonlinear refinement
    # ------------------------------------------------------------------ #
    tracks = build_tracks(matches_per_pair)
    triangulate_gtsam(tracks, keypoints, calibration, poses,
                      max_dist=opts.max_landmark_dist_m,
                      min_parallax_deg=opts.min_parallax_deg)
    good = [
        t for t in tracks
        if t.point3d is not None and all_positive_depth(t.point3d, t.observations, poses)
    ]
    if opts.max_reproj_error_px > 0 and has_priors:
        good = filter_by_reproj(good, keypoints, poses, calibration, opts.max_reproj_error_px)

    good.sort(key=lambda t: len(t.observations), reverse=True)
    triangulated = good[: opts.max_tracks]
    print(
        f"  Tracks: {len(tracks)} built, {len(good)} cheirality-ok / reproj-ok, "
        f"using top {len(triangulated)}",
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


def _chain_essential_matrix(
    img_ids: list[int],
    keypoints: dict[int, np.ndarray],
    matches_per_pair: dict[tuple[int, int], list[tuple[int, int]]],
    K: np.ndarray,
) -> dict[int, gtsam.Pose3]:
    """Initialise poses by chaining essential-matrix relative poses.

    Frame 0 is placed at the origin with optical axis along world X+.
    Translation is unit-scale (GTSAM resolves scale from feature tracks).
    keypoints should be undistorted (caller's responsibility).
    """
    R_wc_0 = gtsam.Rot3(np.array([
        [0.0,  0.0, 1.0],
        [1.0,  0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]))
    poses: dict[int, gtsam.Pose3] = {
        img_ids[0]: gtsam.Pose3(R_wc_0, gtsam.Point3(0.0, 0.0, 0.0))
    }

    for k in range(len(img_ids) - 1):
        id_a, id_b = img_ids[k], img_ids[k + 1]
        pose_a = poses[id_a]

        if (id_a, id_b) not in matches_per_pair or len(matches_per_pair[(id_a, id_b)]) < 5:
            poses[id_b] = pose_a
            continue

        matches = matches_per_pair[(id_a, id_b)]
        pts_a = keypoints[id_a][[i for i, _ in matches]].astype(np.float64)
        pts_b = keypoints[id_b][[j for _, j in matches]].astype(np.float64)

        E, mask = cv2.findEssentialMat(
            pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0,
        )
        if E is None:
            poses[id_b] = pose_a
            continue

        _, R_rel, t_rel, _ = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)

        # cv2.recoverPose: P_camB = R_rel @ P_camA + t_rel
        # R_wc_B = R_wc_A @ R_rel^T  (t is unit-scale)
        R_wc_a = pose_a.rotation().matrix()
        R_wc_b = gtsam.Rot3(R_wc_a @ R_rel.T)
        t_b = pose_a.translation() - R_wc_b.matrix() @ t_rel.flatten()
        poses[id_b] = gtsam.Pose3(R_wc_b, gtsam.Point3(*t_b))

    return poses
