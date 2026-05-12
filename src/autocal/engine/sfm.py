"""
SfM engine — solves camera poses given a fixed calibration.

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
5. Triangulate track 3D positions.
6. Build GTSAM factor graph:
     - PriorFactorPose3 for each camera that has an initial pose.
     - GenericProjectionFactorCal3DS2 per (camera, landmark, 2D observation)
       with calibration as a fixed constant (not a graph variable).
7. Optimise with Levenberg–Marquardt.
8. Return optimised poses and track data.
"""

from __future__ import annotations

import cv2
import gtsam
import numpy as np
from dataclasses import dataclass

from autocal.engine.features import (
    Track,
    build_tracks,
    detect_sift,
    match_sift,
    triangulate_tracks,
)


@dataclass
class SfmOptions:
    """Tunable parameters for the SfM pose solver.

    Attributes:
        sift_features:  Max SIFT features per image (0 = unlimited).
        match_ratio:    Lowe ratio test threshold for SIFT matching.
        min_matches:    Minimum matches required to keep a pair.
        lm_iterations:  Max Levenberg–Marquardt iterations.
        pose_noise_m:   1-sigma translation noise for pose priors (metres).
        pose_noise_rad: 1-sigma rotation noise for pose priors (radians).
        pixel_noise_px:      1-sigma pixel noise on reprojection factors.
        max_tracks:          Max triangulated tracks to include in the graph.
        max_reproj_error_px: Discard tracks whose initial reprojection error
                             (with the provided poses) exceeds this threshold in
                             any camera.  Removes false SIFT matches before the
                             graph is built.  Only applied when initial_poses is
                             provided.  0 = disabled.
    """
    sift_features: int = 0
    match_ratio: float = 0.75
    min_matches: int = 8
    lm_iterations: int = 100
    pose_noise_m: float = 1.0
    pose_noise_rad: float = 0.1
    pixel_noise_px: float = 1.5
    max_tracks: int = 2000
    max_reproj_error_px: float = 0.0


def optimize_poses(
    images: list[tuple[int, bytes]],
    calibration: gtsam.Cal3DS2,
    opts: SfmOptions,
    initial_poses: dict[int, gtsam.Pose3] | None = None,
) -> dict:
    """Solve camera poses with calibration held fixed.

    Args:
        images:         Ordered list of (img_id, jpeg_bytes).  Sequential
                        matching is done between adjacent entries.
        calibration:    Fixed camera intrinsics.  Not a graph variable.
        opts:           Tuning options.
        initial_poses:  If provided, Pose3(R_wc, t) per img_id used as
                        initial values and prior factors.  If None, poses are
                        initialised via essential-matrix chaining and no prior
                        is added (frame 0 is fixed as the gauge).

    Returns:
        Dict with keys:
          "poses"        — dict[int, Pose3] optimised poses (R_wc, t)
          "n_tracks"     — number of triangulated tracks used
          "keypoints"    — dict[int, np.ndarray] SIFT keypoints per image
          "triangulated" — list[Track] with point3d set
    """
    img_ids = [img_id for img_id, _ in images]
    id_to_idx: dict[int, int] = {img_id: i for i, img_id in enumerate(img_ids)}

    # ------------------------------------------------------------------ #
    # Feature detection
    # ------------------------------------------------------------------ #
    print(f"Detecting features in {len(images)} images...", flush=True)
    keypoints: dict[int, np.ndarray] = {}
    descriptors: dict[int, np.ndarray] = {}
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
        f"  {len(matches_per_pair)}/{len(img_ids)-1} pairs passed "
        f"min_matches={opts.min_matches}",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # Track building
    # ------------------------------------------------------------------ #
    tracks = build_tracks(matches_per_pair)

    # ------------------------------------------------------------------ #
    # Initial poses
    # ------------------------------------------------------------------ #
    has_priors = initial_poses is not None
    if has_priors:
        poses: dict[int, gtsam.Pose3] = dict(initial_poses)
    else:
        poses = _chain_essential_matrix(
            img_ids, keypoints, matches_per_pair, calibration,
        )

    # ------------------------------------------------------------------ #
    # Triangulate
    # ------------------------------------------------------------------ #
    K = _cal_to_K(calibration)
    triangulate_tracks(tracks, keypoints, K, poses)
    good = [
        t for t in tracks
        if t.point3d is not None and _all_positive_depth(t.point3d, t.observations, poses)
    ]
    if opts.max_reproj_error_px > 0 and has_priors:
        good = _filter_by_reproj(good, keypoints, calibration, poses, opts.max_reproj_error_px)

    good.sort(key=lambda t: len(t.observations), reverse=True)
    triangulated = good[: opts.max_tracks]
    print(
        f"  Tracks: {len(tracks)} built, {len(good)} cheirality-ok / reproj-ok, "
        f"using top {len(triangulated)}",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # GTSAM factor graph
    # ------------------------------------------------------------------ #
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    X = gtsam.symbol_shorthand.X
    P = gtsam.symbol_shorthand.P

    for img_id, pose in poses.items():
        initial_values.insert(X(id_to_idx[img_id]), pose)

    if has_priors:
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
            opts.pose_noise_rad, opts.pose_noise_rad, opts.pose_noise_rad,
            opts.pose_noise_m, opts.pose_noise_m, opts.pose_noise_m,
        ]))
        for img_id, pose in initial_poses.items():
            graph.add(gtsam.PriorFactorPose3(X(id_to_idx[img_id]), pose, pose_noise))
    else:
        # Fix frame 0 as the gauge (tight prior, no external reference)
        fixed_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
            1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6,
        ]))
        graph.add(gtsam.PriorFactorPose3(X(0), poses[img_ids[0]], fixed_noise))

    pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, opts.pixel_noise_px)
    for j, track in enumerate(triangulated):
        initial_values.insert(P(j), gtsam.Point3(*track.point3d))
        for img_id, kp_idx in track.observations.items():
            if img_id not in id_to_idx:
                continue
            kp = keypoints[img_id][kp_idx]
            measured = np.array([float(kp[0]), float(kp[1])])
            graph.add(gtsam.GenericProjectionFactorCal3DS2(
                measured, pixel_noise,
                X(id_to_idx[img_id]), P(j),
                calibration,
            ))

    # ------------------------------------------------------------------ #
    # Optimise
    # ------------------------------------------------------------------ #
    print(
        f"Optimising ({len(triangulated)} tracks, {len(poses)} cameras)...",
        flush=True,
    )
    lm_params = gtsam.LevenbergMarquardtParams()
    lm_params.setMaxIterations(opts.lm_iterations)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, lm_params)
    result = optimizer.optimize()
    print(
        f"  Error: {graph.error(initial_values):.3e} → {graph.error(result):.3e}",
        flush=True,
    )

    opt_poses: dict[int, gtsam.Pose3] = {
        img_id: result.atPose3(X(idx))
        for img_id, idx in id_to_idx.items()
    }

    return {
        "poses": opt_poses,
        "n_tracks": len(triangulated),
        "keypoints": keypoints,
        "triangulated": triangulated,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cal_to_K(cal: gtsam.Cal3DS2) -> np.ndarray:
    return np.array([
        [cal.fx(), cal.skew(), cal.px()],
        [0.0,      cal.fy(),   cal.py()],
        [0.0,      0.0,        1.0     ],
    ], dtype=np.float64)


def _filter_by_reproj(
    tracks: list[Track],
    keypoints: dict[int, np.ndarray],
    cal: gtsam.Cal3DS2,
    poses: dict[int, gtsam.Pose3],
    max_err_px: float,
) -> list[Track]:
    """Keep only tracks where every observation reprojects within max_err_px."""
    K = _cal_to_K(cal)
    kept = []
    for track in tracks:
        ok = True
        for img_id, kp_idx in track.observations.items():
            if img_id not in poses:
                continue
            pose = poses[img_id]
            R_cw = pose.rotation().matrix().T
            p_cam = R_cw @ (track.point3d - pose.translation())
            if p_cam[2] <= 0:
                ok = False
                break
            proj = K @ p_cam
            proj /= proj[2]
            obs = keypoints[img_id][kp_idx]
            if np.linalg.norm(proj[:2] - obs) > max_err_px:
                ok = False
                break
        if ok:
            kept.append(track)
    return kept


def _all_positive_depth(
    pt3d: np.ndarray,
    observations: dict,
    poses: dict[int, gtsam.Pose3],
) -> bool:
    for img_id in observations:
        if img_id not in poses:
            continue
        pose = poses[img_id]
        R_cw = pose.rotation().matrix().T  # R_wc stored; .T gives R_cw
        z = (R_cw @ (pt3d - pose.translation()))[2]
        if z <= 0:
            return False
    return True


def _chain_essential_matrix(
    img_ids: list[int],
    keypoints: dict[int, np.ndarray],
    matches_per_pair: dict[tuple[int, int], list[tuple[int, int]]],
    calibration: gtsam.Cal3DS2,
) -> dict[int, gtsam.Pose3]:
    """Initialise poses by chaining essential-matrix relative poses.

    Frame 0 is placed at the origin with optical axis along world X+.
    Translation is unit-scale (GTSAM resolves scale from feature tracks).
    """
    K = _cal_to_K(calibration)

    # Camera facing X+: camera Z (optical) = world X+
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

        # cv2.recoverPose gives R, t such that P_camB = R_rel @ P_camA + t_rel
        # R_cw_B = R_rel @ R_cw_A  →  R_wc_B = R_wc_A @ R_rel^T
        R_wc_a = pose_a.rotation().matrix()
        R_wc_b = gtsam.Rot3(R_wc_a @ R_rel.T)

        # t_B = t_A - R_wc_B @ t_rel  (unit scale)
        t_b = pose_a.translation() - R_wc_b.matrix() @ t_rel.flatten()
        poses[id_b] = gtsam.Pose3(R_wc_b, gtsam.Point3(*t_b))

    return poses
