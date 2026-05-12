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
5. Triangulate track 3D positions using gtsam.triangulatePoint3 (multi-view, nonlinear
   refinement) with distorted pixel observations.
6. Build GTSAM factor graph:
     - PriorFactorPose3 for each camera that has an initial pose.
     - GenericProjectionFactorCal3DS2 per (camera, landmark, 2D observation)
       with calibration as a fixed constant (not a graph variable).
7. Optimise with Dogleg (trust-region, more robust than LM for large initial error).
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
    filter_matches_ransac,
    match_sift,
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
        max_landmark_dist_m: Discard triangulated landmarks farther than this
                             from every observing camera.  Eliminates near-
                             degenerate points (nearly-parallel rays) that cause
                             singular Jacobians in the factor graph.  0 = disabled.
        min_parallax_deg:    Discard landmarks where the maximum viewing angle
                             between any two cameras is below this threshold.
                             Enforces a minimum triangulation baseline.  Small
                             angles produce poorly-conditioned Jacobians even
                             when the point is within the distance limit.
                             0 = disabled.
        huber_loss:          Use a Huber robust noise model for projection
                             factors instead of Gaussian.  Downweights
                             observations with large reprojection errors (> k
                             pixels where k = pixel_noise_px) so surviving
                             outlier matches don't dominate the solution.
    """
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
        images:             Ordered list of (img_id, jpeg_bytes).  Image bytes
                            are used for SIFT detection only when
                            preloaded_features is None.
        calibration:        Fixed camera intrinsics.  Not a graph variable.
        opts:               Tuning options.
        initial_poses:      If provided, Pose3(R_wc, t) per img_id used as
                            initial values and prior factors.
        preloaded_features: If provided, {img_id: (kps, descs)} loaded from a
                            cached MCAP topic.  Detection is skipped.
                            Features are truncated to opts.sift_features if set.

    Returns:
        Dict with keys:
          "poses"        — dict[int, Pose3] optimised poses (R_wc, t)
          "n_tracks"     — number of triangulated tracks used
          "keypoints"    — dict[int, np.ndarray] SIFT keypoints per image
          "descriptors"  — dict[int, np.ndarray] SIFT descriptors per image
          "triangulated" — list[Track] with point3d set
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
    K = _cal_to_K(calibration)
    kp_dist = np.array(calibration.k(), dtype=np.float64)
    if np.any(kp_dist != 0.0):
        keypoints_for_geo: dict[int, np.ndarray] = {}
        for img_id, kps in keypoints.items():
            pts = kps.reshape(-1, 1, 2).astype(np.float64)
            if _fisheye:
                D = kp_dist[:4].reshape(4, 1)
                undist = cv2.fisheye.undistortPoints(pts, K, D, P=K)
            else:
                undist = cv2.undistortPoints(pts, K, kp_dist[:4], P=K)
            keypoints_for_geo[img_id] = undist.reshape(-1, 2).astype(np.float32)
    else:
        keypoints_for_geo = keypoints

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
        poses = _chain_essential_matrix(
            img_ids, keypoints_for_geo, matches_per_pair, K,
        )

    # ------------------------------------------------------------------ #
    # Triangulate — GTSAM multi-view with nonlinear refinement
    # Uses distorted pixel observations so the calibration model is applied
    # correctly; all visible cameras contribute (not just the first 2).
    # ------------------------------------------------------------------ #
    _triangulate_gtsam(tracks, keypoints, calibration, poses, id_to_idx,
                       max_dist=opts.max_landmark_dist_m,
                       min_parallax_deg=opts.min_parallax_deg)
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

    base_noise = gtsam.noiseModel.Isotropic.Sigma(2, opts.pixel_noise_px)
    if opts.huber_loss:
        pixel_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(opts.pixel_noise_px),
            base_noise,
        )
    else:
        pixel_noise = base_noise
    for j, track in enumerate(triangulated):
        initial_values.insert(P(j), gtsam.Point3(*track.point3d))
        for img_id, kp_idx in track.observations.items():
            if img_id not in id_to_idx:
                continue
            kp = keypoints[img_id][kp_idx]
            measured = np.array([float(kp[0]), float(kp[1])])
            if _fisheye:
                graph.add(gtsam.GenericProjectionFactorCal3Fisheye(
                    measured, pixel_noise,
                    X(id_to_idx[img_id]), P(j),
                    calibration,
                ))
            else:
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
    dogleg_params = gtsam.DoglegParams()
    dogleg_params.setMaxIterations(opts.lm_iterations)
    optimizer = gtsam.DoglegOptimizer(graph, initial_values, dogleg_params)
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
        "initial_poses": poses,   # pre-optimisation (E-matrix chain or provided priors)
        "n_tracks": len(triangulated),
        "keypoints": keypoints,
        "descriptors": descriptors,
        "triangulated": triangulated,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _triangulate_gtsam(
    tracks: list[Track],
    keypoints: dict[int, np.ndarray],
    calibration: gtsam.Cal3DS2 | gtsam.Cal3Fisheye,
    poses: dict[int, gtsam.Pose3],
    id_to_idx: dict[int, int],
    max_dist: float = 0.0,
    min_parallax_deg: float = 0.0,
) -> None:
    """Triangulate tracks in-place using gtsam.triangulatePoint3.

    Uses all visible cameras (not just 2) and applies nonlinear refinement,
    which is substantially more accurate than 2-view DLT.  Observations are
    distorted pixels — the calibration model handles projection internally.

    max_dist > 0 discards points farther than that from every observing camera.
    min_parallax_deg > 0 requires the maximum viewing angle across all camera
    pairs to exceed this threshold — small angles produce near-zero Jacobians
    that make the landmark information matrix singular.
    """
    for track in tracks:
        visible = [
            (img_id, kp_idx)
            for img_id, kp_idx in track.observations.items()
            if img_id in poses and img_id in id_to_idx
        ]
        if len(visible) < 2:
            continue

        pose_vec = gtsam.Pose3Vector()
        meas_vec = gtsam.Point2Vector()
        for img_id, kp_idx in visible:
            pose_vec.append(poses[img_id])
            kp = keypoints[img_id][kp_idx]
            meas_vec.append(gtsam.Point2(float(kp[0]), float(kp[1])))

        try:
            pt = gtsam.triangulatePoint3(pose_vec, calibration, meas_vec,
                                         rank_tol=1e-9, optimize=True)
            pt_np = np.array([float(pt[0]), float(pt[1]), float(pt[2])])
        except Exception:
            track.point3d = None
            continue

        if max_dist > 0.0:
            min_cam_dist = min(
                np.linalg.norm(pt_np - poses[img_id].translation())
                for img_id, _ in visible
            )
            if min_cam_dist > max_dist:
                track.point3d = None
                continue

        if min_parallax_deg > 0.0:
            cam_positions = np.array([poses[img_id].translation() for img_id, _ in visible])
            rays = pt_np - cam_positions                                   # (N, 3)
            norms = np.linalg.norm(rays, axis=1, keepdims=True)
            if np.any(norms < 1e-10):
                track.point3d = None
                continue
            rays_norm = rays / norms
            cos_mat = rays_norm @ rays_norm.T                              # (N, N)
            np.fill_diagonal(cos_mat, 1.0)
            max_angle_deg = float(np.degrees(np.arccos(np.clip(cos_mat.min(), -1.0, 1.0))))
            if max_angle_deg < min_parallax_deg:
                track.point3d = None
                continue

        track.point3d = pt_np


def _cal_to_K(cal: gtsam.Cal3DS2 | gtsam.Cal3Fisheye) -> np.ndarray:
    return np.array([
        [cal.fx(), cal.skew(), cal.px()],
        [0.0,      cal.fy(),   cal.py()],
        [0.0,      0.0,        1.0     ],
    ], dtype=np.float64)


def _filter_by_reproj(
    tracks: list[Track],
    keypoints: dict[int, np.ndarray],
    cal: gtsam.Cal3DS2 | gtsam.Cal3Fisheye,
    poses: dict[int, gtsam.Pose3],
    max_err_px: float,
) -> list[Track]:
    """Keep only tracks where every observation reprojects within max_err_px."""
    _fisheye = isinstance(cal, gtsam.Cal3Fisheye)
    k_coeffs = np.array(cal.k(), dtype=np.float64)
    kept = []
    for track in tracks:
        ok = True
        for img_id, kp_idx in track.observations.items():
            if img_id not in poses:
                continue
            pose = poses[img_id]
            R_cw = pose.rotation().matrix().T
            p_cam = R_cw @ (track.point3d - pose.translation())
            x, y, z = p_cam
            if z <= 0:
                ok = False
                break
            if _fisheye:
                r = np.sqrt(x*x + y*y)
                if r < 1e-10:
                    pu, pv = cal.px(), cal.py()
                else:
                    theta = np.arctan2(r, z)
                    t2 = theta * theta
                    rd = theta * (1 + k_coeffs[0]*t2 + k_coeffs[1]*t2**2
                                  + k_coeffs[2]*t2**3 + k_coeffs[3]*t2**4)
                    s = rd / r
                    pu = cal.fx() * s * x + cal.px()
                    pv = cal.fy() * s * y + cal.py()
            else:
                xn, yn = x / z, y / z
                pu = cal.fx() * xn + cal.px()
                pv = cal.fy() * yn + cal.py()
            obs = keypoints[img_id][kp_idx]
            if np.hypot(pu - obs[0], pv - obs[1]) > max_err_px:
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
    K: np.ndarray,
) -> dict[int, gtsam.Pose3]:
    """Initialise poses by chaining essential-matrix relative poses.

    Frame 0 is placed at the origin with optical axis along world X+.
    Translation is unit-scale (GTSAM resolves scale from feature tracks).
    keypoints should be undistorted (caller's responsibility).
    """

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
