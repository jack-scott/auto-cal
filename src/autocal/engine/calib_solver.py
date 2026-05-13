"""
Calibration solver — refines camera intrinsics given fixed (or soft-prior) poses.

Coordinate conventions (REP-103 / REP-105)
-------------------------------------------
Camera frame: X right, Y down, Z forward (optical axis).
World frame:  ENU (X east, Y north, Z up) when GPS is present; arbitrary otherwise.
GTSAM Pose3(R_wc, t): R_wc = camera→world rotation, t = camera position in world.

Pipeline
--------
1. discover_topics(input_path)  — inspect MCAP index (no message reads).
2. optimize_sfm(images, initial_poses, initial_cal, opts)
     a. Detect SIFT features; build sequential matches and feature tracks.
     b. Triangulate 3D seed points.
     c. Build GTSAM graph: pose priors + calibration prior +
        GeneralSFMFactor2 projection factors (calibration as variable key).
     d. Optimise with Levenberg–Marquardt.
3. CalibrationEngine.run(input_path, output_path)
     a. Load images, calibration, and GPS fixes from MCAP.
     b. Build GPS-based initial poses (or uniformly spaced if no GPS).
     c. Call optimize_sfm and write results to output MCAP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import gtsam
import numpy as np

from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix

from autocal.engine.calibration_msgs import CalibrationDeltaMsg, PoseErrorMsg
from autocal.engine.features import (
    Track,
    all_positive_depth,
    build_tracks,
    cal_to_K,
    detect_sift,
    filter_by_reproj,
    match_sift,
    triangulate_tracks,
    undistort_keypoints,
)
from autocal.engine.visualization import make_point_cloud, write_camera_path
from autocal.frames.coordinates import gps_to_enu
from autocal.gtsam_bridge.conversions import (
    cal3ds2_from_camera_calibration,
    camera_calibration_from_cal3ds2,
    frame_transform_from_pose3,
)
from autocal.io.mcap_reader import get_topic_map, iter_messages
from autocal.io.mcap_writer import McapWriter, ns_to_timestamp
from autocal.optics.camera import overlay_keypoints


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

CAMERA_IMAGE_TOPIC = "/camera/image"
CAMERA_CAL_TOPIC   = "/camera/calibration"
GPS_TOPIC          = "/gps/fix"

OUT_CAL_TOPIC        = "/camera/calibration"
OUT_TF_TOPIC         = "/tf"
OUT_SIFT_TOPIC       = "/camera/sift_overlay"
OUT_POINTS_TOPIC     = "/points/sfm"
OUT_POSE_ERROR_TOPIC = "/stats/pose_error"
OUT_CAL_DELTA_TOPIC  = "/stats/calibration_delta"


@dataclass
class CalibrationOptions:
    """Tunable parameters for the calibration solver."""
    sift_features: int = 0
    match_ratio: float = 0.75
    min_matches: int = 8
    lm_iterations: int = 50
    pose_noise_m: float = 2.0
    pose_noise_rad: float = 0.1
    cal_noise_frac: float = 0.3
    cal_cx_noise_frac: float = 0.01
    k1_sigma: float = 0.5
    k2_sigma: float = 0.1
    p1_sigma: float = 0.01
    p2_sigma: float = 0.01
    k3_sigma: float = 0.1
    k4_sigma: float = 0.1
    pixel_noise_px: float = 2.0
    reproj_filter_px: float = 20.0
    point_anchor_sigma: float = 0.005
    max_tracks: int = 2000


def discover_topics(path: str | Path) -> dict[str, str]:
    """Return the topic→schema map for an MCAP without reading messages."""
    return get_topic_map(path)


def optimize_sfm(
    images: list[tuple[int, bytes]],
    initial_poses: dict[int, gtsam.Pose3],
    initial_cal: gtsam.Cal3DS2 | gtsam.Cal3Fisheye,
    opts: CalibrationOptions,
    triangulation_cal: gtsam.Cal3DS2 | gtsam.Cal3Fisheye | None = None,
) -> dict:
    """Core SfM optimisation: detect features, build GTSAM graph, optimise calibration.

    Args:
        images:            Ordered list of (id, jpeg_bytes) pairs.
        initial_poses:     Initial Pose3(R_wc, t) per image id.
        initial_cal:       Initial camera intrinsics and prior centre.
        opts:              Tuning options.
        triangulation_cal: If provided, use this calibration for triangulation
                           instead of initial_cal.  Supplying accurate intrinsics
                           here prevents the degenerate case where wrong initial_cal
                           perfectly explains the triangulated structure.

    Returns:
        Dict with keys: "calibration", "poses", "n_tracks",
        "keypoints", "triangulated".
    """
    img_ids = [img_id for img_id, _ in images]
    _fisheye = isinstance(initial_cal, gtsam.Cal3Fisheye)

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
    # Track building + triangulation
    # ------------------------------------------------------------------ #
    tracks = build_tracks(matches_per_pair)
    tri_cal = triangulation_cal if triangulation_cal is not None else initial_cal
    K = cal_to_K(tri_cal)
    keypoints_for_tri = undistort_keypoints(keypoints, tri_cal)

    print(f"  Tracks before triangulation: {len(tracks)}", flush=True)
    triangulate_tracks(tracks, keypoints_for_tri, K, initial_poses)
    attempted = sum(1 for t in tracks if t.point3d is not None)
    good = [
        t for t in tracks
        if t.point3d is not None
        and all_positive_depth(t.point3d, t.observations, initial_poses)
    ]

    if opts.reproj_filter_px > 0:
        before_filter = len(good)
        good = filter_by_reproj(good, keypoints, initial_poses, tri_cal, opts.reproj_filter_px)
        print(
            f"  Reprojection filter ({opts.reproj_filter_px}px): "
            f"{before_filter} → {len(good)} tracks",
            flush=True,
        )

    good.sort(key=lambda t: len(t.observations), reverse=True)
    triangulated = good[: opts.max_tracks]
    print(
        f"  Triangulated: {attempted}, cheirality-ok+filtered: {len(good)}, "
        f"using top {len(triangulated)} (max_tracks={opts.max_tracks})",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # GTSAM factor graph
    # ------------------------------------------------------------------ #
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    X = gtsam.symbol_shorthand.X
    P = gtsam.symbol_shorthand.P
    L = gtsam.symbol_shorthand.L

    id_to_idx: dict[int, int] = {img_id: i for i, img_id in enumerate(img_ids)}

    for img_id, pose in initial_poses.items():
        initial_values.insert(X(id_to_idx[img_id]), pose)
    initial_values.insert(L(0), initial_cal)

    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        opts.pose_noise_rad, opts.pose_noise_rad, opts.pose_noise_rad,
        opts.pose_noise_m,   opts.pose_noise_m,   opts.pose_noise_m,
    ]))
    for img_id, pose in initial_poses.items():
        graph.add(gtsam.PriorFactorPose3(X(id_to_idx[img_id]), pose, pose_noise))

    if _fisheye:
        d_sigmas = [opts.k1_sigma, opts.k2_sigma, opts.k3_sigma, opts.k4_sigma]
    else:
        d_sigmas = [opts.k1_sigma, opts.k2_sigma, opts.p1_sigma, opts.p2_sigma]
    cal_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        initial_cal.fx() * opts.cal_noise_frac,
        initial_cal.fy() * opts.cal_noise_frac,
        1e-6,
        initial_cal.px() * opts.cal_cx_noise_frac,
        initial_cal.py() * opts.cal_cx_noise_frac,
        *d_sigmas,
    ]))
    if _fisheye:
        graph.add(gtsam.PriorFactorCal3Fisheye(L(0), initial_cal, cal_noise))
    else:
        graph.add(gtsam.PriorFactorCal3DS2(L(0), initial_cal, cal_noise))

    pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, opts.pixel_noise_px)
    for j, track in enumerate(triangulated):
        initial_values.insert(P(j), gtsam.Point3(*track.point3d))
        for img_id, kp_idx in track.observations.items():
            if img_id not in id_to_idx:
                continue
            kp = keypoints[img_id][kp_idx]
            measured = gtsam.Point2(float(kp[0]), float(kp[1]))
            if _fisheye:
                graph.add(gtsam.GeneralSFMFactor2Cal3Fisheye(
                    measured, pixel_noise,
                    X(id_to_idx[img_id]), P(j), L(0),
                ))
            else:
                graph.add(gtsam.GeneralSFMFactor2Cal3DS2(
                    measured, pixel_noise,
                    X(id_to_idx[img_id]), P(j), L(0),
                ))

    if triangulation_cal is not None and opts.point_anchor_sigma > 0:
        point_anchor_noise = gtsam.noiseModel.Isotropic.Sigma(3, opts.point_anchor_sigma)
        for j, track in enumerate(triangulated):
            graph.add(gtsam.PriorFactorPoint3(
                P(j), gtsam.Point3(*track.point3d), point_anchor_noise,
            ))

    # ------------------------------------------------------------------ #
    # Optimise
    # ------------------------------------------------------------------ #
    print(
        f"Optimising ({len(triangulated)} tracks, {len(initial_poses)} cameras)...",
        flush=True,
    )
    lm_params = gtsam.LevenbergMarquardtParams()
    lm_params.setMaxIterations(opts.lm_iterations)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, lm_params)
    result = optimizer.optimize()
    print(
        f"  Done. Error: {graph.error(initial_values):.3e} → {graph.error(result):.3e}",
        flush=True,
    )

    if _fisheye:
        opt_cal: gtsam.Cal3DS2 | gtsam.Cal3Fisheye = result.atCal3Fisheye(L(0))
    else:
        opt_cal = result.atCal3DS2(L(0))
    opt_poses: dict[int, gtsam.Pose3] = {
        img_id: result.atPose3(X(idx))
        for img_id, idx in id_to_idx.items()
    }

    return {
        "calibration":  opt_cal,
        "poses":        opt_poses,
        "n_tracks":     len(triangulated),
        "keypoints":    keypoints,
        "triangulated": triangulated,
    }


class CalibrationEngine:
    """Runs the full visual SfM calibration pipeline on an MCAP file."""

    def __init__(self, options: CalibrationOptions | None = None) -> None:
        self._opts = options or CalibrationOptions()

    def run(self, input_path: str | Path, output_path: str | Path) -> dict:
        opts = self._opts
        topics = discover_topics(input_path)
        has_cal = CAMERA_CAL_TOPIC in topics
        has_gps = GPS_TOPIC in topics

        gps_fixes: list[tuple[int, LocationFix]] = []
        raw_images: list[tuple[int, CompressedImage]] = []
        calibration: CameraCalibration | None = None

        load_topics = [CAMERA_IMAGE_TOPIC]
        if has_gps:
            load_topics.append(GPS_TOPIC)
        if has_cal:
            load_topics.append(CAMERA_CAL_TOPIC)

        for topic, t_ns, msg in iter_messages(input_path, topics=load_topics):
            if topic == GPS_TOPIC:
                gps_fixes.append((t_ns, msg))
            elif topic == CAMERA_IMAGE_TOPIC:
                raw_images.append((t_ns, msg))
            elif topic == CAMERA_CAL_TOPIC:
                calibration = msg

        if len(raw_images) < 2:
            raise ValueError(
                f"Need at least 2 images on {CAMERA_IMAGE_TOPIC}; "
                f"found {len(raw_images)}"
            )

        if has_gps and gps_fixes:
            _, fix0 = gps_fixes[0]
            lat0, lon0, alt0 = fix0.latitude, fix0.longitude, fix0.altitude
        else:
            lat0, lon0, alt0 = 0.0, 0.0, 0.0

        if calibration is not None:
            try:
                cal = cal3ds2_from_camera_calibration(calibration)
            except ValueError:
                cal = _default_cal3ds2(raw_images[0][1])
        else:
            cal = _default_cal3ds2(raw_images[0][1])

        R_wc_init = gtsam.Rot3()
        initial_poses: dict[int, gtsam.Pose3] = {}
        for i, (t_ns, _) in enumerate(raw_images):
            gps_msg = _nearest_gps(gps_fixes, t_ns) if has_gps and gps_fixes else None
            if gps_msg is not None:
                enu = gps_to_enu(gps_msg.latitude, gps_msg.longitude, gps_msg.altitude,
                                 lat0, lon0, alt0)
                initial_poses[t_ns] = gtsam.Pose3(R_wc_init, gtsam.Point3(*enu))
            else:
                initial_poses[t_ns] = gtsam.Pose3(R_wc_init, gtsam.Point3(i * 0.5, 0.0, 0.0))

        images = [(t_ns, bytes(img_msg.data)) for t_ns, img_msg in raw_images]
        result = optimize_sfm(images, initial_poses, cal, opts)

        opt_cal: gtsam.Cal3DS2 = result["calibration"]
        opt_poses: dict[int, gtsam.Pose3] = result["poses"]
        keypoints: dict[int, np.ndarray] = result["keypoints"]
        img_ids = [t_ns for t_ns, _ in raw_images]

        if calibration is not None and calibration.width > 0:
            w, h = calibration.width, calibration.height
        else:
            _, first_img = raw_images[0]
            w, h = _jpeg_dimensions(bytes(first_img.data))

        with McapWriter(output_path) as writer:
            t0_ns = img_ids[0]

            cc = camera_calibration_from_cal3ds2(opt_cal, "camera_link", t0_ns, w, h)
            writer.write(OUT_CAL_TOPIC, cc, t0_ns)

            for t_ns, pose in opt_poses.items():
                ft = frame_transform_from_pose3(pose, "map", "camera_link", t_ns)
                writer.write(OUT_TF_TOPIC, ft, t_ns)

            for t_ns, img_msg in raw_images:
                kps = keypoints[t_ns]
                overlay_bytes = overlay_keypoints(bytes(img_msg.data), kps)
                overlay_msg = CompressedImage()
                overlay_msg.timestamp.CopyFrom(ns_to_timestamp(t_ns))
                overlay_msg.frame_id = "camera_link"
                overlay_msg.format = "jpeg"
                overlay_msg.data = overlay_bytes
                writer.write(OUT_SIFT_TOPIC, overlay_msg, t_ns)

            if result["n_tracks"] > 0:
                pts = np.array(
                    [t.point3d for t in result["triangulated"]], dtype=np.float32
                )
                writer.write(OUT_POINTS_TOPIC, make_point_cloud(pts, t0_ns), t0_ns)

            write_camera_path(writer, initial_poses,
                              topic="/scene/cameras/initial",
                              r=0.5, g=0.7, b=1.0, cal=cal)
            write_camera_path(writer, opt_poses,
                              topic="/scene/cameras/optimized",
                              r=0.2, g=1.0, b=0.3, cal=opt_cal)

            for t_ns, pose in opt_poses.items():
                if t_ns in initial_poses:
                    err_m = float(np.linalg.norm(
                        pose.translation() - initial_poses[t_ns].translation()
                    ))
                    writer.write(OUT_POSE_ERROR_TOPIC,
                                 PoseErrorMsg(position_error_m=err_m), t_ns)

            writer.write(OUT_CAL_DELTA_TOPIC, CalibrationDeltaMsg(
                fx_initial=cal.fx(),           fx_optimized=opt_cal.fx(),
                fx_delta=opt_cal.fx()         - cal.fx(),
                fy_initial=cal.fy(),           fy_optimized=opt_cal.fy(),
                fy_delta=opt_cal.fy()         - cal.fy(),
                cx_initial=cal.px(),           cx_optimized=opt_cal.px(),
                cx_delta=opt_cal.px()         - cal.px(),
                cy_initial=cal.py(),           cy_optimized=opt_cal.py(),
                cy_delta=opt_cal.py()         - cal.py(),
                k1_initial=cal.k1(),           k1_optimized=opt_cal.k1(),
                k1_delta=opt_cal.k1()         - cal.k1(),
                k2_initial=cal.k2(),           k2_optimized=opt_cal.k2(),
                k2_delta=opt_cal.k2()         - cal.k2(),
                p1_initial=float(cal.k()[2]),  p1_optimized=float(opt_cal.k()[2]),
                p1_delta=float(opt_cal.k()[2]) - float(cal.k()[2]),
                p2_initial=float(cal.k()[3]),  p2_optimized=float(opt_cal.k()[3]),
                p2_delta=float(opt_cal.k()[3]) - float(cal.k()[3]),
            ), t0_ns)

        return {
            "calibration": opt_cal,
            "poses":       opt_poses,
            "n_tracks":    result["n_tracks"],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img.shape[1], img.shape[0]
    return 640, 480


def _default_cal3ds2(img_msg: CompressedImage) -> gtsam.Cal3DS2:
    w, h = _jpeg_dimensions(bytes(img_msg.data))
    f = float(max(w, h)) * 1.2
    return gtsam.Cal3DS2(f, f, 0.0, w / 2.0, h / 2.0, 0.0, 0.0, 0.0, 0.0)


_NO_GPS_SIGMA_M = 100.0


def _translation_sigmas_from_gps(
    fix: LocationFix | None,
    fallback_m: float,
) -> tuple[float, float, float]:
    if fix is None:
        return _NO_GPS_SIGMA_M, _NO_GPS_SIGMA_M, _NO_GPS_SIGMA_M
    cov_type = fix.position_covariance_type
    if cov_type in (LocationFix.DIAGONAL_KNOWN, LocationFix.KNOWN):
        cov = list(fix.position_covariance)
        if len(cov) >= 9:
            return (
                math.sqrt(max(cov[0], 1e-6)),
                math.sqrt(max(cov[4], 1e-6)),
                math.sqrt(max(cov[8], 1e-6)),
            )
    return fallback_m, fallback_m, fallback_m


def _nearest_gps(
    gps_fixes: list[tuple[int, LocationFix]], t_ns: int
) -> LocationFix | None:
    if not gps_fixes:
        return None
    _, best_msg = min(gps_fixes, key=lambda x: abs(x[0] - t_ns))
    return best_msg
