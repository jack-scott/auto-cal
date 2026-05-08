"""
Calibration engine — reads an MCAP, runs visual SfM, writes calibration output.

Coordinate conventions (REP-103 / REP-105)
-------------------------------------------
Camera frame (ROS REP-103):
  X right, Y down, Z forward (into the scene)

World frame — ENU (ROS REP-103 / REP-105):
  X east, Y north, Z up

GTSAM Pose3(R_cw, t):
  R_cw  — rotation that maps world vectors into camera frame
  t     — camera position in world frame

Nadir (straight-down) camera in ENU:
  Camera Z (forward) = ENU -Z (down)
  Camera X (right)   = ENU +X (east)
  Camera Y (down)    = ENU -Y (south)
  → R_cw = diag(1, -1, -1)

Pipeline
--------
1. discover_topics(input_path)  — inspect MCAP without reading messages.
2. CalibrationEngine.run(input_path, output_path, options)
     a. Load GPS fixes from /gps/fix → ENU origin + camera positions.
     b. Load camera calibration from /camera/calibration (if present).
     c. Load images from /camera/image — detect SIFT, match pairs.
     d. Build tracks via union-find across all pairs.
     e. Initialise GTSAM graph: GPS prior + SfM factors.
     f. Optimise with Levenberg–Marquardt.
     g. Write results to output MCAP: calibrated poses, calibration,
        point cloud, SIFT-overlay images.

Topic convention (Foxglove standard schemas)
--------------------------------------------
  /camera/image           foxglove.CompressedImage
  /camera/calibration     foxglove.CameraCalibration
  /gps/fix                foxglove.LocationFix
  /tf                     foxglove.FrameTransform
  /tf_static              foxglove.FrameTransform

Options
-------
  See CalibrationOptions dataclass.
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
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from google.protobuf.timestamp_pb2 import Timestamp

from autocal.engine.features import (
    Track,
    build_tracks,
    detect_sift,
    draw_matches,
    match_sift,
    triangulate_tracks,
)
from autocal.frames.coordinates import gps_to_enu
from autocal.gtsam_bridge.conversions import (
    cal3bundler_from_camera_calibration,
    camera_calibration_from_cal3bundler,
    frame_transform_from_pose3,
)
from autocal.io.mcap_reader import get_topic_map, iter_messages
from autocal.io.mcap_writer import McapWriter, ns_to_timestamp
from autocal.engine.visualization import write_camera_path
from autocal.optics.camera import overlay_keypoints


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

CAMERA_IMAGE_TOPIC = "/camera/image"
CAMERA_CAL_TOPIC = "/camera/calibration"
GPS_TOPIC = "/gps/fix"


@dataclass
class CalibrationOptions:
    """Tunable parameters for the calibration engine.

    Attributes:
        sift_features:  Max SIFT features per image (0 = unlimited).
        match_ratio:    Lowe ratio test threshold for SIFT matching.
        min_matches:    Minimum matches required to include a pair.
        lm_iterations:  Max Levenberg–Marquardt iterations.
        pose_noise_m:   1-sigma translation noise for GPS prior (metres).
        pose_noise_rad: 1-sigma rotation noise for GPS prior (radians).
        cal_noise_frac: Fractional noise on focal length for initial Cal3Bundler
                        (e.g. 0.3 = ±30 %).
        point_noise_m:  1-sigma noise on triangulated points (metres).
        pixel_noise_px: 1-sigma pixel noise for reprojection factors (pixels).
        nadir_camera:   If True, initialise camera poses looking straight down
                        (REP-103 camera Z+ forward = ENU -Z).  Use for nadir surveys.
        max_tracks:     Maximum number of triangulated tracks to include in the
                        GTSAM graph (top by observation count).  Limits graph size.
    """
    sift_features: int = 0
    match_ratio: float = 0.75
    min_matches: int = 8
    lm_iterations: int = 50
    pose_noise_m: float = 2.0
    pose_noise_rad: float = 0.1
    cal_noise_frac: float = 0.3
    point_noise_m: float = 1.0
    pixel_noise_px: float = 2.0
    nadir_camera: bool = False
    max_tracks: int = 2000


def discover_topics(path: str | Path) -> dict[str, str]:
    """Return the topic→schema map for an MCAP without reading messages.

    Args:
        path: Path to the MCAP file.

    Returns:
        Dict mapping topic string to Foxglove schema name.
    """
    return get_topic_map(path)


class CalibrationEngine:
    """Runs the full visual SfM calibration pipeline on an MCAP file.

    Usage::

        engine = CalibrationEngine()
        engine.run("input.mcap", "output.mcap")
    """

    def __init__(self, options: CalibrationOptions | None = None) -> None:
        self._opts = options or CalibrationOptions()

    def run(self, input_path: str | Path, output_path: str | Path) -> None:
        """Execute the calibration pipeline.

        Args:
            input_path:  Path to input MCAP.
            output_path: Path to write calibration results MCAP.
        """
        opts = self._opts
        topics = discover_topics(input_path)
        has_cal = CAMERA_CAL_TOPIC in topics
        has_gps = GPS_TOPIC in topics

        # ------------------------------------------------------------------ #
        # 1. Load data
        # ------------------------------------------------------------------ #
        gps_fixes: list[tuple[int, LocationFix]] = []
        images: list[tuple[int, CompressedImage]] = []
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
                images.append((t_ns, msg))
            elif topic == CAMERA_CAL_TOPIC:
                calibration = msg

        if len(images) < 2:
            raise ValueError(
                f"Need at least 2 images on {CAMERA_IMAGE_TOPIC}; "
                f"found {len(images)}"
            )

        # ------------------------------------------------------------------ #
        # 2. Build ENU frame from first GPS fix
        # ------------------------------------------------------------------ #
        if has_gps and len(gps_fixes) > 0:
            _, fix0 = gps_fixes[0]
            lat0, lon0, alt0 = fix0.latitude, fix0.longitude, fix0.altitude
        else:
            lat0, lon0, alt0 = 0.0, 0.0, 0.0

        # ------------------------------------------------------------------ #
        # 3. Intrinsics
        # ------------------------------------------------------------------ #
        if calibration is not None:
            try:
                cal = cal3bundler_from_camera_calibration(calibration)
            except ValueError:
                cal = _default_cal3bundler(images[0][1])
        else:
            cal = _default_cal3bundler(images[0][1])

        # ------------------------------------------------------------------ #
        # 4. Feature detection
        # ------------------------------------------------------------------ #
        print(f"Detecting features in {len(images)} images...", flush=True)
        keypoints: dict[int, np.ndarray] = {}
        descriptors: dict[int, np.ndarray] = {}
        for i, (t_ns, img_msg) in enumerate(images):
            kps, descs = detect_sift(bytes(img_msg.data), n_features=opts.sift_features)
            keypoints[t_ns] = kps
            descriptors[t_ns] = descs
            if (i + 1) % 10 == 0 or i + 1 == len(images):
                print(f"  {i+1}/{len(images)}", flush=True)

        # ------------------------------------------------------------------ #
        # 5. Matching — all sequential pairs
        # ------------------------------------------------------------------ #
        print("Matching sequential pairs...", flush=True)
        img_ids = [t for t, _ in images]
        matches_per_pair: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for k in range(len(img_ids) - 1):
            id_a, id_b = img_ids[k], img_ids[k + 1]
            m = match_sift(descriptors[id_a], descriptors[id_b], ratio=opts.match_ratio)
            if len(m) >= opts.min_matches:
                matches_per_pair[(id_a, id_b)] = m
        print(f"  {len(matches_per_pair)}/{len(img_ids)-1} pairs passed min_matches={opts.min_matches}", flush=True)

        # ------------------------------------------------------------------ #
        # 6. Track building
        # ------------------------------------------------------------------ #
        tracks = build_tracks(matches_per_pair)

        # ------------------------------------------------------------------ #
        # 7. Build initial poses from GPS (or identity if no GPS)
        # ------------------------------------------------------------------ #
        # R_cw for a nadir (straight-down) camera in ENU frame:
        #   camera X = ENU East (+X), camera Y = ENU South (-Y), camera Z = ENU Down (-Z)
        #   R_wc cols = [(1,0,0),(0,-1,0),(0,0,-1)]  →  R_cw = R_wc.T = diag(1,-1,-1)
        R_cw_nadir = gtsam.Rot3(np.diag([1.0, -1.0, -1.0]))
        R_cw_init = R_cw_nadir if opts.nadir_camera else gtsam.Rot3()

        initial_poses: dict[int, gtsam.Pose3] = {}
        for i, (t_ns, _) in enumerate(images):
            gps_msg = _nearest_gps(gps_fixes, t_ns) if has_gps and len(gps_fixes) > 0 else None
            if gps_msg is not None:
                enu = gps_to_enu(gps_msg.latitude, gps_msg.longitude, gps_msg.altitude,
                                  lat0, lon0, alt0)
                initial_poses[t_ns] = gtsam.Pose3(R_cw_init, gtsam.Point3(*enu))
            else:
                # No GPS: space cameras 0.5 m apart along X as a bootstrap baseline.
                # The optimizer will refine from here using reprojection factors.
                initial_poses[t_ns] = gtsam.Pose3(R_cw_init, gtsam.Point3(i * 0.5, 0.0, 0.0))

        # ------------------------------------------------------------------ #
        # 8. Triangulate initial 3D points
        # ------------------------------------------------------------------ #
        K = np.array([
            [cal.fx(), 0,        cal.px()],
            [0,        cal.fx(), cal.py()],
            [0,        0,        1       ],
        ], dtype=np.float64)
        print(f"  Tracks before triangulation: {len(tracks)}", flush=True)
        triangulate_tracks(tracks, keypoints, K, initial_poses)
        attempted = sum(1 for t in tracks if t.point3d is not None)
        # Keep only points with positive depth in every observing camera
        good = [
            t for t in tracks
            if t.point3d is not None
            and _all_positive_depth(t.point3d, t.observations, initial_poses)
        ]
        # Select the best max_tracks by number of observations (most-observed = most stable)
        good.sort(key=lambda t: len(t.observations), reverse=True)
        triangulated = good[: opts.max_tracks]
        print(
            f"  Triangulated: {attempted}, cheirality-ok: {len(good)}, "
            f"using top {len(triangulated)} (max_tracks={opts.max_tracks})",
            flush=True,
        )

        # ------------------------------------------------------------------ #
        # 9. GTSAM factor graph
        # ------------------------------------------------------------------ #
        graph = gtsam.NonlinearFactorGraph()
        initial_values = gtsam.Values()

        # Symbol conventions: X(i) = camera pose i, P(j) = 3D point j, K(0) = calibration
        X = gtsam.symbol_shorthand.X
        P = gtsam.symbol_shorthand.P
        L = gtsam.symbol_shorthand.L  # calibration

        # Add initial pose values
        id_to_idx: dict[int, int] = {t: i for i, t in enumerate(img_ids)}
        for t_ns, pose in initial_poses.items():
            initial_values.insert(X(id_to_idx[t_ns]), pose)

        # Add initial calibration
        initial_values.insert(L(0), cal)

        # Pose priors — noise derived from GPS covariance if present in the
        # LocationFix message (DIAGONAL_KNOWN / KNOWN), otherwise fallback to
        # opts.pose_noise_m.  When no GPS data exists at all the translation
        # sigma is set to a very large value so the prior is effectively non-
        # binding and the reprojection factors alone determine camera positions.
        for t_ns, pose in initial_poses.items():
            gps_msg = _nearest_gps(gps_fixes, t_ns) if gps_fixes else None
            t_sigs = _translation_sigmas_from_gps(gps_msg, opts.pose_noise_m)
            pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
                opts.pose_noise_rad, opts.pose_noise_rad, opts.pose_noise_rad,
                *t_sigs,
            ]))
            graph.add(gtsam.PriorFactorPose3(X(id_to_idx[t_ns]), pose, pose_noise))

        # Calibration prior (weak — let it move)
        cal_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
            cal.fx() * opts.cal_noise_frac, 1e-3, 1e-3,
        ]))
        graph.add(gtsam.PriorFactorCal3Bundler(L(0), cal, cal_noise))

        # Reprojection factors
        pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, opts.pixel_noise_px)
        for j, track in enumerate(triangulated):
            initial_values.insert(P(j), gtsam.Point3(*track.point3d))
            for t_ns, kp_idx in track.observations.items():
                if t_ns not in id_to_idx:
                    continue
                kp = keypoints[t_ns][kp_idx]
                measured = gtsam.Point2(float(kp[0]), float(kp[1]))
                graph.add(gtsam.GeneralSFMFactor2Cal3Bundler(
                    measured, pixel_noise,
                    X(id_to_idx[t_ns]), P(j), L(0),
                ))

        # ------------------------------------------------------------------ #
        # 10. Optimise
        # ------------------------------------------------------------------ #
        print(f"Optimising ({len(triangulated)} points, {len(initial_poses)} cameras)...", flush=True)
        lm_params = gtsam.LevenbergMarquardtParams()
        lm_params.setMaxIterations(opts.lm_iterations)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, lm_params)
        # Note: GTSAM prints "CheiralityException: Landmark behind Camera" lines for
        # any point that temporarily goes behind a camera during LM iterations.
        # These are informational — the optimizer continues and they do not indicate failure.
        result = optimizer.optimize()
        print(f"  Done. Error: {graph.error(initial_values):.3e} → {graph.error(result):.3e}", flush=True)

        # ------------------------------------------------------------------ #
        # 11. Extract results
        # ------------------------------------------------------------------ #
        opt_cal: gtsam.Cal3Bundler = result.atCal3Bundler(L(0))
        opt_poses: dict[int, gtsam.Pose3] = {
            t_ns: result.atPose3(X(idx)) for t_ns, idx in id_to_idx.items()
        }

        # ------------------------------------------------------------------ #
        # 12. Write output MCAP
        # ------------------------------------------------------------------ #
        # CompressedImage has no width/height field — get from calibration or decode
        if calibration is not None and calibration.width > 0:
            w, h = calibration.width, calibration.height
        else:
            _, first_img = images[0]
            w, h = _jpeg_dimensions(bytes(first_img.data))

        with McapWriter(output_path) as writer:
            t0_ns = img_ids[0]

            cc = camera_calibration_from_cal3bundler(opt_cal, "camera_link", t0_ns, w, h)
            writer.write("/calibrated/camera/calibration", cc, t0_ns)

            for t_ns, pose in opt_poses.items():
                ft = frame_transform_from_pose3(pose, "map", "camera_link", t_ns)
                writer.write("/calibrated/tf", ft, t_ns)

            # SIFT overlay images
            for t_ns, img_msg in images:
                kps = keypoints[t_ns]
                overlay_bytes = overlay_keypoints(bytes(img_msg.data), kps)
                overlay_msg = CompressedImage()
                overlay_msg.timestamp.CopyFrom(ns_to_timestamp(t_ns))
                overlay_msg.frame_id = "camera_link"
                overlay_msg.format = "jpeg"
                overlay_msg.data = overlay_bytes
                writer.write("/calibrated/camera/sift_overlay", overlay_msg, t_ns)

            if len(triangulated) > 0:
                pts = np.array([t.point3d for t in triangulated], dtype=np.float32)
                pc_msg = _make_point_cloud(pts, t0_ns, "map")
                writer.write("/calibrated/points/sfm", pc_msg, t0_ns)

            # ---------------------------------------------------------- #
            # 3D scene visualisation  (/scene/*)
            # ---------------------------------------------------------- #

            if len(initial_poses) > 0:
                write_camera_path(
                    writer, initial_poses,
                    topic="/scene/cameras/initial",
                    r=0.5, g=0.7, b=1.0,
                    cal=cal,
                )
            write_camera_path(
                writer, opt_poses,
                topic="/scene/cameras/optimized",
                r=0.2, g=1.0, b=0.3,
                cal=opt_cal,
            )

        return {
            "calibration": opt_cal,
            "poses": opt_poses,
            "n_tracks": len(triangulated),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_cal3bundler(img_msg: CompressedImage) -> gtsam.Cal3Bundler:
    """Heuristic focal-length guess when no calibration topic is present."""
    w = img_msg.width if img_msg.width > 0 else 640
    h = img_msg.height if img_msg.height > 0 else 480
    fx = max(w, h) * 1.2
    cx = w / 2.0
    cy = h / 2.0
    return gtsam.Cal3Bundler(fx, 0.0, 0.0, cx, cy)


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Return (width, height) from JPEG bytes without full decoding."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img.shape[1], img.shape[0]
    return 640, 480


def _all_positive_depth(
    pt3d: np.ndarray,
    observations: dict,
    poses: dict,
) -> bool:
    """Return True if pt3d has positive depth in every camera that observes it."""
    for img_id in observations:
        if img_id not in poses:
            continue
        pose = poses[img_id]
        R = pose.rotation().matrix()
        t = pose.translation()
        z = (R @ (pt3d - t))[2]
        if z <= 0:
            return False
    return True


_NO_GPS_SIGMA_M = 100.0  # effectively unconstrained when no GPS data


def _translation_sigmas_from_gps(
    fix: LocationFix | None,
    fallback_m: float,
) -> tuple[float, float, float]:
    """Return (σ_E, σ_N, σ_U) in metres for use as a pose prior.

    Reads the diagonal ENU covariance from the LocationFix if the type is
    DIAGONAL_KNOWN or KNOWN.  Falls back to fallback_m for all axes if the
    covariance type is UNKNOWN / APPROXIMATED, or to _NO_GPS_SIGMA_M if the
    fix itself is None (no GPS in the dataset).
    """
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
    """Return the GPS fix closest in time to t_ns."""
    if not gps_fixes:
        return None
    best_t, best_msg = min(gps_fixes, key=lambda x: abs(x[0] - t_ns))
    return best_msg


def _make_point_cloud(pts: np.ndarray, t_ns: int, frame_id: str) -> PointCloud:
    """Pack a float32 XYZ point cloud into a PointCloud proto.

    Args:
        pts:      shape-(N,3) float32 array.
        t_ns:     Timestamp in Unix nanoseconds.
        frame_id: Frame the points are expressed in.

    Returns:
        Populated PointCloud message with FLOAT32 XYZ fields.
    """
    from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField

    pc = PointCloud()
    pc.timestamp.CopyFrom(ns_to_timestamp(t_ns))
    pc.frame_id = frame_id
    pc.pose.position.x = 0.0
    pc.pose.position.y = 0.0
    pc.pose.position.z = 0.0
    pc.pose.orientation.w = 1.0

    # FLOAT32 = 7 in PackedElementField.NumericType
    FLOAT32 = 7
    stride = 12  # 3 × 4 bytes

    for name, offset in [("x", 0), ("y", 4), ("z", 8)]:
        f = pc.fields.add()
        f.name = name
        f.offset = offset
        f.type = FLOAT32

    pc.point_stride = stride
    pc.data = pts.astype(np.float32).tobytes()
    return pc
