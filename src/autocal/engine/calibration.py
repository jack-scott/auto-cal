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

Input topics (Foxglove standard schemas)
-----------------------------------------
  /camera/image           foxglove.CompressedImage
  /camera/calibration     foxglove.CameraCalibration
  /gps/fix                foxglove.LocationFix
  /tf                     foxglove.FrameTransform
  /tf_static              foxglove.FrameTransform

Output topics (same /namespace/type format as input)
-----------------------------------------------------
  /camera/calibration     foxglove.CameraCalibration  (calibrated intrinsics)
  /tf                     foxglove.FrameTransform     (optimised camera poses)
  /camera/sift_overlay    foxglove.CompressedImage    (SIFT keypoint overlays)
  /points/sfm             foxglove.PointCloud         (SfM 3D points)
  /scene/cameras/initial  foxglove.SceneUpdate        (GPS-initialised frustums)
  /scene/cameras/optimized foxglove.SceneUpdate       (optimised frustums)

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
    cal3ds2_from_camera_calibration,
    camera_calibration_from_cal3ds2,
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

OUT_CAL_TOPIC = "/camera/calibration"
OUT_TF_TOPIC = "/tf"
OUT_SIFT_TOPIC = "/camera/sift_overlay"
OUT_POINTS_TOPIC = "/points/sfm"
OUT_POSE_ERROR_TOPIC = "/stats/pose_error"
OUT_CAL_DELTA_TOPIC = "/stats/calibration_delta"


@dataclass
class CalibrationOptions:
    """Tunable parameters for the calibration engine.

    Attributes:
        sift_features:    Max SIFT features per image (0 = unlimited).
        match_ratio:      Lowe ratio test threshold for SIFT matching.
        min_matches:      Minimum matches required to include a pair.
        lm_iterations:    Max Levenberg–Marquardt iterations.
        pose_noise_m:     1-sigma translation noise for GPS prior (metres).
        pose_noise_rad:   1-sigma rotation noise for GPS prior (radians).
        cal_noise_frac:   Fractional noise on focal lengths (e.g. 0.3 = ±30 %).
        cal_cx_noise_frac: Fractional noise on principal point cx/cy (e.g. 0.01
                          = ±1 % of cx).  Kept tight to prevent degenerate
                          trade-offs between principal point and distortion.
        pixel_noise_px:   1-sigma pixel noise for reprojection factors (pixels).
        nadir_camera:     If True, initialise camera poses looking straight down
                          (REP-103 camera Z+ forward = ENU -Z).  Use for nadir surveys.
        max_tracks:       Maximum number of triangulated tracks to include in the
                          GTSAM graph (top by observation count).  Limits graph size.
    """
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


def optimize_sfm(
    images: list[tuple[int, bytes]],
    initial_poses: dict[int, gtsam.Pose3],
    initial_cal: gtsam.Cal3DS2,
    opts: CalibrationOptions,
    triangulation_cal: gtsam.Cal3DS2 | None = None,
) -> dict:
    """Core SfM optimisation: detect/match features, build GTSAM graph, optimise.

    This is the camera-agnostic heart of the pipeline.  Input and output MCAP
    handling lives in CalibrationEngine.run().

    Args:
        images:           Ordered list of (id, jpeg_bytes) pairs.  The order
                          determines sequential matching pairs (adjacent ids).
        initial_poses:    Initial pose estimate for each image id.
        initial_cal:      Initial camera intrinsics — used as both the prior
                          centre for the calibration factor AND the starting
                          value for the optimiser.
        opts:             Tuning options.
        triangulation_cal: If provided, use this calibration for triangulating
                          3D seed points instead of initial_cal.  Supplying a
                          more accurate calibration here prevents the degenerate
                          case where triangulated points perfectly compensate
                          for a wrong initial_cal, leaving zero initial
                          reprojection error and no gradient for the optimiser.
                          In the normal MCAP pipeline this is left as None
                          (initial_cal used for both stages).

    Returns:
        Dict with keys:
          "calibration"  — optimised Cal3DS2
          "poses"        — dict[int, Pose3] of optimised camera poses
          "n_tracks"     — number of triangulated tracks used
          "keypoints"    — dict[int, np.ndarray] of SIFT keypoints per image
          "triangulated" — list of Track objects with point3d set
    """
    img_ids = [img_id for img_id, _ in images]

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
    # Matching — sequential pairs
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
    # Triangulate initial 3D points
    # ------------------------------------------------------------------ #
    cal = initial_cal
    tri_cal = triangulation_cal if triangulation_cal is not None else initial_cal
    K = np.array([
        [tri_cal.fx(), tri_cal.skew(), tri_cal.px()],
        [0,            tri_cal.fy(),   tri_cal.py()],
        [0,            0,              1            ],
    ], dtype=np.float64)
    print(f"  Tracks before triangulation: {len(tracks)}", flush=True)
    triangulate_tracks(tracks, keypoints, K, initial_poses)
    attempted = sum(1 for t in tracks if t.point3d is not None)
    good = [
        t for t in tracks
        if t.point3d is not None
        and _all_positive_depth(t.point3d, t.observations, initial_poses)
    ]
    good.sort(key=lambda t: len(t.observations), reverse=True)
    triangulated = good[: opts.max_tracks]
    print(
        f"  Triangulated: {attempted}, cheirality-ok: {len(good)}, "
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

    # GTSAM Pose3 convention: rotation() stores R_wc so that
    # transform_to(P) = rotation()^{-1} @ (P - t) = R_cw @ (P - t) = camera coords.
    # Our poses store R_cw, so we invert here and invert back when extracting.
    def _to_gtsam_pose(pose: gtsam.Pose3) -> gtsam.Pose3:
        return gtsam.Pose3(pose.rotation().inverse(), gtsam.Point3(*pose.translation()))

    for img_id, pose in initial_poses.items():
        initial_values.insert(X(id_to_idx[img_id]), _to_gtsam_pose(pose))
    initial_values.insert(L(0), cal)

    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        opts.pose_noise_rad, opts.pose_noise_rad, opts.pose_noise_rad,
        opts.pose_noise_m, opts.pose_noise_m, opts.pose_noise_m,
    ]))
    for img_id, pose in initial_poses.items():
        graph.add(gtsam.PriorFactorPose3(
            X(id_to_idx[img_id]), _to_gtsam_pose(pose), pose_noise,
        ))

    # Calibration prior — ordered as Cal3DS2.vector():
    # [fx, fy, skew, cx, cy, k1, k2, p1, p2]
    cal_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        cal.fx() * opts.cal_noise_frac,      # fx
        cal.fy() * opts.cal_noise_frac,      # fy
        1e-6,                                 # skew (pin to ~0)
        cal.px() * opts.cal_cx_noise_frac,   # cx
        cal.py() * opts.cal_cx_noise_frac,   # cy
        opts.k1_sigma,                        # k1
        opts.k2_sigma,                        # k2
        opts.p1_sigma,                        # p1
        opts.p2_sigma,                        # p2
    ]))
    graph.add(gtsam.PriorFactorCal3DS2(L(0), cal, cal_noise))

    pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, opts.pixel_noise_px)
    for j, track in enumerate(triangulated):
        initial_values.insert(P(j), gtsam.Point3(*track.point3d))
        for img_id, kp_idx in track.observations.items():
            if img_id not in id_to_idx:
                continue
            kp = keypoints[img_id][kp_idx]
            measured = gtsam.Point2(float(kp[0]), float(kp[1]))
            graph.add(gtsam.GeneralSFMFactor2Cal3DS2(
                measured, pixel_noise,
                X(id_to_idx[img_id]), P(j), L(0),
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

    opt_cal: gtsam.Cal3DS2 = result.atCal3DS2(L(0))
    # Convert back from GTSAM R_wc convention to our R_cw storage
    opt_poses: dict[int, gtsam.Pose3] = {
        img_id: gtsam.Pose3(
            result.atPose3(X(idx)).rotation().inverse(),
            gtsam.Point3(*result.atPose3(X(idx)).translation()),
        )
        for img_id, idx in id_to_idx.items()
    }

    return {
        "calibration": opt_cal,
        "poses": opt_poses,
        "n_tracks": len(triangulated),
        "keypoints": keypoints,
        "triangulated": triangulated,
    }


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
                cal = cal3ds2_from_camera_calibration(calibration)
            except ValueError:
                cal = _default_cal3ds2(raw_images[0][1])
        else:
            cal = _default_cal3ds2(raw_images[0][1])

        # ------------------------------------------------------------------ #
        # 4. Build initial poses from GPS (or identity if no GPS)
        # ------------------------------------------------------------------ #
        R_cw_nadir = gtsam.Rot3(np.diag([1.0, -1.0, -1.0]))
        R_cw_init = R_cw_nadir if opts.nadir_camera else gtsam.Rot3()

        initial_poses: dict[int, gtsam.Pose3] = {}
        for i, (t_ns, _) in enumerate(raw_images):
            gps_msg = _nearest_gps(gps_fixes, t_ns) if has_gps and len(gps_fixes) > 0 else None
            if gps_msg is not None:
                enu = gps_to_enu(gps_msg.latitude, gps_msg.longitude, gps_msg.altitude,
                                  lat0, lon0, alt0)
                initial_poses[t_ns] = gtsam.Pose3(R_cw_init, gtsam.Point3(*enu))
            else:
                initial_poses[t_ns] = gtsam.Pose3(R_cw_init, gtsam.Point3(i * 0.5, 0.0, 0.0))

        # ------------------------------------------------------------------ #
        # 5. Run core SfM optimisation
        # ------------------------------------------------------------------ #
        images = [(t_ns, bytes(img_msg.data)) for t_ns, img_msg in raw_images]
        result = optimize_sfm(images, initial_poses, cal, opts)

        opt_cal: gtsam.Cal3DS2 = result["calibration"]
        opt_poses: dict[int, gtsam.Pose3] = result["poses"]
        keypoints: dict[int, np.ndarray] = result["keypoints"]
        img_ids = [t_ns for t_ns, _ in raw_images]

        # ------------------------------------------------------------------ #
        # 6. Write output MCAP
        # ------------------------------------------------------------------ #
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
                writer.write(OUT_POINTS_TOPIC, _make_point_cloud(pts, t0_ns, "map"), t0_ns)

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

            _POSE_ERR_FIELDS = {"position_error_m": "number"}
            for t_ns, pose in opt_poses.items():
                if t_ns in initial_poses:
                    err_m = float(np.linalg.norm(
                        pose.translation() - initial_poses[t_ns].translation()
                    ))
                    writer.write_json(OUT_POSE_ERROR_TOPIC, _POSE_ERR_FIELDS,
                                      {"position_error_m": err_m}, t_ns)

            _CAL_DELTA_FIELDS = {k: "number" for k in (
                "fx_initial", "fx_optimized", "fx_delta",
                "fy_initial", "fy_optimized", "fy_delta",
                "cx_initial", "cx_optimized", "cx_delta",
                "cy_initial", "cy_optimized", "cy_delta",
                "k1_initial", "k1_optimized", "k1_delta",
                "k2_initial", "k2_optimized", "k2_delta",
                "p1_initial", "p1_optimized", "p1_delta",
                "p2_initial", "p2_optimized", "p2_delta",
            )}
            writer.write_json(OUT_CAL_DELTA_TOPIC, _CAL_DELTA_FIELDS, {
                "fx_initial":   cal.fx(),      "fx_optimized": opt_cal.fx(),
                "fx_delta":     opt_cal.fx()  - cal.fx(),
                "fy_initial":   cal.fy(),      "fy_optimized": opt_cal.fy(),
                "fy_delta":     opt_cal.fy()  - cal.fy(),
                "cx_initial":   cal.px(),      "cx_optimized": opt_cal.px(),
                "cx_delta":     opt_cal.px()  - cal.px(),
                "cy_initial":   cal.py(),      "cy_optimized": opt_cal.py(),
                "cy_delta":     opt_cal.py()  - cal.py(),
                "k1_initial":   cal.k1(),      "k1_optimized": opt_cal.k1(),
                "k1_delta":     opt_cal.k1()  - cal.k1(),
                "k2_initial":   cal.k2(),      "k2_optimized": opt_cal.k2(),
                "k2_delta":     opt_cal.k2()  - cal.k2(),
                "p1_initial":   float(cal.k()[2]),      "p1_optimized": float(opt_cal.k()[2]),
                "p1_delta":     float(opt_cal.k()[2])  - float(cal.k()[2]),
                "p2_initial":   float(cal.k()[3]),      "p2_optimized": float(opt_cal.k()[3]),
                "p2_delta":     float(opt_cal.k()[3])  - float(cal.k()[3]),
            }, t0_ns)

        return {
            "calibration": opt_cal,
            "poses": opt_poses,
            "n_tracks": result["n_tracks"],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Return (width, height) from JPEG bytes without full decoding."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img.shape[1], img.shape[0]
    return 640, 480


def _default_cal3ds2(img_msg: CompressedImage) -> gtsam.Cal3DS2:
    """Heuristic intrinsics guess when no calibration topic is present.

    foxglove.CompressedImage has no width/height fields, so we decode
    the JPEG to get the actual image dimensions.  Assumes square pixels
    and no distortion as a starting point.
    """
    w, h = _jpeg_dimensions(bytes(img_msg.data))
    f = float(max(w, h)) * 1.2
    cx, cy = w / 2.0, h / 2.0
    return gtsam.Cal3DS2(f, f, 0.0, cx, cy, 0.0, 0.0, 0.0, 0.0)


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
