"""
Generic SfM CLI: estimate camera poses from images in an MCAP file.

Reads /camera/image and /camera/calibration (and optionally /tf for initial
poses), runs visual SfM (essential-matrix chaining or pose-prior initialisation
→ GTSAM optimisation with calibration held fixed), and writes an output MCAP:

  /camera/image            original images (passthrough)
  /camera/calibration      original calibration (passthrough)
  /tf                      optimised camera poses
  /camera/sift_overlay     SIFT keypoints drawn on each image
  /points/sfm              sparse 3D point cloud
  /scene/cameras/initial   pre-optimisation camera frustums + path (blue)
  /scene/cameras/optimized post-optimisation camera frustums + path (green)
  /ape                     per-frame APE vs GT (only when /tf present in input)
                             fields: translation_m, rotation_deg

If /tf is present in the input MCAP the poses are used as initial values and
soft priors, bypassing essential-matrix chaining.

Usage:
    pixi run python pipelines/sfm.py input.mcap output.mcap [options]

Options:
    --preset {fine,medium,rough,no_pose}
                          Apply a named configuration preset (see PRESETS dict).
                          Individual flags override preset values.
    --ignore-poses        Ignore /tf in input; initialise from essential matrix.
    --sift-features N     Max SIFT features per image (default: 1000)
    --match-ratio R       Lowe ratio test (default: 0.75)
    --min-matches N       Min matches per pair (default: 8)
    --lm-iterations N     Max Dogleg iterations (default: 200)
    --pixel-noise-px P    Reprojection pixel noise σ (default: 1.5)
    --max-tracks N        Max triangulated tracks in graph (default: 1500)
    --pose-noise-m M      Pose prior translation σ (default: 0.1, only with /tf)
    --pose-noise-rad R    Pose prior rotation σ in radians (default: 0.01, only with /tf)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gtsam
import numpy as np

from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform

from autocal.engine.features import decode_sift_features, encode_sift_features
from autocal.engine.sfm_solver import SfmOptions, optimize_poses
from autocal.engine.visualization import make_point_cloud, write_camera_path
from autocal.gtsam_bridge.conversions import (
    calibration_from_mcap_msg,
    frame_transform_from_pose3,
    pose3_from_frame_transform,
)
from autocal.io.mcap_reader import get_topic_map, iter_messages
from autocal.io.mcap_writer import McapWriter, ns_to_timestamp
from autocal.metrics.ape import ApePoseMsg, ApeSummaryMsg, compute_ape
from autocal.optics.camera import overlay_keypoints

CAMERA_IMAGE_TOPIC = "/camera/image"
CAMERA_CAL_TOPIC   = "/camera/calibration"
TF_TOPIC           = "/tf"
TF_GT_TOPIC        = "/tf_gt"
SIFT_TOPIC         = "/camera/sift_overlay"
POINTS_TOPIC       = "/points/sfm"
SCENE_INITIAL_TOPIC   = "/scene/cameras/initial"
SCENE_OPTIMIZED_TOPIC = "/scene/cameras/optimized"
APE_INIT_TOPIC        = "/ape/initial"
APE_INIT_SUMMARY      = "/ape/initial/summary"
APE_OPT_TOPIC         = "/ape/optimized"
APE_OPT_SUMMARY       = "/ape/optimized/summary"
SIFT_FEATURES_TOPIC   = "/camera/sift_features"


_DEFAULTS: dict = {
    "sift_features":    1000,
    "match_ratio":      0.75,
    "min_matches":      8,
    "lm_iterations":    200,
    "pixel_noise_px":   1.5,
    "max_tracks":       1500,
    "pose_noise_m":     0.1,
    "pose_noise_rad":   0.01,
    "reproj_filter_px": 5.0,
    "max_landmark_dist": 0.0,
    "min_parallax_deg": 1.0,
    "huber_loss":       False,
    "ransac_threshold":  2.0,
    "min_track_length":  3,
    "match_window":      3,
    "ignore_poses":      False,
}

# Preset parameter overrides. Keys match the _DEFAULTS keys above.
# ignore_poses=True makes the pipeline skip /tf even if present in the bag.
PRESETS: dict[str, dict] = {
    "fine": {
        # Tight priors, strict reprojection filter — use when initial poses are accurate (< 5 cm).
        "pose_noise_m":     0.05,
        "pose_noise_rad":   0.005,
        "reproj_filter_px": 4.0,
        "pixel_noise_px":   1.0,
        "min_parallax_deg": 2.0,
        "huber_loss":       False,
    },
    "medium": {
        # Moderate priors — use when initial poses have ~10-20 cm of noise.
        # Lenient reprojection filter so noisy triangulations aren't all discarded.
        "pose_noise_m":     0.2,
        "pose_noise_rad":   0.05,
        "reproj_filter_px": 20.0,
        "pixel_noise_px":   1.5,
        "min_parallax_deg": 1.0,
        "huber_loss":       True,
        "max_tracks":       2000,
    },
    "rough": {
        # Very loose priors — use when initial poses are very rough (> 50 cm).
        # No reprojection filter; let the robust loss handle outliers.
        "pose_noise_m":     1.0,
        "pose_noise_rad":   0.2,
        "reproj_filter_px": 0.0,
        "pixel_noise_px":   2.0,
        "min_parallax_deg": 1.0,
        "huber_loss":       True,
        "max_tracks":       2000,
        "ransac_threshold": 3.0,
    },
    "no_pose": {
        # Initialise from essential-matrix chaining; ignore any /tf in the bag.
        "ignore_poses":     True,
        "reproj_filter_px": 0.0,
        "pixel_noise_px":   1.5,
        "min_parallax_deg": 1.0,
        "huber_loss":       True,
        "max_tracks":       2000,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate camera poses from images in an MCAP file."
    )
    parser.add_argument("input",  help="Input MCAP path")
    parser.add_argument("output", help="Output MCAP path")
    parser.add_argument("--preset", choices=list(PRESETS), default=None,
                        help="Apply a named configuration preset. Individual flags override "
                             "preset values.")
    parser.add_argument("--ignore-poses", action="store_true", default=None,
                        help="Ignore /tf poses in the input; initialise from essential matrix.")
    parser.add_argument("--sift-features",  type=int,   default=None)
    parser.add_argument("--match-ratio",    type=float, default=None)
    parser.add_argument("--min-matches",    type=int,   default=None)
    parser.add_argument("--lm-iterations",  type=int,   default=None)
    parser.add_argument("--pixel-noise-px", type=float, default=None)
    parser.add_argument("--max-tracks",     type=int,   default=None)
    parser.add_argument("--pose-noise-m",     type=float, default=None)
    parser.add_argument("--pose-noise-rad",   type=float, default=None)
    parser.add_argument("--reproj-filter-px", type=float, default=None,
                        help="Discard tracks whose initial reprojection error exceeds this "
                             "in any camera (only applied when /tf priors present). 0=disabled.")
    parser.add_argument("--max-landmark-dist", type=float, default=None,
                        help="Discard triangulated landmarks farther than this (metres) from "
                             "every observing camera. Prevents near-parallel-ray singularities. "
                             "0=disabled.")
    parser.add_argument("--min-parallax-deg", type=float, default=None,
                        help="Discard landmarks where the max viewing angle across all camera "
                             "pairs is below this (degrees). Prevents singular Jacobians from "
                             "low-baseline triangulations. 0=disabled.")
    parser.add_argument("--huber-loss", action="store_true", default=None,
                        help="Use a Huber robust noise model for projection factors. "
                             "Downweights observations with reprojection error > pixel-noise-px.")
    parser.add_argument("--ransac-threshold", type=float, default=None,
                        help="RANSAC inlier threshold in pixels for F-matrix geometric "
                             "filtering (USAC_MAGSAC). 0=disabled.")
    parser.add_argument("--min-track-length", type=int, default=None,
                        help="Discard tracks seen in fewer than this many frames before "
                             "triangulation. Default 3. Set to 2 to keep all pair-wise tracks.")
    parser.add_argument("--match-window", type=int, default=None,
                        help="Match each frame against this many following frames (default 3). "
                             "1 = sequential only.")
    args = parser.parse_args()

    # Apply preset first, then explicit CLI flags override, then fall back to _DEFAULTS.
    effective: dict = dict(_DEFAULTS)
    if args.preset:
        effective.update(PRESETS[args.preset])
    cli_overrides = {
        "sift_features":    args.sift_features,
        "match_ratio":      args.match_ratio,
        "min_matches":      args.min_matches,
        "lm_iterations":    args.lm_iterations,
        "pixel_noise_px":   args.pixel_noise_px,
        "max_tracks":       args.max_tracks,
        "pose_noise_m":     args.pose_noise_m,
        "pose_noise_rad":   args.pose_noise_rad,
        "reproj_filter_px": args.reproj_filter_px,
        "max_landmark_dist": args.max_landmark_dist,
        "min_parallax_deg": args.min_parallax_deg,
        "huber_loss":       args.huber_loss if args.huber_loss else None,
        "ransac_threshold":  args.ransac_threshold,
        "min_track_length":  args.min_track_length,
        "match_window":      args.match_window,
        "ignore_poses":      True if args.ignore_poses else None,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            effective[k] = v

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    if args.preset:
        print(f"Preset: {args.preset}")
    print()

    topics = get_topic_map(args.input)
    print("Topics in input MCAP:")
    for t, s in sorted(topics.items()):
        print(f"  {t:<40} {s}")
    print()

    raw_images: list[tuple[int, CompressedImage]] = []
    calibration_msg: CameraCalibration | None = None
    tf_msgs: list[tuple[int, FrameTransform]] = []
    tf_gt_msgs: list[tuple[int, FrameTransform]] = []

    read_topics = [CAMERA_IMAGE_TOPIC, CAMERA_CAL_TOPIC]
    if TF_TOPIC in topics and not effective["ignore_poses"]:
        read_topics.append(TF_TOPIC)
    if TF_GT_TOPIC in topics:
        read_topics.append(TF_GT_TOPIC)

    for topic, t_ns, msg in iter_messages(args.input, topics=read_topics):
        if topic == CAMERA_IMAGE_TOPIC:
            raw_images.append((t_ns, msg))
        elif topic == CAMERA_CAL_TOPIC:
            calibration_msg = msg
        elif topic == TF_TOPIC:
            tf_msgs.append((t_ns, msg))
        elif topic == TF_GT_TOPIC:
            tf_gt_msgs.append((t_ns, msg))

    if len(raw_images) < 2:
        print(f"Error: need ≥2 images on {CAMERA_IMAGE_TOPIC}, found {len(raw_images)}",
              file=sys.stderr)
        sys.exit(1)
    if calibration_msg is None:
        print(f"Error: no calibration on {CAMERA_CAL_TOPIC}", file=sys.stderr)
        sys.exit(1)

    cal = calibration_from_mcap_msg(calibration_msg)
    cal_type = "Cal3Fisheye" if isinstance(cal, gtsam.Cal3Fisheye) else "Cal3DS2"
    print(f"Calibration: {cal_type}")
    print(f"  fx={cal.fx():.1f}  fy={cal.fy():.1f}  cx={cal.px():.1f}  cy={cal.py():.1f}")
    print(f"  {len(raw_images)} images\n")

    opts = SfmOptions(
        sift_features=effective["sift_features"],
        match_ratio=effective["match_ratio"],
        min_matches=effective["min_matches"],
        lm_iterations=effective["lm_iterations"],
        pose_noise_m=effective["pose_noise_m"],
        pose_noise_rad=effective["pose_noise_rad"],
        pixel_noise_px=effective["pixel_noise_px"],
        max_tracks=effective["max_tracks"],
        max_reproj_error_px=effective["reproj_filter_px"],
        max_landmark_dist_m=effective["max_landmark_dist"],
        min_parallax_deg=effective["min_parallax_deg"],
        huber_loss=effective["huber_loss"],
        ransac_threshold=effective["ransac_threshold"],
        min_track_length=effective["min_track_length"],
        match_window=effective["match_window"],
    )

    images = [(t_ns, bytes(msg.data)) for t_ns, msg in raw_images]

    initial_poses: dict[int, gtsam.Pose3] | None = None
    if tf_msgs:
        pose_by_ts = {t_ns: pose3_from_frame_transform(msg) for t_ns, msg in tf_msgs}
        matched = {t_ns: pose_by_ts[t_ns] for t_ns, _ in raw_images if t_ns in pose_by_ts}
        if len(matched) >= 2:
            initial_poses = matched
            print(f"Using {len(initial_poses)}/{len(raw_images)} initial poses from {TF_TOPIC}")
        else:
            print(f"Warning: only {len(matched)} /tf poses matched images; falling back to E-matrix init")

    # Load cached SIFT features if the topic exists in the input MCAP
    preloaded_features = None
    if SIFT_FEATURES_TOPIC in topics:
        print(f"Loading cached SIFT features from {SIFT_FEATURES_TOPIC}...", flush=True)
        preloaded_features = {}
        for _, t_ns, msg in iter_messages(args.input, topics=[SIFT_FEATURES_TOPIC]):
            preloaded_features[t_ns] = decode_sift_features(msg)
        print(f"  Loaded features for {len(preloaded_features)} images", flush=True)

    t_start = time.time()
    result = optimize_poses(images, cal, opts, initial_poses=initial_poses,
                            preloaded_features=preloaded_features)
    elapsed = time.time() - t_start

    opt_poses     = result["poses"]
    initial_poses = result["initial_poses"]
    keypoints     = result["keypoints"]
    descriptors   = result["descriptors"]
    triangulated  = result["triangulated"]
    print(f"\nSfM complete in {elapsed:.1f}s  ({result['n_tracks']} tracks)")

    # Build GT poses from /tf_gt if present
    gt_poses: dict[int, gtsam.Pose3] | None = None
    if tf_gt_msgs:
        gt_by_ts = {t_ns: pose3_from_frame_transform(msg) for t_ns, msg in tf_gt_msgs}
        matched_gt = {t_ns: gt_by_ts[t_ns] for t_ns, _ in raw_images if t_ns in gt_by_ts}
        if len(matched_gt) >= 2:
            gt_poses = matched_gt
            print(f"Ground truth poses loaded from {TF_GT_TOPIC}: {len(gt_poses)} cameras")

    def _print_ape(label: str, result: dict) -> None:
        s = result["stats"]
        print(f"\n{label} ({result['n_poses']} cameras, SE(3)-aligned):")
        print(f"  Translation (m):  mean={s['translation']['mean']:.4f}  "
              f"median={s['translation']['median']:.4f}  "
              f"max={s['translation']['max']:.4f}  "
              f"rmse={s['translation']['rmse']:.4f}")
        print(f"  Rotation (deg):   mean={s['rotation']['mean']:.4f}  "
              f"median={s['rotation']['median']:.4f}  "
              f"max={s['rotation']['max']:.4f}  "
              f"rmse={s['rotation']['rmse']:.4f}")

    ape_init_result: dict = {}
    ape_opt_result:  dict = {}

    if gt_poses:
        ape_init_result = compute_ape(gt_poses, initial_poses)
        ape_opt_result  = compute_ape(gt_poses, opt_poses)
        if ape_init_result["by_key"]:
            _print_ape("APE — initial (noisy) poses vs GT", ape_init_result)
        if ape_opt_result["by_key"]:
            _print_ape("APE — optimised poses vs GT", ape_opt_result)
    elif tf_msgs:
        # No GT available: report how much the optimiser moved the poses
        ape_opt_result = compute_ape(initial_poses, opt_poses)
        if ape_opt_result["by_key"]:
            _print_ape("Pose change (initial → optimised)", ape_opt_result)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with McapWriter(args.output) as writer:
        # Passthrough — calibrate step needs images and calibration
        for t_ns, img_msg in raw_images:
            writer.write(CAMERA_IMAGE_TOPIC, img_msg, t_ns)
        writer.write(CAMERA_CAL_TOPIC, calibration_msg, raw_images[0][0])

        # Optimised poses as /tf
        for t_ns, _ in raw_images:
            if t_ns in opt_poses:
                ft = frame_transform_from_pose3(opt_poses[t_ns], "map", "camera_link", t_ns)
                writer.write(TF_TOPIC, ft, t_ns)

        # SIFT overlay on each image
        for t_ns, img_msg in raw_images:
            if t_ns in keypoints:
                overlay_bytes = overlay_keypoints(bytes(img_msg.data), keypoints[t_ns])
                overlay_msg = CompressedImage()
                overlay_msg.timestamp.CopyFrom(ns_to_timestamp(t_ns))
                overlay_msg.frame_id = "camera_link"
                overlay_msg.format = "jpeg"
                overlay_msg.data = overlay_bytes
                writer.write(SIFT_TOPIC, overlay_msg, t_ns)

        # Sparse point cloud
        pts = [track.point3d for track in triangulated if track.point3d is not None]
        if pts:
            writer.write(POINTS_TOPIC,
                         make_point_cloud(np.array(pts, dtype=np.float32), raw_images[0][0]),
                         raw_images[0][0])

        # Camera frustums + paths
        write_camera_path(
            writer, initial_poses,
            topic=SCENE_INITIAL_TOPIC,
            r=0.5, g=0.7, b=1.0,
            cal=cal,
        )
        write_camera_path(
            writer, opt_poses,
            topic=SCENE_OPTIMIZED_TOPIC,
            r=0.2, g=1.0, b=0.3,
            cal=cal,
        )

        # Pass through GT poses so they're available for comparison in Foxglove
        for t_ns, ft in tf_gt_msgs:
            writer.write(TF_GT_TOPIC, ft, t_ns)

        def _write_ape(topic: str, summary_topic: str, result: dict) -> None:
            t0 = raw_images[0][0]
            for t_ns, (t_err, r_err) in result["by_key"].items():
                writer.write(topic, ApePoseMsg(translation_m=t_err, rotation_deg=r_err), t_ns)
            s = result["stats"]
            writer.write(summary_topic, ApeSummaryMsg(
                trans_mean_m=s["translation"]["mean"],
                trans_median_m=s["translation"]["median"],
                trans_max_m=s["translation"]["max"],
                trans_rmse_m=s["translation"]["rmse"],
                rot_mean_deg=s["rotation"]["mean"],
                rot_median_deg=s["rotation"]["median"],
                rot_max_deg=s["rotation"]["max"],
                rot_rmse_deg=s["rotation"]["rmse"],
                n_poses=result["n_poses"],
            ), t0)

        if ape_init_result.get("by_key"):
            _write_ape(APE_INIT_TOPIC, APE_INIT_SUMMARY, ape_init_result)
        if ape_opt_result.get("by_key"):
            _write_ape(APE_OPT_TOPIC, APE_OPT_SUMMARY, ape_opt_result)

        # SIFT features — write for downstream reuse (cached from input or freshly detected)
        for t_ns, _ in raw_images:
            if t_ns in keypoints and t_ns in descriptors:
                writer.write(SIFT_FEATURES_TOPIC,
                                  encode_sift_features(keypoints[t_ns], descriptors[t_ns]),
                                  t_ns)

    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"Wrote {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
