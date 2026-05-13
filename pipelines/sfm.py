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
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud

from autocal.engine.features import decode_sift_features, encode_sift_features
from autocal.engine.sfm import SfmOptions, optimize_poses
from autocal.engine.visualization import write_camera_path
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
SIFT_TOPIC         = "/camera/sift_overlay"
POINTS_TOPIC       = "/points/sfm"
SCENE_INITIAL_TOPIC   = "/scene/cameras/initial"
SCENE_OPTIMIZED_TOPIC = "/scene/cameras/optimized"
APE_TOPIC             = "/ape"
APE_SUMMARY_TOPIC     = "/ape/summary"
SIFT_FEATURES_TOPIC   = "/camera/sift_features"


def _make_point_cloud(pts: list, t_ns: int) -> PointCloud:
    xyz = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
    pc = PointCloud()
    pc.timestamp.CopyFrom(ns_to_timestamp(t_ns))
    pc.frame_id = "map"
    pc.pose.orientation.w = 1.0
    pc.point_stride = 12
    for name, offset in [("x", 0), ("y", 4), ("z", 8)]:
        f = pc.fields.add()
        f.name = name
        f.offset = offset
        f.type = PackedElementField.FLOAT32
    pc.data = xyz.tobytes()
    return pc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate camera poses from images in an MCAP file."
    )
    parser.add_argument("input",  help="Input MCAP path")
    parser.add_argument("output", help="Output MCAP path")
    parser.add_argument("--sift-features",  type=int,   default=1000)
    parser.add_argument("--match-ratio",    type=float, default=0.75)
    parser.add_argument("--min-matches",    type=int,   default=8)
    parser.add_argument("--lm-iterations",  type=int,   default=200)
    parser.add_argument("--pixel-noise-px", type=float, default=1.5)
    parser.add_argument("--max-tracks",     type=int,   default=1500)
    parser.add_argument("--pose-noise-m",     type=float, default=0.1)
    parser.add_argument("--pose-noise-rad",   type=float, default=0.01)
    parser.add_argument("--reproj-filter-px", type=float, default=5.0,
                        help="Discard tracks whose initial reprojection error exceeds this "
                             "in any camera (only applied when /tf priors present). 0=disabled.")
    parser.add_argument("--max-landmark-dist", type=float, default=0.0,
                        help="Discard triangulated landmarks farther than this (metres) from "
                             "every observing camera. Prevents near-parallel-ray singularities. "
                             "0=disabled.")
    parser.add_argument("--min-parallax-deg", type=float, default=1.0,
                        help="Discard landmarks where the max viewing angle across all camera "
                             "pairs is below this (degrees). Prevents singular Jacobians from "
                             "low-baseline triangulations. 0=disabled.")
    parser.add_argument("--huber-loss", action="store_true",
                        help="Use a Huber robust noise model for projection factors. "
                             "Downweights observations with reprojection error > pixel-noise-px.")
    parser.add_argument("--ransac-threshold", type=float, default=2.0,
                        help="RANSAC inlier threshold in pixels for F-matrix geometric "
                             "filtering (USAC_MAGSAC). 0=disabled.")
    args = parser.parse_args()

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}\n")

    topics = get_topic_map(args.input)
    print("Topics in input MCAP:")
    for t, s in sorted(topics.items()):
        print(f"  {t:<40} {s}")
    print()

    raw_images: list[tuple[int, CompressedImage]] = []
    calibration_msg: CameraCalibration | None = None
    tf_msgs: list[tuple[int, FrameTransform]] = []

    read_topics = [CAMERA_IMAGE_TOPIC, CAMERA_CAL_TOPIC]
    if TF_TOPIC in topics:
        read_topics.append(TF_TOPIC)

    for topic, t_ns, msg in iter_messages(args.input, topics=read_topics):
        if topic == CAMERA_IMAGE_TOPIC:
            raw_images.append((t_ns, msg))
        elif topic == CAMERA_CAL_TOPIC:
            calibration_msg = msg
        elif topic == TF_TOPIC:
            tf_msgs.append((t_ns, msg))

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
        sift_features=args.sift_features,
        match_ratio=args.match_ratio,
        min_matches=args.min_matches,
        lm_iterations=args.lm_iterations,
        pose_noise_m=args.pose_noise_m,
        pose_noise_rad=args.pose_noise_rad,
        pixel_noise_px=args.pixel_noise_px,
        max_tracks=args.max_tracks,
        max_reproj_error_px=args.reproj_filter_px,
        max_landmark_dist_m=args.max_landmark_dist,
        min_parallax_deg=args.min_parallax_deg,
        huber_loss=args.huber_loss,
        ransac_threshold=args.ransac_threshold,
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

    # APE vs GT (only when /tf was present in the input MCAP)
    ape_by_ts: dict[int, tuple[float, float]] = {}   # t_ns → (translation_m, rotation_deg)
    if tf_msgs:
        ape_result = compute_ape(initial_poses, opt_poses)
        ape_by_ts = ape_result["by_key"]
        if ape_by_ts:
            s = ape_result["stats"]
            print(f"\nAPE vs GT ({ape_result['n_poses']} cameras, SE(3)-aligned):")
            print(f"  Translation (m):  mean={s['translation']['mean']:.4f}  "
                  f"median={s['translation']['median']:.4f}  "
                  f"max={s['translation']['max']:.4f}  "
                  f"rmse={s['translation']['rmse']:.4f}")
            print(f"  Rotation (deg):   mean={s['rotation']['mean']:.4f}  "
                  f"median={s['rotation']['median']:.4f}  "
                  f"max={s['rotation']['max']:.4f}  "
                  f"rmse={s['rotation']['rmse']:.4f}")

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
            writer.write(POINTS_TOPIC, _make_point_cloud(pts, raw_images[0][0]),
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

        # Per-frame APE (only present when GT poses were in the input MCAP)
        for t_ns, (t_err, r_err) in ape_by_ts.items():
            writer.write_json(APE_TOPIC,
                              ApePoseMsg(translation_m=t_err, rotation_deg=r_err), t_ns)

        # APE summary statistics — single message at the start timestamp
        if ape_by_ts:
            s = ape_result["stats"]
            writer.write_json(APE_SUMMARY_TOPIC, ApeSummaryMsg(
                trans_mean_m=s["translation"]["mean"],
                trans_median_m=s["translation"]["median"],
                trans_max_m=s["translation"]["max"],
                trans_rmse_m=s["translation"]["rmse"],
                rot_mean_deg=s["rotation"]["mean"],
                rot_median_deg=s["rotation"]["median"],
                rot_max_deg=s["rotation"]["max"],
                rot_rmse_deg=s["rotation"]["rmse"],
                n_poses=ape_result["n_poses"],
            ), raw_images[0][0])

        # SIFT features — write for downstream reuse (cached from input or freshly detected)
        for t_ns, _ in raw_images:
            if t_ns in keypoints and t_ns in descriptors:
                writer.write_json(SIFT_FEATURES_TOPIC,
                                  encode_sift_features(keypoints[t_ns], descriptors[t_ns]),
                                  t_ns)

    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"Wrote {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
