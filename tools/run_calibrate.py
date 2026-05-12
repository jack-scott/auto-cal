"""
Generic calibration refinement CLI: refine camera intrinsics given poses.

Reads /camera/image, /camera/calibration, and /tf from an input MCAP (typically
the output of run_sfm.py), then optimises the calibration with poses held as
soft priors.  Writes a refined /camera/calibration to the output MCAP.

Usage:
    pixi run python tools/run_calibrate.py input.mcap output.mcap [options]

Options:
    --sift-features N       Max SIFT features per image (default: 1000)
    --match-ratio R         Lowe ratio test (default: 0.75)
    --min-matches N         Min matches per pair (default: 8)
    --lm-iterations N       Max LM iterations (default: 200)
    --pose-noise-m M        Pose prior translation σ in scene units (default: 0.01)
    --pose-noise-rad R      Pose prior rotation σ in radians (default: 0.005)
    --cal-noise-frac F      Fractional σ on initial focal length (default: 0.15)
    --pixel-noise-px P      Reprojection pixel noise σ (default: 1.5)
    --max-tracks N          Max triangulated tracks (default: 1500)
    --reproj-filter-px X    Discard tracks with max reprojection error > X px (default: 20)
    --point-anchor-sigma S  3D point anchor σ in scene units (default: 0.005)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gtsam

from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform

from autocal.engine.calibration import CalibrationOptions, optimize_sfm
from autocal.gtsam_bridge.conversions import (
    calibration_from_mcap_msg,
    camera_calibration_from_cal3ds2,
    camera_calibration_from_cal3fisheye,
    pose3_from_frame_transform,
)
from autocal.io.mcap_reader import get_topic_map, iter_messages
from autocal.io.mcap_writer import McapWriter

CAMERA_IMAGE_TOPIC = "/camera/image"
CAMERA_CAL_TOPIC   = "/camera/calibration"
TF_TOPIC           = "/tf"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine camera calibration given poses in an MCAP file."
    )
    parser.add_argument("input",  help="Input MCAP path (images + calibration + /tf poses)")
    parser.add_argument("output", help="Output MCAP path")
    parser.add_argument("--sift-features",      type=int,   default=1000)
    parser.add_argument("--match-ratio",         type=float, default=0.75)
    parser.add_argument("--min-matches",         type=int,   default=8)
    parser.add_argument("--lm-iterations",       type=int,   default=200)
    parser.add_argument("--pose-noise-m",        type=float, default=0.01)
    parser.add_argument("--pose-noise-rad",      type=float, default=0.005)
    parser.add_argument("--cal-noise-frac",      type=float, default=0.15)
    parser.add_argument("--pixel-noise-px",      type=float, default=1.5)
    parser.add_argument("--max-tracks",          type=int,   default=1500)
    parser.add_argument("--reproj-filter-px",    type=float, default=20.0)
    parser.add_argument("--point-anchor-sigma",  type=float, default=0.005)
    args = parser.parse_args()

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}\n")

    topics = get_topic_map(args.input)
    print("Topics in input MCAP:")
    for t, s in sorted(topics.items()):
        print(f"  {t:<40} {s}")
    print()

    if TF_TOPIC not in topics:
        print(
            f"Error: no {TF_TOPIC} in input MCAP.\n"
            "Run run_sfm.py first to estimate poses.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load everything
    raw_images: list[tuple[int, CompressedImage]] = []
    calibration_msg: CameraCalibration | None = None
    tf_msgs: list[tuple[int, FrameTransform]] = []

    for topic, t_ns, msg in iter_messages(
        args.input, topics=[CAMERA_IMAGE_TOPIC, CAMERA_CAL_TOPIC, TF_TOPIC]
    ):
        if topic == CAMERA_IMAGE_TOPIC:
            raw_images.append((t_ns, msg))
        elif topic == CAMERA_CAL_TOPIC:
            calibration_msg = msg
        elif topic == TF_TOPIC:
            tf_msgs.append((t_ns, msg))

    if len(raw_images) < 2:
        print(f"Error: need ≥2 images, found {len(raw_images)}", file=sys.stderr)
        sys.exit(1)
    if calibration_msg is None:
        print(f"Error: no calibration on {CAMERA_CAL_TOPIC}", file=sys.stderr)
        sys.exit(1)
    if not tf_msgs:
        print(f"Error: no pose messages on {TF_TOPIC}", file=sys.stderr)
        sys.exit(1)

    cal = calibration_from_mcap_msg(calibration_msg)
    cal_type = "Cal3Fisheye" if isinstance(cal, gtsam.Cal3Fisheye) else "Cal3DS2"
    print(f"Calibration: {cal_type}")
    print(f"  fx={cal.fx():.1f}  fy={cal.fy():.1f}  cx={cal.px():.1f}  cy={cal.py():.1f}")
    print(f"  {len(raw_images)} images, {len(tf_msgs)} pose messages\n")

    # Match poses to images by timestamp (run_sfm.py writes /tf at image timestamps)
    pose_by_ts: dict[int, gtsam.Pose3] = {
        t_ns: pose3_from_frame_transform(msg) for t_ns, msg in tf_msgs
    }
    initial_poses: dict[int, gtsam.Pose3] = {}
    for t_ns, _ in raw_images:
        if t_ns in pose_by_ts:
            initial_poses[t_ns] = pose_by_ts[t_ns]

    if len(initial_poses) < 2:
        print(
            f"Error: only {len(initial_poses)} image timestamps matched a /tf pose.\n"
            "Ensure the input MCAP was produced by run_sfm.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Matched {len(initial_poses)}/{len(raw_images)} poses to images.")

    _fisheye = isinstance(cal, gtsam.Cal3Fisheye)
    opts = CalibrationOptions(
        sift_features=args.sift_features,
        match_ratio=args.match_ratio,
        min_matches=args.min_matches,
        lm_iterations=args.lm_iterations,
        pose_noise_m=args.pose_noise_m,
        pose_noise_rad=args.pose_noise_rad,
        cal_noise_frac=args.cal_noise_frac,
        cal_cx_noise_frac=0.005,
        k1_sigma=0.1  if _fisheye else 0.05,
        k2_sigma=0.1  if _fisheye else 0.02,
        k3_sigma=0.05 if _fisheye else 0.005,
        k4_sigma=0.1  if _fisheye else 0.005,
        p1_sigma=0.005,
        p2_sigma=0.005,
        pixel_noise_px=args.pixel_noise_px,
        max_tracks=args.max_tracks,
        reproj_filter_px=args.reproj_filter_px,
        point_anchor_sigma=args.point_anchor_sigma,
    )

    images = [(t_ns, bytes(msg.data)) for t_ns, msg in raw_images]

    t_start = time.time()
    result = optimize_sfm(
        images, initial_poses, cal, opts,
        triangulation_cal=cal,
    )
    elapsed = time.time() - t_start

    opt_cal = result["calibration"]
    k = opt_cal.k()
    print(f"\nCalibration complete in {elapsed:.1f}s  ({result['n_tracks']} tracks)")
    print(f"  fx  = {opt_cal.fx():.2f} px  (was {cal.fx():.2f})")
    print(f"  fy  = {opt_cal.fy():.2f} px  (was {cal.fy():.2f})")
    print(f"  cx  = {opt_cal.px():.2f} px")
    print(f"  cy  = {opt_cal.py():.2f} px")
    print(f"  k1  = {k[0]:.6f}")
    print(f"  k2  = {k[1]:.6f}")

    # Write output
    t_ns_out = raw_images[0][0]
    w = calibration_msg.width
    h = calibration_msg.height
    frame_id = calibration_msg.frame_id or "camera_link"

    if _fisheye:
        out_cal_msg = camera_calibration_from_cal3fisheye(opt_cal, frame_id, t_ns_out, w, h)
    else:
        out_cal_msg = camera_calibration_from_cal3ds2(opt_cal, frame_id, t_ns_out, w, h)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with McapWriter(args.output) as writer:
        writer.write(CAMERA_CAL_TOPIC, out_cal_msg, t_ns_out)

    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"\nWrote {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
