"""
Generic SfM CLI: estimate camera poses from images in an MCAP file.

Reads /camera/image and /camera/calibration, runs visual SfM (essential-matrix
chaining → GTSAM optimisation with calibration held fixed), and writes an output
MCAP containing the original images, calibration, estimated /tf poses, and a
sparse /points/sfm point cloud.

Usage:
    pixi run python tools/run_sfm.py input.mcap output.mcap [options]

Options:
    --sift-features N     Max SIFT features per image (default: 1000)
    --match-ratio R       Lowe ratio test (default: 0.75)
    --min-matches N       Min matches per pair (default: 8)
    --lm-iterations N     Max LM iterations (default: 100)
    --pixel-noise-px P    Reprojection pixel noise σ (default: 1.5)
    --max-tracks N        Max triangulated tracks in graph (default: 1500)
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
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud

from autocal.engine.sfm import SfmOptions, optimize_poses
from autocal.gtsam_bridge.conversions import (
    calibration_from_mcap_msg,
    frame_transform_from_pose3,
)
from autocal.io.mcap_reader import get_topic_map, iter_messages
from autocal.io.mcap_writer import McapWriter, ns_to_timestamp

CAMERA_IMAGE_TOPIC = "/camera/image"
CAMERA_CAL_TOPIC   = "/camera/calibration"
TF_TOPIC           = "/tf"
POINTS_TOPIC       = "/points/sfm"


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
    parser.add_argument("--lm-iterations",  type=int,   default=100)
    parser.add_argument("--pixel-noise-px", type=float, default=1.5)
    parser.add_argument("--max-tracks",     type=int,   default=1500)
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

    for topic, t_ns, msg in iter_messages(
        args.input, topics=[CAMERA_IMAGE_TOPIC, CAMERA_CAL_TOPIC]
    ):
        if topic == CAMERA_IMAGE_TOPIC:
            raw_images.append((t_ns, msg))
        elif topic == CAMERA_CAL_TOPIC:
            calibration_msg = msg

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
        pixel_noise_px=args.pixel_noise_px,
        max_tracks=args.max_tracks,
        max_reproj_error_px=0.0,
    )

    images = [(t_ns, bytes(msg.data)) for t_ns, msg in raw_images]

    t_start = time.time()
    result = optimize_poses(images, cal, opts, initial_poses=None)
    elapsed = time.time() - t_start

    opt_poses = result["poses"]
    triangulated = result["triangulated"]
    print(f"\nSfM complete in {elapsed:.1f}s  ({result['n_tracks']} tracks)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with McapWriter(args.output) as writer:
        # Pass through images and calibration so the calibrate step has everything
        for t_ns, img_msg in raw_images:
            writer.write(CAMERA_IMAGE_TOPIC, img_msg, t_ns)
        writer.write(CAMERA_CAL_TOPIC, calibration_msg, raw_images[0][0])

        # Estimated poses as /tf
        for t_ns, _ in raw_images:
            if t_ns in opt_poses:
                ft = frame_transform_from_pose3(opt_poses[t_ns], "map", "camera_link", t_ns)
                writer.write(TF_TOPIC, ft, t_ns)

        # Sparse point cloud
        pts = [track.point3d for track in triangulated if track.point3d is not None]
        if pts:
            pc = _make_point_cloud(pts, raw_images[0][0])
            writer.write(POINTS_TOPIC, pc, raw_images[0][0])

    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"Wrote {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
