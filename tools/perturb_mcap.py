"""
Apply perturbations to poses and calibration in an MCAP file.

Useful for testing pipelines from degraded starting points:
  - Remove /tf entirely to test SfM from scratch (E-matrix initialisation)
  - Add noise to calibration to test calibration refinement
  - Combine both to test the full SfM → calibrate pipeline end-to-end

Usage:
    python tools/perturb_mcap.py input.mcap output.mcap [options]

Options:
    --remove-poses          Drop all /tf messages
    --remove-calibration    Drop all /camera/calibration messages
    --pose-noise-m M        Add Gaussian noise to camera translation (σ = M metres)
    --pose-noise-rad R      Add Gaussian noise to camera rotation (σ = R radians)
    --cal-noise-frac F      Scale fx and fy by (1 + N(0,F)); shift cx/cy by N(0,F·f)
    --seed N                Random seed for reproducibility (default: 42)

All topics not mentioned above are passed through unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gtsam
import numpy as np

from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform

from autocal.gtsam_bridge.conversions import (
    calibration_from_mcap_msg,
    camera_calibration_from_cal3ds2,
    camera_calibration_from_cal3fisheye,
    frame_transform_from_pose3,
    pose3_from_frame_transform,
)
from autocal.io.mcap_reader import get_topic_map, iter_messages, load_sift_features
from autocal.io.mcap_writer import McapWriter

TF_TOPIC            = "/tf"
CAL_TOPIC           = "/camera/calibration"
SIFT_FEATURES_TOPIC = "/camera/sift_features"


def _perturb_pose(
    ft: FrameTransform,
    sigma_m: float,
    sigma_rad: float,
    rng: np.random.Generator,
) -> FrameTransform:
    pose = pose3_from_frame_transform(ft)
    t = pose.translation() + rng.normal(0, sigma_m, 3)

    axis = rng.normal(0, 1.0, 3)
    norm = np.linalg.norm(axis)
    if norm > 1e-10:
        axis = axis / norm
    angle = float(rng.normal(0, sigma_rad))
    delta_R = gtsam.Rot3.AxisAngle(gtsam.Point3(*axis), angle)
    R = pose.rotation().compose(delta_R)

    perturbed = gtsam.Pose3(R, gtsam.Point3(*t))
    t_ns = ft.timestamp.seconds * 1_000_000_000 + ft.timestamp.nanos
    return frame_transform_from_pose3(perturbed, ft.parent_frame_id, ft.child_frame_id, t_ns)


def _perturb_calibration(
    cc: CameraCalibration,
    noise_frac: float,
    rng: np.random.Generator,
) -> CameraCalibration:
    cal = calibration_from_mcap_msg(cc)
    fx = cal.fx() * (1.0 + float(rng.normal(0, noise_frac)))
    fy = cal.fy() * (1.0 + float(rng.normal(0, noise_frac)))
    cx = cal.px() + float(rng.normal(0, noise_frac * cal.fx()))
    cy = cal.py() + float(rng.normal(0, noise_frac * cal.fy()))
    t_ns = cc.timestamp.seconds * 1_000_000_000 + cc.timestamp.nanos

    if isinstance(cal, gtsam.Cal3Fisheye):
        k = cal.k()
        perturbed = gtsam.Cal3Fisheye(fx, fy, cal.skew(), cx, cy, *k)
        return camera_calibration_from_cal3fisheye(perturbed, cc.frame_id, t_ns, cc.width, cc.height)
    else:
        k = cal.k()
        perturbed = gtsam.Cal3DS2(fx, fy, cal.skew(), cx, cy, *k)
        return camera_calibration_from_cal3ds2(perturbed, cc.frame_id, t_ns, cc.width, cc.height)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perturb or remove poses/calibration in an MCAP file."
    )
    parser.add_argument("input",  help="Input MCAP path")
    parser.add_argument("output", help="Output MCAP path")
    parser.add_argument("--remove-poses",       action="store_true")
    parser.add_argument("--remove-calibration", action="store_true")
    parser.add_argument("--pose-noise-m",   type=float, default=0.0)
    parser.add_argument("--pose-noise-rad", type=float, default=0.0)
    parser.add_argument("--cal-noise-frac", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    topics = get_topic_map(args.input)
    print("Topics in input MCAP:")
    for t, s in sorted(topics.items()):
        print(f"  {t:<40} {s}")
    print()

    if args.remove_poses:
        print("Dropping /tf")
    elif args.pose_noise_m > 0 or args.pose_noise_rad > 0:
        print(f"Pose noise: σ_t={args.pose_noise_m}m  σ_R={args.pose_noise_rad}rad")

    if args.remove_calibration:
        print("Dropping /camera/calibration")
    elif args.cal_noise_frac > 0:
        print(f"Calibration noise: frac={args.cal_noise_frac}")

    n_passed = n_modified = n_dropped = 0

    # Separate JSON topics (sift_features) from protobuf topics so each can be
    # handled with the appropriate reader API (iter_decoded_messages can't decode JSON).
    proto_topics = [t for t in topics if t != SIFT_FEATURES_TOPIC]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with McapWriter(args.output) as writer:
        # --- protobuf topics (images, calibration, tf, ...) ---
        for topic, t_ns, msg in iter_messages(args.input, topics=proto_topics):
            if topic == TF_TOPIC:
                if args.remove_poses:
                    n_dropped += 1
                    continue
                if args.pose_noise_m > 0 or args.pose_noise_rad > 0:
                    msg = _perturb_pose(msg, args.pose_noise_m, args.pose_noise_rad, rng)
                    n_modified += 1
                else:
                    n_passed += 1
                writer.write(topic, msg, t_ns)

            elif topic == CAL_TOPIC:
                if args.remove_calibration:
                    n_dropped += 1
                    continue
                if args.cal_noise_frac > 0:
                    msg = _perturb_calibration(msg, args.cal_noise_frac, rng)
                    n_modified += 1
                else:
                    n_passed += 1
                writer.write(topic, msg, t_ns)

            else:
                writer.write(topic, msg, t_ns)
                n_passed += 1

        # --- JSON topics: pass sift features through unchanged ---
        if SIFT_FEATURES_TOPIC in topics:
            features = load_sift_features(args.input, topic=SIFT_FEATURES_TOPIC)
            for t_ns, (kps, descs) in features.items():
                writer.write_sift_features(SIFT_FEATURES_TOPIC, kps, descs, t_ns)
            n_passed += len(features)
            print(f"Passed through {len(features)} cached SIFT feature messages")

    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"\nMessages: {n_passed} passed  {n_modified} modified  {n_dropped} dropped")
    print(f"Wrote {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
