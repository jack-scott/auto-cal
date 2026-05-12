"""
Convert the ETH3D raw DSLR pipes dataset to MCAP format.

Reads from the directory extracted by `pixi run pipes-download-raw`:
  data/eth3d_pipes/pipes/images/dslr_images/       JPEG images
  data/eth3d_pipes/pipes/dslr_calibration_jpg/     COLMAP THIN_PRISM_FISHEYE calibration

Writes:
  data/eth3d_pipes_raw.mcap   images + equidistant (Kannala-Brandt) calibration + COLMAP GT poses

Run:
    pixi run pipe-raw-prepare
"""

from __future__ import annotations

from pathlib import Path

from autocal.gtsam_bridge.conversions import camera_calibration_from_cal3fisheye, frame_transform_from_pose3
from autocal.io.colmap import parse_cameras, parse_images
from autocal.io.mcap_writer import McapWriter, ns_to_timestamp
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage

DATA_ROOT = Path(__file__).parent.parent / "data" / "eth3d_pipes" / "pipes"
IMAGE_DIR = DATA_ROOT / "images" / "dslr_images"
CAL_DIR   = DATA_ROOT / "dslr_calibration_jpg"
OUTPUT    = Path(__file__).parent.parent / "data" / "eth3d_pipes_raw.mcap"


def main() -> None:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}\n"
            "Run: pixi run pipes-download-raw"
        )

    cal, width, height = parse_cameras(CAL_DIR / "cameras.txt")
    poses_named = parse_images(CAL_DIR / "images.txt")

    image_files = sorted(IMAGE_DIR.glob("*.JPG"))
    if not image_files:
        raise FileNotFoundError(f"No .JPG images in {IMAGE_DIR}")

    print(f"Images:      {len(image_files)}")
    print(f"Calibration: {cal}")
    print(f"Output:      {OUTPUT}")

    # Map short filename → full name key used in poses_named
    short_to_full = {Path(n).name: n for n in poses_named.keys()}

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with McapWriter(OUTPUT) as writer:
        t0 = 0
        cal_msg = camera_calibration_from_cal3fisheye(cal, "camera_link", t0, width, height)
        writer.write("/camera/calibration", cal_msg, t0)

        for i, img_path in enumerate(image_files):
            t_ns = i * 1_000_000_000  # 1 s apart

            img_msg = CompressedImage()
            img_msg.timestamp.CopyFrom(ns_to_timestamp(t_ns))
            img_msg.frame_id = "camera_link"
            img_msg.format = "jpeg"
            img_msg.data = img_path.read_bytes()
            writer.write("/camera/image", img_msg, t_ns)

            # Write COLMAP GT pose as /tf
            full_name = short_to_full.get(img_path.name)
            if full_name is not None:
                pose = poses_named.get(full_name)
                if pose is not None:
                    ft = frame_transform_from_pose3(pose, "map", "camera_link", t_ns)
                    writer.write("/tf", ft, t_ns)

            print(f"\r  {i+1}/{len(image_files)}  {img_path.name}", end="", flush=True)

    print(f"\nWrote {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
