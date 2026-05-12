"""
Download and convert the ETH3D "pipes" dataset to MCAP format.

Source:  https://www.eth3d.net/data/pipes_dslr_undistorted.7z
Content: 14 DSLR images of an indoor industrial pipe scene + COLMAP intrinsics.

Topics written:
  /camera/image         foxglove.CompressedImage
  /camera/calibration   foxglove.CameraCalibration  (from COLMAP cameras.txt)
  /tf                   foxglove.FrameTransform      (COLMAP GT poses)

The ETH3D dataset does not include GPS; cameras are expressed in an arbitrary
COLMAP world frame (not ENU).  Timestamps are synthetic (1s apart).

Run:
    pixi run python tools/download_eth3d_pipes.py
"""

import urllib.request
from pathlib import Path

import py7zr

import numpy as np
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage

from autocal.gtsam_bridge.conversions import frame_transform_from_pose3
from autocal.io.colmap import parse_images
from autocal.io.mcap_writer import McapWriter, ns_to_timestamp

URL = "https://www.eth3d.net/data/pipes_dslr_undistorted.7z"
DATA_DIR = Path(__file__).parent.parent / "data"
ARCHIVE = DATA_DIR / "pipes_dslr_undistorted.7z"
EXTRACT_DIR = DATA_DIR / "eth3d_pipes"
OUTPUT = DATA_DIR / "eth3d_pipes.mcap"


def download() -> None:
    if ARCHIVE.exists():
        print(f"Archive already exists: {ARCHIVE}")
        return
    print(f"Downloading {URL} ...")
    last_pct = [-1]

    def _progress(count, block_size, total):
        pct = min(100, int(count * block_size * 100 / total))
        if pct != last_pct[0] and pct % 10 == 0:
            print(f"  {pct}%", flush=True)
            last_pct[0] = pct

    urllib.request.urlretrieve(URL, ARCHIVE, reporthook=_progress)
    print(f"Saved to {ARCHIVE}  ({ARCHIVE.stat().st_size / 1e6:.1f} MB)")


def extract() -> None:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("*.JPG")):
        print(f"Already extracted: {EXTRACT_DIR}")
        return
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {EXTRACT_DIR} ...")
    with py7zr.SevenZipFile(ARCHIVE, mode="r") as z:
        z.extractall(path=EXTRACT_DIR)
    print("Done.")


def _parse_colmap_cameras(cameras_txt: Path) -> dict:
    """Return {camera_id: dict(model, width, height, params...)}."""
    cameras = {}
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = dict(model=model, width=width, height=height, params=params)
    return cameras


def _parse_colmap_images(images_txt: Path) -> list[dict]:
    """Return list of {image_id, camera_id, name} dicts."""
    images = []
    with open(images_txt) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    # Lines alternate: metadata / points2d
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        images.append(dict(
            image_id=int(parts[0]),
            camera_id=int(parts[8]),
            name=parts[9],
        ))
    images.sort(key=lambda x: x["name"])
    return images


def _make_calibration(cam: dict, t_ns: int) -> CameraCalibration:
    """Build CameraCalibration from a COLMAP PINHOLE camera."""
    model = cam["model"]
    w, h = cam["width"], cam["height"]

    if model in ("PINHOLE", "SIMPLE_PINHOLE"):
        if model == "SIMPLE_PINHOLE":
            f, cx, cy = cam["params"]
            fx = fy = f
        else:
            fx, fy, cx, cy = cam["params"]
        k1 = k2 = 0.0
    elif model in ("RADIAL", "SIMPLE_RADIAL"):
        if model == "SIMPLE_RADIAL":
            f, cx, cy, k1 = cam["params"]
            fy = f
            fx = f
            k2 = 0.0
        else:
            f, cx, cy, k1, k2 = cam["params"]
            fx = fy = f
    else:
        # Fall back: use first param as focal length
        fx = fy = cam["params"][0]
        cx, cy = w / 2.0, h / 2.0
        k1 = k2 = 0.0

    cc = CameraCalibration()
    cc.timestamp.CopyFrom(ns_to_timestamp(t_ns))
    cc.frame_id = "camera_link"
    cc.width = w
    cc.height = h
    cc.distortion_model = "plumb_bob"
    cc.D.extend([k1, k2, 0.0, 0.0, 0.0])
    cc.K.extend([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0])
    cc.R.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    cc.P.extend([fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0])
    return cc


def convert() -> None:
    # Find COLMAP sparse reconstruction directory
    colmap_dirs = sorted(EXTRACT_DIR.rglob("cameras.txt"))
    if not colmap_dirs:
        raise FileNotFoundError(f"No cameras.txt found under {EXTRACT_DIR}")
    sparse_dir = colmap_dirs[0].parent
    cameras_txt = sparse_dir / "cameras.txt"
    images_txt = sparse_dir / "images.txt"

    cameras = _parse_colmap_cameras(cameras_txt)
    image_records = _parse_colmap_images(images_txt)
    poses_named = parse_images(images_txt)

    # Find all image files
    img_root = EXTRACT_DIR
    print(f"Found {len(image_records)} image records in {images_txt}")
    print(f"Found {len(poses_named)} poses in {images_txt}")

    print(f"Writing {OUTPUT} ...")
    with McapWriter(OUTPUT) as writer:
        cal_written: set[int] = set()
        for i, rec in enumerate(image_records):
            t_ns = i * 1_000_000_000  # 1s apart

            # Find the image file
            candidates = list(img_root.rglob(rec["name"]))
            if not candidates:
                print(f"  WARNING: image not found: {rec['name']}")
                continue
            img_path = candidates[0]

            cam_id = rec["camera_id"]
            cam = cameras.get(cam_id, cameras.get(1, list(cameras.values())[0]))

            # Write calibration once per camera_id
            if cam_id not in cal_written:
                cc = _make_calibration(cam, t_ns)
                writer.write("/camera/calibration", cc, t_ns)
                cal_written.add(cam_id)

            # Write image
            img_msg = CompressedImage()
            img_msg.timestamp.CopyFrom(ns_to_timestamp(t_ns))
            img_msg.frame_id = "camera_link"
            img_msg.format = "jpeg"
            img_msg.data = img_path.read_bytes()
            writer.write("/camera/image", img_msg, t_ns)

            # Write COLMAP GT pose as /tf
            pose = poses_named.get(rec["name"])
            if pose is not None:
                ft = frame_transform_from_pose3(pose, "map", "camera_link", t_ns)
                writer.write("/tf", ft, t_ns)

            print(f"\r  {i+1}/{len(image_records)}  {rec['name']}", end="", flush=True)

    print(f"\nWrote {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download()
    extract()
    convert()


if __name__ == "__main__":
    main()
