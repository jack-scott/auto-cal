"""
Download and convert a dataset to MCAP format.

Usage:
    python tools/prepare_dataset.py <dataset> [--output PATH]

Available datasets:
    eth3d-pipes      ETH3D pipes, undistorted DSLR (PINHOLE calibration, no distortion)
    eth3d-pipes-raw  ETH3D pipes, raw DSLR JPEGs (THIN_PRISM_FISHEYE / equidistant)
    caliterra        OpenDroneMap Caliterra UAV survey (EXIF calibration + GPS)

All available data is written to the output MCAP:

    eth3d-pipes, eth3d-pipes-raw:
        /camera/image           foxglove.CompressedImage
        /camera/calibration     foxglove.CameraCalibration  (from COLMAP)
        /tf                     foxglove.FrameTransform     (COLMAP GT poses)
        /scene/cameras/gt       foxglove.SceneUpdate        (3D camera frustums + path)

    caliterra:
        /camera/image           foxglove.CompressedImage
        /camera/calibration     foxglove.CameraCalibration  (from EXIF focal length)
        /gps/fix                foxglove.LocationFix        (from EXIF GPS)
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import piexif
import py7zr
import gtsam

from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix

from autocal.engine.features import detect_sift, encode_sift_features
from autocal.engine.visualization import write_camera_path
from autocal.gtsam_bridge.conversions import (
    camera_calibration_from_cal3ds2,
    camera_calibration_from_cal3fisheye,
    frame_transform_from_pose3,
)
from autocal.io.colmap import parse_cameras, parse_images
from autocal.io.mcap_writer import McapWriter, ns_to_timestamp
from autocal.optics.camera import fx_from_exif

SIFT_FEATURES_TOPIC = "/camera/sift_features"
# Detect more features than the SfM default so cached features support higher
# quality runs without re-detection.  The SfM step truncates to --sift-features.
_PREPARE_SIFT_N = 2500

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Shared download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    print(f"Downloading {url} ...")
    last_pct = [-1]

    def _progress(count, block_size, total):
        if total <= 0:
            return
        pct = min(100, int(count * block_size * 100 / total))
        if pct != last_pct[0] and pct % 10 == 0:
            print(f"  {pct}%", flush=True)
            last_pct[0] = pct

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"Saved: {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")


def _extract_7z(archive: Path, dest: Path) -> None:
    print(f"Extracting {archive.name} ...")
    dest.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=dest)
    print("Done.")


# ---------------------------------------------------------------------------
# ETH3D pipes — undistorted PINHOLE
# ---------------------------------------------------------------------------

_ETH3D_UNDIST_URL     = "https://www.eth3d.net/data/pipes_dslr_undistorted.7z"
_ETH3D_UNDIST_ARCHIVE = DATA_DIR / "pipes_dslr_undistorted.7z"
_ETH3D_DIR            = DATA_DIR / "eth3d_pipes"


def prepare_eth3d_pipes(output: Path) -> None:
    _download(_ETH3D_UNDIST_URL, _ETH3D_UNDIST_ARCHIVE)

    if not any(_ETH3D_DIR.rglob("*.JPG")):
        _extract_7z(_ETH3D_UNDIST_ARCHIVE, _ETH3D_DIR)

    # Find COLMAP sparse reconstruction
    cameras_txts = sorted(_ETH3D_DIR.rglob("cameras.txt"))
    if not cameras_txts:
        raise FileNotFoundError(f"No cameras.txt found under {_ETH3D_DIR}")
    sparse_dir = cameras_txts[0].parent
    cameras_txt = sparse_dir / "cameras.txt"
    images_txt  = sparse_dir / "images.txt"

    cal, width, height = parse_cameras(cameras_txt)
    poses = parse_images(images_txt)

    names = sorted(poses.keys())
    print(f"Images: {len(names)}, calibration: {type(cal).__name__}")

    output.parent.mkdir(parents=True, exist_ok=True)
    poses_by_ts: dict[int, gtsam.Pose3] = {}
    with McapWriter(output) as writer:
        t0 = 0
        cal_msg = camera_calibration_from_cal3ds2(cal, "camera_link", t0, width, height)
        writer.write("/camera/calibration", cal_msg, t0)

        for i, name in enumerate(names):
            t_ns = i * 1_000_000_000

            candidates = list(_ETH3D_DIR.rglob(name))
            if not candidates:
                print(f"  WARNING: image not found: {name}")
                continue
            img_path = candidates[0]

            img_data = img_path.read_bytes()
            img_msg = CompressedImage()
            img_msg.timestamp.CopyFrom(ns_to_timestamp(t_ns))
            img_msg.frame_id = "camera_link"
            img_msg.format = "jpeg"
            img_msg.data = img_data
            writer.write("/camera/image", img_msg, t_ns)

            ft = frame_transform_from_pose3(poses[name], "map", "camera_link", t_ns)
            writer.write("/tf", ft, t_ns)
            poses_by_ts[t_ns] = poses[name]

            kps, descs = detect_sift(img_data, n_features=_PREPARE_SIFT_N)
            writer.write(SIFT_FEATURES_TOPIC, encode_sift_features(kps, descs), t_ns)

            print(f"\r  {i+1}/{len(names)}  {Path(name).name}  ({len(kps)} kps)",
                  end="", flush=True)

        write_camera_path(writer, poses_by_ts, topic="/scene/cameras/gt",
                          r=1.0, g=0.8, b=0.1, cal=cal)

    print(f"\nWrote {output}  ({output.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# ETH3D raw DSLR — shared helper (THIN_PRISM_FISHEYE / Kannala-Brandt)
# ---------------------------------------------------------------------------
# All raw-DSLR archives follow the same layout:
#   <scene>/images/dslr_images/*.JPG
#   <scene>/dslr_calibration_jpg/cameras.txt
#   <scene>/dslr_calibration_jpg/images.txt

def _prepare_eth3d_raw(url: str, archive: Path, extract_dir: Path,
                       scene_name: str, output: Path) -> None:
    """Download, extract, and convert one ETH3D raw-DSLR dataset to MCAP."""
    images_dir = extract_dir / scene_name / "images" / "dslr_images"
    cal_dir    = extract_dir / scene_name / "dslr_calibration_jpg"

    _download(url, archive)

    if not (images_dir.exists() and any(images_dir.glob("*.JPG"))):
        _extract_7z(archive, extract_dir)

    cameras_txt = cal_dir / "cameras.txt"
    images_txt  = cal_dir / "images.txt"
    if not cameras_txt.exists():
        raise FileNotFoundError(f"Calibration not found: {cameras_txt}")

    cal, width, height = parse_cameras(cameras_txt)
    poses = parse_images(images_txt)

    image_files = sorted(images_dir.glob("*.JPG"))
    if not image_files:
        raise FileNotFoundError(f"No .JPG images in {images_dir}")

    short_to_full = {Path(n).name: n for n in poses}

    n_workers = min(4, os.cpu_count() or 1)
    print(f"Images: {len(image_files)}, calibration: {type(cal).__name__}")
    print(f"Detecting SIFT features ({_PREPARE_SIFT_N} max) on {n_workers} threads...")

    def _load_and_detect(img_path: Path):
        data = img_path.read_bytes()
        kps, descs = detect_sift(data, n_features=_PREPARE_SIFT_N)
        return data, kps, descs

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        detections = list(pool.map(_load_and_detect, image_files))
    print(f"  Done — {sum(len(d[1]) for d in detections)} total keypoints")

    output.parent.mkdir(parents=True, exist_ok=True)
    poses_by_ts: dict[int, gtsam.Pose3] = {}
    with McapWriter(output) as writer:
        t0 = 0
        cal_msg = camera_calibration_from_cal3fisheye(cal, "camera_link", t0, width, height)
        writer.write("/camera/calibration", cal_msg, t0)

        for i, (img_path, (img_data, kps, descs)) in enumerate(zip(image_files, detections)):
            t_ns = i * 1_000_000_000

            img_msg = CompressedImage()
            img_msg.timestamp.CopyFrom(ns_to_timestamp(t_ns))
            img_msg.frame_id = "camera_link"
            img_msg.format = "jpeg"
            img_msg.data = img_data
            writer.write("/camera/image", img_msg, t_ns)

            full_name = short_to_full.get(img_path.name)
            if full_name is not None and full_name in poses:
                ft = frame_transform_from_pose3(poses[full_name], "map", "camera_link", t_ns)
                writer.write("/tf", ft, t_ns)
                poses_by_ts[t_ns] = poses[full_name]

            writer.write(SIFT_FEATURES_TOPIC, encode_sift_features(kps, descs), t_ns)

            print(f"\r  Writing {i+1}/{len(image_files)}  {img_path.name}",
                  end="", flush=True)

        write_camera_path(writer, poses_by_ts, topic="/scene/cameras/gt",
                          r=1.0, g=0.8, b=0.1, cal=cal)

    print(f"\nWrote {output}  ({output.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# ETH3D pipes — raw DSLR JPEGs
# ---------------------------------------------------------------------------

def prepare_eth3d_pipes_raw(output: Path) -> None:
    _prepare_eth3d_raw(
        url        = "https://www.eth3d.net/data/pipes_dslr_jpg.7z",
        archive    = DATA_DIR / "pipes_dslr_jpg.7z",
        extract_dir= DATA_DIR / "eth3d_pipes",
        scene_name = "pipes",
        output     = output,
    )


# ---------------------------------------------------------------------------
# ETH3D exhibition hall — raw DSLR JPEGs
# ---------------------------------------------------------------------------

def prepare_eth3d_exhibition_hall(output: Path) -> None:
    _prepare_eth3d_raw(
        url        = "https://www.eth3d.net/data/exhibition_hall_dslr_jpg.7z",
        archive    = DATA_DIR / "exhibition_hall_dslr_jpg.7z",
        extract_dir= DATA_DIR / "eth3d_exhibition_hall",
        scene_name = "exhibition_hall",
        output     = output,
    )


# ---------------------------------------------------------------------------
# Caliterra — Canon SX260 HS drone survey with GPS
# ---------------------------------------------------------------------------

_CALITERRA_URL      = "https://github.com/OpenDroneMap/odm_data_caliterra/archive/master.zip"
_CALITERRA_ZIP      = DATA_DIR / "caliterra_raw.zip"
_CALITERRA_IMAGES   = DATA_DIR / "caliterra"

# Canon SX260 HS: 1/2.3-inch CCD, 6.17mm × 4.55mm
_SENSOR_WIDTH_MM = 6.17
_GPS_SIGMA_H_M   = 3.0
_GPS_SIGMA_V_M   = 5.0


def _rational(v) -> float:
    return v[0] / v[1] if v[1] else 0.0


def _dms_to_deg(dms, ref: bytes) -> float:
    deg = sum(_rational(x) / (60 ** i) for i, x in enumerate(dms))
    return -deg if ref in (b"S", b"W") else deg


def _gps_to_unix_ns(date_stamp: bytes, time_stamp) -> int:
    y, mo, d = date_stamp.decode().split(":")
    h = int(_rational(time_stamp[0]))
    m = int(_rational(time_stamp[1]))
    s_frac = _rational(time_stamp[2])
    s = int(s_frac)
    us = int((s_frac - s) * 1_000_000)
    dt = datetime(int(y), int(mo), int(d), h, m, s, us, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def prepare_caliterra(output: Path) -> None:
    if not (_CALITERRA_IMAGES.exists() and any(_CALITERRA_IMAGES.glob("*.jpg"))):
        _download(_CALITERRA_URL, _CALITERRA_ZIP)
        print(f"Extracting images to {_CALITERRA_IMAGES} ...")
        _CALITERRA_IMAGES.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(_CALITERRA_ZIP) as zf:
            for name in zf.namelist():
                if name.lower().endswith((".jpg", ".jpeg")):
                    dest = _CALITERRA_IMAGES / Path(name).name
                    dest.write_bytes(zf.read(name))
        count = len(list(_CALITERRA_IMAGES.glob("*.jpg")))
        print(f"Extracted {count} images.")

    jpgs = sorted(_CALITERRA_IMAGES.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No images in {_CALITERRA_IMAGES}")

    print(f"Images: {len(jpgs)}")

    seen: set[int] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with McapWriter(output) as writer:
        cal_written = False

        for i, path in enumerate(jpgs):
            exif = piexif.load(str(path))
            gps  = exif.get("GPS", {})

            date_stamp = gps.get(piexif.GPSIFD.GPSDateStamp)
            time_stamp = gps.get(piexif.GPSIFD.GPSTimeStamp)
            if date_stamp and time_stamp:
                t_ns = _gps_to_unix_ns(date_stamp, time_stamp)
            else:
                t_ns = i * 1_000_000_000
            while t_ns in seen:
                t_ns += 1_000_000
            seen.add(t_ns)

            exif_img = exif.get("Exif", {})
            width  = exif_img.get(piexif.ExifIFD.PixelXDimension, 640)
            height = exif_img.get(piexif.ExifIFD.PixelYDimension, 480)

            if not cal_written:
                raw_focal = exif_img.get(piexif.ExifIFD.FocalLength)
                focal_mm = _rational(raw_focal) if raw_focal else None
                if focal_mm and focal_mm > 0:
                    fx = fx_from_exif(focal_mm, _SENSOR_WIDTH_MM, width)
                else:
                    fx = max(width, height) * 1.2
                cx, cy = width / 2.0, height / 2.0
                cal = gtsam.Cal3DS2(fx, fx, 0.0, cx, cy, 0.0, 0.0, 0.0, 0.0)
                cal_msg = camera_calibration_from_cal3ds2(cal, "camera_link", t_ns, width, height)
                writer.write("/camera/calibration", cal_msg, t_ns)
                cal_written = True

            img_msg = CompressedImage()
            img_msg.timestamp.CopyFrom(ns_to_timestamp(t_ns))
            img_msg.frame_id = "camera_link"
            img_msg.format = "jpeg"
            img_msg.data = path.read_bytes()
            writer.write("/camera/image", img_msg, t_ns)

            lat = _dms_to_deg(gps[piexif.GPSIFD.GPSLatitude], gps[piexif.GPSIFD.GPSLatitudeRef])
            lon = _dms_to_deg(gps[piexif.GPSIFD.GPSLongitude], gps[piexif.GPSIFD.GPSLongitudeRef])
            alt = _rational(gps[piexif.GPSIFD.GPSAltitude])

            sh2 = _GPS_SIGMA_H_M ** 2
            sv2 = _GPS_SIGMA_V_M ** 2
            fix = LocationFix()
            fix.timestamp.CopyFrom(ns_to_timestamp(t_ns))
            fix.frame_id = "camera_link"
            fix.latitude  = lat
            fix.longitude = lon
            fix.altitude  = alt
            fix.position_covariance[:] = [sh2, 0, 0, 0, sh2, 0, 0, 0, sv2]
            fix.position_covariance_type = LocationFix.DIAGONAL_KNOWN
            writer.write("/gps/fix", fix, t_ns)

            print(f"\r  {i+1}/{len(jpgs)}  {path.name}", end="", flush=True)

    print(f"\nWrote {output}  ({output.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASETS: dict[str, tuple] = {
    "eth3d-pipes":            (prepare_eth3d_pipes,            DATA_DIR / "eth3d_pipes.mcap"),
    "eth3d-pipes-raw":        (prepare_eth3d_pipes_raw,        DATA_DIR / "eth3d_pipes_raw.mcap"),
    "eth3d-exhibition-hall":  (prepare_eth3d_exhibition_hall,  DATA_DIR / "eth3d_exhibition_hall.mcap"),
    "caliterra":              (prepare_caliterra,              DATA_DIR / "caliterra.mcap"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and convert a dataset to MCAP format."
    )
    parser.add_argument("dataset", choices=list(DATASETS), help="Dataset to prepare")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output MCAP path (default: data/<dataset>.mcap)")
    parser.add_argument("--force", action="store_true",
                        help="Re-prepare even if the output MCAP already exists.")
    args = parser.parse_args()

    prepare_fn, default_output = DATASETS[args.dataset]
    output = args.output or default_output

    print(f"Dataset: {args.dataset}")
    print(f"Output:  {output}\n")

    if output.exists() and not args.force:
        print(f"Already prepared: {output}  (use --force to regenerate)")
        return

    prepare_fn(output)


if __name__ == "__main__":
    main()
