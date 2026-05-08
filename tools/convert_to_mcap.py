"""
Convert Caliterra EXIF drone images → input.mcap for Foxglove.

Topics written:
  /camera/image         foxglove.CompressedImage   JPEG at native resolution
  /camera/calibration   foxglove.CameraCalibration estimated from EXIF focal length
  /gps/fix              foxglove.LocationFix        lat/lon/alt from EXIF GPS

Timestamps come from the GPS UTC date+time embedded in each image.
"""

from datetime import datetime, timezone
from pathlib import Path

import piexif
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix

from autocal.io.mcap_writer import McapWriter, ns_to_timestamp
from autocal.optics.camera import fx_from_exif

DATA_DIR = Path(__file__).parent.parent / "data" / "caliterra"
OUTPUT = Path(__file__).parent.parent / "data" / "caliterra.mcap"

# Canon SX260 HS: 1/2.3-inch CCD sensor, 6.17mm × 4.55mm
SENSOR_WIDTH_MM = 6.17


# ---------------------------------------------------------------------------
# EXIF helpers
# ---------------------------------------------------------------------------

def _rational(v) -> float:
    return v[0] / v[1] if v[1] else 0.0


def _dms_to_deg(dms, ref: bytes) -> float:
    deg = sum(_rational(x) / (60 ** i) for i, x in enumerate(dms))
    return -deg if ref in (b"S", b"W") else deg


def _gps_to_unix_ns(date_stamp: bytes, time_stamp) -> int:
    """Parse EXIF GPS date + time into UTC Unix nanoseconds."""
    y, mo, d = date_stamp.decode().split(":")
    h = int(_rational(time_stamp[0]))
    m = int(_rational(time_stamp[1]))
    s_frac = _rational(time_stamp[2])
    s = int(s_frac)
    us = int((s_frac - s) * 1_000_000)
    dt = datetime(int(y), int(mo), int(d), h, m, s, us, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _build_calibration(exif: dict, t_ns: int, width: int, height: int) -> CameraCalibration:
    """Build a CameraCalibration proto from EXIF focal length."""
    focal_mm = None
    exif_img = exif.get("Exif", {})
    raw_focal = exif_img.get(piexif.ExifIFD.FocalLength)
    if raw_focal:
        focal_mm = _rational(raw_focal)

    if focal_mm and focal_mm > 0:
        fx = fx_from_exif(focal_mm, SENSOR_WIDTH_MM, width)
    else:
        fx = max(width, height) * 1.2  # rough fallback

    cx = width / 2.0
    cy = height / 2.0

    cc = CameraCalibration()
    cc.timestamp.CopyFrom(ns_to_timestamp(t_ns))
    cc.frame_id = "camera_link"
    cc.width = width
    cc.height = height
    cc.distortion_model = "plumb_bob"
    cc.D.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    cc.K.extend([fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0])
    cc.R.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    cc.P.extend([fx, 0.0, cx, 0.0, 0.0, fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0])
    return cc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Canon SX260 HS consumer GPS accuracy (1-sigma, metres)
_GPS_SIGMA_HORIZONTAL_M = 3.0
_GPS_SIGMA_VERTICAL_M   = 5.0


def main() -> None:
    jpgs = sorted(DATA_DIR.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No images in {DATA_DIR}. Run: pixi run download-data")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(jpgs)} images → {OUTPUT}")

    seen_timestamps: set[int] = set()

    with McapWriter(OUTPUT) as writer:
        cal_written = False

        for i, path in enumerate(jpgs):
            exif = piexif.load(str(path))
            gps = exif.get("GPS", {})

            # Timestamp from GPS UTC; deduplicate by advancing 1 ms if needed
            date_stamp = gps.get(piexif.GPSIFD.GPSDateStamp)
            time_stamp = gps.get(piexif.GPSIFD.GPSTimeStamp)
            if date_stamp and time_stamp:
                t_ns = _gps_to_unix_ns(date_stamp, time_stamp)
            else:
                t_ns = i * 1_000_000_000
            while t_ns in seen_timestamps:
                t_ns += 1_000_000  # advance 1 ms
            seen_timestamps.add(t_ns)

            lat = _dms_to_deg(
                gps[piexif.GPSIFD.GPSLatitude],
                gps[piexif.GPSIFD.GPSLatitudeRef],
            )
            lon = _dms_to_deg(
                gps[piexif.GPSIFD.GPSLongitude],
                gps[piexif.GPSIFD.GPSLongitudeRef],
            )
            alt = _rational(gps[piexif.GPSIFD.GPSAltitude])

            # Image dimensions from EXIF
            exif_img = exif.get("Exif", {})
            width = exif_img.get(piexif.ExifIFD.PixelXDimension, 640)
            height = exif_img.get(piexif.ExifIFD.PixelYDimension, 480)

            # Write calibration once (first image)
            if not cal_written:
                cc = _build_calibration(exif, t_ns, width, height)
                writer.write("/camera/calibration", cc, t_ns)
                cal_written = True

            # CompressedImage
            img_msg = CompressedImage()
            img_msg.timestamp.CopyFrom(ns_to_timestamp(t_ns))
            img_msg.frame_id = "camera_link"
            img_msg.format = "jpeg"
            img_msg.data = path.read_bytes()
            writer.write("/camera/image", img_msg, t_ns)

            # LocationFix — include diagonal position covariance so the
            # calibration engine knows how tightly to constrain these poses.
            # Row-major ENU covariance [σ_E², 0, 0, 0, σ_N², 0, 0, 0, σ_U²]
            sh2 = _GPS_SIGMA_HORIZONTAL_M ** 2
            sv2 = _GPS_SIGMA_VERTICAL_M ** 2
            fix_msg = LocationFix()
            fix_msg.timestamp.CopyFrom(ns_to_timestamp(t_ns))
            fix_msg.frame_id = "camera_link"
            fix_msg.latitude = lat
            fix_msg.longitude = lon
            fix_msg.altitude = alt
            fix_msg.position_covariance[:] = [sh2, 0, 0,  0, sh2, 0,  0, 0, sv2]
            fix_msg.position_covariance_type = LocationFix.DIAGONAL_KNOWN
            writer.write("/gps/fix", fix_msg, t_ns)

            print(f"\r  {i+1}/{len(jpgs)}  {path.name}", end="", flush=True)

    print(f"\nWrote {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
