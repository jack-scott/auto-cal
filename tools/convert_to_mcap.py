"""
Convert Caliterra EXIF drone images → input.mcap for Foxglove.

Topics written:
  /camera/image   foxglove.CompressedImage   JPEG at native 4000×3000
  /gps/fix        foxglove.LocationFix       lat/lon/alt from EXIF GPS

Timestamps come from the GPS UTC date+time embedded in each image.
"""

from datetime import datetime, timezone
from pathlib import Path

import foxglove
import piexif
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
from google.protobuf import descriptor_pb2
from google.protobuf.timestamp_pb2 import Timestamp

DATA_DIR = Path(__file__).parent.parent / "data" / "caliterra"
OUTPUT = Path(__file__).parent.parent / "data" / "input.mcap"


# ---------------------------------------------------------------------------
# Helpers
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


def _proto_schema(msg_class) -> foxglove.Schema:
    desc = msg_class.DESCRIPTOR
    fds = descriptor_pb2.FileDescriptorSet()
    seen: set = set()

    def _collect(fd):
        if fd.name in seen:
            return
        seen.add(fd.name)
        for dep in fd.dependencies:
            _collect(dep)
        fd.CopyToProto(fds.file.add())

    _collect(desc.file)
    return foxglove.Schema(
        name=desc.full_name,
        encoding="protobuf",
        data=fds.SerializeToString(),
    )


def _ns_to_ts(ns: int) -> Timestamp:
    ts = Timestamp()
    ts.seconds = ns // 1_000_000_000
    ts.nanos = ns % 1_000_000_000
    return ts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    jpgs = sorted(DATA_DIR.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No images in {DATA_DIR}. Run: pixi run download-data")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(jpgs)} images → {OUTPUT}")

    with foxglove.open_mcap(str(OUTPUT), allow_overwrite=True):
        img_ch = foxglove.Channel(
            "/camera/image",
            schema=_proto_schema(CompressedImage),
            message_encoding="protobuf",
        )
        gps_ch = foxglove.Channel(
            "/gps/fix",
            schema=_proto_schema(LocationFix),
            message_encoding="protobuf",
        )

        for i, path in enumerate(jpgs):
            exif = piexif.load(str(path))
            gps = exif.get("GPS", {})

            # Timestamp from GPS UTC
            date_stamp = gps.get(piexif.GPSIFD.GPSDateStamp)
            time_stamp = gps.get(piexif.GPSIFD.GPSTimeStamp)
            if date_stamp and time_stamp:
                t_ns = _gps_to_unix_ns(date_stamp, time_stamp)
            else:
                # Fallback: 1-second increments from epoch if no GPS time
                t_ns = i * 1_000_000_000

            lat = _dms_to_deg(
                gps[piexif.GPSIFD.GPSLatitude],
                gps[piexif.GPSIFD.GPSLatitudeRef],
            )
            lon = _dms_to_deg(
                gps[piexif.GPSIFD.GPSLongitude],
                gps[piexif.GPSIFD.GPSLongitudeRef],
            )
            alt = _rational(gps[piexif.GPSIFD.GPSAltitude])

            # CompressedImage
            img_msg = CompressedImage()
            img_msg.timestamp.CopyFrom(_ns_to_ts(t_ns))
            img_msg.frame_id = "camera_link"
            img_msg.format = "jpeg"
            img_msg.data = path.read_bytes()
            img_ch.log(img_msg.SerializeToString(), log_time=t_ns)

            # LocationFix
            fix_msg = LocationFix()
            fix_msg.timestamp.CopyFrom(_ns_to_ts(t_ns))
            fix_msg.frame_id = "camera_link"
            fix_msg.latitude = lat
            fix_msg.longitude = lon
            fix_msg.altitude = alt
            gps_ch.log(fix_msg.SerializeToString(), log_time=t_ns)

            print(f"\r  {i+1}/{len(jpgs)}  {path.name}", end="", flush=True)

    print(f"\nWrote {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
