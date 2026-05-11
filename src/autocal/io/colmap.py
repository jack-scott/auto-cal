"""COLMAP text-format file readers (cameras.txt, images.txt).

Pose convention (COLMAP → GTSAM)
---------------------------------
COLMAP stores poses as (QW, QX, QY, QZ, TX, TY, TZ) where:
  - QW..QZ is the unit quaternion for R_cw (world → camera rotation)
  - TX, TY, TZ is the translational part of the world→camera transform:
      X_cam = R_cw @ X_world + T_colmap

GTSAM Pose3(R_cw, t) where t = camera position in world:
  t = -R_cw^T @ T_colmap  (= R_cw.inverse().rotate(-T_colmap))

Supported camera models
-----------------------
  PINHOLE  : fx fy cx cy               → Cal3DS2 with zero distortion
  OPENCV   : fx fy cx cy k1 k2 p1 p2   → Cal3DS2 with full distortion
  RADIAL   : f cx cy k1 k2             → Cal3DS2 with k1 k2 only (p=0)
"""

from __future__ import annotations

from pathlib import Path

import gtsam
import numpy as np


def parse_cameras(path: str | Path) -> tuple[gtsam.Cal3DS2, int, int]:
    """Parse COLMAP cameras.txt and return (Cal3DS2, width, height).

    Uses the first camera entry in the file.

    Args:
        path: Path to cameras.txt.

    Returns:
        Tuple of (Cal3DS2 calibration, image width, height).

    Raises:
        ValueError: Unsupported camera model or empty file.
    """
    data_lines = _read_data_lines(path)
    if not data_lines:
        raise ValueError(f"No cameras in {path}")

    parts = data_lines[0].split()
    model = parts[1]
    width = int(parts[2])
    height = int(parts[3])
    params = [float(x) for x in parts[4:]]

    if model == "PINHOLE":
        fx, fy, cx, cy = params
        cal = gtsam.Cal3DS2(fx, fy, 0.0, cx, cy, 0.0, 0.0, 0.0, 0.0)
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        cal = gtsam.Cal3DS2(fx, fy, 0.0, cx, cy, k1, k2, p1, p2)
    elif model == "RADIAL":
        # COLMAP RADIAL: f cx cy k1 k2  (single focal length)
        f, cx, cy, k1, k2 = params
        cal = gtsam.Cal3DS2(f, f, 0.0, cx, cy, k1, k2, 0.0, 0.0)
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model!r}")

    return cal, width, height


def parse_images(path: str | Path) -> dict[str, gtsam.Pose3]:
    """Parse COLMAP images.txt and return {image_name: Pose3}.

    images.txt alternates between a header line and an observations line
    for each image.  Only header lines are parsed.

    Args:
        path: Path to images.txt.

    Returns:
        Dict mapping image name (the NAME column, e.g.
        "dslr_images_undistorted/DSC_0634.JPG") to a GTSAM Pose3 with
        R_cw and camera position in world.
    """
    data_lines = _read_data_lines(path)
    poses: dict[str, gtsam.Pose3] = {}

    for i in range(0, len(data_lines), 2):
        parts = data_lines[i].split()
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        qw = float(parts[1])
        qx = float(parts[2])
        qy = float(parts[3])
        qz = float(parts[4])
        tx = float(parts[5])
        ty = float(parts[6])
        tz = float(parts[7])
        name = parts[9]

        R_cw = gtsam.Rot3.Quaternion(qw, qx, qy, qz)
        T_colmap = np.array([tx, ty, tz])
        cam_pos = -R_cw.matrix().T @ T_colmap
        poses[name] = gtsam.Pose3(R_cw, gtsam.Point3(*cam_pos))

    return poses


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_data_lines(path: str | Path) -> list[str]:
    """Read a COLMAP text file, skipping blank lines and comments."""
    lines = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    return lines
