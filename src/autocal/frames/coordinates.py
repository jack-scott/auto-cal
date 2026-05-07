"""
Coordinate frame conversions.

All angles are in degrees unless a parameter name ends in _rad.
All distances are in metres.

GPS ↔ ECEF uses pyproj (EPSG:4979 geodetic ↔ EPSG:4978 ECEF).
Transformer is created once at module level and is thread-safe for reads.

ENU ↔ ECEF uses the standard rotation matrix built from the map origin:
  R_ecef_enu columns are [East, North, Up] unit vectors in ECEF.
  v_ecef = R_ecef_enu @ v_enu
  v_enu  = R_ecef_enu.T @ (v_ecef - origin_ecef)

pyproj Transformer notes:
  always_xy=True   →  input/output order is always (lon, lat, alt) / (X, Y, Z)
                       regardless of the CRS axis order convention.
  EPSG:4979        →  WGS-84 geographic 3D (lat, lon, ellipsoidal height)
  EPSG:4978        →  WGS-84 geocentric (ECEF X, Y, Z)
"""

import math

import numpy as np
from pyproj import Transformer

# Module-level transformers — created once, reused (thread-safe for transforms).
_GEO_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
_ECEF_TO_GEO = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)


def gps_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert WGS-84 geodetic coordinates to ECEF (metres).

    Args:
        lat_deg: Latitude in degrees (north positive).
        lon_deg: Longitude in degrees (east positive).
        alt_m:   Ellipsoidal height in metres.

    Returns:
        shape-(3,) float64 array [X, Y, Z] in metres.
    """
    x, y, z = _GEO_TO_ECEF.transform(lon_deg, lat_deg, alt_m)
    return np.array([x, y, z])


def ecef_to_gps(xyz: np.ndarray) -> tuple[float, float, float]:
    """Convert ECEF (metres) to WGS-84 geodetic (lat_deg, lon_deg, alt_m).

    Args:
        xyz: shape-(3,) float64 array [X, Y, Z] in metres.

    Returns:
        (lat_deg, lon_deg, alt_m)
    """
    lon, lat, alt = _ECEF_TO_GEO.transform(xyz[0], xyz[1], xyz[2])
    return lat, lon, alt


def _R_ecef_enu(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """3×3 rotation matrix whose columns are [East, North, Up] in ECEF."""
    lat = math.radians(lat0_deg)
    lon = math.radians(lon0_deg)
    return np.array([
        [-math.sin(lon), -math.sin(lat) * math.cos(lon), math.cos(lat) * math.cos(lon)],
        [ math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat) * math.sin(lon)],
        [           0.0,               math.cos(lat),                 math.sin(lat)    ],
    ])


def enu_origin_ecef(
    lat0_deg: float, lon0_deg: float, alt0_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz_ecef, R_ecef_enu) for the ENU map origin.

    R_ecef_enu is 3×3 rotation; columns are [East, North, Up] unit vectors
    expressed in ECEF.  Transforms ENU vectors to ECEF:
        v_ecef = R_ecef_enu @ v_enu
    """
    return gps_to_ecef(lat0_deg, lon0_deg, alt0_m), _R_ecef_enu(lat0_deg, lon0_deg)


def ecef_to_enu(
    xyz: np.ndarray, lat0_deg: float, lon0_deg: float, alt0_m: float
) -> np.ndarray:
    """Convert ECEF xyz to local ENU relative to origin (lat0, lon0, alt0).

    Full WGS-84 accuracy — no flat-earth approximation.
    """
    origin = gps_to_ecef(lat0_deg, lon0_deg, alt0_m)
    R = _R_ecef_enu(lat0_deg, lon0_deg)
    return R.T @ (xyz - origin)


def gps_to_enu(
    lat_deg: float, lon_deg: float, alt_m: float,
    lat0_deg: float, lon0_deg: float, alt0_m: float,
) -> np.ndarray:
    """Convert GPS to local ENU relative to origin.

    Uses full WGS-84 via ECEF.  Accurate at any distance from the origin.

    Args:
        lat_deg, lon_deg, alt_m:    Point to convert.
        lat0_deg, lon0_deg, alt0_m: ENU origin (map frame origin).

    Returns:
        shape-(3,) [east, north, up] in metres.
    """
    xyz = gps_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_to_enu(xyz, lat0_deg, lon0_deg, alt0_m)
