"""Tests for autocal.frames.coordinates."""

import math

import numpy as np
import pytest

from autocal.frames.coordinates import (
    ecef_to_enu,
    ecef_to_gps,
    enu_origin_ecef,
    gps_to_ecef,
    gps_to_enu,
)

# ---------------------------------------------------------------------------
# gps_to_ecef / ecef_to_gps round-trip
# ---------------------------------------------------------------------------

def test_gps_to_ecef_equator_prime_meridian():
    """At (0°lat, 0°lon, 0m) ECEF X = WGS-84 semi-major axis."""
    a = 6_378_137.0
    result = gps_to_ecef(0.0, 0.0, 0.0)
    assert abs(result[0] - a) < 1e-3
    assert abs(result[1]) < 1e-3
    assert abs(result[2]) < 1e-3


def test_gps_to_ecef_north_pole():
    """At (90°lat, 0°lon, 0m) ECEF Z = WGS-84 semi-minor axis."""
    b = 6_356_752.314_245  # WGS-84 b
    result = gps_to_ecef(90.0, 0.0, 0.0)
    assert abs(result[0]) < 1e-3
    assert abs(result[1]) < 1e-3
    assert abs(result[2] - b) < 1e-1  # pyproj is sub-mm accurate


def test_ecef_to_gps_round_trip():
    lat_in, lon_in, alt_in = 37.7749, -122.4194, 100.0
    xyz = gps_to_ecef(lat_in, lon_in, alt_in)
    lat_out, lon_out, alt_out = ecef_to_gps(xyz)
    assert abs(lat_out - lat_in) < 1e-8
    assert abs(lon_out - lon_in) < 1e-8
    assert abs(alt_out - alt_in) < 1e-4  # sub-mm


def test_gps_to_ecef_altitude_offset():
    """Adding 1000 m altitude increases |ECEF| by ~1000 m."""
    xyz0 = gps_to_ecef(45.0, 0.0, 0.0)
    xyz1 = gps_to_ecef(45.0, 0.0, 1000.0)
    delta = np.linalg.norm(xyz1) - np.linalg.norm(xyz0)
    assert abs(delta - 1000.0) < 0.1


# ---------------------------------------------------------------------------
# enu_origin_ecef — rotation matrix properties
# ---------------------------------------------------------------------------

def test_enu_rotation_orthogonal():
    _, R = enu_origin_ecef(37.7749, -122.4194, 100.0)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_enu_rotation_det_one():
    _, R = enu_origin_ecef(37.7749, -122.4194, 100.0)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_enu_rotation_up_column():
    """The Up column of R_ecef_enu at (lat=0, lon=0) should point along +X in ECEF."""
    _, R = enu_origin_ecef(0.0, 0.0, 0.0)
    up_col = R[:, 2]
    assert abs(up_col[0] - 1.0) < 1e-10  # X
    assert abs(up_col[1]) < 1e-10         # Y
    assert abs(up_col[2]) < 1e-10         # Z


# ---------------------------------------------------------------------------
# gps_to_enu / ecef_to_enu
# ---------------------------------------------------------------------------

def test_gps_to_enu_at_origin():
    lat0, lon0, alt0 = 37.7749, -122.4194, 100.0
    result = gps_to_enu(lat0, lon0, alt0, lat0, lon0, alt0)
    assert np.allclose(result, [0.0, 0.0, 0.0], atol=1e-6)


def test_gps_to_enu_altitude():
    lat0, lon0, alt0 = 37.7749, -122.4194, 0.0
    result = gps_to_enu(lat0, lon0, alt0 + 50.0, lat0, lon0, alt0)
    assert abs(result[0]) < 1e-4
    assert abs(result[1]) < 1e-4
    assert abs(result[2] - 50.0) < 1e-4


def test_gps_to_enu_east():
    """Moving east gives positive East component."""
    lat0, lon0, alt0 = 0.0, 0.0, 0.0
    result = gps_to_enu(lat0, lon0 + 0.001, alt0, lat0, lon0, alt0)
    assert result[0] > 0
    assert abs(result[1]) < 1.0
    assert abs(result[2]) < 1e-3


def test_gps_to_enu_north():
    """Moving north gives positive North component."""
    lat0, lon0, alt0 = 0.0, 0.0, 0.0
    result = gps_to_enu(lat0 + 0.001, lon0, alt0, lat0, lon0, alt0)
    assert result[1] > 0
    assert abs(result[0]) < 1.0
    assert abs(result[2]) < 1e-3


def test_gps_to_enu_known_distance():
    """1° latitude ≈ 111 320 m north at the equator."""
    lat0, lon0, alt0 = 0.0, 0.0, 0.0
    result = gps_to_enu(1.0, 0.0, 0.0, lat0, lon0, alt0)
    assert abs(result[1] - 110_574) < 500   # varies slightly with ellipsoid


def test_ecef_to_enu_at_origin():
    lat0, lon0, alt0 = 51.5074, -0.1278, 10.0
    xyz = gps_to_ecef(lat0, lon0, alt0)
    enu = ecef_to_enu(xyz, lat0, lon0, alt0)
    assert np.allclose(enu, [0.0, 0.0, 0.0], atol=1e-6)


def test_ecef_to_enu_matches_gps_to_enu():
    """ecef_to_enu and gps_to_enu should agree to sub-millimetre."""
    lat0, lon0, alt0 = 37.7749, -122.4194, 100.0
    lat,  lon,  alt  = 37.7849, -122.4094, 110.0
    enu_via_gps  = gps_to_enu(lat, lon, alt, lat0, lon0, alt0)
    enu_via_ecef = ecef_to_enu(gps_to_ecef(lat, lon, alt), lat0, lon0, alt0)
    assert np.allclose(enu_via_gps, enu_via_ecef, atol=1e-3)
