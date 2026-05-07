"""
Camera / optics utilities.

Projection model: pinhole with plumb_bob radial distortion (k1, k2).
All pixel coordinates are (col, row) = (x, y) with origin at top-left.
3D points in the camera frame have +Z pointing forward (along optical axis).
"""

import math

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

def fx_from_exif(fl_mm: float, sensor_w_mm: float, image_w_px: int) -> float:
    """Convert focal length from mm to pixels using sensor width metadata.

    Args:
        fl_mm:       Focal length in millimetres (EXIF FocalLength tag).
        sensor_w_mm: Physical sensor width in millimetres.
        image_w_px:  Image width in pixels.

    Returns:
        Focal length fx in pixels.
    """
    return fl_mm / sensor_w_mm * image_w_px


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build the 3×3 camera intrinsic matrix K."""
    return np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ])


# ---------------------------------------------------------------------------
# Distortion
# ---------------------------------------------------------------------------

def apply_distortion(r2: float, k1: float, k2: float) -> float:
    """Radial distortion multiplier: 1 + k1·r² + k2·r⁴."""
    return 1.0 + k1 * r2 + k2 * r2 * r2


def distort_point(pt_norm: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """Apply radial distortion to a normalised (undistorted) image point.

    Args:
        pt_norm: shape-(2,) normalised point [x_n, y_n] (camera frame, z=1).
        k1, k2:  Radial distortion coefficients.

    Returns:
        Distorted normalised point, shape-(2,).
    """
    r2 = float(pt_norm[0] ** 2 + pt_norm[1] ** 2)
    return pt_norm * apply_distortion(r2, k1, k2)


def undistort_point(
    pt_dist: np.ndarray, k1: float, k2: float, max_iter: int = 20, tol: float = 1e-8
) -> np.ndarray:
    """Iteratively remove radial distortion from a normalised point.

    Args:
        pt_dist: shape-(2,) distorted normalised point.
        k1, k2:  Radial distortion coefficients.
        max_iter: Maximum Newton iterations.
        tol:     Convergence tolerance (pixel distance).

    Returns:
        Undistorted normalised point, shape-(2,).
    """
    pt = pt_dist.copy().astype(float)
    for _ in range(max_iter):
        r2 = float(pt[0] ** 2 + pt[1] ** 2)
        factor = apply_distortion(r2, k1, k2)
        pt_new = pt_dist / factor
        if np.linalg.norm(pt_new - pt) < tol:
            return pt_new
        pt = pt_new
    return pt


# ---------------------------------------------------------------------------
# Projection / unprojection
# ---------------------------------------------------------------------------

def project(
    pt3d: np.ndarray,
    K: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
) -> np.ndarray:
    """Project a 3D point in the camera frame to pixel coordinates.

    Args:
        pt3d: shape-(3,) point in camera frame (z > 0 for visible points).
        K:    3×3 intrinsic matrix.
        k1, k2: Radial distortion coefficients.

    Returns:
        shape-(2,) pixel coordinates [col, row].

    Raises:
        ValueError: if z <= 0 (point is behind the camera).
    """
    z = float(pt3d[2])
    if z <= 0:
        raise ValueError(f"Point is behind camera: z={z}")
    pt_norm = pt3d[:2] / z
    pt_dist = distort_point(pt_norm, k1, k2)
    return np.array([
        K[0, 0] * pt_dist[0] + K[0, 2],
        K[1, 1] * pt_dist[1] + K[1, 2],
    ])


def unproject(
    pt2d: np.ndarray,
    K: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
) -> np.ndarray:
    """Unproject a pixel to a unit bearing vector in the camera frame.

    Args:
        pt2d: shape-(2,) pixel coordinates [col, row].
        K:    3×3 intrinsic matrix.
        k1, k2: Radial distortion coefficients.

    Returns:
        shape-(3,) unit bearing vector.
    """
    pt_dist = np.array([
        (pt2d[0] - K[0, 2]) / K[0, 0],
        (pt2d[1] - K[1, 2]) / K[1, 1],
    ])
    pt_norm = undistort_point(pt_dist, k1, k2)
    bearing = np.array([pt_norm[0], pt_norm[1], 1.0])
    return bearing / np.linalg.norm(bearing)


# ---------------------------------------------------------------------------
# Image overlays
# ---------------------------------------------------------------------------

def overlay_keypoints(image_bytes: bytes, keypoints) -> bytes:
    """Draw SIFT keypoints on a JPEG image.

    Args:
        image_bytes: Raw JPEG bytes.
        keypoints:   Either a list of cv2.KeyPoint objects or a shape-(N,2)
                     numpy array of [u, v] pixel coordinates.

    Returns:
        JPEG bytes with keypoints drawn as circles.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # Accept either cv2.KeyPoint list or (N,2) numpy array
    if hasattr(keypoints, '__len__') and len(keypoints) > 0 and not isinstance(keypoints[0], cv2.KeyPoint):
        kps = [cv2.KeyPoint(float(u), float(v), 6.0) for u, v in keypoints]
    else:
        kps = list(keypoints)
    out = cv2.drawKeypoints(
        img, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def overlay_projected_points(
    image_bytes: bytes,
    pts3d: np.ndarray,
    K: np.ndarray,
    R_cw: np.ndarray,
    t_cw: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    colour: tuple[int, int, int] = (0, 255, 0),
    radius: int = 3,
) -> bytes:
    """Project 3D world points onto an image and draw them.

    Args:
        image_bytes: Raw JPEG bytes.
        pts3d:  shape-(N, 3) 3D points in world frame.
        K:      3×3 intrinsic matrix.
        R_cw:   3×3 rotation matrix (world→camera).
        t_cw:   shape-(3,) translation (camera position in world).
        k1, k2: Radial distortion coefficients.
        colour: BGR colour for drawn points.
        radius: Circle radius in pixels.

    Returns:
        JPEG bytes with projected points drawn.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    for pt in pts3d:
        # Transform to camera frame
        p_cam = R_cw @ (pt - t_cw)
        if p_cam[2] <= 0:
            continue
        try:
            px = project(p_cam, K, k1, k2)
        except ValueError:
            continue
        x, y = int(round(px[0])), int(round(px[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius, colour, -1)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()
