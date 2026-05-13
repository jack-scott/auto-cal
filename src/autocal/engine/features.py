"""
Feature detection, matching, track building, and triangulation.

All functions operate on raw bytes (JPEG/PNG) or numpy arrays — no MCAP I/O.

Coordinate conventions
----------------------
- Keypoints: pixel coordinates (u, v) as returned by OpenCV, shape-(N,2) float32.
- Normalised points: (u - cx) / fx, etc. — used internally for triangulation.
- 3D points: shape-(3,) or shape-(N,3) float64 in whichever world frame the
  caller supplies camera poses in.

Detection
---------
    kps, descs = detect_sift(image_bytes)
    # kps:   shape-(N,2) float32  [u, v] pixel coords
    # descs: shape-(N,128) float32  SIFT descriptors

Matching
--------
    matches = match_sift(descs_a, descs_b, ratio=0.75)
    # matches: list of (i, j) index pairs passing Lowe ratio test

Track building
--------------
    tracks = build_tracks(matches_per_pair)
    # matches_per_pair: {(img_a, img_b): [(i, j), ...]}
    # tracks: list of Track — each Track holds {image_id: keypoint_idx}

Triangulation
-------------
    pt3d = triangulate(kp_a, kp_b, K, pose_a, pose_b)
    # pose_*: gtsam.Pose3 (R_wc, t = camera in world)
    # Returns shape-(3,) point or None if degenerate.

Visualisation
-------------
    jpeg_bytes = draw_matches(img_a_bytes, kps_a, img_b_bytes, kps_b, matches)
"""

from __future__ import annotations

import base64
import io
import zlib
from dataclasses import dataclass, field
from typing import Any

import cv2
import gtsam
import numpy as np

from autocal.io.mcap_reader import register_message


# ---------------------------------------------------------------------------
# SIFT feature wire format
# ---------------------------------------------------------------------------

@register_message
@dataclass
class SiftFeaturesMsg:
    """Wire format for cached SIFT features stored as a JSON MCAP message."""
    n:    int  # number of keypoints
    kps:  str  # base64+zlib float32 array, shape (n, 2)
    desc: str  # base64+zlib float32 array, shape (n, 128)


def _encode_array(arr: np.ndarray) -> str:
    return base64.b64encode(
        zlib.compress(arr.astype(np.float32).tobytes(), level=1)
    ).decode()


def _decode_array(s: str, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(
        zlib.decompress(base64.b64decode(s)), dtype=np.float32
    ).reshape(shape)


def encode_sift_features(kps: np.ndarray, descs: np.ndarray) -> SiftFeaturesMsg:
    """Encode SIFT keypoints and descriptors into the MCAP wire format."""
    return SiftFeaturesMsg(n=len(kps), kps=_encode_array(kps), desc=_encode_array(descs))


def decode_sift_features(msg: SiftFeaturesMsg) -> tuple[np.ndarray, np.ndarray]:
    """Decode a SiftFeaturesMsg back to (kps, descs) numpy arrays."""
    kps   = _decode_array(msg.kps,  (msg.n, 2))
    descs = _decode_array(msg.desc, (msg.n, 128))
    return kps, descs


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_sift(
    image_bytes: bytes,
    n_features: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect SIFT keypoints and descriptors in a JPEG/PNG image.

    Args:
        image_bytes: Raw image bytes (any format OpenCV can decode).
        n_features:  Max features to detect; 0 = unlimited.

    Returns:
        kps:   shape-(N,2) float32 pixel coordinates [u, v].
        descs: shape-(N,128) float32 SIFT descriptors.
               Both arrays are empty (shape (0,2) / (0,128)) if no features found.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)

    sift = cv2.SIFT_create(nfeatures=n_features)
    cv_kps, descs = sift.detectAndCompute(img, None)

    if cv_kps is None or len(cv_kps) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)

    kps = np.array([[kp.pt[0], kp.pt[1]] for kp in cv_kps], dtype=np.float32)
    if descs is None:
        descs = np.zeros((len(kps), 128), dtype=np.float32)
    return kps, descs.astype(np.float32)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_sift(
    descs_a: np.ndarray,
    descs_b: np.ndarray,
    ratio: float = 0.75,
) -> list[tuple[int, int]]:
    """Match SIFT descriptors using Lowe's ratio test.

    Args:
        descs_a: shape-(Na,128) float32 descriptors for image A.
        descs_b: shape-(Nb,128) float32 descriptors for image B.
        ratio:   Lowe ratio threshold (keep match if best < ratio * second-best).

    Returns:
        List of (i, j) index pairs where i ∈ descs_a, j ∈ descs_b.
    """
    if len(descs_a) == 0 or len(descs_b) == 0:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(descs_a, descs_b, k=2)
    matches = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            matches.append((m.queryIdx, m.trainIdx))
    return matches


# ---------------------------------------------------------------------------
# Geometric verification
# ---------------------------------------------------------------------------

def filter_matches_ransac(
    kps_a: np.ndarray,
    kps_b: np.ndarray,
    matches: list[tuple[int, int]],
    ransac_threshold: float = 2.0,
    min_inliers: int = 15,
) -> list[tuple[int, int]]:
    """Filter matches using RANSAC + fundamental matrix (USAC_MAGSAC).

    Args:
        kps_a, kps_b:      Undistorted keypoints for images A and B (shape-(N,2)
                           float32).  Must be undistorted so epipolar geometry
                           is correct.
        matches:           Initial matches from the ratio test.
        ransac_threshold:  Inlier reprojection threshold in pixels.
        min_inliers:       Discard the pair if fewer inliers survive.

    Returns:
        Inlier matches, or an empty list if the pair fails.
    """
    if len(matches) < 8:
        return []

    pts_a = kps_a[[i for i, _ in matches]].astype(np.float64)
    pts_b = kps_b[[j for _, j in matches]].astype(np.float64)

    _, mask = cv2.findFundamentalMat(
        pts_a, pts_b,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_threshold,
        confidence=0.999,
        maxIters=10000,
    )
    if mask is None:
        return []

    inliers = [m for m, ok in zip(matches, mask.ravel()) if ok]
    return inliers if len(inliers) >= min_inliers else []


# ---------------------------------------------------------------------------
# Track building
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """A 3D feature track across multiple images.

    Attributes:
        observations: {image_id: keypoint_index} mapping.
        point3d:      Triangulated 3D position (None until triangulated).
    """
    observations: dict[Any, int] = field(default_factory=dict)
    point3d: np.ndarray | None = None  # shape (3,) float64


def build_tracks(
    matches_per_pair: dict[tuple[Any, Any], list[tuple[int, int]]],
) -> list[Track]:
    """Merge per-pair matches into consistent feature tracks using union-find.

    Args:
        matches_per_pair: {(img_a, img_b): [(i, j), ...]} where i/j are
                          keypoint indices within their respective images.

    Returns:
        List of Track objects, each with ≥2 observations.
    """
    # Nodes are (image_id, keypoint_idx) pairs.
    parent: dict[tuple[Any, int], tuple[Any, int]] = {}

    def find(x: tuple[Any, int]) -> tuple[Any, int]:
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: tuple[Any, int], b: tuple[Any, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (img_a, img_b), matches in matches_per_pair.items():
        for i, j in matches:
            union((img_a, i), (img_b, j))

    # Group by root
    groups: dict[tuple[Any, int], Track] = {}
    for node in parent:
        root = find(node)
        if root not in groups:
            groups[root] = Track()
        img_id, kp_idx = node
        groups[root].observations[img_id] = kp_idx

    return [t for t in groups.values() if len(t.observations) >= 2]


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def triangulate(
    kp_a: np.ndarray,
    kp_b: np.ndarray,
    K: np.ndarray,
    pose_a: Any,  # gtsam.Pose3
    pose_b: Any,  # gtsam.Pose3
) -> np.ndarray | None:
    """Triangulate a single 3D point from two observations.

    Uses the DLT (linear) method via OpenCV's triangulatePoints.

    Args:
        kp_a:   shape-(2,) pixel [u, v] in image A.
        kp_b:   shape-(2,) pixel [u, v] in image B.
        K:      3×3 intrinsic matrix (same for both cameras, square pixels assumed).
        pose_a: gtsam.Pose3 for camera A (R_wc, t = camera in world).
        pose_b: gtsam.Pose3 for camera B.

    Returns:
        shape-(3,) float64 point in world frame, or None if degenerate.
    """
    # Build 3×4 projection matrices P = K @ [R_cw | -R_cw @ t_world]
    def _projection(pose: Any) -> np.ndarray:
        R_wc = pose.rotation().matrix()  # camera→world, shape (3,3)
        R_cw = R_wc.T                    # world→camera
        t = pose.translation()            # camera position in world
        t_cam = -R_cw @ t
        Rt = np.hstack([R_cw, t_cam[:, None]])
        return K @ Rt

    P_a = _projection(pose_a)
    P_b = _projection(pose_b)

    pts_a = np.array([[kp_a[0]], [kp_a[1]]], dtype=np.float64)
    pts_b = np.array([[kp_b[0]], [kp_b[1]]], dtype=np.float64)

    pts4d = cv2.triangulatePoints(P_a, P_b, pts_a, pts_b)
    w = pts4d[3, 0]
    if abs(w) < 1e-12:
        return None
    pt = pts4d[:3, 0] / w
    return pt.astype(np.float64)


def triangulate_tracks(
    tracks: list[Track],
    keypoints_per_image: dict[Any, np.ndarray],
    K: np.ndarray,
    poses: dict[Any, Any],  # image_id → gtsam.Pose3
) -> None:
    """Triangulate all tracks in-place using the first two visible images.

    Only triangulates tracks that have at least 2 observations with known poses.
    Sets track.point3d on success.

    Args:
        tracks:               List of Track objects (modified in-place).
        keypoints_per_image:  {image_id: shape-(N,2) float32 keypoints}.
        K:                    3×3 intrinsic matrix.
        poses:                {image_id: gtsam.Pose3}.
    """
    for track in tracks:
        visible = [(img_id, kp_idx)
                   for img_id, kp_idx in track.observations.items()
                   if img_id in poses and img_id in keypoints_per_image]
        if len(visible) < 2:
            continue
        (img_a, idx_a), (img_b, idx_b) = visible[0], visible[1]
        kp_a = keypoints_per_image[img_a][idx_a].astype(np.float64)
        kp_b = keypoints_per_image[img_b][idx_b].astype(np.float64)
        pt = triangulate(kp_a, kp_b, K, poses[img_a], poses[img_b])
        if pt is not None:
            track.point3d = pt


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_matches(
    img_a_bytes: bytes,
    kps_a: np.ndarray,
    img_b_bytes: bytes,
    kps_b: np.ndarray,
    matches: list[tuple[int, int]],
    max_matches: int = 100,
) -> bytes:
    """Draw matched keypoints side-by-side and return JPEG bytes.

    Args:
        img_a_bytes:  Raw image bytes for image A.
        img_b_bytes:  Raw image bytes for image B.
        kps_a:        shape-(Na,2) float32 keypoints for A.
        kps_b:        shape-(Nb,2) float32 keypoints for B.
        matches:      List of (i, j) index pairs.
        max_matches:  Cap on number of matches drawn (random subset if more).

    Returns:
        JPEG bytes of the side-by-side match image.
    """
    def _decode(b: bytes) -> np.ndarray:
        arr = np.frombuffer(b, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    img_a = _decode(img_a_bytes)
    img_b = _decode(img_b_bytes)

    cv_kps_a = [cv2.KeyPoint(float(u), float(v), 1.0) for u, v in kps_a]
    cv_kps_b = [cv2.KeyPoint(float(u), float(v), 1.0) for u, v in kps_b]

    subset = matches[:max_matches]
    cv_matches = [cv2.DMatch(i, j, 0.0) for i, j in subset]

    out = cv2.drawMatches(img_a, cv_kps_a, img_b, cv_kps_b, cv_matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bytes(buf)


# ---------------------------------------------------------------------------
# Geometry utilities (GTSAM-aware)
# ---------------------------------------------------------------------------

def cal_to_K(cal: gtsam.Cal3DS2 | gtsam.Cal3Fisheye) -> np.ndarray:
    """Return the 3×3 intrinsic matrix K for a GTSAM calibration object."""
    return np.array([
        [cal.fx(), cal.skew(), cal.px()],
        [0.0,      cal.fy(),   cal.py()],
        [0.0,      0.0,        1.0     ],
    ], dtype=np.float64)


def undistort_keypoints(
    keypoints: dict[Any, np.ndarray],
    cal: gtsam.Cal3DS2 | gtsam.Cal3Fisheye,
) -> dict[Any, np.ndarray]:
    """Undistort pixel keypoints using the camera calibration model.

    Returns the input dict unchanged when there is no distortion.

    Args:
        keypoints: {img_id: shape-(N,2) float32 pixel coordinates}.
        cal:       gtsam.Cal3DS2 or gtsam.Cal3Fisheye.
    """
    k = np.array(cal.k(), dtype=np.float64)
    if not np.any(k != 0.0):
        return keypoints
    K = cal_to_K(cal)
    fisheye = isinstance(cal, gtsam.Cal3Fisheye)
    result: dict[Any, np.ndarray] = {}
    for img_id, kps in keypoints.items():
        pts = kps.reshape(-1, 1, 2).astype(np.float64)
        if fisheye:
            D = k[:4].reshape(4, 1)
            undist = cv2.fisheye.undistortPoints(pts, K, D, P=K)
        else:
            undist = cv2.undistortPoints(pts, K, k[:4], P=K)
        result[img_id] = undist.reshape(-1, 2).astype(np.float32)
    return result


def all_positive_depth(
    pt3d: np.ndarray,
    observations: dict,
    poses: dict,
) -> bool:
    """Return True if pt3d has positive depth in every observing camera.

    Args:
        pt3d:         shape-(3,) world-frame 3D point.
        observations: {img_id: kp_idx} track observations.
        poses:        {img_id: gtsam.Pose3(R_wc, t)}.
    """
    for img_id in observations:
        if img_id not in poses:
            continue
        pose = poses[img_id]
        R_cw = pose.rotation().matrix().T
        z = (R_cw @ (pt3d - pose.translation()))[2]
        if z <= 0:
            return False
    return True


def filter_by_reproj(
    tracks: list[Track],
    keypoints: dict[Any, np.ndarray],
    poses: dict,
    cal: gtsam.Cal3DS2 | gtsam.Cal3Fisheye,
    max_err_px: float,
) -> list[Track]:
    """Keep tracks whose max reprojection error across all observations is ≤ max_err_px.

    Uses the full distortion model (Kannala-Brandt for fisheye,
    Brown-Conrady for Cal3DS2).

    Args:
        tracks:     Track list with point3d set.
        keypoints:  {img_id: shape-(N,2) float32 pixel coordinates}.
        poses:      {img_id: gtsam.Pose3(R_wc, t)}.
        cal:        gtsam.Cal3DS2 or gtsam.Cal3Fisheye.
        max_err_px: Discard tracks with any observation error above this.
    """
    fisheye = isinstance(cal, gtsam.Cal3Fisheye)
    fx, fy, cx, cy = cal.fx(), cal.fy(), cal.px(), cal.py()
    k = np.array(cal.k(), dtype=np.float64)

    def _max_err(track: Track) -> float:
        max_e = 0.0
        for img_id, kp_idx in track.observations.items():
            if img_id not in poses or img_id not in keypoints:
                continue
            pose = poses[img_id]
            R_cw = pose.rotation().matrix().T
            pc = R_cw @ (track.point3d - pose.translation())
            x, y, z = pc
            if z <= 0:
                return float("inf")
            if fisheye:
                r = np.sqrt(x*x + y*y)
                if r < 1e-10:
                    pu, pv = cx, cy
                else:
                    theta = np.arctan2(r, z)
                    t2 = theta * theta
                    rd = theta * (1 + k[0]*t2 + k[1]*t2**2 + k[2]*t2**3 + k[3]*t2**4)
                    s = rd / r
                    pu = fx * s * x + cx
                    pv = fy * s * y + cy
            else:
                xn, yn = x / z, y / z
                r2 = xn*xn + yn*yn
                radial = 1.0 + k[0]*r2 + k[1]*r2*r2
                pu = fx * (radial*xn + 2*k[2]*xn*yn + k[3]*(r2 + 2*xn*xn)) + cx
                pv = fy * (radial*yn + k[2]*(r2 + 2*yn*yn) + 2*k[3]*xn*yn) + cy
            obs_kp = keypoints[img_id][kp_idx]
            e = np.hypot(pu - obs_kp[0], pv - obs_kp[1])
            if e > max_e:
                max_e = e
        return max_e

    return [t for t in tracks if _max_err(t) <= max_err_px]


def max_reproj_error(
    track: Track,
    keypoints: dict[Any, np.ndarray],
    poses: dict,
    cal: gtsam.Cal3DS2 | gtsam.Cal3Fisheye,
) -> float:
    """Return the maximum reprojection error in pixels for a single track.

    Uses the full distortion model.  Returns infinity if any camera sees
    the point behind it.  Useful for diagnostics and per-track inspection.
    """
    fisheye = isinstance(cal, gtsam.Cal3Fisheye)
    fx, fy, cx, cy = cal.fx(), cal.fy(), cal.px(), cal.py()
    k = np.array(cal.k(), dtype=np.float64)
    max_e = 0.0
    for img_id, kp_idx in track.observations.items():
        if img_id not in poses or img_id not in keypoints:
            continue
        pose = poses[img_id]
        R_cw = pose.rotation().matrix().T
        pc = R_cw @ (track.point3d - pose.translation())
        x, y, z = pc
        if z <= 0:
            return float("inf")
        if fisheye:
            r = np.sqrt(x*x + y*y)
            if r < 1e-10:
                pu, pv = cx, cy
            else:
                theta = np.arctan2(r, z)
                t2 = theta * theta
                rd = theta * (1 + k[0]*t2 + k[1]*t2**2 + k[2]*t2**3 + k[3]*t2**4)
                s = rd / r
                pu = fx * s * x + cx
                pv = fy * s * y + cy
        else:
            xn, yn = x / z, y / z
            r2 = xn*xn + yn*yn
            radial = 1.0 + k[0]*r2 + k[1]*r2*r2
            pu = fx * (radial*xn + 2*k[2]*xn*yn + k[3]*(r2 + 2*xn*xn)) + cx
            pv = fy * (radial*yn + k[2]*(r2 + 2*yn*yn) + 2*k[3]*xn*yn) + cy
        obs_kp = keypoints[img_id][kp_idx]
        e = np.hypot(pu - obs_kp[0], pv - obs_kp[1])
        if e > max_e:
            max_e = e
    return max_e


def triangulate_gtsam(
    tracks: list[Track],
    keypoints: dict[Any, np.ndarray],
    cal: gtsam.Cal3DS2 | gtsam.Cal3Fisheye,
    poses: dict,
    max_dist: float = 0.0,
    min_parallax_deg: float = 0.0,
) -> None:
    """Triangulate tracks in-place using GTSAM multi-view triangulation.

    Uses all visible cameras and applies nonlinear refinement via
    gtsam.triangulatePoint3.  Observations are distorted pixels — the
    calibration model handles projection internally.

    Args:
        tracks:            Track list; point3d is set in-place on success.
        keypoints:         {img_id: shape-(N,2) float32 distorted pixels}.
        cal:               gtsam.Cal3DS2 or gtsam.Cal3Fisheye.
        poses:             {img_id: gtsam.Pose3(R_wc, t)}.
        max_dist:          Discard points farther than this from every
                           observing camera (metres).  0 = disabled.
        min_parallax_deg:  Discard points where the max viewing angle
                           across all camera pairs is below this.  0 = disabled.
    """
    for track in tracks:
        visible = [
            (img_id, kp_idx)
            for img_id, kp_idx in track.observations.items()
            if img_id in poses
        ]
        if len(visible) < 2:
            continue

        pose_vec = gtsam.Pose3Vector()
        meas_vec = gtsam.Point2Vector()
        for img_id, kp_idx in visible:
            pose_vec.append(poses[img_id])
            kp = keypoints[img_id][kp_idx]
            meas_vec.append(gtsam.Point2(float(kp[0]), float(kp[1])))

        try:
            pt = gtsam.triangulatePoint3(pose_vec, cal, meas_vec,
                                         rank_tol=1e-9, optimize=True)
            pt_np = np.array([float(pt[0]), float(pt[1]), float(pt[2])])
        except Exception:
            track.point3d = None
            continue

        if max_dist > 0.0:
            if min(np.linalg.norm(pt_np - poses[img_id].translation())
                   for img_id, _ in visible) > max_dist:
                track.point3d = None
                continue

        if min_parallax_deg > 0.0:
            cam_pos = np.array([poses[img_id].translation() for img_id, _ in visible])
            rays = pt_np - cam_pos
            norms = np.linalg.norm(rays, axis=1, keepdims=True)
            if np.any(norms < 1e-10):
                track.point3d = None
                continue
            rays_norm = rays / norms
            cos_mat = rays_norm @ rays_norm.T
            np.fill_diagonal(cos_mat, 1.0)
            max_angle = float(np.degrees(np.arccos(np.clip(cos_mat.min(), -1.0, 1.0))))
            if max_angle < min_parallax_deg:
                track.point3d = None
                continue

        track.point3d = pt_np


def chain_essential_matrix(
    img_ids: list,
    keypoints: dict[Any, np.ndarray],
    matches_per_pair: dict[tuple[Any, Any], list[tuple[int, int]]],
    K: np.ndarray,
) -> dict:
    """Initialise camera poses by chaining essential-matrix relative poses.

    Frame 0 is placed at the origin with optical axis along world X+.
    Translation is unit-scale only — GTSAM resolves scale from feature tracks.
    keypoints must be undistorted before calling (caller's responsibility).

    Args:
        img_ids:          Ordered list of image ids.
        keypoints:        {img_id: shape-(N,2) undistorted pixel coordinates}.
        matches_per_pair: {(id_a, id_b): [(i,j), ...]} sequential matches.
        K:                3×3 intrinsic matrix.

    Returns:
        {img_id: gtsam.Pose3(R_wc, t)} initial poses.
    """
    R_wc_0 = gtsam.Rot3(np.array([
        [0.0,  0.0, 1.0],
        [1.0,  0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]))
    poses: dict = {img_ids[0]: gtsam.Pose3(R_wc_0, gtsam.Point3(0.0, 0.0, 0.0))}

    for k in range(len(img_ids) - 1):
        id_a, id_b = img_ids[k], img_ids[k + 1]
        pose_a = poses[id_a]

        if (id_a, id_b) not in matches_per_pair or len(matches_per_pair[(id_a, id_b)]) < 5:
            poses[id_b] = pose_a
            continue

        matches = matches_per_pair[(id_a, id_b)]
        pts_a = keypoints[id_a][[i for i, _ in matches]].astype(np.float64)
        pts_b = keypoints[id_b][[j for _, j in matches]].astype(np.float64)

        E, mask = cv2.findEssentialMat(pts_a, pts_b, K, method=cv2.RANSAC,
                                        prob=0.999, threshold=1.0)
        if E is None:
            poses[id_b] = pose_a
            continue

        _, R_rel, t_rel, _ = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)

        # cv2.recoverPose gives R,t such that P_camB = R_rel @ P_camA + t_rel
        # R_wc_B = R_wc_A @ R_rel^T  (unit-scale translation)
        R_wc_a = pose_a.rotation().matrix()
        R_wc_b = gtsam.Rot3(R_wc_a @ R_rel.T)
        t_b = pose_a.translation() - R_wc_b.matrix() @ t_rel.flatten()
        poses[id_b] = gtsam.Pose3(R_wc_b, gtsam.Point3(*t_b))

    return poses
