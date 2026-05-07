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
    # pose_*: gtsam.Pose3 (R_cw, t = camera in world)
    # Returns shape-(3,) point or None if degenerate.

Visualisation
-------------
    jpeg_bytes = draw_matches(img_a_bytes, kps_a, img_b_bytes, kps_b, matches)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


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
        pose_a: gtsam.Pose3 for camera A (R_cw, t = camera in world).
        pose_b: gtsam.Pose3 for camera B.

    Returns:
        shape-(3,) float64 point in world frame, or None if degenerate.
    """
    # Build 3×4 projection matrices P = K @ [R_cw | -R_cw @ t_world]
    def _projection(pose: Any) -> np.ndarray:
        R = pose.rotation().matrix()   # R_cw, shape (3,3)
        t = pose.translation()          # camera in world, shape (3,)
        t_cam = -R @ t                  # translation in camera frame
        Rt = np.hstack([R, t_cam[:, None]])
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
