"""
Self-calibrating SfM pipeline using GTSAM.

Estimates camera intrinsics (focal length + 2 radial distortion params)
without calibration targets, using GPS-positioned drone images.

Pipeline:
  1. Load images, extract GPS + focal length from EXIF
  2. Detect SIFT features, match GPS-adjacent image pairs
  3. Build feature tracks via union-find
  4. Initialize camera poses from GPS + nadir rotation assumption
  5. Triangulate 3D points from initial poses
  6. Optimize poses + landmarks + shared Cal3Bundler in GTSAM,
     writing per-iteration PointCloud + CameraCalibration convergence data
     plus final camera_link trajectory to data/reconstruction.mcap.

Frame convention (REP-105):
  earth       WGS-84 ECEF — global reference
  map         Local ENU, origin at first GPS fix — all SfM quantities live here
  camera_link Moving camera frame — one TF message per image at its GPS timestamp
"""

from datetime import datetime, timezone
import math
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import cv2
import foxglove
import gtsam
import numpy as np
import piexif
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration as FgCameraCalibration
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from google.protobuf import descriptor_pb2
from google.protobuf.timestamp_pb2 import Timestamp

DATA_DIR = Path(__file__).parent.parent / "data" / "caliterra"
OUTPUT_MCAP = Path(__file__).parent.parent / "data" / "reconstruction.mcap"
IMAGE_W, IMAGE_H = 4000, 3000

# Noise parameters
GPS_NOISE_XY_M = 3.0
GPS_NOISE_Z_M = 5.0
REPROJ_NOISE_PX = 1.5

# Fallback sensor width if EXIF FocalPlaneXResolution tag is absent (mm).
SENSOR_W_MM_FALLBACK = 6.17


# ---------------------------------------------------------------------------
# GPS helpers
# ---------------------------------------------------------------------------

def _rational(v) -> float:
    return v[0] / v[1] if v[1] else 0.0


def _dms_to_deg(dms, ref: bytes) -> float:
    deg = sum(_rational(x) / (60 ** i) for i, x in enumerate(dms))
    return -deg if ref in (b'S', b'W') else deg


def _gps_to_unix_ns(date_stamp: bytes, time_stamp) -> int:
    y, mo, d = date_stamp.decode().split(":")
    h = int(_rational(time_stamp[0]))
    m = int(_rational(time_stamp[1]))
    s_frac = _rational(time_stamp[2])
    s = int(s_frac)
    us = int((s_frac - s) * 1_000_000)
    dt = datetime(int(y), int(mo), int(d), h, m, s, us, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def gps_to_enu(lat, lon, alt, lat0, lon0, alt0) -> np.ndarray:
    """Flat-earth GPS → local ENU (accurate to < 1m within a 5km radius)."""
    R = 6_371_000.0
    dx = (lon - lon0) * math.radians(1) * R * math.cos(math.radians(lat0))
    dy = (lat - lat0) * math.radians(1) * R
    dz = alt - alt0
    return np.array([dx, dy, dz])


def _enu_origin_ecef(lat0_deg: float, lon0_deg: float, alt0_m: float):
    """Return (xyz_ecef, R_ecef_enu) for the ENU origin point.

    R_ecef_enu transforms map (ENU) vectors into ECEF vectors.
    Columns are [East, North, Up] unit vectors expressed in ECEF.
    """
    a = 6_378_137.0
    e2 = 0.00669437999014
    lat = math.radians(lat0_deg)
    lon = math.radians(lon0_deg)
    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    xyz = np.array([
        (N + alt0_m) * math.cos(lat) * math.cos(lon),
        (N + alt0_m) * math.cos(lat) * math.sin(lon),
        (N * (1 - e2) + alt0_m) * math.sin(lat),
    ])
    R_ecef_enu = np.array([
        [-math.sin(lon), -math.sin(lat) * math.cos(lon), math.cos(lat) * math.cos(lon)],
        [ math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat) * math.sin(lon)],
        [           0.0,              math.cos(lat),               math.sin(lat)       ],
    ])
    return xyz, R_ecef_enu


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_images(image_dir: Path) -> list[dict]:
    jpgs = sorted(image_dir.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No images found in {image_dir}. Run: pixi run download-data")

    images = []
    for i, path in enumerate(jpgs):
        exif = piexif.load(str(path))
        gps = exif.get("GPS", {})
        exif_ifd = exif.get("Exif", {})
        lat = _dms_to_deg(gps[piexif.GPSIFD.GPSLatitude], gps[piexif.GPSIFD.GPSLatitudeRef])
        lon = _dms_to_deg(gps[piexif.GPSIFD.GPSLongitude], gps[piexif.GPSIFD.GPSLongitudeRef])
        alt = _rational(gps[piexif.GPSIFD.GPSAltitude])

        date_stamp = gps.get(piexif.GPSIFD.GPSDateStamp)
        time_stamp = gps.get(piexif.GPSIFD.GPSTimeStamp)
        t_ns = _gps_to_unix_ns(date_stamp, time_stamp) if (date_stamp and time_stamp) else i * 1_000_000_000

        fl_tag = exif_ifd.get(piexif.ExifIFD.FocalLength)
        fl_mm = _rational(fl_tag) if fl_tag else None

        # Compute exact sensor width from FocalPlaneXResolution if present
        fpr_tag = exif_ifd.get(piexif.ExifIFD.FocalPlaneXResolution)
        fpr_unit = exif_ifd.get(piexif.ExifIFD.FocalPlaneResolutionUnit, 2)  # 2=inch, 3=cm
        if fpr_tag:
            pixels_per_unit = _rational(fpr_tag)
            unit_mm = 25.4 if fpr_unit == 2 else 10.0
            sensor_w_mm = IMAGE_W / pixels_per_unit * unit_mm
        else:
            sensor_w_mm = SENSOR_W_MM_FALLBACK

        images.append({
            "path": path, "lat": lat, "lon": lon, "alt": alt,
            "t_ns": t_ns, "fl_mm": fl_mm, "sensor_w_mm": sensor_w_mm,
        })

    lat0, lon0, alt0 = images[0]["lat"], images[0]["lon"], images[0]["alt"]
    for img in images:
        img["enu"] = gps_to_enu(img["lat"], img["lon"], img["alt"], lat0, lon0, alt0)

    return images


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def detect_features(images: list[dict], max_side: int = 1600) -> list[dict]:
    sift = cv2.SIFT_create(nfeatures=8000)
    feats = []
    for i, img_data in enumerate(images):
        img = cv2.imread(str(img_data["path"]), cv2.IMREAD_GRAYSCALE)
        scale = min(max_side / max(img.shape), 1.0)
        small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img
        kps, descs = sift.detectAndCompute(small, None)
        pts = np.array([[kp.pt[0] / scale, kp.pt[1] / scale] for kp in kps], dtype=np.float32)
        feats.append({"pts": pts, "descs": descs})
        print(f"\r  {i+1}/{len(images)}: {len(kps)} keypoints", end="", flush=True)
    print()
    return feats


# ---------------------------------------------------------------------------
# Feature matching
# ---------------------------------------------------------------------------

def match_pairs(images: list[dict], feats: list[dict], max_dist_m: float = 120.0) -> list[dict]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = []
    n = len(images)

    candidate_pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if np.linalg.norm(images[i]["enu"][:2] - images[j]["enu"][:2]) < max_dist_m
    ]
    print(f"  {len(candidate_pairs)} candidate pairs within {max_dist_m}m GPS distance")

    for i, j in candidate_pairs:
        matches = matcher.knnMatch(feats[i]["descs"], feats[j]["descs"], k=2)
        good = [m for m, n_ in matches if m.distance < 0.75 * n_.distance]
        if len(good) < 20:
            continue

        pts1 = feats[i]["pts"][[m.queryIdx for m in good]]
        pts2 = feats[j]["pts"][[m.trainIdx for m in good]]
        _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.999)
        if mask is None:
            continue

        ok = mask.ravel().astype(bool)
        if ok.sum() < 30:
            continue

        pairs.append({
            "i": i, "j": j,
            "pts1": pts1[ok],
            "pts2": pts2[ok],
            "qi": np.array([good[k].queryIdx for k in np.where(ok)[0]]),
            "qj": np.array([good[k].trainIdx for k in np.where(ok)[0]]),
        })

    return pairs


# ---------------------------------------------------------------------------
# Track building (union-find)
# ---------------------------------------------------------------------------

def build_tracks(pairs: list[dict], feats: list[dict]) -> list[list]:
    parent: dict = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for pair in pairs:
        for qi, qj in zip(pair["qi"], pair["qj"]):
            union((pair["i"], int(qi)), (pair["j"], int(qj)))

    groups: dict = defaultdict(list)
    for key in parent:
        groups[find(key)].append(key)

    tracks = []
    for members in groups.values():
        if len({img for img, _ in members}) < 2:
            continue
        tracks.append([(img, feats[img]["pts"][kp]) for img, kp in members])

    return tracks


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _proj_matrix(pose: gtsam.Pose3, K: np.ndarray) -> np.ndarray:
    R_cw = pose.rotation().matrix().T
    t_c = -R_cw @ pose.translation()
    return K @ np.hstack([R_cw, t_c.reshape(3, 1)])


def triangulate(obs: list, poses: list, K: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation from multiple views."""
    A = []
    for img_idx, pt2d in obs:
        P = _proj_matrix(poses[img_idx], K)
        x, y = pt2d
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    p = Vt[-1]
    return p[:3] / p[3]


def _ray(pose: gtsam.Pose3, pt3d: np.ndarray) -> np.ndarray:
    v = pt3d - pose.translation()
    return v / (np.linalg.norm(v) + 1e-12)


def min_angle_deg(obs: list, poses: list, pt3d: np.ndarray) -> float:
    """Minimum angle (degrees) between any two viewing rays to pt3d."""
    cam_positions = list({img_idx for img_idx, _ in obs})
    if len(cam_positions) < 2:
        return 0.0
    rays = [_ray(poses[i], pt3d) for i in cam_positions]
    min_cos = 1.0
    for a in range(len(rays)):
        for b in range(a + 1, len(rays)):
            min_cos = min(min_cos, np.dot(rays[a], rays[b]))
    return math.degrees(math.acos(np.clip(min_cos, -1, 1)))


def is_valid(pt3d: np.ndarray, obs: list, poses: list, K: np.ndarray,
             max_reproj: float = 50.0, min_angle: float = 1.0) -> bool:
    """Returns True if pt3d is in front of all cameras, reprojects reasonably,
    and has enough triangulation baseline (min viewing angle)."""
    if min_angle_deg(obs, poses, pt3d) < min_angle:
        return False
    for img_idx, pt2d in obs:
        R_cw = poses[img_idx].rotation().matrix().T
        t_c = -R_cw @ poses[img_idx].translation()
        p_cam = R_cw @ pt3d + t_c
        if p_cam[2] < 0.5:
            return False
        proj = K[:2, :2] @ (p_cam[:2] / p_cam[2]) + K[:2, 2]
        if np.linalg.norm(proj - pt2d) > max_reproj:
            return False
    return True


# ---------------------------------------------------------------------------
# GTSAM factor graph
# ---------------------------------------------------------------------------

def build_and_optimise(images, valid_tracks, pts3d, poses_init, cal_init, mcap_channels):
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    K_key = gtsam.symbol('k', 0)
    values.insert(K_key, cal_init)

    # Soft prior on calibration: ±30% on focal length, tiny on distortion.
    # Cal3Bundler has 3 DOF: {fx, k1, k2} — principal point is fixed.
    fx0 = cal_init.fx()
    cal_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([fx0 * 0.3, 1e-3, 1e-3])
    )
    graph.add(gtsam.PriorFactorCal3Bundler(K_key, cal_init, cal_noise))

    # Camera pose priors from GPS (tight position, loose yaw)
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        0.3, 0.3, math.pi,          # roll, pitch: constrained; yaw: free
        GPS_NOISE_XY_M, GPS_NOISE_XY_M, GPS_NOISE_Z_M
    ]))
    # Anchor first camera more tightly to remove gauge freedom
    anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.5])
    )
    for i, (img, pose) in enumerate(zip(images, poses_init)):
        x_key = gtsam.symbol('x', i)
        values.insert(x_key, pose)
        graph.add(gtsam.PriorFactorPose3(x_key, pose, pose_noise))

    graph.add(gtsam.PriorFactorPose3(gtsam.symbol('x', 0), poses_init[0], anchor_noise))

    # Observation factors — use Huber robust kernel to downweight outliers
    # that would otherwise cause persistent CheiralityExceptions.
    obs_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.5),
        gtsam.noiseModel.Isotropic.Sigma(2, REPROJ_NOISE_PX),
    )
    n_obs = 0
    for j, (track, pt3d) in enumerate(zip(valid_tracks, pts3d)):
        l_key = gtsam.symbol('l', j)
        values.insert(l_key, gtsam.Point3(*pt3d))
        for img_idx, pt2d in track:
            graph.add(gtsam.GeneralSFMFactor2Cal3Bundler(
                gtsam.Point2(*pt2d), obs_noise,
                gtsam.symbol('x', img_idx), l_key, K_key
            ))
            n_obs += 1

    print(f"  {graph.size()} factors | {n_obs} observations | "
          f"{len(valid_tracks)} landmarks | {len(images)} cameras")

    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SILENT")

    # Place iteration snapshots in the 102 seconds before the first image so
    # they appear as a "pre-flight convergence" window in the Foxglove timeline.
    max_iter = 100
    t_iter_base = images[0]["t_ns"] - (max_iter + 2) * 1_000_000_000

    with _suppress_cpp_stderr():
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
        _write_mcap_iteration(mcap_channels, t_iter_base, optimizer.values(), len(valid_tracks))
        prev_error = optimizer.error()
        for iteration in range(1, max_iter + 1):
            optimizer.iterate()
            curr_error = optimizer.error()
            print(f"\r  iter {iteration:3d}  error: {curr_error:.2f}", end="", flush=True)
            t_iter = t_iter_base + iteration * 1_000_000_000
            _write_mcap_iteration(mcap_channels, t_iter, optimizer.values(), len(valid_tracks))
            if abs(prev_error - curr_error) / (prev_error + 1e-9) < 1e-5:
                break
            prev_error = curr_error
        print()
        result = optimizer.values()

    _write_final_result(mcap_channels, result, images, len(valid_tracks))
    return graph, values, result


@contextmanager
def _suppress_cpp_stderr():
    """Silence C++ and Python stderr (e.g. CheiralityException spam from GTSAM)."""
    # Redirect the OS-level fd 2 (catches C++ std::cerr)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    # Also redirect Python-level stderr (catches pybind11 prints)
    saved_py = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = saved_py
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


# ---------------------------------------------------------------------------
# Foxglove MCAP helpers
# ---------------------------------------------------------------------------

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


def _write_mcap_iteration(channels: dict, t_ns: int,
                           values: gtsam.Values, n_landmarks: int) -> None:
    """Write PointCloud + CameraCalibration snapshot at one optimizer iteration."""
    ts = _ns_to_ts(t_ns)

    # PointCloud — landmark positions in the map frame
    pts = []
    for j in range(n_landmarks):
        try:
            pts.append(values.atPoint3(gtsam.symbol('l', j)))
        except Exception:
            continue
    if pts:
        xyz = np.array(pts, dtype=np.float32)
        pc = PointCloud()
        pc.timestamp.CopyFrom(ts)
        pc.frame_id = "map"
        pc.pose.orientation.w = 1.0
        pc.point_stride = 12
        for name, offset in [("x", 0), ("y", 4), ("z", 8)]:
            f = pc.fields.add()
            f.name = name
            f.offset = offset
            f.type = PackedElementField.FLOAT32
        pc.data = xyz.tobytes()
        channels["points"].log(pc.SerializeToString(), log_time=t_ns)

    # CameraCalibration — current intrinsics, referenced to camera_link
    try:
        cal = values.atCal3Bundler(gtsam.symbol('k', 0))
        cc = FgCameraCalibration()
        cc.timestamp.CopyFrom(ts)
        cc.frame_id = "camera_link"
        cc.width = IMAGE_W
        cc.height = IMAGE_H
        cc.distortion_model = "plumb_bob"
        cc.D.extend([cal.k1(), cal.k2(), 0.0, 0.0, 0.0])
        fx, px, py = cal.fx(), cal.px(), cal.py()
        cc.K.extend([fx, 0.0, px, 0.0, fx, py, 0.0, 0.0, 1.0])
        cc.R.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        cc.P.extend([fx, 0.0, px, 0.0, 0.0, fx, py, 0.0, 0.0, 0.0, 1.0, 0.0])
        channels["calibration"].log(cc.SerializeToString(), log_time=t_ns)
    except Exception:
        pass


def _write_final_result(channels: dict, result: gtsam.Values,
                         images: list[dict], n_landmarks: int) -> None:
    """Write the converged reconstruction at real GPS timestamps.

    - /tf_static : earth → map  (WGS-84 ECEF origin + ENU orientation)
    - /tf        : map  → camera_link  at each image's GPS timestamp
    - /sfm/points       : final landmark cloud at the first image timestamp
    - /sfm/calibration  : final intrinsics at the first image timestamp
    """
    t0_ns = images[0]["t_ns"]
    ts0 = _ns_to_ts(t0_ns)

    # --- Static transform: earth → map ---
    xyz_ecef, R_ecef_enu = _enu_origin_ecef(
        images[0]["lat"], images[0]["lon"], images[0]["alt"]
    )
    ft_static = FrameTransform()
    ft_static.timestamp.CopyFrom(ts0)
    ft_static.parent_frame_id = "earth"
    ft_static.child_frame_id = "map"
    ft_static.translation.x = float(xyz_ecef[0])
    ft_static.translation.y = float(xyz_ecef[1])
    ft_static.translation.z = float(xyz_ecef[2])
    q_earth_map = gtsam.Rot3(R_ecef_enu).toQuaternion()
    ft_static.rotation.w = float(q_earth_map.w())
    ft_static.rotation.x = float(q_earth_map.x())
    ft_static.rotation.y = float(q_earth_map.y())
    ft_static.rotation.z = float(q_earth_map.z())
    channels["tf_static"].log(ft_static.SerializeToString(), log_time=t0_ns)

    # --- Final PointCloud at first image time ---
    pts = []
    for j in range(n_landmarks):
        try:
            pts.append(result.atPoint3(gtsam.symbol('l', j)))
        except Exception:
            continue
    if pts:
        xyz = np.array(pts, dtype=np.float32)
        pc = PointCloud()
        pc.timestamp.CopyFrom(ts0)
        pc.frame_id = "map"
        pc.pose.orientation.w = 1.0
        pc.point_stride = 12
        for name, offset in [("x", 0), ("y", 4), ("z", 8)]:
            f = pc.fields.add()
            f.name = name
            f.offset = offset
            f.type = PackedElementField.FLOAT32
        pc.data = xyz.tobytes()
        channels["points"].log(pc.SerializeToString(), log_time=t0_ns)

    # --- Final CameraCalibration at first image time ---
    try:
        cal = result.atCal3Bundler(gtsam.symbol('k', 0))
        cc = FgCameraCalibration()
        cc.timestamp.CopyFrom(ts0)
        cc.frame_id = "camera_link"
        cc.width = IMAGE_W
        cc.height = IMAGE_H
        cc.distortion_model = "plumb_bob"
        cc.D.extend([cal.k1(), cal.k2(), 0.0, 0.0, 0.0])
        fx, px, py = cal.fx(), cal.px(), cal.py()
        cc.K.extend([fx, 0.0, px, 0.0, fx, py, 0.0, 0.0, 1.0])
        cc.R.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        cc.P.extend([fx, 0.0, px, 0.0, 0.0, fx, py, 0.0, 0.0, 0.0, 1.0, 0.0])
        channels["calibration"].log(cc.SerializeToString(), log_time=t0_ns)
    except Exception:
        pass

    # --- Dynamic transforms: map → camera_link at each image's GPS timestamp ---
    # pose.rotation() = R_cw (camera-from-world); TF wants R_wc (world-from-camera).
    for i, img in enumerate(images):
        t_ns = img["t_ns"]
        try:
            pose = result.atPose3(gtsam.symbol('x', i))
            ft = FrameTransform()
            ft.timestamp.CopyFrom(_ns_to_ts(t_ns))
            ft.parent_frame_id = "map"
            ft.child_frame_id = "camera_link"
            t = pose.translation()
            ft.translation.x = float(t[0])
            ft.translation.y = float(t[1])
            ft.translation.z = float(t[2])
            q = pose.rotation().inverse().toQuaternion()
            ft.rotation.w = float(q.w())
            ft.rotation.x = float(q.x())
            ft.rotation.y = float(q.y())
            ft.rotation.z = float(q.z())
            channels["tf"].log(ft.SerializeToString(), log_time=t_ns)
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading images and GPS data...")
    images = load_images(DATA_DIR)
    ext_e = max(img["enu"][0] for img in images) - min(img["enu"][0] for img in images)
    ext_n = max(img["enu"][1] for img in images) - min(img["enu"][1] for img in images)
    print(f"  {len(images)} images  |  coverage: {ext_e:.0f}m E × {ext_n:.0f}m N")

    fl_mm = images[0]["fl_mm"] or 4.5
    sensor_w_mm = images[0]["sensor_w_mm"]
    fx_init = fl_mm / sensor_w_mm * IMAGE_W
    cx_init, cy_init = IMAGE_W / 2.0, IMAGE_H / 2.0
    print(f"  EXIF focal length: {fl_mm}mm  |  sensor width: {sensor_w_mm:.2f}mm  →  fx₀ ≈ {fx_init:.0f} px")

    print("\nDetecting SIFT features...")
    feats = detect_features(images)

    print("\nMatching GPS-adjacent pairs...")
    pairs = match_pairs(images, feats)
    print(f"  {len(pairs)} pairs with ≥30 inliers")

    print("\nBuilding feature tracks...")
    tracks = build_tracks(pairs, feats)
    print(f"  {len(tracks)} tracks spanning ≥2 images")

    # Initial poses: GPS translation + nadir (downward-looking) rotation
    R_nadir = gtsam.Rot3.Rx(math.pi)
    poses_init = [gtsam.Pose3(R_nadir, gtsam.Point3(*img["enu"])) for img in images]

    K_mat = np.array([[fx_init, 0, cx_init], [0, fx_init, cy_init], [0, 0, 1.0]])

    print("\nTriangulating 3D points...")
    valid_tracks, pts3d = [], []
    for track in tracks:
        pt = triangulate(track, poses_init, K_mat)
        if is_valid(pt, track, poses_init, K_mat):
            valid_tracks.append(track)
            pts3d.append(pt)
    print(f"  {len(valid_tracks)} valid 3D points")

    if len(valid_tracks) < 100:
        raise RuntimeError(f"Only {len(valid_tracks)} valid 3D points — too few for calibration.")

    # Keep the best-observed tracks to stay within memory limits
    max_landmarks = 5000
    if len(valid_tracks) > max_landmarks:
        order = sorted(range(len(valid_tracks)), key=lambda i: -len(valid_tracks[i]))
        valid_tracks = [valid_tracks[i] for i in order[:max_landmarks]]
        pts3d = [pts3d[i] for i in order[:max_landmarks]]
        print(f"  Keeping top {len(valid_tracks)} tracks by observation count")

    cal_init = gtsam.Cal3Bundler(fx_init, 0.0, 0.0, cx_init, cy_init)
    print(f"\nBuilding factor graph and optimising...")
    OUTPUT_MCAP.parent.mkdir(parents=True, exist_ok=True)
    with foxglove.open_mcap(str(OUTPUT_MCAP), allow_overwrite=True):
        ft_schema = _proto_schema(FrameTransform)
        channels = {
            "points": foxglove.Channel(
                "/sfm/points",
                schema=_proto_schema(PointCloud),
                message_encoding="protobuf",
            ),
            "calibration": foxglove.Channel(
                "/sfm/calibration",
                schema=_proto_schema(FgCameraCalibration),
                message_encoding="protobuf",
            ),
            "tf": foxglove.Channel(
                "/tf",
                schema=ft_schema,
                message_encoding="protobuf",
            ),
            "tf_static": foxglove.Channel(
                "/tf_static",
                schema=ft_schema,
                message_encoding="protobuf",
            ),
        }
        graph, initial_values, result = build_and_optimise(
            images, valid_tracks, pts3d, poses_init, cal_init, channels
        )
    print(f"  Wrote {OUTPUT_MCAP}  ({OUTPUT_MCAP.stat().st_size / 1e6:.1f} MB)")

    cal_out = result.atCal3Bundler(gtsam.symbol('k', 0))
    fl_result_mm = cal_out.fx() * sensor_w_mm / IMAGE_W

    print("\n" + "=" * 52)
    print("  Self-Calibration Results")
    print("=" * 52)
    print(f"  fx (focal length)   : {cal_out.fx():.2f} px")
    print(f"  k1 (radial dist 1)  : {cal_out.k1():.6f}")
    print(f"  k2 (radial dist 2)  : {cal_out.k2():.6f}")
    print(f"  cx (principal pt x) : {cal_out.px():.2f} px  (centre: {cx_init:.0f})")
    print(f"  cy (principal pt y) : {cal_out.py():.2f} px  (centre: {cy_init:.0f})")
    print(f"\n  ≈ {fl_result_mm:.2f} mm  (EXIF tag: {fl_mm} mm, sensor width: {sensor_w_mm:.2f} mm)")
    print(f"\n  Initial graph error : {graph.error(initial_values):.2f}")
    print(f"  Final graph error   : {graph.error(result):.2f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
