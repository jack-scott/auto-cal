"""Tests for autocal.io.mcap_writer, mcap_reader, and autocal.frames.tf_tree."""

import math

import numpy as np
import pytest

from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix

from autocal.frames.tf_tree import TFTree, Transform
from autocal.io.mcap_reader import build_tf_tree, get_topic_map, iter_messages
from autocal.io.mcap_writer import McapWriter, ns_to_timestamp


# ---------------------------------------------------------------------------
# ns_to_timestamp
# ---------------------------------------------------------------------------

def test_ns_to_timestamp_zero():
    ts = ns_to_timestamp(0)
    assert ts.seconds == 0
    assert ts.nanos == 0


def test_ns_to_timestamp_round_trip():
    ns = 1_413_742_851_123_456_789
    ts = ns_to_timestamp(ns)
    recovered = ts.seconds * 1_000_000_000 + ts.nanos
    assert recovered == ns


# ---------------------------------------------------------------------------
# McapWriter + iter_messages round-trip
# ---------------------------------------------------------------------------

def _make_fix(t_ns: int, lat: float, lon: float) -> LocationFix:
    fix = LocationFix()
    fix.timestamp.CopyFrom(ns_to_timestamp(t_ns))
    fix.frame_id = "camera_link"
    fix.latitude = lat
    fix.longitude = lon
    fix.altitude = 100.0
    return fix


def test_mcap_round_trip_single_message(tmp_path):
    path = tmp_path / "test.mcap"
    t_ns = 1_000_000_000
    fix = _make_fix(t_ns, 37.7749, -122.4194)

    with McapWriter(path) as w:
        w.write("/gps/fix", fix, t_ns)

    msgs = list(iter_messages(path, topics=["/gps/fix"]))
    assert len(msgs) == 1
    topic, t, msg = msgs[0]
    assert topic == "/gps/fix"
    assert t == t_ns
    assert abs(msg.latitude - 37.7749) < 1e-9
    assert abs(msg.longitude - -122.4194) < 1e-9


def test_mcap_round_trip_multiple_topics(tmp_path):
    path = tmp_path / "test.mcap"
    t1, t2 = 1_000_000_000, 2_000_000_000

    fix = _make_fix(t1, 37.0, -122.0)
    img = CompressedImage()
    img.timestamp.CopyFrom(ns_to_timestamp(t2))
    img.frame_id = "camera_link"
    img.format = "jpeg"
    img.data = b"\xff\xd8\xff"  # minimal JPEG header bytes

    with McapWriter(path) as w:
        w.write("/gps/fix", fix, t1)
        w.write("/camera/image", img, t2)

    all_msgs = list(iter_messages(path))
    assert len(all_msgs) == 2
    topics = {m[0] for m in all_msgs}
    assert topics == {"/gps/fix", "/camera/image"}


def test_mcap_topic_filter(tmp_path):
    path = tmp_path / "test.mcap"
    with McapWriter(path) as w:
        w.write("/gps/fix", _make_fix(1_000_000_000, 0.0, 0.0), 1_000_000_000)
        w.write("/gps/fix", _make_fix(2_000_000_000, 1.0, 0.0), 2_000_000_000)
        w.write("/gps/fix", _make_fix(3_000_000_000, 2.0, 0.0), 3_000_000_000)

    msgs = list(iter_messages(path, topics=["/gps/fix"]))
    assert len(msgs) == 3
    assert all(t == "/gps/fix" for t, _, _ in msgs)


def test_mcap_time_filter(tmp_path):
    path = tmp_path / "test.mcap"
    for i in range(5):
        t = (i + 1) * 1_000_000_000
        with McapWriter(path, allow_overwrite=(i > 0)) as w:
            pass  # just test the filter logic after writing

    # Write all 5 in one writer, then filter
    path2 = tmp_path / "test2.mcap"
    with McapWriter(path2) as w:
        for i in range(5):
            t = (i + 1) * 1_000_000_000
            w.write("/gps/fix", _make_fix(t, float(i), 0.0), t)

    # Only messages in [2s, 4s]
    msgs = list(iter_messages(path2, start_ns=2_000_000_000, end_ns=4_000_000_000))
    assert len(msgs) == 3


def test_get_topic_map(tmp_path):
    path = tmp_path / "test.mcap"
    with McapWriter(path) as w:
        w.write("/gps/fix", _make_fix(1_000_000_000, 0.0, 0.0), 1_000_000_000)
        img = CompressedImage()
        img.frame_id = "camera_link"
        w.write("/camera/image", img, 1_000_000_000)

    topic_map = get_topic_map(path)
    assert "/gps/fix" in topic_map
    assert topic_map["/gps/fix"] == "foxglove.LocationFix"
    assert "/camera/image" in topic_map
    assert topic_map["/camera/image"] == "foxglove.CompressedImage"


# ---------------------------------------------------------------------------
# TFTree
# ---------------------------------------------------------------------------

def _make_tf(parent: str, child: str, t_ns: int,
             tx: float, ty: float, tz: float,
             qx: float = 0.0, qy: float = 0.0,
             qz: float = 0.0, qw: float = 1.0) -> FrameTransform:
    ft = FrameTransform()
    ft.timestamp.CopyFrom(ns_to_timestamp(t_ns))
    ft.parent_frame_id = parent
    ft.child_frame_id = child
    ft.translation.x = tx
    ft.translation.y = ty
    ft.translation.z = tz
    ft.rotation.x = qx
    ft.rotation.y = qy
    ft.rotation.z = qz
    ft.rotation.w = qw
    return ft


def test_tf_tree_known_frames():
    tree = TFTree()
    tree.add(_make_tf("map", "camera_link", 0, 0, 0, 0), 0)
    tree.add(_make_tf("earth", "map", 0, 0, 0, 0), 0)
    assert "map" in tree.known_frames()
    assert "camera_link" in tree.known_frames()
    assert "earth" in tree.known_frames()


def test_tf_tree_static_lookup():
    """Single-entry edge (static TF) returns same value at any t_ns."""
    tree = TFTree()
    tree.add(_make_tf("map", "camera_link", 5_000_000_000, 1.0, 2.0, 3.0), 5_000_000_000)
    tf = tree.lookup_transform("map", "camera_link", 0)
    assert np.allclose(tf.translation, [1.0, 2.0, 3.0])
    tf2 = tree.lookup_transform("map", "camera_link", 99_999_999_999)
    assert np.allclose(tf2.translation, [1.0, 2.0, 3.0])


def test_tf_tree_translation_interpolation():
    tree = TFTree()
    tree.add(_make_tf("map", "cam", 0, 0.0, 0.0, 0.0), 0)
    tree.add(_make_tf("map", "cam", 2_000_000_000, 2.0, 0.0, 0.0), 2_000_000_000)
    tf = tree.lookup_transform("map", "cam", 1_000_000_000)
    assert np.allclose(tf.translation, [1.0, 0.0, 0.0], atol=1e-9)


def test_tf_tree_interpolation_endpoints():
    tree = TFTree()
    tree.add(_make_tf("map", "cam", 0, 0.0, 0.0, 0.0), 0)
    tree.add(_make_tf("map", "cam", 1_000_000_000, 1.0, 0.0, 0.0), 1_000_000_000)
    tf0 = tree.lookup_transform("map", "cam", 0)
    tf1 = tree.lookup_transform("map", "cam", 1_000_000_000)
    assert np.allclose(tf0.translation, [0.0, 0.0, 0.0])
    assert np.allclose(tf1.translation, [1.0, 0.0, 0.0])


def test_tf_tree_unknown_edge_raises():
    tree = TFTree()
    tree.add(_make_tf("map", "cam", 0, 0, 0, 0), 0)
    with pytest.raises(KeyError):
        tree.lookup_transform("map", "nonexistent", 0)


def test_tf_tree_out_of_range_raises():
    tree = TFTree()
    tree.add(_make_tf("map", "cam", 1_000_000_000, 0, 0, 0), 1_000_000_000)
    tree.add(_make_tf("map", "cam", 3_000_000_000, 1, 0, 0), 3_000_000_000)
    with pytest.raises(ValueError, match="outside buffered range"):
        tree.lookup_transform("map", "cam", 0)
    with pytest.raises(ValueError, match="outside buffered range"):
        tree.lookup_transform("map", "cam", 4_000_000_000)


def test_tf_tree_rotation_slerp():
    """SLERP between identity and 180° rotation should give 90° at alpha=0.5."""
    # 180° rotation around Z: quat [0, 0, sin(90°), cos(90°)] = [0,0,1,0]
    tree = TFTree()
    tree.add(_make_tf("map", "cam", 0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0), 0)
    tree.add(_make_tf("map", "cam", 2_000_000_000, 0, 0, 0,
                      0.0, 0.0, 1.0, 0.0), 2_000_000_000)  # 180° around Z
    tf = tree.lookup_transform("map", "cam", 1_000_000_000)
    # Midpoint should be 90° around Z: [0, 0, sin(45°), cos(45°)]
    s = math.sin(math.pi / 4)
    expected = np.array([0.0, 0.0, s, s])
    assert np.allclose(np.abs(tf.rotation), np.abs(expected), atol=1e-6)


def test_tf_tree_transform_point_identity():
    tree = TFTree()
    tree.add(_make_tf("map", "cam", 0, 5.0, 0.0, 0.0), 0)
    # Point at [0,0,0] in camera_link → [5,0,0] in map
    pt = tree.transform_point(np.array([0.0, 0.0, 0.0]), "cam", "map", 0)
    assert np.allclose(pt, [5.0, 0.0, 0.0], atol=1e-9)


def test_build_tf_tree_from_mcap(tmp_path):
    path = tmp_path / "tf.mcap"
    with McapWriter(path) as w:
        w.write("/tf", _make_tf("map", "cam", 1_000_000_000, 1.0, 2.0, 3.0), 1_000_000_000)
        w.write("/tf_static", _make_tf("earth", "map", 0, 0.0, 0.0, 0.0), 0)

    tree = build_tf_tree(path)
    assert "map" in tree.known_frames()
    assert "cam" in tree.known_frames()
    assert "earth" in tree.known_frames()
