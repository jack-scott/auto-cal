"""
MCAP read helpers.

Wraps the mcap + mcap_protobuf libraries:
  make_reader(stream, decoder_factories=[DecoderFactory()])
  reader.iter_decoded_messages() → (schema, channel, message, proto_msg)
  reader.get_summary()           → Summary with .channels, .schemas, .statistics

Usage::

    from autocal.io.mcap_reader import iter_messages, get_topic_map, build_tf_tree

    # Iterate all messages on two topics
    for topic, t_ns, msg in iter_messages("input.mcap", topics=["/gps/fix"]):
        print(topic, t_ns, msg.latitude, msg.longitude)

    # Discover what topics exist without reading messages
    topic_map = get_topic_map("input.mcap")  # {"/camera/image": "foxglove.CompressedImage", ...}

    # Build a TFTree from all transform messages
    tf_tree = build_tf_tree("input.mcap")

mcap package notes:
  make_reader returns an McapReader; call within an open file context.
  iter_decoded_messages() yields 4-tuples (schema, channel, message, decoded).
    schema.name  = e.g. "foxglove.CompressedImage"
    channel.topic = e.g. "/camera/image"
    message.log_time = Unix nanoseconds (int)
    decoded      = the decoded protobuf message object
  get_summary() reads only the index (no message data) and returns a Summary
    with .channels (dict[int, Channel]) and .schemas (dict[int, Schema]).
"""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from autocal.frames.tf_tree import TFTree


def _decode_array(s: str, shape: tuple) -> np.ndarray:
    """Decode a base64+zlib compressed float32 array produced by _encode_array."""
    return np.frombuffer(
        zlib.decompress(base64.b64decode(s)), dtype=np.float32
    ).reshape(shape)

# Known FrameTransform schema names in Foxglove protobuf encoding
_TF_SCHEMA_NAMES = {"foxglove.FrameTransform"}


def iter_messages(
    path: str | Path,
    topics: list[str] | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> Iterator[tuple[str, int, Any]]:
    """Iterate decoded protobuf messages from an MCAP file.

    Args:
        path:     Path to the MCAP file.
        topics:   If given, only yield messages on these topics.
        start_ns: Only yield messages at or after this Unix nanosecond time.
        end_ns:   Only yield messages at or before this Unix nanosecond time.

    Yields:
        (topic, log_time_ns, decoded_proto_msg) in log-time order.
    """
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, proto_msg in reader.iter_decoded_messages(
            topics=topics,
            start_time=start_ns,
            end_time=end_ns,
        ):
            yield channel.topic, message.log_time, proto_msg


def get_topic_map(path: str | Path) -> dict[str, str]:
    """Return a mapping of topic → schema name without reading any messages.

    Uses the MCAP summary/index only — O(1) in message count.

    Args:
        path: Path to the MCAP file.

    Returns:
        dict mapping topic string to schema name, e.g.
        {"/camera/image": "foxglove.CompressedImage", ...}
    """
    with open(path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary is None:
            return {}
        schema_map = {s.id: s.name for s in summary.schemas.values()}
        return {
            ch.topic: schema_map.get(ch.schema_id, "")
            for ch in summary.channels.values()
        }


def load_sift_features(
    path: str | Path,
    topic: str = "/camera/sift_features",
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load cached SIFT features from an MCAP file.

    Reads raw message bytes directly (bypassing the protobuf decoder) and
    decodes the base64+zlib compressed arrays written by McapWriter.write_sift_features.

    Args:
        path:  Path to the MCAP file.
        topic: Topic name for SIFT features.

    Returns:
        dict mapping log_time_ns → (kps shape-(N,2) float32, descs shape-(N,128) float32).
        Empty dict if the topic is not present.
    """
    features: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with open(path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=[topic]):
            data = json.loads(bytes(message.data))
            n = data["n"]
            kps   = _decode_array(data["kps"],  (n, 2))
            descs = _decode_array(data["desc"],  (n, 128))
            features[message.log_time] = (kps, descs)
    return features


def build_tf_tree(path: str | Path) -> TFTree:
    """Read all /tf and /tf_static messages and return a populated TFTree.

    Args:
        path: Path to the MCAP file.

    Returns:
        TFTree populated with all FrameTransform messages found.
    """
    tree = TFTree()
    for topic, t_ns, msg in iter_messages(path, topics=["/tf", "/tf_static"]):
        tree.add(msg, t_ns)
    return tree
