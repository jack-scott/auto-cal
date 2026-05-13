"""
MCAP read helpers.

A single iter_messages(path, topics, ...) handles both protobuf and JSON
channels, mirroring how McapWriter.write handles both in a single call.

  Protobuf channels  → yields the decoded protobuf message object
  JSON channels      → yields a plain dict (json.loads of the raw bytes)

Usage::

    from autocal.io.mcap_reader import iter_messages, get_topic_map

    for topic, t_ns, msg in iter_messages("input.mcap", topics=["/gps/fix"]):
        print(topic, t_ns, msg.latitude, msg.longitude)  # protobuf field access

    for topic, t_ns, msg in iter_messages("input.mcap", topics=["/camera/sift_features"]):
        kps = decode_array(msg["kps"], (msg["n"], 2))     # dict field access

Implementation note
-------------------
iter_messages reads the file index to split topics by encoding, then opens a
second pass for each encoding and merges the two streams via heapq.merge.
MCAP guarantees messages are written in log-time order so merging two monotone
streams is correct and O(N).
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any, Iterator

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from autocal.frames.tf_tree import TFTree


# ---------------------------------------------------------------------------
# Message registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def register_message(cls: type) -> type:
    """Register a dataclass as a known JSON message schema.

    Use as a decorator on the dataclass definition.  When iter_messages
    encounters a JSON channel whose schema name matches cls.__name__, it
    instantiates cls(**dict) so callers get typed attribute access instead
    of a plain dict.

        @register_message
        @dataclasses.dataclass
        class MyMsg:
            value: float
    """
    _REGISTRY[cls.__name__] = cls
    return cls


def iter_messages(
    path: str | Path,
    topics: list[str] | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> Iterator[tuple[str, int, Any]]:
    """Iterate messages from an MCAP file.

    Handles both protobuf and JSON-encoded channels transparently.

    Args:
        path:     Path to the MCAP file.
        topics:   If given, only yield messages on these topics.
        start_ns: Only yield messages at or after this Unix nanosecond timestamp.
        end_ns:   Only yield messages at or before this Unix nanosecond timestamp.

    Yields:
        (topic, log_time_ns, msg) where msg is a decoded protobuf object for
        protobuf channels, or a plain dict for JSON channels.
    """
    # Read the index to classify every topic by encoding.
    json_topics: set[str] = set()
    all_topics: set[str] = set()
    schema_name_map: dict[int, str] = {}
    with open(path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary is not None:
            schema_enc  = {s.id: s.encoding for s in summary.schemas.values()}
            schema_name_map = {s.id: s.name for s in summary.schemas.values()}
            for ch in summary.channels.values():
                all_topics.add(ch.topic)
                if schema_enc.get(ch.schema_id) == "jsonschema":
                    json_topics.add(ch.topic)

    # Split requested topics into the two encoding buckets.
    requested = set(topics) if topics is not None else all_topics
    proto_filter = [t for t in requested if t not in json_topics]
    json_filter  = [t for t in requested if t in json_topics]

    def _proto() -> Iterator[tuple[int, str, Any]]:
        if not proto_filter:
            return
        with open(path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, channel, message, decoded in reader.iter_decoded_messages(
                topics=proto_filter, start_time=start_ns, end_time=end_ns
            ):
                yield message.log_time, channel.topic, decoded

    def _json() -> Iterator[tuple[int, str, Any]]:
        if not json_filter:
            return
        with open(path, "rb") as f:
            reader = make_reader(f)
            for _, channel, message in reader.iter_messages(
                topics=json_filter, start_time=start_ns, end_time=end_ns
            ):
                raw = json.loads(bytes(message.data))
                name = schema_name_map.get(channel.schema_id, "")
                cls = _REGISTRY.get(name)
                yield message.log_time, channel.topic, cls(**raw) if cls else raw

    for t_ns, topic, msg in heapq.merge(_proto(), _json(), key=lambda x: x[0]):
        yield topic, t_ns, msg


def get_topic_map(path: str | Path) -> dict[str, str]:
    """Return topic → schema name using the MCAP index only (no message reads)."""
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


def build_tf_tree(path: str | Path) -> TFTree:
    """Read all /tf and /tf_static messages and return a populated TFTree."""
    tree = TFTree()
    for topic, t_ns, msg in iter_messages(path, topics=["/tf", "/tf_static"]):
        tree.add(msg, t_ns)
    return tree
