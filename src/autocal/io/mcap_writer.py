"""
MCAP write helpers.

Wraps the foxglove SDK (foxglove.open_mcap / foxglove.Channel).

A single write(topic, msg, t_ns) method handles three message kinds:

  Protobuf   any generated protobuf instance (has a DESCRIPTOR attribute)
             Schema inferred from the generated class via make_proto_schema.

  Dataclass  any dataclass instance
             Schema inferred from type annotations (float→"number",
             int→"integer", str→"string", bool→"boolean").

  dict       plain Python dict — for pass-through of JSON messages that
             were read from another MCAP.  Schema inferred from value types
             on first write to each topic.

Usage::

    from autocal.io.mcap_writer import McapWriter

    with McapWriter("out.mcap") as w:
        w.write("/tf", frame_transform_msg, t_ns)            # protobuf
        w.write("/ape", ApePoseMsg(translation_m=0.01), t_ns) # dataclass
"""

from __future__ import annotations

import dataclasses
import json
import typing
from pathlib import Path
from typing import Any

import foxglove
from google.protobuf import descriptor_pb2
from google.protobuf.timestamp_pb2 import Timestamp


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_JSON_TYPE_MAP: dict[type, str] = {
    float: "number",
    int:   "integer",
    str:   "string",
    bool:  "boolean",
}


def _schema_from_dataclass(cls: type) -> foxglove.Schema:
    hints = typing.get_type_hints(cls)
    properties = {
        name: {"type": _JSON_TYPE_MAP.get(tp, "string")}
        for name, tp in hints.items()
    }
    return foxglove.Schema(
        name=cls.__name__,
        encoding="jsonschema",
        data=json.dumps({"type": "object", "properties": properties}).encode(),
    )


def _schema_from_dict(topic: str, data: dict) -> foxglove.Schema:
    properties = {
        k: {"type": _JSON_TYPE_MAP.get(type(v), "string")}
        for k, v in data.items()
    }
    return foxglove.Schema(
        name=topic.lstrip("/").replace("/", "_"),
        encoding="jsonschema",
        data=json.dumps({"type": "object", "properties": properties}).encode(),
    )


def make_proto_schema(msg_class: type) -> foxglove.Schema:
    """Build a foxglove.Schema from a protobuf message class."""
    desc = msg_class.DESCRIPTOR
    fds = descriptor_pb2.FileDescriptorSet()
    seen: set[str] = set()

    def _collect(fd: Any) -> None:
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


def ns_to_timestamp(ns: int) -> Timestamp:
    """Convert Unix nanoseconds to a google.protobuf.Timestamp."""
    ts = Timestamp()
    ts.seconds = ns // 1_000_000_000
    ts.nanos = ns % 1_000_000_000
    return ts


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class McapWriter:
    """Context manager that writes messages to an MCAP file.

    A single write(topic, msg, t_ns) handles all message kinds:
      - Protobuf instances  (have a DESCRIPTOR attribute)
      - Dataclass instances (schema inferred from type annotations)
      - Plain dicts         (schema inferred from value types; for pass-through)
    """

    def __init__(self, path: str | Path, allow_overwrite: bool = True) -> None:
        self._path = str(path)
        self._allow_overwrite = allow_overwrite
        self._ctx = None
        self._channels: dict[str, foxglove.Channel] = {}

    def __enter__(self) -> "McapWriter":
        self._ctx = foxglove.open_mcap(self._path, allow_overwrite=self._allow_overwrite)
        self._ctx.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(*args)

    def write(self, topic: str, msg: Any, t_ns: int) -> None:
        """Write one message to the given topic.

        Args:
            topic: MCAP topic string.
            msg:   Protobuf instance, dataclass instance, or plain dict.
            t_ns:  Log timestamp in Unix nanoseconds.
        """
        if hasattr(msg, "DESCRIPTOR"):
            # Protobuf
            if topic not in self._channels:
                self._channels[topic] = foxglove.Channel(
                    topic,
                    schema=make_proto_schema(type(msg)),
                    message_encoding="protobuf",
                )
            self._channels[topic].log(msg.SerializeToString(), log_time=t_ns)

        elif dataclasses.is_dataclass(msg) and not isinstance(msg, type):
            # Dataclass → JSON, schema from type annotations
            if topic not in self._channels:
                self._channels[topic] = foxglove.Channel(
                    topic,
                    schema=_schema_from_dataclass(type(msg)),
                    message_encoding="json",
                )
            self._channels[topic].log(
                json.dumps(dataclasses.asdict(msg)).encode(), log_time=t_ns
            )

        elif isinstance(msg, dict):
            # Plain dict → JSON, schema inferred from value types on first write
            if topic not in self._channels:
                self._channels[topic] = foxglove.Channel(
                    topic,
                    schema=_schema_from_dict(topic, msg),
                    message_encoding="json",
                )
            self._channels[topic].log(json.dumps(msg).encode(), log_time=t_ns)

        else:
            raise TypeError(f"Unsupported message type for topic {topic!r}: {type(msg)}")
