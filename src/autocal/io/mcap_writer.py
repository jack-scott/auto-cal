"""
MCAP write helpers.

Wraps the foxglove SDK (foxglove.open_mcap / foxglove.Channel) with:
  - Automatic FileDescriptorSet schema registration (fixes the wire-type-6
    error that occurs when passing a bare FileDescriptorProto).
  - A channel registry so the same topic is never registered twice.
  - A helper to convert int nanoseconds → google.protobuf.Timestamp.

Usage::

    from autocal.io.mcap_writer import McapWriter

    with McapWriter("out.mcap") as w:
        w.write("/camera/image", img_msg, t_ns)
        w.write("/gps/fix", fix_msg, t_ns)

foxglove SDK notes:
  foxglove.open_mcap(path, allow_overwrite=True) returns a context manager.
  foxglove.Channel(topic, schema=Schema, message_encoding="protobuf") creates
    a channel inside the active open_mcap context.
  channel.log(bytes, log_time=ns) writes one message.

Schema encoding:
  MCAP protobuf channels require a FileDescriptorSet (not a bare
  FileDescriptorProto) as the schema data.  make_proto_schema collects all
  transitive file dependencies before serialising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import foxglove
from google.protobuf import descriptor_pb2
from google.protobuf.timestamp_pb2 import Timestamp


def make_proto_schema(msg_class: type) -> foxglove.Schema:
    """Build a foxglove.Schema from a protobuf message class.

    Collects all transitive FileDescriptorProto dependencies into a
    FileDescriptorSet so Foxglove Studio can decode the messages.

    Args:
        msg_class: A protobuf message class (not an instance).

    Returns:
        foxglove.Schema with encoding="protobuf".
    """
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


class McapWriter:
    """Context manager that writes protobuf messages to an MCAP file.

    Channels are registered automatically on first write to each topic.
    The schema is inferred from the message class.

    Example::

        with McapWriter("out.mcap") as w:
            w.write("/tf", frame_transform_msg, t_ns)

    Args:
        path:            Output file path.
        allow_overwrite: If False, raise if the file already exists.
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
        """Serialise and write one protobuf message.

        Args:
            topic: MCAP topic string (e.g. "/tf").
            msg:   Protobuf message instance.
            t_ns:  Log timestamp in Unix nanoseconds.
        """
        if topic not in self._channels:
            self._channels[topic] = foxglove.Channel(
                topic,
                schema=make_proto_schema(type(msg)),
                message_encoding="protobuf",
            )
        self._channels[topic].log(msg.SerializeToString(), log_time=t_ns)

    def write_json(self, topic: str, fields: dict[str, str], msg: dict, t_ns: int) -> None:
        """Write a JSON-encoded message — use for numeric values Foxglove can plot.

        The channel is registered once using a minimal JSON Schema built from
        *fields*, then each call serialises *msg* as JSON bytes.

        Args:
            topic:  MCAP topic string.
            fields: {field_name: json_schema_type} for schema registration,
                    e.g. {"position_error_m": "number"}.  Only used on the
                    first call per topic.
            msg:    Message data as a plain dict.
            t_ns:   Log timestamp in Unix nanoseconds.
        """
        import json
        if topic not in self._channels:
            schema_doc = {
                "type": "object",
                "properties": {k: {"type": v} for k, v in fields.items()},
            }
            self._channels[topic] = foxglove.Channel(
                topic,
                schema=foxglove.Schema(
                    name=topic.lstrip("/").replace("/", "_"),
                    encoding="jsonschema",
                    data=json.dumps(schema_doc).encode(),
                ),
                message_encoding="json",
            )
        self._channels[topic].log(json.dumps(msg).encode(), log_time=t_ns)
