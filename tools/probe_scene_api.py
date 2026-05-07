"""Probe foxglove SceneUpdate proto API."""
import sys

# List available scene-related modules
import foxglove_schemas_protobuf
import pkgutil, importlib
mods = [m.name for m in pkgutil.iter_modules(foxglove_schemas_protobuf.__path__) if 'scene' in m.name.lower() or 'primitive' in m.name.lower() or 'line' in m.name.lower() or 'arrow' in m.name.lower() or 'sphere' in m.name.lower() or 'color' in m.name.lower() or 'pose' in m.name.lower() or 'point' in m.name.lower() or 'vector' in m.name.lower() or 'quat' in m.name.lower() or 'duration' in m.name.lower()]
print("Relevant modules:")
for m in sorted(mods):
    print(f"  {m}")

print()

# Check SceneUpdate fields
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
print("SceneUpdate fields:", list(SceneUpdate.DESCRIPTOR.fields_by_name.keys()))

from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
print("SceneEntity fields:", list(SceneEntity.DESCRIPTOR.fields_by_name.keys()))

from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive
print("LinePrimitive fields:", list(LinePrimitive.DESCRIPTOR.fields_by_name.keys()))
print("LinePrimitive.Type:", {v.name: v.number for v in LinePrimitive.DESCRIPTOR.enum_types_by_name['Type'].values})

from foxglove_schemas_protobuf.ArrowPrimitive_pb2 import ArrowPrimitive
print("ArrowPrimitive fields:", list(ArrowPrimitive.DESCRIPTOR.fields_by_name.keys()))

from foxglove_schemas_protobuf.SpherePrimitive_pb2 import SpherePrimitive
print("SpherePrimitive fields:", list(SpherePrimitive.DESCRIPTOR.fields_by_name.keys()))

from foxglove_schemas_protobuf.Color_pb2 import Color
print("Color fields:", list(Color.DESCRIPTOR.fields_by_name.keys()))

from foxglove_schemas_protobuf.Point3_pb2 import Point3
print("Point3 fields:", list(Point3.DESCRIPTOR.fields_by_name.keys()))

from foxglove_schemas_protobuf.Pose_pb2 import Pose
print("Pose fields:", list(Pose.DESCRIPTOR.fields_by_name.keys()))

# Check duration type
from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
e = SceneEntity()
print("lifetime type:", type(e.lifetime).__name__, list(e.lifetime.DESCRIPTOR.fields_by_name.keys()))
