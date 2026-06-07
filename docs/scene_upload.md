# Scene Upload

Milestone 3 makes the ROCm renderer consume uploaded `sim-core::Scene` data
instead of hardcoded kernel geometry.

## Supported Geometry

The current uploaded primitive subset is intentionally small:

- `PrimitiveShape::Sphere`
- `PrimitiveShape::Plane`

`PrimitiveShape::Box` is present in `sim-core`, but the ROCm renderer rejects it
for now with an unsupported-primitive error. Triangle meshes, BVH acceleration,
OpenUSD, URDF, physics bodies, LiDAR, and ROS2 integration are not part of this
milestone.

## GPU Buffer Layout

`sim-render-rocm` defines `#[repr(C)]` GPU ABI structs:

```text
GpuVec3      x, y, z, _pad                 16 bytes
GpuSphere    center, radius, material_id, object_id, pad
GpuPlane     point, normal, material_id, object_id, pads
GpuMaterial  base_color, emission, roughness, material_kind, pads
```

`GpuVec3` always uses four `f32` values to avoid Rust/C `float3` layout
surprises. The HIPRTC kernel declares matching structs and the Rust tests assert
the expected ABI sizes.

## Upload Path

The host conversion is:

```text
sim_core::Scene
  -> HostSceneBuffers
  -> RocmScene {
       DeviceBuffer<GpuSphere>,
       DeviceBuffer<GpuPlane>,
       DeviceBuffer<GpuMaterial>,
       counts
     }
```

Each entity contributes one material record. Sphere centers come from the
entity transform translation, with radius scaled by the largest absolute scale
component. Plane normals are transformed and normalized, and plane points are
derived from the local plane offset and entity transform.

## Object IDs

Segmentation IDs are copied from each entity's `ObjectId`:

- `0` is background/miss.
- Nonzero IDs are written into the segmentation frame on hit.
- Metadata is generated from the active scene, not from a hardcoded table.

The default smoke scene still uses:

```text
1 ground
2 red sphere
3 green sphere
4 blue sphere
```

## Kernel Approach

The HIPRTC kernel receives sphere, plane, and material buffers plus a compact
render parameter struct. It performs simple linear intersection loops and writes
RGB, linear ray-distance depth, and segmentation in one pass.

There is no acceleration structure, mesh path, dynamic GPU mutation API, or
zero-copy presentation path yet.
