# Sensor Outputs

Milestone 1 defined three image-like sensor outputs. Later milestones keep the
same contracts while sourcing hits from uploaded `sim-core::Scene` geometry.
Milestone 6A adds a single-return LiDAR/raycast output using the same uploaded
scene buffers. Milestone 6B lets those sensors be mounted on a shared
`SensorRig` and loaded through `ScenarioConfig`.

## RGB

RGB pixels are packed `u32` values in `0x00RRGGBB` order. Dataset previews write
these values as binary PPM (`P6`) files.

## Depth

Depth pixels are `f32` values containing linear distance in meters along the
camera ray. A value of `0.0` means background or miss.

Raw depth files use little-endian `f32` values with no header:

```text
width * height * 4 bytes
```

Depth preview files are binary PGM (`P5`) images. Miss pixels map to black.
Finite positive depths are normalized per frame with nearest samples brighter
and farthest samples dimmer.

## Segmentation

Segmentation pixels are stable `u32` object IDs. `0` means background or miss.
Raw segmentation files use little-endian `u32` values with no header:

```text
width * height * 4 bytes
```

Segmentation preview files are binary PPM (`P6`) images using a stable color
mapping.

## LiDAR

LiDAR frames are deterministic spherical scans. Horizontal samples sweep yaw and
vertical channels sweep pitch around the sensor transform's local `-Z` forward
axis. The current default is:

```text
horizontal_samples = 512
vertical_channels = 32
horizontal_fov_degrees = 360.0
vertical_fov_degrees = 30.0
min_range_m = 0.1
max_range_m = 50.0
```

Each emitted ray stores:

- `ranges_m`: `f32` linear ray distance in meters.
- `points_xyz`: world-space XYZ point for the first hit.
- `object_ids`: stable `u32` object ID for the hit primitive.

Miss/no-return samples use:

```text
range = 0.0
point = 0.0 0.0 0.0
object_id = 0
```

Dataset LiDAR files are:

```text
lidar_range/frame_000001.f32          raw little-endian f32 values
lidar_points/frame_000001.xyz         text XYZ rows, one point per ray
lidar_object_ids/frame_000001.u32     raw little-endian u32 values
lidar_preview/frame_000001.pgm        per-frame normalized range preview
```

The preview maps misses to black and finite positive returns from bright
near-range values to dim farther-range values.

## Object IDs

Segmentation IDs come from each `sim-core::Entity` `ObjectId`. Metadata is
generated from the active scene:

- `0` is always background/miss.
- Nonzero IDs are stable per scene and are copied into the segmentation buffer
  on hit.
- Object metadata includes the entity label plus primitive and material kind
  when it comes from a `sim-core::Scene`.
- The default scene keeps the original smoke-test IDs:

```text
0 = background
1 = ground
2 = red sphere
3 = green sphere
4 = blue sphere
```

`examples/scenes/boxes_scene.json` includes object IDs for axis-aligned boxes:

```text
5 = red box
6 = metal preview box
7 = blue emissive sphere
```

The renderer currently supports uploaded sphere, plane, and axis-aligned box
primitives. Box rotation is ignored. Meshes, OpenUSD assets, URDF robot
geometry, dynamic GPU scene mutation, LiDAR noise, multi-return LiDAR, and
rolling scan timing are not implemented yet.

Dataset metadata records the active scene object ID map for every generated
frame and repeats the depth/segmentation conventions in `dataset_manifest.json`
for reproducible downstream parsing.

Scenario datasets also record rig name, stable sensor mount names, mount
transforms, and world sensor transforms in the manifest and per-frame metadata.
