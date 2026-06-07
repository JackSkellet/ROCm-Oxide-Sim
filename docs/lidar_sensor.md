# LiDAR / Raycast Sensor

Milestone 6A adds a deterministic single-return LiDAR path to the ROCm sensor
renderer. It is a framework-level sensor primitive, not a full automotive
LiDAR simulator.

## Scan Model

`sim-sensors::LidarConfig` defines a spherical scan:

```text
horizontal_samples
vertical_channels
horizontal_fov_degrees
vertical_fov_degrees
min_range_m
max_range_m
pose
```

Horizontal samples sweep yaw and vertical channels sweep pitch around the
sensor pose. The pose uses the transform's local `-Z` direction as forward,
`+X` as right, and `+Y` as up.

Defaults:

```text
512 horizontal samples
32 vertical channels
360 degree horizontal FOV
30 degree vertical FOV
0.1 m minimum range
50.0 m maximum range
identity pose
```

## Output Contract

Each ray produces one return:

```text
range_m: f32
point_xyz: Vec3
object_id: u32
```

Miss/no-return convention:

```text
range_m = 0.0
point_xyz = Vec3::ZERO
object_id = 0
```

Ranges are linear ray distance in meters. Points are world-space hit positions.
Object IDs are copied from `sim-core::Entity::object_id`.

## ROCm Renderer

The ROCm backend uploads `sim-core::Scene` primitives into GPU buffers and
launches a HIPRTC LiDAR kernel over the ray grid. The kernel currently uses
linear first-hit loops over:

- spheres
- planes
- world-space axis-aligned boxes

This matches the RGB/depth/segmentation renderer's supported primitive subset.

## Dataset Layout

When `DatasetConfig.lidar.enabled` is true, `dataset_generator` writes:

```text
lidar_range/frame_000001.f32
lidar_points/frame_000001.xyz
lidar_object_ids/frame_000001.u32
lidar_preview/frame_000001.pgm
```

The manifest records the LiDAR config and convention. Per-frame metadata records
the LiDAR output paths and the same miss convention.

## Current Limitations

- Single return only.
- No noise model.
- No intensity output.
- No rolling scan timing or motion distortion.
- No multi-echo returns.
- No meshes or BVH.
- No robot-relative sensor rig abstraction yet.
- No physics, ROS2, URDF, or OpenUSD integration.
