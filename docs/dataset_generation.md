# Dataset Generation

The dataset writer keeps formats simple and dependency-light. The current layout
is:

```text
dataset/
  rgb/frame_000001.ppm
  depth/frame_000001.f32
  depth_preview/frame_000001.pgm
  segmentation/frame_000001.u32
  segmentation_preview/frame_000001.ppm
  metadata/frame_000001.json
  dataset_manifest.json
```

Generate a ROCm-backed dataset:

```bash
cargo run -p dataset_generator --features rocm -- --frames 16 --out target/sim_dataset
cargo run -p dataset_generator --features rocm -- --frames 4 --out target/sim_dataset --scene examples/scenes/basic_scene.json
```

Generate a CPU-preview smoke dataset:

```bash
cargo run -p dataset_generator -- --frames 4 --out target/sim_dataset
```

## Metadata

Each per-frame metadata JSON includes:

- Frame index and timestamp.
- Sensor ID.
- Depth convention: linear camera ray distance in meters, `0.0` for misses.
- Output file paths and formats.
- Active scene object ID mapping.

## Current Limitations

- The ROCm renderer uploads only sphere and plane primitives today.
- Box, mesh, BVH, OpenUSD, and URDF dataset geometry paths are deferred.
- COCO, KITTI, ROS2 bags, and OpenUSD exports are not implemented.
- Depth and segmentation are single-camera outputs, not LiDAR or physics output.
