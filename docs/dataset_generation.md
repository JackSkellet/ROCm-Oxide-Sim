# Dataset Generation

Milestone 4 turned `apps/dataset_generator` into a reproducible synthetic
dataset pipeline for RGB/depth/segmentation camera outputs. Milestone 6A adds
optional single-return LiDAR/raycast outputs that use the same uploaded
`sim-core::Scene` geometry as the camera renderer. Milestone 6B adds scenario
files that combine a scene path, sensor rig, and dataset job.

## Quickstart

Generate four frames with the default scene and camera settings:

```bash
cargo run -p dataset_generator --features rocm -- --frames 4 --out target/sim_dataset --overwrite
```

Generate from a config file while overriding the output directory:

```bash
cargo run -p dataset_generator --features rocm -- \
  --config examples/datasets/basic_orbit.json \
  --out target/sim_dataset_config \
  --overwrite
```

Generate a dataset from the box/material fixture:

```bash
cargo run -p dataset_generator --features rocm -- \
  --scene examples/scenes/boxes_scene.json \
  --frames 4 \
  --out target/boxes_dataset \
  --overwrite
```

Generate a deterministic randomized dataset:

```bash
cargo run -p dataset_generator --features rocm -- \
  --config examples/datasets/randomized_boxes.json \
  --out target/randomized_boxes \
  --overwrite
```

Generate a deterministic randomized box dataset with LiDAR:

```bash
cargo run -p dataset_generator --features rocm -- \
  --config examples/datasets/randomized_boxes_lidar.json \
  --out target/lidar_dataset \
  --overwrite
```

Generate from a shared scenario:

```bash
cargo run -p dataset_generator --features rocm -- \
  --scenario examples/scenarios/basic_sensor_rig.json \
  --out target/scenario_dataset \
  --overwrite
```

Validate a generated dataset:

```bash
cargo run -p dataset_generator --features rocm -- validate --dataset target/sim_dataset
```

Preview a run without rendering or writing files:

```bash
cargo run -p dataset_generator -- --camera-path random --seed 1234 --frames 4 --dry-run
```

## CLI Options

Generation accepts:

```text
--scene <PATH>
--out <DIR>
--frames <N>
--width <N>
--height <N>
--camera-path static|orbit|line|random
--seed <u64>
--config <PATH>
--scenario <PATH>
--randomize
--overwrite
--dry-run
```

Existing simple usage still works:

```bash
cargo run -p dataset_generator --features rocm -- --frames 4 --out target/sim_dataset --overwrite
```

The generator refuses to write into a non-empty output directory unless
`--overwrite` is passed.

## Config Files

Config files are JSON and map to `DatasetConfig`:

```json
{
  "scene_path": "examples/scenes/basic_scene.json",
  "output_dir": "target/sim_dataset_config",
  "frame_count": 16,
  "width": 640,
  "height": 360,
  "camera_path": {
    "kind": "orbit",
    "target": { "x": 0.0, "y": 0.55, "z": -1.45 },
    "radius": 4.1,
    "height": 1.35,
    "start_angle_degrees": 0.0,
    "end_angle_degrees": 360.0,
    "fov_y_degrees": 55.0
  },
  "seed": 20260608,
  "outputs": {
    "rgb": true,
    "depth": true,
    "depth_preview": true,
    "segmentation": true,
    "segmentation_preview": true,
    "metadata": true
  },
  "lidar": {
    "enabled": true,
    "horizontal_samples": 512,
    "vertical_channels": 32,
    "horizontal_fov_degrees": 360.0,
    "vertical_fov_degrees": 30.0,
    "min_range_m": 0.1,
    "max_range_m": 50.0
  }
}
```

CLI flags override config values where they refer to the same setting.

## Scenario Files

Scenario files map to `ScenarioConfig`:

```json
{
  "name": "basic_sensor_rig",
  "scene_path": "examples/scenes/boxes_scene.json",
  "rig": {
    "name": "front_rig",
    "base_transform": {
      "translation": { "x": 0.0, "y": 1.1, "z": 4.5 },
      "rotation": { "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
      "scale": { "x": 1.0, "y": 1.0, "z": 1.0 }
    },
    "mounts": []
  },
  "dataset": {
    "frame_count": 4,
    "seed": 20260608,
    "outputs": {
      "rgb": true,
      "depth": true,
      "segmentation": true,
      "metadata": true
    }
  }
}
```

The current dataset generator uses the first camera-like sensor mount and the
first LiDAR mount from the rig. Scenario datasets keep the existing single
camera/LiDAR output layout.

See [Domain randomization](domain_randomization.md) for the
`domain_randomization` config block used by
`examples/datasets/randomized_boxes.json`.

## Camera Paths

`static`

Uses the same camera pose for every frame.

`orbit`

Moves around a target point at a fixed radius and height. This is the default
config-file example path.

`line`

Interpolates camera position from a start pose to an end pose while looking at a
fixed target.

`random`

Chooses deterministic pseudo-random camera positions inside configured bounds.
The sequence is reproducible for the same `--seed`, frame count, image size, and
path configuration.

## Output Layout

Frame numbering is 6 digits and starts at 1:

```text
dataset/
  rgb/frame_000001.ppm
  depth/frame_000001.f32
  depth_preview/frame_000001.pgm
  segmentation/frame_000001.u32
  segmentation_preview/frame_000001.ppm
  lidar_range/frame_000001.f32
  lidar_points/frame_000001.xyz
  lidar_object_ids/frame_000001.u32
  lidar_preview/frame_000001.pgm
  metadata/frame_000001.json
  dataset_manifest.json
```

The LiDAR directories are present only when `lidar.enabled` is true.

## Per-Frame Metadata

Each `metadata/frame_000001.json` includes:

- Frame index and timestamp.
- Sensor ID.
- Seed and camera path type.
- Camera position, forward/right/up vectors, and intrinsics.
- Width and height.
- Relative output file paths.
- Scene file path when a scene file was used.
- Object ID map.
- Primitive type and material kind for scene-derived object IDs.
- Domain randomization seed, frame seed, per-frame flag, object transforms, and
  randomized material state when enabled.
- Renderer backend label, such as `rocm:gfx1201` or `cpu-preview`.
- Depth convention: linear camera ray distance in meters, `0.0` for miss.
- Segmentation convention: `u32` object IDs, `0` background.
- LiDAR config, pose, output paths, and convention when enabled: single-return
  linear ray range in meters, `0.0` for miss/no return, zero XYZ point for miss,
  and object ID `0` for miss/background.
- Scenario name, rig name, rig base transform, sensor mount transforms, and
  sensor world transforms when generated from a scenario.

## Manifest

`dataset_manifest.json` includes:

- Dataset format version.
- Generator name.
- Scene path and config path when present.
- Frame count, width, height, and seed.
- Camera path config.
- Enabled output set.
- Object ID map.
- Domain randomization config when enabled.
- Relative metadata path for every frame.
- Depth and segmentation conventions.
- LiDAR config and convention when enabled.
- Scenario name, scene path, rig name, and sensor list when generated from a
  scenario.
- Renderer backend label.

Timestamps are intentionally omitted so manifest output remains deterministic.

## Validation

The validation command checks:

- Manifest exists and parses.
- Manifest frame count matches the frame list.
- Expected output files exist.
- Per-frame metadata files parse.
- Object ID maps exist.
- Metadata dimensions match manifest dimensions when present.
- Randomized datasets include per-frame `domain_randomization` metadata.
- LiDAR datasets include range, XYZ point, object ID, preview files, and
  per-frame LiDAR metadata.
- Scenario datasets include scenario metadata in the manifest and every frame.

## Current Limitations

- ROCm rendering supports sphere, plane, and world-space axis-aligned box
  primitives.
- Box rotation is ignored; only translation and scale affect uploaded boxes.
- No meshes or BVH.
- Domain randomization is deterministic but not collision-aware.
- Per-frame domain randomization may upload a fresh scene every frame.
- LiDAR is single-return raycast only; no noise model, multi-return, rolling
  timing, or motion distortion yet.
- Scenario output layout is currently single camera plus optional single LiDAR.
- No physics, ROS2, OpenUSD, or URDF.
- Viewer presentation still uses ROCm -> host copy -> pixels upload.
