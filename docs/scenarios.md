# Scenarios

Milestone 6B adds `ScenarioConfig` as the shared configuration layer for app
startup and dataset generation.

## What A Scenario Contains

```text
ScenarioConfig
  name
  scene_path
  rig
  dataset
```

The scene path points at a serde JSON `sim-core::Scene`. The rig describes
mounted camera and LiDAR sensors. The dataset block reuses `DatasetConfig` for
frame count, output toggles, seed, and domain randomization.

## Examples

```bash
cargo run -p dataset_generator --features rocm -- \
  --scenario examples/scenarios/basic_sensor_rig.json \
  --out target/scenario_dataset \
  --overwrite

cargo run -p dataset_generator --features rocm -- \
  validate --dataset target/scenario_dataset

cargo run -p sim_viewer --features rocm -- \
  --scenario examples/scenarios/basic_sensor_rig.json
```

`examples/scenarios/basic_sensor_rig.json` contains one camera mount and one
LiDAR mount on `examples/scenes/boxes_scene.json`.

`examples/scenarios/randomized_boxes_rig.json` adds deterministic domain
randomization to the same scene and rig.

## Dataset Behavior

The first scenario implementation intentionally keeps the existing single-camera
dataset layout:

```text
rgb/
depth/
segmentation/
lidar_range/
lidar_points/
lidar_object_ids/
metadata/
dataset_manifest.json
```

The generator uses the first camera-like sensor mount and the first LiDAR mount.
Manifest and frame metadata include scenario name, rig name, rig base transform,
and sensor mount/world transforms.

## Current Limitations

- One rig per scenario.
- Apps use the first camera-like mount and first LiDAR mount.
- No per-sensor output directory layout yet.
- No robot placeholder or robot-mounted rig semantics yet.
- No physics, ROS2, URDF, OpenUSD, meshes, or BVH.
