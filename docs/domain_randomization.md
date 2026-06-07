# Domain Randomization

Milestone 5B adds deterministic domain randomization to
`apps/dataset_generator`. The goal is varied synthetic datasets that are still
reproducible from the same seed and config.

## Quickstart

```bash
cargo run -p dataset_generator --features rocm -- \
  --config examples/datasets/randomized_boxes.json \
  --out target/randomized_boxes \
  --overwrite

cargo run -p dataset_generator --features rocm -- \
  validate --dataset target/randomized_boxes
```

The LiDAR variant uses the same randomized scene controls plus LiDAR outputs:

```bash
cargo run -p dataset_generator --features rocm -- \
  --config examples/datasets/randomized_boxes_lidar.json \
  --out target/lidar_dataset \
  --overwrite
```

Two runs with the same config and seed should be byte-identical:

```bash
cargo run -p dataset_generator --features rocm -- \
  --config examples/datasets/randomized_boxes.json \
  --out target/randomized_boxes_a \
  --overwrite
cargo run -p dataset_generator --features rocm -- \
  --config examples/datasets/randomized_boxes.json \
  --out target/randomized_boxes_b \
  --overwrite
diff -qr target/randomized_boxes_a target/randomized_boxes_b
```

## Seed Model

The dataset seed comes from `DatasetConfig.seed` or `--seed`. A
`domain_randomization.seed` can override that for randomization only. When
`per_frame` is `true`, frame variations use a deterministic seed derived from
the base seed and frame index. When `per_frame` is `false`, every frame uses the
same randomized scene.

Manifests omit timestamps, and output paths are relative to the dataset root, so
same-seed runs can compare cleanly across different output directories.

## Config Shape

```json
{
  "seed": 1234,
  "domain_randomization": {
    "enabled": true,
    "per_frame": true,
    "object_transforms": {
      "enabled": true,
      "position_jitter": { "x": 0.25, "y": 0.0, "z": 0.25 },
      "scale_range": [0.8, 1.2],
      "include_planes": false
    },
    "materials": {
      "enabled": true,
      "base_color_jitter": 0.18,
      "randomize_kind": false,
      "emissive_intensity_range": [0.8, 1.35]
    },
    "lights": {
      "enabled": true,
      "position_jitter": { "x": 0.35, "y": 0.2, "z": 0.35 },
      "intensity_range": [0.85, 1.3]
    },
    "camera": {
      "enabled": true,
      "pose_jitter": { "x": 0.05, "y": 0.02, "z": 0.05 },
      "fov_degrees_range": [42.0, 50.0]
    }
  }
}
```

`--randomize` enables the default randomization config from the CLI. Config
files remain the preferred path for reproducible runs.

## Randomized Properties

Object transforms:

- Position jitter for spheres and boxes.
- Uniform scale jitter for spheres and boxes.
- Planes remain fixed unless `include_planes` is enabled.
- Object IDs remain stable.

Materials:

- Base color jitter.
- Optional material-kind randomization for non-emissive objects.
- Emissive intensity jitter for emissive materials.

Lights:

- There is no separate light entity type yet.
- Current light randomization applies to emissive primitives by jittering their
  transform and emission intensity.

Camera:

- Pose jitter is applied on top of the configured camera path.
- FOV can be sampled from a deterministic range.

## Metadata

`dataset_manifest.json` includes the enabled domain randomization config.

Each randomized frame metadata file includes:

```json
"domain_randomization": {
  "enabled": true,
  "seed": 1234,
  "frame_seed": 15031427735924447607,
  "per_frame": true,
  "objects": [
    {
      "object_id": 5,
      "name": "red box",
      "primitive": "box",
      "material": "matte",
      "transform": {},
      "material_state": {},
      "randomized": true
    }
  ]
}
```

The normal camera metadata records the camera after randomization.

## Current Limitations

- No collision-aware placement.
- No physics.
- No mesh assets or BVH.
- No material texture randomization.
- No OpenUSD or URDF import.
- No ROS2 integration.
- No GUI/editor for randomization configs.
- Per-frame randomization may upload a fresh scene to the GPU every frame.
