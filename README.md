# rocm-oxide-sim

`rocm-oxide-sim` is an early Rust-first simulation and sensor framework that
uses [ROCm-Oxide](../ROCm-Oxide) as its AMD GPU backend.

This repository is downstream from ROCm-Oxide. It should consume ROCm-Oxide as a
dependency and should not copy ROCm-Oxide internals. The current renderer is a
small experimental camera sensor backend, not a full simulator, scene editor, or
Isaac Sim replacement.

## Current Status

- CPU-only scene, transform, camera, material, sensor, dataset, and placeholder
  physics types are available.
- `sim-render-rocm` contains an opt-in ROCm-Oxide HIPRTC renderer for RGB,
  linear depth, and segmentation outputs from uploaded `sim-core::Scene`
  sphere and plane primitives.
- `apps/sim_viewer` provides a live `winit + pixels` viewer for RGB, depth, and
  segmentation modes using a ROCm render -> host copy -> presentation path.
- `examples/scenes/basic_scene.json` can be loaded by the CLIs with `--scene`.
- LiDAR/raycast sensors, full physics, ROS2, OpenUSD, and robot model
  integration are planned but not implemented yet.

## Workspace Structure

```text
crates/sim-core          CPU-only scene, transform, camera, material, and IDs
crates/sim-sensors       Sensor traits, intrinsics, poses, frames, and metadata
crates/sim-render-rocm   Optional ROCm-Oxide RGB/depth/segmentation renderer
crates/sim-datasets      PPM RGB and metadata/manifest dataset writer
crates/sim-physics       Placeholder physics backend trait
apps/sensor_lab          One-frame sensor rendering CLI
apps/dataset_generator   Simple N-frame dataset generation CLI
apps/sim_viewer          Live sensor output viewer
examples/scenes/         Serde JSON scene fixtures
docs/                    Architecture, roadmap, and physics notes
```

## ROCm-Oxide Path Dependency

ROCm-Oxide is not crates.io-ready yet, so this workspace uses a local path
dependency. The expected sibling layout is:

```text
workspace-root/
  ROCm-Oxide/
  rocm-oxide-sim/
```

The editable path is in the root `Cargo.toml`:

```toml
[workspace.dependencies]
rocm-oxide = { path = "../ROCm-Oxide" }
```

If your checkout uses another layout, edit that path before building with the
`rocm` feature.

## Quick Commands

CPU-only tests:

```bash
cargo test
```

One-frame sensor lab. Without the `rocm` feature, this writes deterministic CPU
preview outputs so the CLI is still smoke-testable:

```bash
cargo run -p sensor_lab
```

Dataset generator help:

```bash
cargo run -p dataset_generator -- --help
```

ROCm RGB/depth/segmentation sensor demo, requiring ROCm, a visible AMD GPU, and
the local ROCm-Oxide sibling dependency. The renderer uploads the `sim-core`
scene to GPU primitive buffers before rendering:

```bash
cargo run -p sensor_lab --features rocm
cargo run -p sensor_lab --features rocm -- --scene examples/scenes/basic_scene.json
```

The ROCm path writes:

```text
target/sensor_lab/rgb.ppm
target/sensor_lab/depth.f32
target/sensor_lab/depth_preview.pgm
target/sensor_lab/segmentation.u32
target/sensor_lab/segmentation_preview.ppm
target/sensor_lab/metadata.json
```

Generate a small dataset:

```bash
cargo run -p dataset_generator --features rocm -- --frames 16 --out target/sim_dataset
cargo run -p dataset_generator --features rocm -- --frames 4 --out target/sim_dataset --scene examples/scenes/basic_scene.json
```

Run the live viewer:

```bash
cargo run -p sim_viewer --features rocm
cargo run -p sim_viewer --features rocm -- --mode depth
cargo run -p sim_viewer --features rocm -- --mode segmentation
cargo run -p sim_viewer --features rocm -- --scene examples/scenes/basic_scene.json
```

Finite-frame smoke runs do not open a window:

```bash
cargo run -p sim_viewer --features rocm -- --frames 4 --mode rgb
```

If ROCm-Oxide cannot open a device, compile the HIPRTC kernel, launch it, or copy
the output buffer back, the renderer returns a clear error with the failed step.

## Documentation

- [Architecture](docs/architecture.md)
- [Scene upload](docs/scene_upload.md)
- [Sensor outputs](docs/sensor_outputs.md)
- [Dataset generation](docs/dataset_generation.md)
- [Live viewer](docs/live_viewer.md)
- [Roadmap](docs/roadmap.md)
- [Physics plan](docs/physics_plan.md)
