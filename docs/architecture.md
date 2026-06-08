# Architecture

`rocm-oxide-sim` is the library/framework layer for simple AMD GPU-backed
simulation and sensor generation. It is not the final robotics lab application,
not a scene editor, and not a full Isaac Sim replacement.

## Crate Responsibilities

`sim-core`

CPU-only foundational data types: `Vec3`, `EulerRotation`, `Transform`,
`EntityId`, `ObjectId`, `Entity`, `Scene`, `Camera`, `PrimitiveShape`, and
`Material`. This crate must not depend on ROCm-Oxide.

`sim-sensors`

Sensor interfaces and frame contracts: `Sensor`, RGB/depth/segmentation sensor
types, LiDAR config/frame types, `SensorRig`, `SensorMount`, `SensorFrame<T>`,
`CameraIntrinsics`, `SensorPose`, and `FrameMetadata`. This crate depends on
`sim-core` only.

`sim-render-rocm`

The optional AMD GPU backend. It depends on ROCm-Oxide behind the `rocm` feature
and currently renders deterministic RGB, linear depth, segmentation, and
single-return LiDAR/raycast buffers. It uploads supported `sim-core::Scene`
primitives into GPU buffers before rendering. It owns ROCm-Oxide
device/module/kernel/buffer interactions but does not duplicate ROCm-Oxide
runtime internals.

`sim-datasets`

Dataset export helpers and scenario configs. The layout writes
RGB/depth/segmentation/LiDAR files, previews, per-frame JSON metadata, and
`dataset_manifest.json`.

`sim-physics`

Placeholder physics boundary. It defines a `PhysicsWorld` trait and no-op
backend without adding a heavy physics engine yet.

Apps

`sensor_lab` runs one sensor capture, `dataset_generator` writes a multi-frame
dataset, and `sim_viewer` displays live RGB/depth/segmentation previews.

## Data Flow

```text
Scenario -> Scene + SensorRig -> Sensor -> ROCm-Oxide GPU backend -> Frame buffers -> Viewer/export
```

Current milestone detail:

1. `sim-datasets` can load a `ScenarioConfig` that names a scene path, sensor
   rig, and dataset job.
2. `sim-core` builds a CPU `Scene`; `sim-sensors` composes rig base and mount
   transforms into world sensor poses.
3. `sim-sensors` wraps the primary camera as an RGB sensor and creates frame
   metadata.
4. `sim-render-rocm`, when built with `--features rocm`, converts supported
   scene primitives to GPU ABI structs and uploads sphere, plane, box, and
   material buffers with ROCm-Oxide `DeviceBuffer`.
5. The renderer opens ROCm-Oxide, compiles a deterministic HIPRTC kernel,
   launches linear primitive intersection over the uploaded buffers, and returns
   `DeviceBuffer<u32>` RGB, `DeviceBuffer<f32>` depth, and `DeviceBuffer<u32>`
   segmentation outputs.
6. The LiDAR renderer launches a separate ray-grid kernel over the same
   uploaded scene buffers and returns range, point, and object ID buffers.
7. Apps copy the buffers to host through renderer helpers.
8. `sim-datasets` writes PPM/PGM previews, raw depth/segmentation/LiDAR files,
   and JSON metadata/manifest outputs.
9. `sim_viewer` uses the same host-copy output path and uploads the selected
   preview mode through `winit + pixels`.

Without the `rocm` feature, the renderer reports `BackendUnavailable`. The apps
use deterministic CPU preview images only for smoke-testable CLI output.

## Current Renderer Scope

The ROCm renderer now uses uploaded `sim-core::Scene` data for the supported
primitive subset:

- Sphere.
- Plane.
- World-space axis-aligned box.
- One simple material record per entity, using base color and simple material
  kind for preview shading.
- Stable nonzero `ObjectId` values for segmentation.

The default deterministic scene is now just a `sim-core::Scene` fixture that
goes through the same upload path. Box rotation is ignored for now. Meshes, BVH
acceleration, richer material systems, OpenUSD, URDF, physics synchronization,
robot articulation, and dynamic GPU scene mutation are deferred.

The viewer is also intentionally not a direct HIP/Vulkan interop path yet. It
copies selected ROCm outputs to host memory, converts them to an RGBA preview,
and uploads that preview for presentation.

## Repo Boundary

This repo owns:

- Simulator-facing scene and sensor abstractions.
- Sensor output data contracts.
- Dataset export layout.
- Renderer/backend integration code that calls ROCm-Oxide public APIs.
- Future viewer, physics adapter, robotics model, and ROS2 bridge interfaces.

ROCm-Oxide owns:

- HIP/ROCm runtime bindings.
- Device discovery and architecture detection.
- HIPRTC/COMGR compilation and code-object loading.
- Device memory buffers, streams, events, and kernel launch APIs.
- ROCm library interop and lower-level GPU runtime validation.

`rocm-oxide-sim` should use ROCm-Oxide as a dependency. It should not copy
ROCm-Oxide internals or generate its own competing HIP runtime layer.

## Future Robotics-Lab App Relationship

A future robotics-lab application can sit above this workspace. That app should
compose scenes, sensors, datasets, viewer controls, robot models, and ROS2
integration. This repository should remain the reusable framework layer that
the app calls into, with clear crate boundaries and CPU-only tests that do not
need AMD GPU hardware.
