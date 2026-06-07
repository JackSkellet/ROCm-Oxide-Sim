# Roadmap

## Milestone 0: Skeleton + RGB Output (Completed)

- Cargo workspace and crate boundaries.
- CPU-only scene, camera, sensor, dataset, and placeholder physics types.
- Opt-in ROCm-Oxide RGB renderer.
- `sensor_lab` one-frame RGB output.
- CPU-only tests pass without ROCm.

## Milestone 1: Depth + Segmentation (Completed)

- Add one-pass RGB/depth/segmentation renderer output.
- Define depth miss/clip conventions.
- Define segmentation/object ID buffer conventions.
- Add dataset exports for depth and segmentation metadata.
- Keep renderer limited to the deterministic built-in scene.

## Milestone 2: Live Viewer (Completed)

- Add a separate viewer app.
- Display RGB, depth preview, and segmentation preview modes.
- Add finite-frame smoke mode for noninteractive verification.
- Use host-copy presentation through `winit + pixels`.
- Keep direct HIP/Vulkan external memory interop as future work.

## Milestone 2.5: Direct Presentation Interop

- Investigate direct HIP/Vulkan external memory interop.
- Avoid overfitting the renderer to a single presentation backend.

## Milestone 3: Arbitrary Scene Geometry Upload (Completed)

- Upload a small subset of `sim-core::Scene` geometry into the ROCm renderer.
- Preserve the deterministic built-in scene as a smoke-test fixture through the
  same upload path.
- Keep geometry upload narrow: spheres, planes, simple material colors, and
  stable object IDs.
- Reject unsupported boxes clearly until box intersection semantics are chosen.
- Use the live viewer to inspect RGB/depth/segmentation correctness immediately.

## Milestone 4: Dataset Generator Expansion (Completed)

- Expand camera paths and deterministic dataset seeds.
- Add JSON config files and CLI overrides.
- Add stronger per-frame metadata and dataset manifests.
- Add validation command for generated datasets.
- Keep train/validation split metadata deferred until dataset consumers need it.
- Keep COCO/KITTI export deferred until the native frame contracts are stable.

## Milestone 5A: AABB/Box Support + Materials (Completed)

- Added ROCm renderer support for `PrimitiveShape::Box` as world-space AABBs.
- Defined the current box transform convention: translation and scale are used,
  rotation is ignored.
- Added simple material metadata and deterministic preview material kinds.
- Improved scene variety for all camera sensors and datasets with
  `examples/scenes/boxes_scene.json`.

## Milestone 5B: Domain Randomization (Completed)

- Added deterministic scene/config randomization for primitive positions, scale,
  material colors/kinds, emissive intensity, and camera pose/FOV parameters.
- Preserved reproducibility through explicit seeds, frame seeds, manifest
  metadata, and relative output paths.
- Kept randomization limited to supported primitives until mesh/BVH support
  exists.
- Documented the current per-frame scene upload strategy.

## Milestone 5C: LiDAR/Raycast Sensor

- Add raycast/LiDAR sensor traits and output contracts.
- Reuse uploaded scene primitive intersection where practical.
- Keep mesh/BVH support deferred until primitive raycasts are stable.

## Milestone 6: Physics Adapter

- Add rigid body descriptors and scene synchronization.
- Integrate a first lightweight backend, likely Rapier.
- Keep the backend behind traits.

## Milestone 7: Robot Model / URDF

- Define robot model data structures.
- Import URDF or a minimal subset.
- Map robot links to renderable entities and physics bodies.

## Milestone 8: ROS2 Bridge

- Add optional ROS2 publishing/subscription crates.
- Export sensor streams with explicit timestamp and frame ID conventions.
- Keep ROS2 optional so core tests remain dependency-light.

## Milestone 9: Robotics-Lab App Integration

- Build the higher-level application on top of this framework.
- Compose scenes, sensors, datasets, viewer, physics, robots, and ROS2.
- Keep app-specific UX and orchestration out of the reusable crates.
