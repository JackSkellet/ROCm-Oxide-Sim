# Changelog

## Unreleased

- Added Milestone 6B `SensorRig`, `SensorMount`, mounted sensor configs, and
  `ScenarioConfig`.
- Added scenario dataset generation with shared scene path, rig, dataset job,
  domain randomization, and LiDAR settings.
- Added scenario metadata/manifest fields plus validation for scenario sensor
  lists and per-frame scenario sections.
- Added `--scenario` support to `dataset_generator`, `sensor_lab`, and
  `sim_viewer`.
- Added `examples/scenarios/basic_sensor_rig.json` and
  `examples/scenarios/randomized_boxes_rig.json`.
- Added Milestone 6A LiDAR/raycast sensor contracts with deterministic
  spherical scan configuration.
- Added ROCm LiDAR raycast rendering over uploaded sphere, plane, and AABB box
  scene buffers.
- Added LiDAR dataset outputs for range `.f32`, XYZ point `.xyz`, object ID
  `.u32`, and normalized range preview `.pgm` files.
- Added LiDAR metadata, manifest conventions, validation checks, and
  `examples/datasets/randomized_boxes_lidar.json`.
- Added `sensor_lab --lidar` for one-frame LiDAR smoke output.
- Added Milestone 5B deterministic domain randomization for dataset generation.
- Added randomized object position/scale, material color/kind, emissive
  intensity, and camera pose/FOV controls.
- Added randomization metadata to frame JSON and manifests, plus validation for
  randomized datasets.
- Added `examples/datasets/randomized_boxes.json`.
- Added Milestone 5A ROCm scene upload support for `PrimitiveShape::Box` as a
  world-space AABB primitive.
- Added `GpuBox` buffers and HIPRTC ray/AABB slab intersection alongside the
  existing sphere and plane loops.
- Added simple material kinds (`diffuse`, `matte`, `emissive`,
  `metal_preview`) with deterministic preview shading.
- Expanded object ID metadata with primitive type and material kind fields.
- Added `examples/scenes/boxes_scene.json` for box/material dataset and viewer
  smoke runs.
- Added Milestone 4 dataset config support through `DatasetConfig` and
  `examples/datasets/basic_orbit.json`.
- Added deterministic `static`, `orbit`, `line`, and seeded `random` camera
  paths for dataset generation.
- Added stronger per-frame dataset metadata with camera pose, intrinsics,
  output paths, scene path, backend label, seed, and sensor conventions.
- Expanded `dataset_manifest.json` with reproducibility fields, camera path
  config, output selection, object IDs, and convention metadata.
- Added `dataset_generator validate --dataset <DIR>` for simple dataset checks.
- Added `--overwrite` and `--dry-run` dataset generator behavior.
- Added Milestone 3 ROCm scene upload from `sim-core::Scene` to GPU primitive
  buffers.
- Added GPU-facing sphere, plane, material, and render parameter ABI structs for
  the HIPRTC renderer.
- Updated the renderer kernel to intersect uploaded sphere and plane buffers and
  preserve stable `ObjectId` segmentation from scene entities.
- Added `RocmSensorRenderer::upload_scene` and
  `render_uploaded_scene_to_device` while preserving the existing convenience
  render methods.
- Added `examples/scenes/basic_scene.json` plus `--scene` loading for
  `sensor_lab`, `dataset_generator`, and `sim_viewer`.
- Updated output metadata to derive object ID labels from the active scene.
- Added Milestone 2 `sim_viewer` live viewer using `winit + pixels`.
- Added RGB, depth preview, and segmentation preview viewer modes.
- Added static/orbit camera modes plus keyboard controls for the windowed path.
- Added finite-frame viewer smoke mode for noninteractive ROCm verification.
- Passed camera parameters into the deterministic ROCm renderer so viewer camera
  movement affects the rendered scene without arbitrary geometry upload.
- Added Milestone 1 RGB/depth/segmentation sensor outputs in the ROCm renderer.
- Added raw depth `.f32`, depth preview `.pgm`, raw segmentation `.u32`, and
  segmentation preview `.ppm` dataset output conventions.
- Added frame metadata for output files, depth convention, and built-in object
  IDs.
- Updated `sensor_lab` to write RGB, depth, segmentation, previews, and metadata.
- Updated `dataset_generator` to write the full Milestone 1 dataset layout.
