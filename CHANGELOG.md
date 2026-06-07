# Changelog

## Unreleased

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
