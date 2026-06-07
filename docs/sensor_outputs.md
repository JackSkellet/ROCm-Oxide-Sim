# Sensor Outputs

Milestone 1 defines three image-like sensor outputs. Later milestones keep the
same contracts while sourcing hits from uploaded `sim-core::Scene` geometry. The
ROCm renderer writes all three in one deterministic HIPRTC pass when built with
`--features rocm`.

## RGB

RGB pixels are packed `u32` values in `0x00RRGGBB` order. Dataset previews write
these values as binary PPM (`P6`) files.

## Depth

Depth pixels are `f32` values containing linear distance in meters along the
camera ray. A value of `0.0` means background or miss.

Raw depth files use little-endian `f32` values with no header:

```text
width * height * 4 bytes
```

Depth preview files are binary PGM (`P5`) images. Miss pixels map to black.
Finite positive depths are normalized per frame with nearest samples brighter
and farthest samples dimmer.

## Segmentation

Segmentation pixels are stable `u32` object IDs. `0` means background or miss.
Raw segmentation files use little-endian `u32` values with no header:

```text
width * height * 4 bytes
```

Segmentation preview files are binary PPM (`P6`) images using a stable color
mapping.

## Object IDs

Segmentation IDs come from each `sim-core::Entity` `ObjectId`. Metadata is
generated from the active scene:

- `0` is always background/miss.
- Nonzero IDs are stable per scene and are copied into the segmentation buffer
  on hit.
- Object metadata includes the entity label plus primitive and material kind
  when it comes from a `sim-core::Scene`.
- The default scene keeps the original smoke-test IDs:

```text
0 = background
1 = ground
2 = red sphere
3 = green sphere
4 = blue sphere
```

`examples/scenes/boxes_scene.json` includes object IDs for axis-aligned boxes:

```text
5 = red box
6 = metal preview box
7 = blue emissive sphere
```

The renderer currently supports uploaded sphere, plane, and axis-aligned box
primitives. Box rotation is ignored. Meshes, OpenUSD assets, URDF robot
geometry, and dynamic GPU scene mutation are not implemented yet.

Dataset metadata records the active scene object ID map for every generated
frame and repeats the depth/segmentation conventions in `dataset_manifest.json`
for reproducible downstream parsing.
