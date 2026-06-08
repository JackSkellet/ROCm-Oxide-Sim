# Sensor Rigs

Milestone 6B adds a reusable sensor rig layer for grouping cameras and LiDAR
sensors under a shared transform.

## Structure

```text
SensorRig
  name
  base_transform
  mounts[]

SensorMount
  name
  sensor
  transform
```

Mount names are stable and appear in scenario metadata. Mount transforms are
relative to the rig base transform.

World pose calculation:

```text
world_sensor_transform = rig.base_transform.compose(mount.transform)
```

For cameras, the world transform's local `-Z` axis is forward and `+Y` is up.
For LiDAR, the mounted LiDAR config receives the same world transform as its
pose. LiDAR still uses the Milestone 6A single-return raycast contract.

## Supported Sensor Configs

The first rig format supports:

- `rgb_camera`
- `depth_camera`
- `segmentation_camera`
- `lidar`

The current apps use the first camera-like mount and the first LiDAR mount. A
future multi-sensor dataset layout can add per-sensor output directories without
changing the core rig definition.

## Current Limitations

- No robot link attachment yet.
- No articulated robot model.
- No physics-driven rig motion.
- No per-sensor time offsets.
- No multi-camera output layout yet.
