//! Sensor traits and frame data types.
//!
//! Sensor outputs are represented as [`SensorFrame<T>`] values. Initial frame
//! payload conventions are:
//!
//! - RGB: packed `u32` pixels in `0x00RRGGBB` order.
//! - Depth: linear `f32` distance in meters along the camera ray, with `0.0`
//!   for background/miss pixels.
//! - Segmentation: stable `u32` object IDs, with `0` for background/miss pixels.
//! - LiDAR: linear range in meters per ray, with `0.0` for miss/no return;
//!   miss points are `Vec3::ZERO` and miss object IDs are `0`.

use serde::{Deserialize, Serialize};
use sim_core::{Camera, ObjectId, PrimitiveShape, Scene, Transform, Vec3};
use std::collections::BTreeMap;

/// Shared interface for configured simulator sensors.
pub trait Sensor {
    type Output;

    fn id(&self) -> &str;
    fn pose(&self) -> SensorPose;
}

/// Camera intrinsic parameters using pixel-center coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub width: u32,
    pub height: u32,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl CameraIntrinsics {
    pub fn from_camera(camera: &Camera) -> Self {
        let half_fov_y = (camera.vertical_fov_degrees.to_radians() * 0.5).tan();
        let fy = camera.height as f32 / (2.0 * half_fov_y);
        let fx = fy;
        Self {
            width: camera.width,
            height: camera.height,
            fx,
            fy,
            cx: (camera.width.saturating_sub(1)) as f32 * 0.5,
            cy: (camera.height.saturating_sub(1)) as f32 * 0.5,
        }
    }
}

/// Sensor pose in world space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensorPose {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
}

impl SensorPose {
    pub fn from_camera(camera: &Camera) -> Self {
        Self {
            position: camera.position,
            forward: camera.forward,
            up: camera.up,
        }
    }

    pub fn from_transform(transform: Transform) -> Self {
        Self {
            position: transform.translation,
            forward: transform.transform_direction(Vec3::new(0.0, 0.0, -1.0)),
            up: transform.transform_direction(Vec3::Y),
        }
    }
}

/// Per-output metadata written next to a sensor frame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameOutputMetadata {
    pub name: String,
    pub format: String,
    pub path: String,
}

impl FrameOutputMetadata {
    pub fn new(
        name: impl Into<String>,
        format: impl Into<String>,
        path: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            format: format.into(),
            path: path.into(),
        }
    }
}

/// Depth convention metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthMetadata {
    pub convention: String,
    pub units: String,
    pub miss_value: f32,
}

impl DepthMetadata {
    pub fn linear_ray_distance_meters() -> Self {
        Self {
            convention: "linear camera ray distance".to_string(),
            units: "meters".to_string(),
            miss_value: 0.0,
        }
    }
}

/// Human-readable label for a segmentation object ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectIdMetadata {
    pub id: u32,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primitive: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
}

impl ObjectIdMetadata {
    pub fn new(id: u32, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            primitive: None,
            material: None,
        }
    }

    pub fn with_primitive(mut self, primitive: impl Into<String>) -> Self {
        self.primitive = Some(primitive.into());
        self
    }

    pub fn with_material(mut self, material: impl Into<String>) -> Self {
        self.material = Some(material.into());
        self
    }
}

pub fn builtin_scene_object_ids() -> Vec<ObjectIdMetadata> {
    vec![
        ObjectIdMetadata::new(0, "background"),
        ObjectIdMetadata::new(1, "ground"),
        ObjectIdMetadata::new(2, "red sphere"),
        ObjectIdMetadata::new(3, "green sphere"),
        ObjectIdMetadata::new(4, "blue sphere"),
    ]
}

pub fn scene_object_ids(scene: &Scene) -> Vec<ObjectIdMetadata> {
    let mut object_ids = vec![ObjectIdMetadata::new(0, "background")];
    let mut scene_ids = BTreeMap::new();

    for entity in scene.entities() {
        let id = entity.object_id.get();
        if id == 0 {
            continue;
        }
        scene_ids.entry(id).or_insert_with(|| {
            ObjectIdMetadata::new(id, entity.name.clone())
                .with_primitive(primitive_label(entity.shape))
                .with_material(entity.material.kind.as_str())
        });
    }

    object_ids.extend(scene_ids.into_values());
    object_ids
}

fn primitive_label(shape: PrimitiveShape) -> &'static str {
    match shape {
        PrimitiveShape::Sphere { .. } => "sphere",
        PrimitiveShape::Box { .. } => "box",
        PrimitiveShape::Plane { .. } => "plane",
    }
}

/// Metadata attached to every sensor frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameMetadata {
    pub frame_index: u64,
    pub timestamp_seconds: f64,
    pub sensor_id: String,
    pub depth: Option<DepthMetadata>,
    pub outputs: Vec<FrameOutputMetadata>,
    pub object_ids: Vec<ObjectIdMetadata>,
}

impl FrameMetadata {
    pub fn new(frame_index: u64, timestamp_seconds: f64, sensor_id: impl Into<String>) -> Self {
        Self {
            frame_index,
            timestamp_seconds,
            sensor_id: sensor_id.into(),
            depth: None,
            outputs: Vec::new(),
            object_ids: Vec::new(),
        }
    }

    pub fn with_depth(mut self, depth: DepthMetadata) -> Self {
        self.depth = Some(depth);
        self
    }

    pub fn with_output(mut self, output: FrameOutputMetadata) -> Self {
        self.outputs.push(output);
        self
    }

    pub fn with_object_id(mut self, object_id: ObjectIdMetadata) -> Self {
        self.object_ids.push(object_id);
        self
    }

    pub fn with_builtin_scene_object_ids(mut self) -> Self {
        self.object_ids = builtin_scene_object_ids();
        self
    }

    pub fn with_scene_object_ids(mut self, scene: &Scene) -> Self {
        self.object_ids = scene_object_ids(scene);
        self
    }
}

/// Generic image-like frame payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorFrame<T> {
    pub width: u32,
    pub height: u32,
    pub metadata: FrameMetadata,
    pub pixels: Vec<T>,
}

impl<T> SensorFrame<T> {
    pub fn new(width: u32, height: u32, metadata: FrameMetadata, pixels: Vec<T>) -> Self {
        Self {
            width,
            height,
            metadata,
            pixels,
        }
    }

    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// Packed `0x00RRGGBB` RGB frame.
pub type RgbFrame = SensorFrame<u32>;

/// Linear camera ray distance in meters. `0.0` means background/miss.
pub type DepthFrame = SensorFrame<f32>;

/// Stable `u32` segmentation object IDs. `0` means background/miss.
pub type SegmentationFrame = SensorFrame<u32>;

pub type SegmentationId = u32;

/// Stable `u32` LiDAR return object ID. `0` means miss/background.
pub type LidarObjectId = u32;

/// Configured single-return spinning/raster LiDAR.
///
/// The scan is a deterministic spherical grid: horizontal samples sweep yaw,
/// vertical channels sweep pitch, both centered around the sensor transform's
/// local `-Z` forward axis. Range values are linear ray distance in meters.
/// Miss/no-return samples use `0.0` range, `Vec3::ZERO` point, and object ID
/// `0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LidarConfig {
    #[serde(default = "default_lidar_horizontal_samples")]
    pub horizontal_samples: u32,
    #[serde(default = "default_lidar_vertical_channels")]
    pub vertical_channels: u32,
    #[serde(default = "default_lidar_horizontal_fov_degrees")]
    pub horizontal_fov_degrees: f32,
    #[serde(default = "default_lidar_vertical_fov_degrees")]
    pub vertical_fov_degrees: f32,
    #[serde(default = "default_lidar_min_range_m")]
    pub min_range_m: f32,
    #[serde(default = "default_lidar_max_range_m")]
    pub max_range_m: f32,
    #[serde(default)]
    pub pose: Transform,
}

fn default_lidar_horizontal_samples() -> u32 {
    512
}

fn default_lidar_vertical_channels() -> u32 {
    32
}

fn default_lidar_horizontal_fov_degrees() -> f32 {
    360.0
}

fn default_lidar_vertical_fov_degrees() -> f32 {
    30.0
}

fn default_lidar_min_range_m() -> f32 {
    0.1
}

fn default_lidar_max_range_m() -> f32 {
    50.0
}

impl Default for LidarConfig {
    fn default() -> Self {
        Self {
            horizontal_samples: default_lidar_horizontal_samples(),
            vertical_channels: default_lidar_vertical_channels(),
            horizontal_fov_degrees: default_lidar_horizontal_fov_degrees(),
            vertical_fov_degrees: default_lidar_vertical_fov_degrees(),
            min_range_m: default_lidar_min_range_m(),
            max_range_m: default_lidar_max_range_m(),
            pose: Transform::default(),
        }
    }
}

impl LidarConfig {
    pub fn normalized(mut self) -> Self {
        self.horizontal_samples = self.horizontal_samples.max(1);
        self.vertical_channels = self.vertical_channels.max(1);
        self.horizontal_fov_degrees = self.horizontal_fov_degrees.clamp(0.0, 360.0);
        self.vertical_fov_degrees = self.vertical_fov_degrees.clamp(0.0, 180.0);
        self.min_range_m = self.min_range_m.max(0.0);
        self.max_range_m = self.max_range_m.max(self.min_range_m);
        self
    }

    pub fn sample_count(self) -> usize {
        self.horizontal_samples as usize * self.vertical_channels as usize
    }

    pub fn pose(self) -> SensorPose {
        SensorPose::from_transform(self.pose)
    }
}

/// Host LiDAR frame with one range, point, and object ID per emitted ray.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LidarFrame {
    pub width: u32,
    pub height: u32,
    pub metadata: FrameMetadata,
    pub ranges_m: Vec<f32>,
    pub points_xyz: Vec<Vec3>,
    pub object_ids: Vec<LidarObjectId>,
}

impl LidarFrame {
    pub fn new(
        width: u32,
        height: u32,
        metadata: FrameMetadata,
        ranges_m: Vec<f32>,
        points_xyz: Vec<Vec3>,
        object_ids: Vec<LidarObjectId>,
    ) -> Self {
        Self {
            width,
            height,
            metadata,
            ranges_m,
            points_xyz,
            object_ids,
        }
    }

    pub fn from_config(
        config: LidarConfig,
        metadata: FrameMetadata,
        ranges_m: Vec<f32>,
        points_xyz: Vec<Vec3>,
        object_ids: Vec<LidarObjectId>,
    ) -> Self {
        let config = config.normalized();
        Self::new(
            config.horizontal_samples,
            config.vertical_channels,
            metadata,
            ranges_m,
            points_xyz,
            object_ids,
        )
    }

    pub fn sample_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub fn miss_sample_count(&self) -> usize {
        self.ranges_m
            .iter()
            .zip(&self.points_xyz)
            .zip(&self.object_ids)
            .filter(|&((&range, &point), &object_id)| {
                range == 0.0 && point == Vec3::ZERO && object_id == 0
            })
            .count()
    }
}

/// Configured LiDAR/raycast sensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LidarSensor {
    id: String,
    config: LidarConfig,
}

impl LidarSensor {
    pub fn new(id: impl Into<String>, config: LidarConfig) -> Self {
        Self {
            id: id.into(),
            config: config.normalized(),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn config(&self) -> LidarConfig {
        self.config
    }

    pub fn sample_count(&self) -> usize {
        self.config.sample_count()
    }
}

impl Sensor for LidarSensor {
    type Output = LidarFrame;

    fn id(&self) -> &str {
        self.id()
    }

    fn pose(&self) -> SensorPose {
        self.config.pose()
    }
}

/// Pinhole camera configuration used by sensor rigs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraSensorConfig {
    #[serde(default = "default_camera_width")]
    pub width: u32,
    #[serde(default = "default_camera_height")]
    pub height: u32,
    #[serde(default = "default_camera_vertical_fov_degrees")]
    pub vertical_fov_degrees: f32,
}

fn default_camera_width() -> u32 {
    640
}

fn default_camera_height() -> u32 {
    360
}

fn default_camera_vertical_fov_degrees() -> f32 {
    55.0
}

impl Default for CameraSensorConfig {
    fn default() -> Self {
        Self {
            width: default_camera_width(),
            height: default_camera_height(),
            vertical_fov_degrees: default_camera_vertical_fov_degrees(),
        }
    }
}

impl CameraSensorConfig {
    pub fn normalized(mut self) -> Self {
        self.width = self.width.max(1);
        self.height = self.height.max(1);
        self.vertical_fov_degrees = self.vertical_fov_degrees.max(1.0);
        self
    }

    pub fn to_camera(self, world_transform: Transform) -> Camera {
        let config = self.normalized();
        Camera {
            position: world_transform.translation,
            forward: world_transform.transform_direction(Vec3::new(0.0, 0.0, -1.0)),
            up: world_transform.transform_direction(Vec3::Y),
            vertical_fov_degrees: config.vertical_fov_degrees,
            aspect_ratio: config.width as f32 / config.height as f32,
            near: 0.01,
            far: 1_000.0,
            width: config.width,
            height: config.height,
        }
    }
}

/// Sensor configuration that can be mounted on a shared rig.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SensorConfig {
    RgbCamera(CameraSensorConfig),
    DepthCamera(CameraSensorConfig),
    SegmentationCamera(CameraSensorConfig),
    Lidar(LidarConfig),
}

impl SensorConfig {
    pub fn sensor_type(&self) -> &'static str {
        match self {
            Self::RgbCamera(_) => "rgb_camera",
            Self::DepthCamera(_) => "depth_camera",
            Self::SegmentationCamera(_) => "segmentation_camera",
            Self::Lidar(_) => "lidar",
        }
    }

    pub fn camera_config(&self) -> Option<CameraSensorConfig> {
        match self {
            Self::RgbCamera(config)
            | Self::DepthCamera(config)
            | Self::SegmentationCamera(config) => Some(*config),
            Self::Lidar(_) => None,
        }
    }

    pub fn lidar_config(&self) -> Option<LidarConfig> {
        match self {
            Self::Lidar(config) => Some(*config),
            _ => None,
        }
    }
}

/// A named sensor mount relative to a [`SensorRig`] base transform.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorMount {
    pub name: String,
    pub sensor: SensorConfig,
    #[serde(default)]
    pub transform: Transform,
}

/// A group of sensors mounted relative to a common base transform.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorRig {
    pub name: String,
    #[serde(default)]
    pub base_transform: Transform,
    #[serde(default)]
    pub mounts: Vec<SensorMount>,
}

impl SensorRig {
    pub fn sensor_names(&self) -> Vec<&str> {
        self.mounts
            .iter()
            .map(|mount| mount.name.as_str())
            .collect()
    }

    pub fn world_transform_for_mount(&self, name: &str) -> Option<Transform> {
        self.mounts
            .iter()
            .find(|mount| mount.name == name)
            .map(|mount| self.base_transform.compose(mount.transform))
    }

    pub fn sensor_summary(&self) -> Vec<SensorSummary> {
        self.mounts
            .iter()
            .map(|mount| SensorSummary {
                name: mount.name.clone(),
                sensor_type: mount.sensor.sensor_type().to_string(),
                mount_transform: mount.transform,
                world_transform: self.base_transform.compose(mount.transform),
            })
            .collect()
    }

    pub fn primary_camera(&self) -> Option<(&SensorMount, Camera)> {
        self.mounts.iter().find_map(|mount| {
            let config = mount.sensor.camera_config()?;
            let world = self.base_transform.compose(mount.transform);
            Some((mount, config.to_camera(world)))
        })
    }

    pub fn primary_lidar(&self) -> Option<(&SensorMount, LidarConfig)> {
        self.mounts.iter().find_map(|mount| {
            let mut config = mount.sensor.lidar_config()?;
            let world = self.base_transform.compose(mount.transform);
            config.pose = world.compose(config.pose);
            Some((mount, config.normalized()))
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorSummary {
    pub name: String,
    pub sensor_type: String,
    pub mount_transform: Transform,
    pub world_transform: Transform,
}

/// Configured RGB camera sensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RgbCameraSensor {
    id: String,
    camera: Camera,
    intrinsics: CameraIntrinsics,
    pose: SensorPose,
}

impl RgbCameraSensor {
    pub fn new(id: impl Into<String>, camera: Camera) -> Self {
        Self {
            id: id.into(),
            intrinsics: CameraIntrinsics::from_camera(&camera),
            pose: SensorPose::from_camera(&camera),
            camera,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn intrinsics(&self) -> CameraIntrinsics {
        self.intrinsics
    }

    pub fn pose(&self) -> SensorPose {
        self.pose
    }
}

impl Sensor for RgbCameraSensor {
    type Output = u32;

    fn id(&self) -> &str {
        self.id()
    }

    fn pose(&self) -> SensorPose {
        self.pose()
    }
}

/// Configured depth camera sensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthCameraSensor {
    id: String,
    camera: Camera,
    intrinsics: CameraIntrinsics,
    pose: SensorPose,
}

impl DepthCameraSensor {
    pub fn new(id: impl Into<String>, camera: Camera) -> Self {
        Self {
            id: id.into(),
            intrinsics: CameraIntrinsics::from_camera(&camera),
            pose: SensorPose::from_camera(&camera),
            camera,
        }
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn intrinsics(&self) -> CameraIntrinsics {
        self.intrinsics
    }
}

impl Sensor for DepthCameraSensor {
    type Output = f32;

    fn id(&self) -> &str {
        &self.id
    }

    fn pose(&self) -> SensorPose {
        self.pose
    }
}

/// Configured segmentation camera sensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentationSensor {
    id: String,
    camera: Camera,
    intrinsics: CameraIntrinsics,
    pose: SensorPose,
}

impl SegmentationSensor {
    pub fn new(id: impl Into<String>, camera: Camera) -> Self {
        Self {
            id: id.into(),
            intrinsics: CameraIntrinsics::from_camera(&camera),
            pose: SensorPose::from_camera(&camera),
            camera,
        }
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn intrinsics(&self) -> CameraIntrinsics {
        self.intrinsics
    }
}

impl Sensor for SegmentationSensor {
    type Output = ObjectId;

    fn id(&self) -> &str {
        &self.id
    }

    fn pose(&self) -> SensorPose {
        self.pose
    }
}
