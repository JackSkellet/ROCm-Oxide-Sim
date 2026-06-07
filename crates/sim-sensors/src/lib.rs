//! Sensor traits and frame data types.
//!
//! Sensor outputs are represented as [`SensorFrame<T>`] values. Initial frame
//! payload conventions are:
//!
//! - RGB: packed `u32` pixels in `0x00RRGGBB` order.
//! - Depth: linear `f32` distance in meters along the camera ray, with `0.0`
//!   for background/miss pixels.
//! - Segmentation: stable `u32` object IDs, with `0` for background/miss pixels.

use serde::{Deserialize, Serialize};
use sim_core::{Camera, ObjectId, PrimitiveShape, Scene, Vec3};
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
