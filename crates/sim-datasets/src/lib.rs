//! Dataset export helpers.
//!
//! The initial dataset layout is intentionally small:
//!
//! ```text
//! dataset_root/
//!   rgb/frame_000001.ppm
//!   depth/frame_000001.f32
//!   depth_preview/frame_000001.pgm
//!   segmentation/frame_000001.u32
//!   segmentation_preview/frame_000001.ppm
//!   lidar_range/frame_000001.f32
//!   lidar_points/frame_000001.xyz
//!   lidar_object_ids/frame_000001.u32
//!   lidar_preview/frame_000001.pgm
//!   metadata/frame_000001.json
//!   dataset_manifest.json
//! ```

use serde::{Deserialize, Serialize};
use sim_core::{Camera, MaterialKind, PrimitiveShape, Scene, Transform, Vec3};
use sim_sensors::{
    CameraIntrinsics, DepthFrame, FrameMetadata, FrameOutputMetadata, LidarConfig, LidarFrame,
    ObjectIdMetadata, RgbFrame, SegmentationFrame,
};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid image dimensions: {0}")]
    InvalidImage(String),
}

pub type Result<T> = std::result::Result<T, DatasetError>;

fn default_output_dir() -> PathBuf {
    PathBuf::from("target/dataset_generator")
}

fn default_frame_count() -> u32 {
    8
}

fn default_width() -> u32 {
    640
}

fn default_height() -> u32 {
    360
}

/// Output toggles for generated datasets.
///
/// Metadata is treated as a core dataset output by the writer and defaults to
/// enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputSelection {
    #[serde(default = "default_true")]
    pub rgb: bool,
    #[serde(default = "default_true")]
    pub depth: bool,
    #[serde(default = "default_true")]
    pub depth_preview: bool,
    #[serde(default = "default_true")]
    pub segmentation: bool,
    #[serde(default = "default_true")]
    pub segmentation_preview: bool,
    #[serde(default = "default_true")]
    pub metadata: bool,
}

fn default_true() -> bool {
    true
}

impl OutputSelection {
    pub const fn all() -> Self {
        Self {
            rgb: true,
            depth: true,
            depth_preview: true,
            segmentation: true,
            segmentation_preview: true,
            metadata: true,
        }
    }

    pub fn normalized(mut self) -> Self {
        self.metadata = true;
        self
    }
}

impl Default for OutputSelection {
    fn default() -> Self {
        Self::all()
    }
}

/// Optional LiDAR output configuration for dataset generation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DatasetLidarConfig {
    #[serde(default)]
    pub enabled: bool,
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
    LidarConfig::default().horizontal_samples
}

fn default_lidar_vertical_channels() -> u32 {
    LidarConfig::default().vertical_channels
}

fn default_lidar_horizontal_fov_degrees() -> f32 {
    LidarConfig::default().horizontal_fov_degrees
}

fn default_lidar_vertical_fov_degrees() -> f32 {
    LidarConfig::default().vertical_fov_degrees
}

fn default_lidar_min_range_m() -> f32 {
    LidarConfig::default().min_range_m
}

fn default_lidar_max_range_m() -> f32 {
    LidarConfig::default().max_range_m
}

impl DatasetLidarConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            horizontal_samples: default_lidar_horizontal_samples(),
            vertical_channels: default_lidar_vertical_channels(),
            horizontal_fov_degrees: default_lidar_horizontal_fov_degrees(),
            vertical_fov_degrees: default_lidar_vertical_fov_degrees(),
            min_range_m: default_lidar_min_range_m(),
            max_range_m: default_lidar_max_range_m(),
            pose: Transform::default(),
        }
    }

    pub fn normalized(mut self) -> Self {
        let config = self.to_lidar_config().normalized();
        self.horizontal_samples = config.horizontal_samples;
        self.vertical_channels = config.vertical_channels;
        self.horizontal_fov_degrees = config.horizontal_fov_degrees;
        self.vertical_fov_degrees = config.vertical_fov_degrees;
        self.min_range_m = config.min_range_m;
        self.max_range_m = config.max_range_m;
        self.pose = config.pose;
        self
    }

    pub fn is_disabled(&self) -> bool {
        !self.enabled
    }

    pub fn to_lidar_config(self) -> LidarConfig {
        LidarConfig {
            horizontal_samples: self.horizontal_samples,
            vertical_channels: self.vertical_channels,
            horizontal_fov_degrees: self.horizontal_fov_degrees,
            vertical_fov_degrees: self.vertical_fov_degrees,
            min_range_m: self.min_range_m,
            max_range_m: self.max_range_m,
            pose: self.pose,
        }
        .normalized()
    }
}

impl Default for DatasetLidarConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Camera path configuration used by the dataset generator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CameraPathConfig {
    Static {
        position: Vec3,
        target: Vec3,
        #[serde(default = "default_up")]
        up: Vec3,
        #[serde(default = "default_fov")]
        fov_y_degrees: f32,
    },
    Orbit {
        target: Vec3,
        #[serde(default = "default_orbit_radius")]
        radius: f32,
        #[serde(default = "default_orbit_height")]
        height: f32,
        #[serde(default)]
        start_angle_degrees: f32,
        #[serde(default = "default_orbit_end_angle")]
        end_angle_degrees: f32,
        #[serde(default = "default_fov")]
        fov_y_degrees: f32,
    },
    Line {
        start_position: Vec3,
        end_position: Vec3,
        target: Vec3,
        #[serde(default = "default_fov")]
        fov_y_degrees: f32,
    },
    Random {
        target: Vec3,
        min_position: Vec3,
        max_position: Vec3,
        #[serde(default = "default_fov")]
        fov_y_degrees: f32,
    },
}

fn default_up() -> Vec3 {
    Vec3::Y
}

fn default_fov() -> f32 {
    55.0
}

fn default_orbit_radius() -> f32 {
    4.1
}

fn default_orbit_height() -> f32 {
    1.35
}

fn default_orbit_end_angle() -> f32 {
    360.0
}

impl CameraPathConfig {
    pub fn static_default() -> Self {
        Self::Static {
            position: Vec3::new(0.0, 1.1, 4.5),
            target: Vec3::new(0.0, 0.55, -1.45),
            up: Vec3::Y,
            fov_y_degrees: 55.0,
        }
    }

    pub fn orbit_default() -> Self {
        Self::Orbit {
            target: Vec3::new(0.0, 0.55, -1.45),
            radius: 4.1,
            height: 1.35,
            start_angle_degrees: 0.0,
            end_angle_degrees: 360.0,
            fov_y_degrees: 55.0,
        }
    }

    pub fn line_default() -> Self {
        Self::Line {
            start_position: Vec3::new(-0.9, 1.1, 4.6),
            end_position: Vec3::new(0.9, 1.2, 3.7),
            target: Vec3::new(0.0, 0.55, -1.45),
            fov_y_degrees: 55.0,
        }
    }

    pub fn random_default() -> Self {
        Self::Random {
            target: Vec3::new(0.0, 0.55, -1.45),
            min_position: Vec3::new(-1.4, 0.85, 3.4),
            max_position: Vec3::new(1.4, 1.75, 5.2),
            fov_y_degrees: 55.0,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Static { .. } => "static",
            Self::Orbit { .. } => "orbit",
            Self::Line { .. } => "line",
            Self::Random { .. } => "random",
        }
    }

    pub fn start_position(&self) -> Option<Vec3> {
        match self {
            Self::Line { start_position, .. } => Some(*start_position),
            _ => None,
        }
    }

    pub fn end_position(&self) -> Option<Vec3> {
        match self {
            Self::Line { end_position, .. } => Some(*end_position),
            _ => None,
        }
    }
}

impl Default for CameraPathConfig {
    fn default() -> Self {
        Self::static_default()
    }
}

/// Deterministic domain randomization controls for dataset generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DomainRandomizationConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default = "default_true")]
    pub per_frame: bool,
    #[serde(default)]
    pub object_transforms: ObjectTransformRandomization,
    #[serde(default)]
    pub materials: MaterialRandomization,
    #[serde(default)]
    pub lights: LightRandomization,
    #[serde(default)]
    pub camera: CameraRandomization,
}

impl DomainRandomizationConfig {
    pub fn is_disabled(&self) -> bool {
        !self.enabled
    }

    pub fn effective_seed(&self, dataset_seed: u64) -> u64 {
        self.seed.unwrap_or(dataset_seed)
    }

    pub fn frame_seed(&self, dataset_seed: u64, frame_index: u64) -> u64 {
        let seed = self.effective_seed(dataset_seed);
        if self.per_frame {
            seed ^ frame_index.wrapping_mul(0xd1b5_4a32_d192_ed03)
        } else {
            seed
        }
    }
}

impl Default for DomainRandomizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            seed: None,
            per_frame: true,
            object_transforms: ObjectTransformRandomization::default(),
            materials: MaterialRandomization::default(),
            lights: LightRandomization::default(),
            camera: CameraRandomization::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObjectTransformRandomization {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub position_jitter: Vec3,
    #[serde(default = "unit_range")]
    pub scale_range: [f32; 2],
    #[serde(default)]
    pub include_planes: bool,
}

impl Default for ObjectTransformRandomization {
    fn default() -> Self {
        Self {
            enabled: false,
            position_jitter: Vec3::ZERO,
            scale_range: [1.0, 1.0],
            include_planes: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MaterialRandomization {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub base_color_jitter: f32,
    #[serde(default)]
    pub randomize_kind: bool,
    #[serde(default = "unit_range")]
    pub emissive_intensity_range: [f32; 2],
}

impl Default for MaterialRandomization {
    fn default() -> Self {
        Self {
            enabled: false,
            base_color_jitter: 0.0,
            randomize_kind: false,
            emissive_intensity_range: [1.0, 1.0],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LightRandomization {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub position_jitter: Vec3,
    #[serde(default = "unit_range")]
    pub intensity_range: [f32; 2],
}

impl Default for LightRandomization {
    fn default() -> Self {
        Self {
            enabled: false,
            position_jitter: Vec3::ZERO,
            intensity_range: [1.0, 1.0],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraRandomization {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub pose_jitter: Vec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fov_degrees_range: Option<[f32; 2]>,
}

impl Default for CameraRandomization {
    fn default() -> Self {
        Self {
            enabled: false,
            pose_jitter: Vec3::ZERO,
            fov_degrees_range: None,
        }
    }
}

fn unit_range() -> [f32; 2] {
    [1.0, 1.0]
}

/// Top-level generator configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene_path: Option<PathBuf>,
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
    #[serde(default = "default_frame_count")]
    pub frame_count: u32,
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default)]
    pub camera_path: CameraPathConfig,
    #[serde(default)]
    pub seed: u64,
    #[serde(
        default,
        skip_serializing_if = "DomainRandomizationConfig::is_disabled"
    )]
    pub domain_randomization: DomainRandomizationConfig,
    #[serde(default)]
    pub outputs: OutputSelection,
    #[serde(default, skip_serializing_if = "DatasetLidarConfig::is_disabled")]
    pub lidar: DatasetLidarConfig,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            scene_path: None,
            output_dir: default_output_dir(),
            frame_count: default_frame_count(),
            width: default_width(),
            height: default_height(),
            camera_path: CameraPathConfig::default(),
            seed: 0,
            domain_randomization: DomainRandomizationConfig::default(),
            outputs: OutputSelection::all(),
            lidar: DatasetLidarConfig::default(),
        }
    }
}

impl DatasetConfig {
    pub fn normalized(mut self) -> Self {
        self.frame_count = self.frame_count.max(1);
        self.width = self.width.max(1);
        self.height = self.height.max(1);
        self.outputs = self.outputs.normalized();
        self.lidar = self.lidar.normalized();
        self
    }
}

pub fn camera_for_dataset_frame(
    config: &CameraPathConfig,
    frame_index: u32,
    frame_count: u32,
    width: u32,
    height: u32,
    seed: u64,
) -> Camera {
    let frame_count = frame_count.max(1);
    let frame_index = frame_index.clamp(1, frame_count);
    let progress = if frame_count <= 1 {
        0.0
    } else {
        (frame_index - 1) as f32 / (frame_count - 1) as f32
    };
    let aspect_ratio = width.max(1) as f32 / height.max(1) as f32;

    let (position, target, fov_y_degrees) = match *config {
        CameraPathConfig::Static {
            position,
            target,
            fov_y_degrees,
            ..
        } => (position, target, fov_y_degrees),
        CameraPathConfig::Orbit {
            target,
            radius,
            height,
            start_angle_degrees,
            end_angle_degrees,
            fov_y_degrees,
        } => {
            let angle = (start_angle_degrees
                + (end_angle_degrees - start_angle_degrees) * progress)
                .to_radians();
            (
                Vec3::new(
                    angle.sin() * radius,
                    height,
                    target.z + angle.cos() * radius,
                ),
                target,
                fov_y_degrees,
            )
        }
        CameraPathConfig::Line {
            start_position,
            end_position,
            target,
            fov_y_degrees,
        } => (
            start_position + (end_position - start_position) * progress,
            target,
            fov_y_degrees,
        ),
        CameraPathConfig::Random {
            target,
            min_position,
            max_position,
            fov_y_degrees,
        } => {
            let mut rng = DeterministicRng::new(seed ^ ((frame_index as u64) << 32));
            let position = Vec3::new(
                rng.range_f32(min_position.x, max_position.x),
                rng.range_f32(min_position.y, max_position.y),
                rng.range_f32(min_position.z, max_position.z),
            );
            (position, target, fov_y_degrees)
        }
    };

    Camera::look_at(position, target, fov_y_degrees, aspect_ratio).with_resolution(width, height)
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / 16_777_215.0
    }

    fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    fn signed_range_f32(&mut self, extent: f32) -> f32 {
        self.range_f32(-extent.abs(), extent.abs())
    }

    fn jitter_vec3(&mut self, extents: Vec3) -> Vec3 {
        Vec3::new(
            self.signed_range_f32(extents.x),
            self.signed_range_f32(extents.y),
            self.signed_range_f32(extents.z),
        )
    }

    fn index(&mut self, len: usize) -> usize {
        if len == 0 {
            0
        } else {
            (self.next_u64() as usize) % len
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomizedSceneFrame {
    pub scene: Scene,
    pub frame_seed: u64,
    pub objects: Vec<RandomizedObjectMetadata>,
    pub metadata: DomainRandomizationFrameMetadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DomainRandomizationFrameMetadata {
    pub enabled: bool,
    pub seed: u64,
    pub frame_seed: u64,
    pub per_frame: bool,
    pub objects: Vec<RandomizedObjectMetadata>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomizedObjectMetadata {
    pub object_id: u32,
    pub name: String,
    pub primitive: String,
    pub material: String,
    pub transform: Transform,
    pub material_state: RandomizedMaterialMetadata,
    pub randomized: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomizedMaterialMetadata {
    pub base_color: Vec3,
    pub emission: Vec3,
    pub roughness: f32,
    pub metallic: f32,
    pub kind: MaterialKind,
}

pub fn randomize_scene_for_frame(
    base_scene: &Scene,
    config: &DomainRandomizationConfig,
    dataset_seed: u64,
    frame_index: u64,
) -> RandomizedSceneFrame {
    let seed = config.effective_seed(dataset_seed);
    let frame_seed = config.frame_seed(dataset_seed, frame_index);
    let mut scene = base_scene.clone();

    if config.enabled {
        let mut rng = DeterministicRng::new(frame_seed);
        let ids = scene.entities().map(|entity| entity.id).collect::<Vec<_>>();
        for id in ids {
            let Some(entity) = scene.entity_mut(id) else {
                continue;
            };

            if config.object_transforms.enabled
                && randomize_transform_for_shape(
                    entity.shape,
                    config.object_transforms.include_planes,
                )
            {
                entity.transform.translation = entity.transform.translation
                    + rng.jitter_vec3(config.object_transforms.position_jitter);
                let [scale_min, scale_max] = sorted_range(config.object_transforms.scale_range);
                let scale = rng.range_f32(scale_min, scale_max).max(0.001);
                entity.transform.scale = entity.transform.scale * scale;
            }

            let is_emissive = entity.material.kind == MaterialKind::Emissive
                || entity.material.emission.length_squared() > 0.0;
            if config.lights.enabled && is_emissive {
                entity.transform.translation =
                    entity.transform.translation + rng.jitter_vec3(config.lights.position_jitter);
                let [intensity_min, intensity_max] = sorted_range(config.lights.intensity_range);
                let intensity = rng.range_f32(intensity_min, intensity_max).max(0.0);
                entity.material.emission = entity.material.emission * intensity;
            }

            if config.materials.enabled {
                let jitter = config.materials.base_color_jitter.abs();
                if jitter > 0.0 {
                    entity.material.base_color = Vec3::new(
                        clamp01(entity.material.base_color.x + rng.signed_range_f32(jitter)),
                        clamp01(entity.material.base_color.y + rng.signed_range_f32(jitter)),
                        clamp01(entity.material.base_color.z + rng.signed_range_f32(jitter)),
                    );
                }

                if config.materials.randomize_kind && entity.material.kind != MaterialKind::Emissive
                {
                    let kinds = [
                        MaterialKind::Diffuse,
                        MaterialKind::Matte,
                        MaterialKind::MetalPreview,
                    ];
                    entity.material.kind = kinds[rng.index(kinds.len())];
                }

                if is_emissive {
                    let [intensity_min, intensity_max] =
                        sorted_range(config.materials.emissive_intensity_range);
                    let intensity = rng.range_f32(intensity_min, intensity_max).max(0.0);
                    entity.material.emission = entity.material.emission * intensity;
                }
            }
        }
    }

    let objects: Vec<RandomizedObjectMetadata> = scene
        .entities()
        .filter(|entity| entity.object_id.get() != 0)
        .map(|entity| RandomizedObjectMetadata {
            object_id: entity.object_id.get(),
            name: entity.name.clone(),
            primitive: primitive_label(entity.shape).to_string(),
            material: entity.material.kind.as_str().to_string(),
            transform: entity.transform,
            material_state: RandomizedMaterialMetadata {
                base_color: entity.material.base_color,
                emission: entity.material.emission,
                roughness: entity.material.roughness,
                metallic: entity.material.metallic,
                kind: entity.material.kind,
            },
            randomized: config.enabled,
        })
        .collect();

    RandomizedSceneFrame {
        scene,
        frame_seed,
        objects: objects.clone(),
        metadata: DomainRandomizationFrameMetadata {
            enabled: config.enabled,
            seed,
            frame_seed,
            per_frame: config.per_frame,
            objects,
        },
    }
}

pub fn randomize_camera_for_frame(
    mut camera: Camera,
    config: &DomainRandomizationConfig,
    dataset_seed: u64,
    frame_index: u64,
) -> Camera {
    if !config.enabled || !config.camera.enabled {
        return camera;
    }

    let mut rng =
        DeterministicRng::new(config.frame_seed(dataset_seed, frame_index) ^ 0x4341_4d45_5241);
    camera.position = camera.position + rng.jitter_vec3(config.camera.pose_jitter);
    if let Some(range) = config.camera.fov_degrees_range {
        let [min_fov, max_fov] = sorted_range(range);
        camera.vertical_fov_degrees = rng.range_f32(min_fov, max_fov).max(1.0);
    }
    camera
}

fn randomize_transform_for_shape(shape: PrimitiveShape, include_planes: bool) -> bool {
    match shape {
        PrimitiveShape::Sphere { .. } | PrimitiveShape::Box { .. } => true,
        PrimitiveShape::Plane { .. } => include_planes,
    }
}

fn primitive_label(shape: PrimitiveShape) -> &'static str {
    match shape {
        PrimitiveShape::Sphere { .. } => "sphere",
        PrimitiveShape::Box { .. } => "box",
        PrimitiveShape::Plane { .. } => "plane",
    }
}

fn sorted_range(range: [f32; 2]) -> [f32; 2] {
    if range[0] <= range[1] {
        range
    } else {
        [range[1], range[0]]
    }
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameOutputPaths {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rgb: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segmentation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segmentation_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lidar_range: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lidar_points: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lidar_object_ids: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lidar_preview: Option<String>,
    pub metadata: String,
}

impl FrameOutputPaths {
    pub fn to_manifest_frame(&self, frame_index: u64) -> ManifestFrame {
        ManifestFrame {
            frame_index,
            rgb: self.rgb.clone(),
            depth: self.depth.clone(),
            depth_preview: self.depth_preview.clone(),
            segmentation: self.segmentation.clone(),
            segmentation_preview: self.segmentation_preview.clone(),
            lidar_range: self.lidar_range.clone(),
            lidar_points: self.lidar_points.clone(),
            lidar_object_ids: self.lidar_object_ids.clone(),
            lidar_preview: self.lidar_preview.clone(),
            metadata: self.metadata.clone(),
        }
    }

    pub fn to_frame_outputs(&self) -> Vec<FrameOutputMetadata> {
        let mut outputs = Vec::new();
        if let Some(path) = &self.rgb {
            outputs.push(FrameOutputMetadata::new("rgb", "ppm", path.clone()));
        }
        if let Some(path) = &self.depth {
            outputs.push(FrameOutputMetadata::new(
                "depth",
                "raw-f32-little-endian",
                path.clone(),
            ));
        }
        if let Some(path) = &self.depth_preview {
            outputs.push(FrameOutputMetadata::new(
                "depth_preview",
                "pgm",
                path.clone(),
            ));
        }
        if let Some(path) = &self.segmentation {
            outputs.push(FrameOutputMetadata::new(
                "segmentation",
                "raw-u32-little-endian",
                path.clone(),
            ));
        }
        if let Some(path) = &self.segmentation_preview {
            outputs.push(FrameOutputMetadata::new(
                "segmentation_preview",
                "ppm",
                path.clone(),
            ));
        }
        if let Some(path) = &self.lidar_range {
            outputs.push(FrameOutputMetadata::new(
                "lidar_range",
                "raw-f32-little-endian",
                path.clone(),
            ));
        }
        if let Some(path) = &self.lidar_points {
            outputs.push(FrameOutputMetadata::new(
                "lidar_points",
                "xyz",
                path.clone(),
            ));
        }
        if let Some(path) = &self.lidar_object_ids {
            outputs.push(FrameOutputMetadata::new(
                "lidar_object_ids",
                "raw-u32-little-endian",
                path.clone(),
            ));
        }
        if let Some(path) = &self.lidar_preview {
            outputs.push(FrameOutputMetadata::new(
                "lidar_preview",
                "pgm",
                path.clone(),
            ));
        }
        outputs.push(FrameOutputMetadata::new(
            "metadata",
            "json",
            self.metadata.clone(),
        ));
        outputs
    }
}

pub fn frame_output_paths(frame_index: u64) -> FrameOutputPaths {
    frame_output_paths_for_selection(frame_index, OutputSelection::all())
}

pub fn frame_output_paths_for_selection(
    frame_index: u64,
    outputs: OutputSelection,
) -> FrameOutputPaths {
    frame_output_paths_for_selection_and_lidar(frame_index, outputs, false)
}

pub fn frame_output_paths_for_config(frame_index: u64, config: &DatasetConfig) -> FrameOutputPaths {
    frame_output_paths_for_selection_and_lidar(frame_index, config.outputs, config.lidar.enabled)
}

pub fn frame_output_paths_for_selection_and_lidar(
    frame_index: u64,
    outputs: OutputSelection,
    lidar_enabled: bool,
) -> FrameOutputPaths {
    let file_stem = format!("frame_{frame_index:06}");
    let outputs = outputs.normalized();
    FrameOutputPaths {
        rgb: outputs.rgb.then(|| format!("rgb/{file_stem}.ppm")),
        depth: outputs.depth.then(|| format!("depth/{file_stem}.f32")),
        depth_preview: outputs
            .depth_preview
            .then(|| format!("depth_preview/{file_stem}.pgm")),
        segmentation: outputs
            .segmentation
            .then(|| format!("segmentation/{file_stem}.u32")),
        segmentation_preview: outputs
            .segmentation_preview
            .then(|| format!("segmentation_preview/{file_stem}.ppm")),
        lidar_range: lidar_enabled.then(|| format!("lidar_range/{file_stem}.f32")),
        lidar_points: lidar_enabled.then(|| format!("lidar_points/{file_stem}.xyz")),
        lidar_object_ids: lidar_enabled.then(|| format!("lidar_object_ids/{file_stem}.u32")),
        lidar_preview: lidar_enabled.then(|| format!("lidar_preview/{file_stem}.pgm")),
        metadata: format!("metadata/{file_stem}.json"),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthConventionMetadata {
    pub convention: String,
    pub units: String,
    pub miss_value: f32,
}

impl DepthConventionMetadata {
    pub fn linear_ray_distance_meters() -> Self {
        Self {
            convention: "linear camera ray distance".to_string(),
            units: "meters".to_string(),
            miss_value: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentationConventionMetadata {
    pub convention: String,
    pub background_id: u32,
}

impl SegmentationConventionMetadata {
    pub fn object_ids_u32() -> Self {
        Self {
            convention: "u32 object IDs".to_string(),
            background_id: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LidarConventionMetadata {
    pub range_convention: String,
    pub point_convention: String,
    pub object_id_convention: String,
    pub range_units: String,
    pub miss_range_m: f32,
    pub miss_point: Vec3,
    pub miss_object_id: u32,
}

impl LidarConventionMetadata {
    pub fn single_return_linear_range() -> Self {
        Self {
            range_convention: "linear LiDAR ray distance".to_string(),
            point_convention: "world-space XYZ point at first hit".to_string(),
            object_id_convention: "u32 object IDs".to_string(),
            range_units: "meters".to_string(),
            miss_range_m: 0.0,
            miss_point: Vec3::ZERO,
            miss_object_id: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LidarFrameMetadata {
    pub enabled: bool,
    pub horizontal_samples: u32,
    pub vertical_channels: u32,
    pub horizontal_fov_degrees: f32,
    pub vertical_fov_degrees: f32,
    pub min_range_m: f32,
    pub max_range_m: f32,
    pub pose: Transform,
    pub convention: LidarConventionMetadata,
    pub outputs: Vec<FrameOutputMetadata>,
}

impl LidarFrameMetadata {
    pub fn new(config: DatasetLidarConfig, paths: &FrameOutputPaths) -> Self {
        let config = config.normalized();
        let mut outputs = Vec::new();
        if let Some(path) = &paths.lidar_range {
            outputs.push(FrameOutputMetadata::new(
                "range",
                "raw-f32-little-endian",
                path.clone(),
            ));
        }
        if let Some(path) = &paths.lidar_points {
            outputs.push(FrameOutputMetadata::new("points", "xyz", path.clone()));
        }
        if let Some(path) = &paths.lidar_object_ids {
            outputs.push(FrameOutputMetadata::new(
                "object_ids",
                "raw-u32-little-endian",
                path.clone(),
            ));
        }
        if let Some(path) = &paths.lidar_preview {
            outputs.push(FrameOutputMetadata::new("preview", "pgm", path.clone()));
        }

        Self {
            enabled: config.enabled,
            horizontal_samples: config.horizontal_samples,
            vertical_channels: config.vertical_channels,
            horizontal_fov_degrees: config.horizontal_fov_degrees,
            vertical_fov_degrees: config.vertical_fov_degrees,
            min_range_m: config.min_range_m,
            max_range_m: config.max_range_m,
            pose: config.pose,
            convention: LidarConventionMetadata::single_return_linear_range(),
            outputs,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraFrameMetadata {
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub intrinsics: CameraIntrinsics,
}

impl CameraFrameMetadata {
    pub fn from_camera(camera: &Camera) -> Self {
        Self {
            position: camera.position,
            forward: camera.forward,
            right: camera.right(),
            up: camera.up,
            intrinsics: CameraIntrinsics::from_camera(camera),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetFrameMetadata {
    pub frame_index: u64,
    pub timestamp_seconds: f64,
    pub sensor_id: String,
    pub seed: u64,
    pub camera_path: String,
    pub camera: CameraFrameMetadata,
    pub width: u32,
    pub height: u32,
    pub outputs: Vec<FrameOutputMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene_path: Option<String>,
    pub object_ids: Vec<ObjectIdMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub renderer_backend: Option<String>,
    pub depth_convention: DepthConventionMetadata,
    pub segmentation_convention: SegmentationConventionMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lidar: Option<LidarFrameMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_randomization: Option<DomainRandomizationFrameMetadata>,
}

impl DatasetFrameMetadata {
    pub fn new(
        frame_index: u64,
        timestamp_seconds: f64,
        sensor_id: impl Into<String>,
        seed: u64,
        camera_path: &CameraPathConfig,
        camera: &Camera,
        paths: &FrameOutputPaths,
        scene_path: Option<String>,
        object_ids: Vec<ObjectIdMetadata>,
        renderer_backend: Option<String>,
    ) -> Self {
        Self {
            frame_index,
            timestamp_seconds,
            sensor_id: sensor_id.into(),
            seed,
            camera_path: camera_path.kind().to_string(),
            camera: CameraFrameMetadata::from_camera(camera),
            width: camera.width,
            height: camera.height,
            outputs: paths.to_frame_outputs(),
            scene_path,
            object_ids,
            renderer_backend,
            depth_convention: DepthConventionMetadata::linear_ray_distance_meters(),
            segmentation_convention: SegmentationConventionMetadata::object_ids_u32(),
            lidar: None,
            domain_randomization: None,
        }
    }

    pub fn with_randomization(mut self, randomization: DomainRandomizationFrameMetadata) -> Self {
        self.domain_randomization = Some(randomization);
        self
    }

    pub fn with_lidar(mut self, lidar: LidarFrameMetadata) -> Self {
        self.lidar = Some(lidar);
        self
    }
}

/// Host RGB image in packed `0x00RRGGBB` format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
}

impl RgbImage {
    pub fn new(width: u32, height: u32, pixels: Vec<u32>) -> Result<Self> {
        let expected = width as usize * height as usize;
        if pixels.len() != expected {
            return Err(DatasetError::InvalidImage(format!(
                "expected {expected} pixels for {width}x{height}, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn from_frame(frame: RgbFrame) -> Result<Self> {
        Self::new(frame.width, frame.height, frame.pixels)
    }

    /// Deterministic CPU preview image used only when the ROCm backend is not
    /// compiled in. This keeps CLI smoke tests useful on machines without ROCm.
    pub fn synthetic_preview(width: u32, height: u32, frame_index: u64) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let mut pixels = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let checker = ((x / 32) + (y / 32) + frame_index as u32) & 1;
                let red = ((x as f32 / width as f32) * 180.0 + 40.0) as u32;
                let green = ((y as f32 / height as f32) * 160.0 + 55.0) as u32;
                let blue = if checker == 0 { 210 } else { 120 };
                pixels.push((red.min(255) << 16) | (green.min(255) << 8) | blue);
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }
}

/// Host depth image. Values are linear camera ray distance in meters.
/// `0.0` means background/miss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<f32>,
}

impl DepthImage {
    pub fn new(width: u32, height: u32, pixels: Vec<f32>) -> Result<Self> {
        let expected = width as usize * height as usize;
        if pixels.len() != expected {
            return Err(DatasetError::InvalidImage(format!(
                "expected {expected} depth samples for {width}x{height}, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn from_frame(frame: DepthFrame) -> Result<Self> {
        Self::new(frame.width, frame.height, frame.pixels)
    }

    pub fn synthetic_preview(width: u32, height: u32, frame_index: u64) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let mut pixels = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let miss = ((x / 80) + (y / 80) + frame_index as u32) % 7 == 0;
                if miss {
                    pixels.push(0.0);
                } else {
                    let nx = x as f32 / width as f32;
                    let ny = y as f32 / height as f32;
                    pixels.push(1.0 + nx * 2.5 + ny * 4.0);
                }
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }
}

/// Host segmentation image with stable `u32` object IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentationImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
}

impl SegmentationImage {
    pub fn new(width: u32, height: u32, pixels: Vec<u32>) -> Result<Self> {
        let expected = width as usize * height as usize;
        if pixels.len() != expected {
            return Err(DatasetError::InvalidImage(format!(
                "expected {expected} segmentation samples for {width}x{height}, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn from_frame(frame: SegmentationFrame) -> Result<Self> {
        Self::new(frame.width, frame.height, frame.pixels)
    }

    pub fn synthetic_preview(width: u32, height: u32, frame_index: u64) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let mut pixels = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let id = match ((x * 4 / width) + (y * 3 / height) + frame_index as u32) % 5 {
                    0 => 0,
                    1 => 1,
                    2 => 2,
                    3 => 3,
                    _ => 4,
                };
                pixels.push(id);
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorImageSet {
    pub rgb: RgbImage,
    pub depth: DepthImage,
    pub segmentation: SegmentationImage,
}

impl SensorImageSet {
    pub fn from_frames(
        rgb: RgbFrame,
        depth: DepthFrame,
        segmentation: SegmentationFrame,
    ) -> Result<Self> {
        Ok(Self {
            rgb: RgbImage::from_frame(rgb)?,
            depth: DepthImage::from_frame(depth)?,
            segmentation: SegmentationImage::from_frame(segmentation)?,
        })
    }

    pub fn synthetic_preview(width: u32, height: u32, frame_index: u64) -> Self {
        Self {
            rgb: RgbImage::synthetic_preview(width, height, frame_index),
            depth: DepthImage::synthetic_preview(width, height, frame_index),
            segmentation: SegmentationImage::synthetic_preview(width, height, frame_index),
        }
    }
}

pub fn synthetic_lidar_frame(
    config: LidarConfig,
    frame_index: u64,
    metadata: FrameMetadata,
) -> LidarFrame {
    let config = config.normalized();
    let mut ranges_m = Vec::with_capacity(config.sample_count());
    let mut points_xyz = Vec::with_capacity(config.sample_count());
    let mut object_ids = Vec::with_capacity(config.sample_count());
    for y in 0..config.vertical_channels {
        for x in 0..config.horizontal_samples {
            let miss = ((x / 16) + y + frame_index as u32) % 11 == 0;
            if miss {
                ranges_m.push(0.0);
                points_xyz.push(Vec3::ZERO);
                object_ids.push(0);
            } else {
                let nx = x as f32 / config.horizontal_samples.max(1) as f32;
                let ny = y as f32 / config.vertical_channels.max(1) as f32;
                let range = 2.0 + nx * 8.0 + ny * 3.0;
                ranges_m.push(range);
                points_xyz.push(Vec3::new(nx - 0.5, ny - 0.5, -range));
                object_ids.push(1 + ((x + y + frame_index as u32) % 4));
            }
        }
    }
    LidarFrame::new(
        config.horizontal_samples,
        config.vertical_channels,
        metadata,
        ranges_m,
        points_xyz,
        object_ids,
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestFrame {
    pub frame_index: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rgb: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segmentation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segmentation_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lidar_range: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lidar_points: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lidar_object_ids: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lidar_preview: Option<String>,
    pub metadata: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub dataset_format_version: u32,
    pub generator: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    pub frame_count: u32,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub camera_path: CameraPathConfig,
    pub outputs: OutputSelection,
    pub object_ids: Vec<ObjectIdMetadata>,
    pub frames: Vec<ManifestFrame>,
    pub depth_convention: DepthConventionMetadata,
    pub segmentation_convention: SegmentationConventionMetadata,
    #[serde(default, skip_serializing_if = "DatasetLidarConfig::is_disabled")]
    pub lidar: DatasetLidarConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lidar_convention: Option<LidarConventionMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_randomization: Option<DomainRandomizationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub renderer_backend: Option<String>,
}

impl DatasetManifest {
    pub fn new(
        frame_count: u32,
        width: u32,
        height: u32,
        seed: u64,
        camera_path: CameraPathConfig,
        outputs: OutputSelection,
    ) -> Self {
        Self {
            dataset_format_version: 1,
            generator: "rocm-oxide-sim dataset_generator".to_string(),
            scene_path: None,
            config_path: None,
            frame_count,
            width,
            height,
            seed,
            camera_path,
            outputs: outputs.normalized(),
            object_ids: Vec::new(),
            frames: Vec::new(),
            depth_convention: DepthConventionMetadata::linear_ray_distance_meters(),
            segmentation_convention: SegmentationConventionMetadata::object_ids_u32(),
            lidar: DatasetLidarConfig::default(),
            lidar_convention: None,
            domain_randomization: None,
            renderer_backend: None,
        }
    }

    pub fn with_scene_path(mut self, scene_path: Option<String>) -> Self {
        self.scene_path = scene_path;
        self
    }

    pub fn with_config_path(mut self, config_path: Option<String>) -> Self {
        self.config_path = config_path;
        self
    }

    pub fn with_renderer_backend(mut self, renderer_backend: impl Into<String>) -> Self {
        self.renderer_backend = Some(renderer_backend.into());
        self
    }

    pub fn with_object_ids(mut self, object_ids: Vec<ObjectIdMetadata>) -> Self {
        self.object_ids = object_ids;
        self
    }

    pub fn with_frames(mut self, frames: Vec<ManifestFrame>) -> Self {
        self.frame_count = frames.len() as u32;
        self.frames = frames;
        self
    }

    pub fn with_domain_randomization(mut self, randomization: DomainRandomizationConfig) -> Self {
        self.domain_randomization = randomization.enabled.then_some(randomization);
        self
    }

    pub fn with_lidar(mut self, lidar: DatasetLidarConfig) -> Self {
        self.lidar = lidar.normalized();
        self.lidar_convention = self
            .lidar
            .enabled
            .then(LidarConventionMetadata::single_return_linear_range);
        self
    }
}

/// Writer for the initial RGB + metadata dataset layout.
#[derive(Debug)]
pub struct DatasetWriter {
    root: PathBuf,
    config: DatasetConfig,
    scene_path: Option<String>,
    config_path: Option<String>,
    renderer_backend: String,
    object_ids: Vec<ObjectIdMetadata>,
    frames: Vec<ManifestFrame>,
}

impl DatasetWriter {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        Self::new_with_config(
            root,
            DatasetConfig::default(),
            None,
            None,
            "unknown".to_string(),
            Vec::new(),
        )
    }

    pub fn new_with_config(
        root: impl AsRef<Path>,
        config: DatasetConfig,
        scene_path: Option<String>,
        config_path: Option<String>,
        renderer_backend: String,
        object_ids: Vec<ObjectIdMetadata>,
    ) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let config = config.normalized();
        if config.outputs.rgb {
            fs::create_dir_all(root.join("rgb"))?;
        }
        if config.outputs.depth {
            fs::create_dir_all(root.join("depth"))?;
        }
        if config.outputs.depth_preview {
            fs::create_dir_all(root.join("depth_preview"))?;
        }
        if config.outputs.segmentation {
            fs::create_dir_all(root.join("segmentation"))?;
        }
        if config.outputs.segmentation_preview {
            fs::create_dir_all(root.join("segmentation_preview"))?;
        }
        if config.lidar.enabled {
            fs::create_dir_all(root.join("lidar_range"))?;
            fs::create_dir_all(root.join("lidar_points"))?;
            fs::create_dir_all(root.join("lidar_object_ids"))?;
            fs::create_dir_all(root.join("lidar_preview"))?;
        }
        fs::create_dir_all(root.join("metadata"))?;
        Ok(Self {
            root,
            config,
            scene_path,
            config_path,
            renderer_backend,
            object_ids,
            frames: Vec::new(),
        })
    }

    pub fn write_rgb_frame(
        &mut self,
        frame_index: u64,
        image: &RgbImage,
        metadata: &FrameMetadata,
    ) -> Result<()> {
        let paths = frame_output_paths_for_selection(
            frame_index,
            OutputSelection {
                rgb: true,
                depth: false,
                depth_preview: false,
                segmentation: false,
                segmentation_preview: false,
                metadata: true,
            },
        );

        if let Some(path) = &paths.rgb {
            write_ppm(self.root.join(path), image)?;
        }
        write_metadata_json(self.root.join(&paths.metadata), metadata)?;

        self.frames.push(paths.to_manifest_frame(frame_index));
        Ok(())
    }

    pub fn write_sensor_outputs(
        &mut self,
        frame_index: u64,
        images: &SensorImageSet,
        metadata: &impl Serialize,
    ) -> Result<()> {
        self.write_dataset_outputs(frame_index, images, None, metadata)
    }

    pub fn write_dataset_outputs(
        &mut self,
        frame_index: u64,
        images: &SensorImageSet,
        lidar: Option<&LidarFrame>,
        metadata: &impl Serialize,
    ) -> Result<()> {
        let paths = frame_output_paths_for_config(frame_index, &self.config);

        if let Some(path) = &paths.rgb {
            write_ppm(self.root.join(path), &images.rgb)?;
        }
        if let Some(path) = &paths.depth {
            write_depth_f32(self.root.join(path), &images.depth)?;
        }
        if let Some(path) = &paths.depth_preview {
            write_depth_preview_pgm(self.root.join(path), &images.depth)?;
        }
        if let Some(path) = &paths.segmentation {
            write_segmentation_u32(self.root.join(path), &images.segmentation)?;
        }
        if let Some(path) = &paths.segmentation_preview {
            write_segmentation_preview_ppm(self.root.join(path), &images.segmentation)?;
        }
        if self.config.lidar.enabled {
            let lidar = lidar.ok_or_else(|| {
                DatasetError::InvalidImage(
                    "LiDAR is enabled in dataset config, but no LiDAR frame was provided"
                        .to_string(),
                )
            })?;
            if let Some(path) = &paths.lidar_range {
                write_lidar_range_f32(self.root.join(path), lidar)?;
            }
            if let Some(path) = &paths.lidar_points {
                write_lidar_points_xyz(self.root.join(path), lidar)?;
            }
            if let Some(path) = &paths.lidar_object_ids {
                write_lidar_object_ids_u32(self.root.join(path), lidar)?;
            }
            if let Some(path) = &paths.lidar_preview {
                write_lidar_preview_pgm(self.root.join(path), lidar)?;
            }
        }
        write_metadata_json(self.root.join(&paths.metadata), metadata)?;

        self.frames.push(paths.to_manifest_frame(frame_index));
        Ok(())
    }

    pub fn finish(&self) -> Result<DatasetManifest> {
        let mut manifest = DatasetManifest::new(
            self.frames.len() as u32,
            self.config.width,
            self.config.height,
            self.config.seed,
            self.config.camera_path.clone(),
            self.config.outputs,
        )
        .with_scene_path(self.scene_path.clone())
        .with_config_path(self.config_path.clone())
        .with_object_ids(self.object_ids.clone())
        .with_frames(self.frames.clone())
        .with_lidar(self.config.lidar)
        .with_domain_randomization(self.config.domain_randomization.clone());
        manifest.renderer_backend = Some(self.renderer_backend.clone());
        let path = self.root.join("dataset_manifest.json");
        let json = serde_json::to_string_pretty(&manifest)?;
        fs::write(path, json)?;
        Ok(manifest)
    }
}

pub fn write_ppm(path: impl AsRef<Path>, image: &RgbImage) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    write!(writer, "P6\n{} {}\n255\n", image.width, image.height)?;
    for &pixel in &image.pixels {
        let red = ((pixel >> 16) & 0xff) as u8;
        let green = ((pixel >> 8) & 0xff) as u8;
        let blue = (pixel & 0xff) as u8;
        writer.write_all(&[red, green, blue])?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_depth_f32(path: impl AsRef<Path>, image: &DepthImage) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for &sample in &image.pixels {
        writer.write_all(&sample.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_depth_preview_pgm(path: impl AsRef<Path>, image: &DepthImage) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    write!(writer, "P5\n{} {}\n255\n", image.width, image.height)?;
    writer.write_all(&depth_preview_pixels(image))?;
    writer.flush()?;
    Ok(())
}

pub fn depth_preview_pixels(image: &DepthImage) -> Vec<u8> {
    let finite_positive = image
        .pixels
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0);
    let (mut min_depth, mut max_depth) = (f32::INFINITY, f32::NEG_INFINITY);
    for value in finite_positive {
        min_depth = min_depth.min(value);
        max_depth = max_depth.max(value);
    }

    if !min_depth.is_finite() || !max_depth.is_finite() {
        return vec![0; image.pixels.len()];
    }

    image
        .pixels
        .iter()
        .map(|&value| {
            if !value.is_finite() || value <= 0.0 {
                return 0;
            }
            if (max_depth - min_depth).abs() <= f32::EPSILON {
                return 255;
            }
            let normalized = (value - min_depth) / (max_depth - min_depth);
            (255.0 - normalized * 223.0).round().clamp(32.0, 255.0) as u8
        })
        .collect()
}

pub fn write_segmentation_u32(path: impl AsRef<Path>, image: &SegmentationImage) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for &sample in &image.pixels {
        writer.write_all(&sample.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_segmentation_preview_ppm(
    path: impl AsRef<Path>,
    image: &SegmentationImage,
) -> Result<()> {
    let preview = RgbImage::new(
        image.width,
        image.height,
        image
            .pixels
            .iter()
            .copied()
            .map(segmentation_color)
            .collect(),
    )?;
    write_ppm(path, &preview)
}

pub fn write_lidar_range_f32(path: impl AsRef<Path>, frame: &LidarFrame) -> Result<()> {
    validate_lidar_frame(frame)?;
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for &sample in &frame.ranges_m {
        writer.write_all(&sample.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_lidar_object_ids_u32(path: impl AsRef<Path>, frame: &LidarFrame) -> Result<()> {
    validate_lidar_frame(frame)?;
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for &sample in &frame.object_ids {
        writer.write_all(&sample.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_lidar_points_xyz(path: impl AsRef<Path>, frame: &LidarFrame) -> Result<()> {
    validate_lidar_frame(frame)?;
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for point in &frame.points_xyz {
        writeln!(writer, "{} {} {}", point.x, point.y, point.z)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_lidar_preview_pgm(path: impl AsRef<Path>, frame: &LidarFrame) -> Result<()> {
    validate_lidar_frame(frame)?;
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    write!(writer, "P5\n{} {}\n255\n", frame.width, frame.height)?;
    writer.write_all(&lidar_range_preview_pixels(frame))?;
    writer.flush()?;
    Ok(())
}

pub fn lidar_range_preview_pixels(frame: &LidarFrame) -> Vec<u8> {
    let finite_positive = frame
        .ranges_m
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0);
    let (mut min_range, mut max_range) = (f32::INFINITY, f32::NEG_INFINITY);
    for value in finite_positive {
        min_range = min_range.min(value);
        max_range = max_range.max(value);
    }

    if !min_range.is_finite() || !max_range.is_finite() {
        return vec![0; frame.ranges_m.len()];
    }

    frame
        .ranges_m
        .iter()
        .map(|&value| {
            if !value.is_finite() || value <= 0.0 {
                return 0;
            }
            if (max_range - min_range).abs() <= f32::EPSILON {
                return 255;
            }
            let normalized = (value - min_range) / (max_range - min_range);
            (255.0 - normalized * 223.0).round().clamp(32.0, 255.0) as u8
        })
        .collect()
}

fn validate_lidar_frame(frame: &LidarFrame) -> Result<()> {
    let expected = frame.sample_count();
    if frame.ranges_m.len() != expected {
        return Err(DatasetError::InvalidImage(format!(
            "expected {expected} LiDAR range samples for {}x{}, got {}",
            frame.width,
            frame.height,
            frame.ranges_m.len()
        )));
    }
    if frame.points_xyz.len() != expected {
        return Err(DatasetError::InvalidImage(format!(
            "expected {expected} LiDAR point samples for {}x{}, got {}",
            frame.width,
            frame.height,
            frame.points_xyz.len()
        )));
    }
    if frame.object_ids.len() != expected {
        return Err(DatasetError::InvalidImage(format!(
            "expected {expected} LiDAR object ID samples for {}x{}, got {}",
            frame.width,
            frame.height,
            frame.object_ids.len()
        )));
    }
    Ok(())
}

pub fn segmentation_color(object_id: u32) -> u32 {
    match object_id {
        0 => 0x0000_0000,
        1 => 0x0080_8080,
        2 => 0x00e6_1f1a,
        3 => 0x001a_9e38,
        4 => 0x001a_47e6,
        other => {
            let hash = other.wrapping_mul(0x45d9_f3b);
            let red = 48 + (hash & 0x7f);
            let green = 48 + ((hash >> 8) & 0x7f);
            let blue = 48 + ((hash >> 16) & 0x7f);
            (red << 16) | (green << 8) | blue
        }
    }
}

pub fn write_metadata_json<T: Serialize>(path: impl AsRef<Path>, metadata: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(metadata)?;
    fs::write(path, json)?;
    Ok(())
}

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("missing expected file `{0}`")]
    MissingFile(String),
    #[error("invalid dataset manifest: {0}")]
    InvalidManifest(String),
    #[error("invalid frame metadata `{path}`: {message}")]
    InvalidMetadata { path: String, message: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub frame_count: u32,
}

pub fn validate_dataset(
    root: impl AsRef<Path>,
) -> std::result::Result<ValidationReport, ValidationError> {
    let root = root.as_ref();
    let manifest_path = root.join("dataset_manifest.json");
    if !manifest_path.exists() {
        return Err(ValidationError::MissingFile(
            "dataset_manifest.json".to_string(),
        ));
    }

    let manifest: DatasetManifest = serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
    if manifest.frames.len() as u32 != manifest.frame_count {
        return Err(ValidationError::InvalidManifest(format!(
            "frame_count is {}, but frames has {} entries",
            manifest.frame_count,
            manifest.frames.len()
        )));
    }
    if manifest.object_ids.is_empty() {
        return Err(ValidationError::InvalidManifest(
            "object ID map is empty".to_string(),
        ));
    }
    let randomization_enabled = manifest
        .domain_randomization
        .as_ref()
        .is_some_and(|config| config.enabled);
    let lidar_enabled = manifest.lidar.enabled;

    for frame in &manifest.frames {
        check_optional_file(root, frame.rgb.as_deref())?;
        check_optional_file(root, frame.depth.as_deref())?;
        check_optional_file(root, frame.depth_preview.as_deref())?;
        check_optional_file(root, frame.segmentation.as_deref())?;
        check_optional_file(root, frame.segmentation_preview.as_deref())?;
        if lidar_enabled {
            check_required_optional_file(root, frame.lidar_range.as_deref(), "lidar_range")?;
            check_required_optional_file(root, frame.lidar_points.as_deref(), "lidar_points")?;
            check_required_optional_file(
                root,
                frame.lidar_object_ids.as_deref(),
                "lidar_object_ids",
            )?;
            check_required_optional_file(root, frame.lidar_preview.as_deref(), "lidar_preview")?;
        } else {
            check_optional_file(root, frame.lidar_range.as_deref())?;
            check_optional_file(root, frame.lidar_points.as_deref())?;
            check_optional_file(root, frame.lidar_object_ids.as_deref())?;
            check_optional_file(root, frame.lidar_preview.as_deref())?;
        }
        check_file(root, &frame.metadata)?;

        let metadata_path = root.join(&frame.metadata);
        let metadata_text = fs::read_to_string(&metadata_path)?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata_text)?;
        let object_ids = metadata
            .get("object_ids")
            .and_then(|value| value.as_array())
            .ok_or_else(|| ValidationError::InvalidMetadata {
                path: frame.metadata.clone(),
                message: "missing object_ids array".to_string(),
            })?;
        if object_ids.is_empty() {
            return Err(ValidationError::InvalidMetadata {
                path: frame.metadata.clone(),
                message: "object_ids array is empty".to_string(),
            });
        }
        if let Some(width) = metadata.get("width").and_then(|value| value.as_u64())
            && width as u32 != manifest.width
        {
            return Err(ValidationError::InvalidMetadata {
                path: frame.metadata.clone(),
                message: format!(
                    "width {width} does not match manifest width {}",
                    manifest.width
                ),
            });
        }
        if let Some(height) = metadata.get("height").and_then(|value| value.as_u64())
            && height as u32 != manifest.height
        {
            return Err(ValidationError::InvalidMetadata {
                path: frame.metadata.clone(),
                message: format!(
                    "height {height} does not match manifest height {}",
                    manifest.height
                ),
            });
        }
        if randomization_enabled {
            let randomization = metadata.get("domain_randomization").ok_or_else(|| {
                ValidationError::InvalidMetadata {
                    path: frame.metadata.clone(),
                    message: "missing domain_randomization section".to_string(),
                }
            })?;
            if !randomization
                .get("enabled")
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
            {
                return Err(ValidationError::InvalidMetadata {
                    path: frame.metadata.clone(),
                    message: "domain_randomization.enabled is not true".to_string(),
                });
            }
            if randomization
                .get("frame_seed")
                .and_then(|value| value.as_u64())
                .is_none()
            {
                return Err(ValidationError::InvalidMetadata {
                    path: frame.metadata.clone(),
                    message: "missing domain_randomization.frame_seed".to_string(),
                });
            }
            let objects = randomization
                .get("objects")
                .and_then(|value| value.as_array())
                .ok_or_else(|| ValidationError::InvalidMetadata {
                    path: frame.metadata.clone(),
                    message: "missing domain_randomization.objects array".to_string(),
                })?;
            if objects.is_empty() {
                return Err(ValidationError::InvalidMetadata {
                    path: frame.metadata.clone(),
                    message: "domain_randomization.objects array is empty".to_string(),
                });
            }
        }
        if lidar_enabled {
            let lidar = metadata
                .get("lidar")
                .ok_or_else(|| ValidationError::InvalidMetadata {
                    path: frame.metadata.clone(),
                    message: "missing lidar section".to_string(),
                })?;
            if !lidar
                .get("enabled")
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
            {
                return Err(ValidationError::InvalidMetadata {
                    path: frame.metadata.clone(),
                    message: "lidar.enabled is not true".to_string(),
                });
            }
            let object_id_convention = lidar
                .get("convention")
                .and_then(|value| value.get("miss_object_id"))
                .and_then(|value| value.as_u64());
            if object_id_convention != Some(0) {
                return Err(ValidationError::InvalidMetadata {
                    path: frame.metadata.clone(),
                    message: "lidar convention must use object ID 0 for misses".to_string(),
                });
            }
        }
    }

    Ok(ValidationReport {
        frame_count: manifest.frame_count,
    })
}

fn check_optional_file(
    root: &Path,
    path: Option<&str>,
) -> std::result::Result<(), ValidationError> {
    if let Some(path) = path {
        check_file(root, path)?;
    }
    Ok(())
}

fn check_required_optional_file(
    root: &Path,
    path: Option<&str>,
    label: &str,
) -> std::result::Result<(), ValidationError> {
    let path = path.ok_or_else(|| {
        ValidationError::InvalidManifest(format!(
            "LiDAR is enabled, but manifest frame is missing {label}"
        ))
    })?;
    check_file(root, path)
}

fn check_file(root: &Path, path: &str) -> std::result::Result<(), ValidationError> {
    if root.join(path).exists() {
        Ok(())
    } else {
        Err(ValidationError::MissingFile(path.to_string()))
    }
}
