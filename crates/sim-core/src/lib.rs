//! CPU-only foundational simulator types.
//!
//! This crate deliberately has no ROCm dependency. It contains the small scene,
//! transform, camera, material, and ID types that higher-level sensor, renderer,
//! dataset, and future physics crates can share.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A compact 3-D vector used for positions, directions, scales, and colors.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 1.0, 1.0);
    pub const X: Self = Self::new(1.0, 0.0, 0.0);
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub const fn splat(value: f32) -> Self {
        Self::new(value, value, value)
    }

    pub fn dot(self, rhs: Self) -> f32 {
        self.x.mul_add(rhs.x, self.y.mul_add(rhs.y, self.z * rhs.z))
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len <= f32::EPSILON {
            Self::ZERO
        } else {
            self / len
        }
    }

    pub fn component_mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }

    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }

    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// A simple Euler rotation helper in radians.
///
/// The fields are applied as roll around X, pitch around Y, then yaw around Z.
/// This is intentionally simple for early simulator scaffolding; a quaternion
/// type can be added once interpolation or articulated bodies need it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EulerRotation {
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
}

impl EulerRotation {
    pub const IDENTITY: Self = Self::new(0.0, 0.0, 0.0);

    pub const fn new(roll: f32, pitch: f32, yaw: f32) -> Self {
        Self { roll, pitch, yaw }
    }

    pub fn from_degrees(roll: f32, pitch: f32, yaw: f32) -> Self {
        Self::new(roll.to_radians(), pitch.to_radians(), yaw.to_radians())
    }

    pub fn rotate_vector(self, vector: Vec3) -> Vec3 {
        let (sin_x, cos_x) = self.roll.sin_cos();
        let (sin_y, cos_y) = self.pitch.sin_cos();
        let (sin_z, cos_z) = self.yaw.sin_cos();

        let after_x = Vec3::new(
            vector.x,
            vector.y * cos_x - vector.z * sin_x,
            vector.y * sin_x + vector.z * cos_x,
        );
        let after_y = Vec3::new(
            after_x.x * cos_y + after_x.z * sin_y,
            after_x.y,
            -after_x.x * sin_y + after_x.z * cos_y,
        );
        Vec3::new(
            after_y.x * cos_z - after_y.y * sin_z,
            after_y.x * sin_z + after_y.y * cos_z,
            after_y.z,
        )
    }
}

impl Default for EulerRotation {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Position, rotation, and scale for a scene entity or sensor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: EulerRotation,
    pub scale: Vec3,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: EulerRotation::IDENTITY,
        scale: Vec3::ONE,
    };

    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            rotation: EulerRotation::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    pub fn transform_point(self, point: Vec3) -> Vec3 {
        self.rotation.rotate_vector(point.component_mul(self.scale)) + self.translation
    }

    pub fn transform_direction(self, direction: Vec3) -> Vec3 {
        self.rotation.rotate_vector(direction).normalized()
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Stable per-scene entity handle.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct EntityId(u64);

impl EntityId {
    pub const UNASSIGNED: Self = Self(0);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Segmentation ID assigned to an object.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct ObjectId(u32);

impl ObjectId {
    pub const BACKGROUND: Self = Self(0);

    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Minimal primitive shapes for early rendering and scene queries.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PrimitiveShape {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
    Plane { normal: Vec3, offset: f32 },
}

impl PrimitiveShape {
    pub const fn sphere(radius: f32) -> Self {
        Self::Sphere { radius }
    }

    pub const fn box_with_half_extents(half_extents: Vec3) -> Self {
        Self::Box { half_extents }
    }

    pub const fn plane(normal: Vec3, offset: f32) -> Self {
        Self::Plane { normal, offset }
    }
}

/// Small material classifier used by preview renderers.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialKind {
    Diffuse = 0,
    Emissive = 1,
    Matte = 2,
    MetalPreview = 3,
}

impl MaterialKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Diffuse => "diffuse",
            Self::Emissive => "emissive",
            Self::Matte => "matte",
            Self::MetalPreview => "metal_preview",
        }
    }

    pub const fn gpu_id(self) -> u32 {
        self as u32
    }
}

impl Default for MaterialKind {
    fn default() -> Self {
        Self::Matte
    }
}

/// A small physically-inspired material descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Material {
    pub base_color: Vec3,
    #[serde(default)]
    pub emission: Vec3,
    pub roughness: f32,
    pub metallic: f32,
    #[serde(default)]
    pub kind: MaterialKind,
}

impl Material {
    pub const fn new(base_color: Vec3, roughness: f32, metallic: f32) -> Self {
        Self {
            base_color,
            emission: Vec3::ZERO,
            roughness,
            metallic,
            kind: MaterialKind::Diffuse,
        }
    }

    pub const fn matte(base_color: Vec3) -> Self {
        Self {
            base_color,
            emission: Vec3::ZERO,
            roughness: 0.8,
            metallic: 0.0,
            kind: MaterialKind::Matte,
        }
    }

    pub const fn emissive(base_color: Vec3, emission: Vec3) -> Self {
        Self {
            base_color,
            emission,
            roughness: 0.0,
            metallic: 0.0,
            kind: MaterialKind::Emissive,
        }
    }

    pub const fn metal_preview(base_color: Vec3, roughness: f32) -> Self {
        Self {
            base_color,
            emission: Vec3::ZERO,
            roughness,
            metallic: 1.0,
            kind: MaterialKind::MetalPreview,
        }
    }

    pub const fn with_kind(mut self, kind: MaterialKind) -> Self {
        self.kind = kind;
        self
    }

    pub const fn with_emission(mut self, emission: Vec3) -> Self {
        self.emission = emission;
        self
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::matte(Vec3::splat(0.8))
    }
}

/// A scene object with shape, transform, material, and segmentation ID.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub name: String,
    pub shape: PrimitiveShape,
    pub transform: Transform,
    pub material: Material,
    pub object_id: ObjectId,
}

impl Entity {
    pub fn new(
        name: impl Into<String>,
        shape: PrimitiveShape,
        transform: Transform,
        material: Material,
        object_id: ObjectId,
    ) -> Self {
        Self {
            id: EntityId::UNASSIGNED,
            name: name.into(),
            shape,
            transform,
            material,
            object_id,
        }
    }
}

/// A simple collection of entities with ID assignment and query helpers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    entities: Vec<Entity>,
    next_id: u64,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            next_id: 1,
        }
    }

    pub fn default_sensor_scene() -> Self {
        let mut scene = Self::new();
        scene.add_entity(Entity::new(
            "ground",
            PrimitiveShape::plane(Vec3::Y, 0.0),
            Transform::default(),
            Material::matte(Vec3::new(0.55, 0.56, 0.52)),
            ObjectId::new(1),
        ));
        scene.add_entity(Entity::new(
            "red sphere",
            PrimitiveShape::sphere(0.55),
            Transform::from_translation(Vec3::new(-0.9, 0.55, -1.2)),
            Material::matte(Vec3::new(0.9, 0.12, 0.1)),
            ObjectId::new(2),
        ));
        scene.add_entity(Entity::new(
            "green sphere",
            PrimitiveShape::sphere(0.45),
            Transform::from_translation(Vec3::new(0.25, 0.45, -1.9)),
            Material::matte(Vec3::new(0.1, 0.62, 0.22)),
            ObjectId::new(3),
        ));
        scene.add_entity(Entity::new(
            "blue sphere",
            PrimitiveShape::sphere(0.35),
            Transform::from_translation(Vec3::new(1.05, 0.35, -1.45)),
            Material::matte(Vec3::new(0.1, 0.28, 0.9)),
            ObjectId::new(4),
        ));
        scene
    }

    pub fn add_entity(&mut self, mut entity: Entity) -> EntityId {
        let id = EntityId::new(self.next_id);
        self.next_id += 1;
        entity.id = id;
        self.entities.push(entity);
        id
    }

    pub fn entity(&self, id: EntityId) -> Option<&Entity> {
        self.entities.iter().find(|entity| entity.id == id)
    }

    pub fn entity_mut(&mut self, id: EntityId) -> Option<&mut Entity> {
        self.entities.iter_mut().find(|entity| entity.id == id)
    }

    pub fn by_object_id(&self, object_id: ObjectId) -> Option<&Entity> {
        self.entities
            .iter()
            .find(|entity| entity.object_id == object_id)
    }

    pub fn entities(&self) -> impl Iterator<Item = &Entity> {
        self.entities.iter()
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

/// Pinhole camera description used by sensor crates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub vertical_fov_degrees: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
    pub width: u32,
    pub height: u32,
}

impl Camera {
    pub fn look_at(
        position: Vec3,
        target: Vec3,
        vertical_fov_degrees: f32,
        aspect_ratio: f32,
    ) -> Self {
        let width = 640;
        let height = ((width as f32 / aspect_ratio).round() as u32).max(1);
        Self {
            position,
            forward: (target - position).normalized(),
            up: Vec3::Y,
            vertical_fov_degrees,
            aspect_ratio,
            near: 0.01,
            far: 1_000.0,
            width,
            height,
        }
    }

    pub fn default_rgb() -> Self {
        Self::look_at(
            Vec3::new(0.0, 1.1, 4.5),
            Vec3::new(0.0, 0.55, -1.4),
            55.0,
            16.0 / 9.0,
        )
        .with_resolution(640, 360)
    }

    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = width.max(1);
        self.height = height.max(1);
        self.aspect_ratio = self.width as f32 / self.height as f32;
        self
    }

    pub fn right(self) -> Vec3 {
        self.forward.cross(self.up).normalized()
    }
}
