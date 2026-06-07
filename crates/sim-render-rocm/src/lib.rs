//! ROCm-Oxide sensor renderer.
//!
//! The first backend is intentionally narrow: an opt-in HIPRTC renderer for
//! uploaded [`sim_core::Scene`] sphere and plane primitives. It writes RGB,
//! linear ray depth, and segmentation buffers in one pass. It is not a path
//! tracer and does not build acceleration structures yet.

use sim_core::{PrimitiveShape, Scene, Vec3};
use sim_sensors::{DepthFrame, FrameMetadata, RgbCameraSensor, RgbFrame, SegmentationFrame};
use thiserror::Error;

/// GPU ABI vector with 16-byte stride.
///
/// The HIPRTC kernel declares the same shape as `struct GpuVec3 { float x, y, z,
/// _pad; }`. Keeping this as `repr(C)` and four `f32` fields avoids Rust/C
/// `float3` layout surprises.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

impl GpuVec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, _pad: 0.0 }
    }
}

impl From<Vec3> for GpuVec3 {
    fn from(value: Vec3) -> Self {
        Self::new(value.x, value.y, value.z)
    }
}

/// GPU sphere primitive.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuSphere {
    pub center: GpuVec3,
    pub radius: f32,
    pub material_id: u32,
    pub object_id: u32,
    pub _pad0: u32,
}

/// GPU plane primitive represented by a point and normalized normal.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuPlane {
    pub point: GpuVec3,
    pub normal: GpuVec3,
    pub material_id: u32,
    pub object_id: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// GPU material. Only `base_color` is used by Milestone 3 shading.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuMaterial {
    pub base_color: GpuVec3,
    pub emission: GpuVec3,
    pub roughness: f32,
    pub material_kind: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[cfg(feature = "rocm")]
unsafe impl rocm_oxide::DevicePod for GpuVec3 {}
#[cfg(feature = "rocm")]
unsafe impl rocm_oxide::DevicePod for GpuSphere {}
#[cfg(feature = "rocm")]
unsafe impl rocm_oxide::DevicePod for GpuPlane {}
#[cfg(feature = "rocm")]
unsafe impl rocm_oxide::DevicePod for GpuMaterial {}

#[cfg(feature = "rocm")]
unsafe impl rocm_oxide::DevicePod for CameraParams {}
#[cfg(feature = "rocm")]
unsafe impl rocm_oxide::DevicePod for RenderParams {}

#[derive(Debug, Clone, PartialEq)]
pub struct HostSceneBuffers {
    pub spheres: Vec<GpuSphere>,
    pub planes: Vec<GpuPlane>,
    pub materials: Vec<GpuMaterial>,
}

impl HostSceneBuffers {
    pub fn sphere_count(&self) -> u32 {
        self.spheres.len() as u32
    }

    pub fn plane_count(&self) -> u32 {
        self.planes.len() as u32
    }

    pub fn material_count(&self) -> u32 {
        self.materials.len() as u32
    }
}

#[cfg(feature = "rocm")]
pub struct RocmScene {
    spheres: rocm_oxide::DeviceBuffer<GpuSphere>,
    planes: rocm_oxide::DeviceBuffer<GpuPlane>,
    materials: rocm_oxide::DeviceBuffer<GpuMaterial>,
    sphere_count: u32,
    plane_count: u32,
    material_count: u32,
}

#[cfg(feature = "rocm")]
impl RocmScene {
    pub fn sphere_count(&self) -> u32 {
        self.sphere_count
    }

    pub fn plane_count(&self) -> u32 {
        self.plane_count
    }

    pub fn material_count(&self) -> u32 {
        self.material_count
    }
}

#[cfg(not(feature = "rocm"))]
pub struct RocmScene {
    sphere_count: u32,
    plane_count: u32,
    material_count: u32,
}

#[cfg(not(feature = "rocm"))]
impl RocmScene {
    pub fn sphere_count(&self) -> u32 {
        self.sphere_count
    }

    pub fn plane_count(&self) -> u32 {
        self.plane_count
    }

    pub fn material_count(&self) -> u32 {
        self.material_count
    }
}

#[cfg(feature = "rocm")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CameraParams {
    origin: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    forward: [f32; 4],
    tan_half_fov_y: f32,
    aspect: f32,
    _padding: [f32; 2],
}

#[cfg(feature = "rocm")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct RenderParams {
    camera: CameraParams,
    width: u32,
    height: u32,
    sphere_count: u32,
    plane_count: u32,
    material_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[cfg(feature = "rocm")]
impl CameraParams {
    fn from_sensor(sensor: &RgbCameraSensor) -> Self {
        let camera = sensor.camera();
        let right = camera.right();
        let up = camera.up.normalized();
        let forward = camera.forward.normalized();
        Self {
            origin: [camera.position.x, camera.position.y, camera.position.z, 0.0],
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            forward: [forward.x, forward.y, forward.z, 0.0],
            tan_half_fov_y: (camera.vertical_fov_degrees.to_radians() * 0.5).tan(),
            aspect: camera.aspect_ratio,
            _padding: [0.0, 0.0],
        }
    }
}

#[cfg(feature = "rocm")]
const SENSOR_OUTPUTS_KERNEL: &str = r#"
struct GpuVec3 {
    float x;
    float y;
    float z;
    float _pad;
};

struct GpuSphere {
    GpuVec3 center;
    float radius;
    unsigned int material_id;
    unsigned int object_id;
    unsigned int _pad0;
};

struct GpuPlane {
    GpuVec3 point;
    GpuVec3 normal;
    unsigned int material_id;
    unsigned int object_id;
    unsigned int _pad0;
    unsigned int _pad1;
};

struct GpuMaterial {
    GpuVec3 base_color;
    GpuVec3 emission;
    float roughness;
    unsigned int material_kind;
    unsigned int _pad0;
    unsigned int _pad1;
};

struct V3 {
    float x;
    float y;
    float z;
};

__device__ V3 v3(float x, float y, float z) {
    V3 out = {x, y, z};
    return out;
}

__device__ V3 add(V3 a, V3 b) {
    return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ V3 sub(V3 a, V3 b) {
    return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ V3 mul(V3 a, float s) {
    return v3(a.x * s, a.y * s, a.z * s);
}

__device__ float dot3(V3 a, V3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ V3 norm(V3 a) {
    float len = sqrtf(fmaxf(dot3(a, a), 1.0e-20f));
    return mul(a, 1.0f / len);
}

struct CameraParams {
    float4 origin;
    float4 right;
    float4 up;
    float4 forward;
    float tan_half_fov_y;
    float aspect;
    float2 padding;
};

struct RenderParams {
    CameraParams camera;
    unsigned int width;
    unsigned int height;
    unsigned int sphere_count;
    unsigned int plane_count;
    unsigned int material_count;
    unsigned int _pad0;
    unsigned int _pad1;
    unsigned int _pad2;
};

__device__ V3 from_float4(float4 value) {
    return v3(value.x, value.y, value.z);
}

__device__ V3 from_gpu_vec3(GpuVec3 value) {
    return v3(value.x, value.y, value.z);
}

__device__ float clamp01(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ unsigned int pack_rgb(V3 color) {
    unsigned int r = (unsigned int)(clamp01(color.x) * 255.0f + 0.5f);
    unsigned int g = (unsigned int)(clamp01(color.y) * 255.0f + 0.5f);
    unsigned int b = (unsigned int)(clamp01(color.z) * 255.0f + 0.5f);
    return (r << 16) | (g << 8) | b;
}

__device__ float hit_sphere(V3 ray_origin, V3 ray_dir, V3 center, float radius) {
    V3 oc = sub(ray_origin, center);
    float b = dot3(oc, ray_dir);
    float c = dot3(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0f) {
        return -1.0f;
    }
    h = sqrtf(h);
    float t = -b - h;
    if (t > 0.001f) {
        return t;
    }
    t = -b + h;
    return t > 0.001f ? t : -1.0f;
}

__device__ float hit_plane(V3 ray_origin, V3 ray_dir, V3 point, V3 normal) {
    float denom = dot3(ray_dir, normal);
    if (fabsf(denom) < 1.0e-6f) {
        return -1.0f;
    }
    float t = dot3(sub(point, ray_origin), normal) / denom;
    return t > 0.001f ? t : -1.0f;
}

__device__ V3 material_color(const GpuMaterial* materials, unsigned int material_count, unsigned int material_id) {
    if (material_id < material_count) {
        return from_gpu_vec3(materials[material_id].base_color);
    }
    return v3(0.8f, 0.2f, 0.8f);
}

extern "C" __global__
void render_sensor_outputs(
    unsigned int* rgb,
    float* depth,
    unsigned int* segmentation,
    const GpuSphere* spheres,
    const GpuPlane* planes,
    const GpuMaterial* materials,
    RenderParams params
) {
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int width = params.width;
    unsigned int height = params.height;
    unsigned long pixel_count = (unsigned long)width * (unsigned long)height;
    if (i >= pixel_count) {
        return;
    }

    unsigned int x = (unsigned int)(i % width);
    unsigned int y = (unsigned int)(i / width);

    float px = (((float)x + 0.5f) / (float)width) * 2.0f - 1.0f;
    float py = 1.0f - (((float)y + 0.5f) / (float)height) * 2.0f;

    CameraParams camera = params.camera;
    V3 ray_origin = from_float4(camera.origin);
    V3 ray_dir = norm(add(
        from_float4(camera.forward),
        add(
            mul(from_float4(camera.right), px * camera.aspect * camera.tan_half_fov_y),
            mul(from_float4(camera.up), py * camera.tan_half_fov_y)
        )
    ));
    V3 light_dir = norm(v3(-0.45f, 0.9f, 0.25f));

    float best_t = 1.0e20f;
    V3 best_color = v3(0.55f + 0.20f * py, 0.70f + 0.12f * py, 0.92f);
    V3 best_normal = v3(0.0f, 1.0f, 0.0f);
    unsigned int best_object_id = 0;
    int hit = 0;

    for (unsigned int sphere = 0; sphere < params.sphere_count; ++sphere) {
        GpuSphere primitive = spheres[sphere];
        float t = hit_sphere(ray_origin, ray_dir, from_gpu_vec3(primitive.center), primitive.radius);
        if (t > 0.0f && t < best_t) {
            best_t = t;
            V3 point = add(ray_origin, mul(ray_dir, t));
            best_normal = norm(sub(point, from_gpu_vec3(primitive.center)));
            best_color = material_color(materials, params.material_count, primitive.material_id);
            best_object_id = primitive.object_id;
            hit = 1;
        }
    }

    for (unsigned int plane = 0; plane < params.plane_count; ++plane) {
        GpuPlane primitive = planes[plane];
        V3 plane_point = from_gpu_vec3(primitive.point);
        V3 plane_normal = norm(from_gpu_vec3(primitive.normal));
        float t = hit_plane(ray_origin, ray_dir, plane_point, plane_normal);
        if (t > 0.0f && t < best_t) {
            best_t = t;
            best_color = material_color(materials, params.material_count, primitive.material_id);
            best_normal = plane_normal;
            best_object_id = primitive.object_id;
            hit = 1;
        }
    }

    if (hit) {
        float diffuse = fmaxf(dot3(best_normal, light_dir), 0.0f);
        float shade = 0.16f + 0.84f * diffuse;
        best_color = mul(best_color, shade);
    }

    rgb[i] = pack_rgb(best_color);
    depth[i] = hit ? best_t : 0.0f;
    segmentation[i] = best_object_id;
}
"#;

#[derive(Debug, Error)]
pub enum RocmRenderError {
    #[error(
        "sim-render-rocm was built without the `rocm` feature; rerun with `--features rocm` on this package or the app package"
    )]
    BackendUnavailable,
    #[error("invalid sensor render target: {0}")]
    InvalidTarget(String),
    #[error("scene entity `{entity}` uses unsupported primitive `{primitive}`")]
    UnsupportedPrimitive { entity: String, primitive: String },
    #[cfg(feature = "rocm")]
    #[error("ROCm-Oxide failed while {context}: {message}")]
    RocmOxide { context: String, message: String },
}

pub type Result<T> = std::result::Result<T, RocmRenderError>;

pub fn build_gpu_scene_buffers(scene: &Scene) -> Result<HostSceneBuffers> {
    let mut spheres = Vec::new();
    let mut planes = Vec::new();
    let mut materials = Vec::new();

    for entity in scene.entities() {
        let material_id = u32::try_from(materials.len()).map_err(|_| {
            RocmRenderError::InvalidTarget("scene has more than u32::MAX materials".to_string())
        })?;
        materials.push(GpuMaterial {
            base_color: entity.material.base_color.into(),
            emission: GpuVec3::default(),
            roughness: entity.material.roughness,
            material_kind: 0,
            _pad0: 0,
            _pad1: 0,
        });

        match entity.shape {
            PrimitiveShape::Sphere { radius } => {
                let scale = entity.transform.scale;
                let radius_scale = scale.x.abs().max(scale.y.abs()).max(scale.z.abs()).max(0.0);
                spheres.push(GpuSphere {
                    center: entity.transform.translation.into(),
                    radius: radius * radius_scale,
                    material_id,
                    object_id: entity.object_id.get(),
                    _pad0: 0,
                });
            }
            PrimitiveShape::Plane { normal, offset } => {
                let local_point = normal * (-offset);
                let normal = entity.transform.transform_direction(normal).normalized();
                let point = entity.transform.transform_point(local_point);
                planes.push(GpuPlane {
                    point: point.into(),
                    normal: normal.into(),
                    material_id,
                    object_id: entity.object_id.get(),
                    _pad0: 0,
                    _pad1: 0,
                });
            }
            PrimitiveShape::Box { .. } => {
                return Err(RocmRenderError::UnsupportedPrimitive {
                    entity: entity.name.clone(),
                    primitive: format!("{:?}", entity.shape),
                });
            }
        }
    }

    Ok(HostSceneBuffers {
        spheres,
        planes,
        materials,
    })
}

/// Camera configuration prepared for the ROCm RGB renderer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RocmRgbCamera {
    pub width: u32,
    pub height: u32,
}

impl RocmRgbCamera {
    pub fn from_sensor(sensor: &RgbCameraSensor) -> Self {
        let intrinsics = sensor.intrinsics();
        Self {
            width: intrinsics.width,
            height: intrinsics.height,
        }
    }

    pub fn pixel_count(self) -> usize {
        self.width as usize * self.height as usize
    }
}

#[cfg(feature = "rocm")]
pub struct RocmRgbFrame {
    width: u32,
    height: u32,
    buffer: rocm_oxide::DeviceBuffer<u32>,
}

#[cfg(feature = "rocm")]
pub struct RocmSensorOutput {
    pub width: u32,
    pub height: u32,
    pub metadata: FrameMetadata,
    pub rgb: rocm_oxide::DeviceBuffer<u32>,
    pub depth: rocm_oxide::DeviceBuffer<f32>,
    pub segmentation: rocm_oxide::DeviceBuffer<u32>,
}

#[cfg(not(feature = "rocm"))]
pub struct RocmSensorOutput {
    pub width: u32,
    pub height: u32,
    pub metadata: FrameMetadata,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HostSensorOutput {
    pub rgb: RgbFrame,
    pub depth: DepthFrame,
    pub segmentation: SegmentationFrame,
}

#[cfg(not(feature = "rocm"))]
pub struct RocmRgbFrame {
    width: u32,
    height: u32,
}

impl RocmRgbFrame {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    #[cfg(feature = "rocm")]
    pub fn device_buffer(&self) -> &rocm_oxide::DeviceBuffer<u32> {
        &self.buffer
    }

    #[cfg(feature = "rocm")]
    pub fn into_device_buffer(self) -> rocm_oxide::DeviceBuffer<u32> {
        self.buffer
    }
}

#[cfg(feature = "rocm")]
pub struct RocmSensorRenderer {
    device: rocm_oxide::Device,
    _module: rocm_oxide::Module,
    kernel: rocm_oxide::Kernel,
}

#[cfg(not(feature = "rocm"))]
pub struct RocmSensorRenderer;

impl RocmSensorRenderer {
    #[cfg(feature = "rocm")]
    pub fn new() -> Result<Self> {
        let device = rocm_oxide::Device::first().map_err(|err| RocmRenderError::RocmOxide {
            context: "opening the first ROCm device".to_string(),
            message: err.to_string(),
        })?;
        let module = device
            .compile_hip_source(SENSOR_OUTPUTS_KERNEL)
            .map_err(|err| RocmRenderError::RocmOxide {
                context: "compiling the deterministic RGB/depth/segmentation HIPRTC kernel"
                    .to_string(),
                message: err.to_string(),
            })?;
        let kernel =
            module
                .kernel(c"render_sensor_outputs")
                .map_err(|err| RocmRenderError::RocmOxide {
                    context: "loading the render_sensor_outputs kernel".to_string(),
                    message: err.to_string(),
                })?;
        Ok(Self {
            device,
            _module: module,
            kernel,
        })
    }

    #[cfg(not(feature = "rocm"))]
    pub fn new() -> Result<Self> {
        Err(RocmRenderError::BackendUnavailable)
    }

    #[cfg(feature = "rocm")]
    pub fn device_arch(&self) -> &str {
        self.device.arch()
    }

    #[cfg(not(feature = "rocm"))]
    pub fn device_arch(&self) -> &str {
        "unavailable"
    }

    #[cfg(feature = "rocm")]
    pub fn upload_scene(&self, scene: &Scene) -> Result<RocmScene> {
        let host = build_gpu_scene_buffers(scene)?;
        let sphere_count = host.sphere_count();
        let plane_count = host.plane_count();
        let material_count = host.material_count();
        let spheres = rocm_oxide::DeviceBuffer::from_slice(&host.spheres).map_err(|err| {
            RocmRenderError::RocmOxide {
                context: format!("uploading {sphere_count} sphere primitives"),
                message: err.to_string(),
            }
        })?;
        let planes = rocm_oxide::DeviceBuffer::from_slice(&host.planes).map_err(|err| {
            RocmRenderError::RocmOxide {
                context: format!("uploading {plane_count} plane primitives"),
                message: err.to_string(),
            }
        })?;
        let materials = rocm_oxide::DeviceBuffer::from_slice(&host.materials).map_err(|err| {
            RocmRenderError::RocmOxide {
                context: format!("uploading {material_count} material records"),
                message: err.to_string(),
            }
        })?;

        Ok(RocmScene {
            spheres,
            planes,
            materials,
            sphere_count,
            plane_count,
            material_count,
        })
    }

    #[cfg(not(feature = "rocm"))]
    pub fn upload_scene(&self, scene: &Scene) -> Result<RocmScene> {
        let host = build_gpu_scene_buffers(scene)?;
        let _ = host;
        Err(RocmRenderError::BackendUnavailable)
    }

    #[cfg(feature = "rocm")]
    pub fn render_rgb_to_device(
        &self,
        scene: &Scene,
        sensor: &RgbCameraSensor,
    ) -> Result<RocmRgbFrame> {
        let metadata = FrameMetadata::new(0, 0.0, sensor.id());
        let output = self.render_all_to_device(scene, sensor, metadata)?;
        Ok(RocmRgbFrame {
            width: output.width,
            height: output.height,
            buffer: output.rgb,
        })
    }

    #[cfg(not(feature = "rocm"))]
    pub fn render_rgb_to_device(
        &self,
        _scene: &Scene,
        sensor: &RgbCameraSensor,
    ) -> Result<RocmRgbFrame> {
        let camera = RocmRgbCamera::from_sensor(sensor);
        let _ = camera;
        Err(RocmRenderError::BackendUnavailable)
    }

    #[cfg(feature = "rocm")]
    pub fn render_all_to_device(
        &self,
        scene: &Scene,
        sensor: &RgbCameraSensor,
        metadata: FrameMetadata,
    ) -> Result<RocmSensorOutput> {
        let uploaded = self.upload_scene(scene)?;
        self.render_uploaded_scene_to_device(&uploaded, sensor, metadata)
    }

    #[cfg(feature = "rocm")]
    pub fn render_uploaded_scene_to_device(
        &self,
        scene: &RocmScene,
        sensor: &RgbCameraSensor,
        metadata: FrameMetadata,
    ) -> Result<RocmSensorOutput> {
        let camera = RocmRgbCamera::from_sensor(sensor);
        let camera_params = CameraParams::from_sensor(sensor);
        let render_params = RenderParams {
            camera: camera_params,
            width: camera.width,
            height: camera.height,
            sphere_count: scene.sphere_count,
            plane_count: scene.plane_count,
            material_count: scene.material_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let pixel_count = camera.pixel_count();
        if pixel_count == 0 {
            return Err(RocmRenderError::InvalidTarget(format!(
                "{}x{}",
                camera.width, camera.height
            )));
        }

        let buffer = rocm_oxide::DeviceBuffer::<u32>::new(pixel_count).map_err(|err| {
            RocmRenderError::RocmOxide {
                context: format!(
                    "allocating RGB output buffer for {}x{}",
                    camera.width, camera.height
                ),
                message: err.to_string(),
            }
        })?;
        let depth = rocm_oxide::DeviceBuffer::<f32>::new(pixel_count).map_err(|err| {
            RocmRenderError::RocmOxide {
                context: format!(
                    "allocating depth output buffer for {}x{}",
                    camera.width, camera.height
                ),
                message: err.to_string(),
            }
        })?;
        let segmentation = rocm_oxide::DeviceBuffer::<u32>::new(pixel_count).map_err(|err| {
            RocmRenderError::RocmOxide {
                context: format!(
                    "allocating segmentation output buffer for {}x{}",
                    camera.width, camera.height
                ),
                message: err.to_string(),
            }
        })?;

        unsafe {
            rocm_oxide::launch_1d!(
                self.kernel,
                pixel_count,
                buffer.as_mut_ptr(),
                depth.as_mut_ptr(),
                segmentation.as_mut_ptr(),
                scene.spheres.as_ptr(),
                scene.planes.as_ptr(),
                scene.materials.as_ptr(),
                render_params,
            )
            .map_err(|err| RocmRenderError::RocmOxide {
                context: "launching render_sensor_outputs".to_string(),
                message: err.to_string(),
            })?;
        }
        rocm_oxide::hip::synchronize().map_err(|err| RocmRenderError::RocmOxide {
            context: "synchronizing the RGB/depth/segmentation render".to_string(),
            message: err.to_string(),
        })?;

        Ok(RocmSensorOutput {
            width: camera.width,
            height: camera.height,
            metadata,
            rgb: buffer,
            depth,
            segmentation,
        })
    }

    #[cfg(not(feature = "rocm"))]
    pub fn render_all_to_device(
        &self,
        _scene: &Scene,
        sensor: &RgbCameraSensor,
        metadata: FrameMetadata,
    ) -> Result<RocmSensorOutput> {
        let camera = RocmRgbCamera::from_sensor(sensor);
        let _ = (camera, metadata);
        Err(RocmRenderError::BackendUnavailable)
    }

    #[cfg(feature = "rocm")]
    pub fn copy_rgb_to_host(
        &self,
        frame: &RocmRgbFrame,
        metadata: FrameMetadata,
    ) -> Result<RgbFrame> {
        let pixels = frame
            .buffer
            .copy_to_vec()
            .map_err(|err| RocmRenderError::RocmOxide {
                context: "copying RGB device buffer to host".to_string(),
                message: err.to_string(),
            })?;
        Ok(RgbFrame::new(frame.width, frame.height, metadata, pixels))
    }

    #[cfg(not(feature = "rocm"))]
    pub fn copy_rgb_to_host(
        &self,
        _frame: &RocmRgbFrame,
        _metadata: FrameMetadata,
    ) -> Result<RgbFrame> {
        Err(RocmRenderError::BackendUnavailable)
    }

    #[cfg(feature = "rocm")]
    pub fn copy_depth_to_host(&self, output: &RocmSensorOutput) -> Result<DepthFrame> {
        let pixels = output
            .depth
            .copy_to_vec()
            .map_err(|err| RocmRenderError::RocmOxide {
                context: "copying depth device buffer to host".to_string(),
                message: err.to_string(),
            })?;
        Ok(DepthFrame::new(
            output.width,
            output.height,
            output.metadata.clone(),
            pixels,
        ))
    }

    #[cfg(not(feature = "rocm"))]
    pub fn copy_depth_to_host(&self, _output: &RocmSensorOutput) -> Result<DepthFrame> {
        Err(RocmRenderError::BackendUnavailable)
    }

    #[cfg(feature = "rocm")]
    pub fn copy_segmentation_to_host(
        &self,
        output: &RocmSensorOutput,
    ) -> Result<SegmentationFrame> {
        let pixels =
            output
                .segmentation
                .copy_to_vec()
                .map_err(|err| RocmRenderError::RocmOxide {
                    context: "copying segmentation device buffer to host".to_string(),
                    message: err.to_string(),
                })?;
        Ok(SegmentationFrame::new(
            output.width,
            output.height,
            output.metadata.clone(),
            pixels,
        ))
    }

    #[cfg(not(feature = "rocm"))]
    pub fn copy_segmentation_to_host(
        &self,
        _output: &RocmSensorOutput,
    ) -> Result<SegmentationFrame> {
        Err(RocmRenderError::BackendUnavailable)
    }

    #[cfg(feature = "rocm")]
    pub fn copy_all_to_host(&self, output: &RocmSensorOutput) -> Result<HostSensorOutput> {
        let rgb_pixels = output
            .rgb
            .copy_to_vec()
            .map_err(|err| RocmRenderError::RocmOxide {
                context: "copying RGB device buffer to host".to_string(),
                message: err.to_string(),
            })?;
        let depth = self.copy_depth_to_host(output)?;
        let segmentation = self.copy_segmentation_to_host(output)?;
        Ok(HostSensorOutput {
            rgb: RgbFrame::new(
                output.width,
                output.height,
                output.metadata.clone(),
                rgb_pixels,
            ),
            depth,
            segmentation,
        })
    }

    #[cfg(not(feature = "rocm"))]
    pub fn copy_all_to_host(&self, _output: &RocmSensorOutput) -> Result<HostSensorOutput> {
        Err(RocmRenderError::BackendUnavailable)
    }

    pub fn render_rgb_host(
        &self,
        scene: &Scene,
        sensor: &RgbCameraSensor,
        metadata: FrameMetadata,
    ) -> Result<RgbFrame> {
        let device_frame = self.render_rgb_to_device(scene, sensor)?;
        self.copy_rgb_to_host(&device_frame, metadata)
    }

    pub fn render_all_host(
        &self,
        scene: &Scene,
        sensor: &RgbCameraSensor,
        metadata: FrameMetadata,
    ) -> Result<HostSensorOutput> {
        let device_output = self.render_all_to_device(scene, sensor, metadata)?;
        self.copy_all_to_host(&device_output)
    }
}

pub fn rocm_feature_enabled() -> bool {
    cfg!(feature = "rocm")
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_core::{Camera, Entity, Material, ObjectId, Transform, Vec3};
    use std::mem::{align_of, size_of};

    #[test]
    fn rocm_rgb_camera_uses_sensor_resolution() {
        let camera = Camera::look_at(Vec3::new(0.0, 0.0, 1.0), Vec3::ZERO, 60.0, 16.0 / 9.0)
            .with_resolution(320, 180);
        let sensor = RgbCameraSensor::new("rgb", camera);

        let rocm_camera = RocmRgbCamera::from_sensor(&sensor);

        assert_eq!(rocm_camera.width, 320);
        assert_eq!(rocm_camera.height, 180);
        assert_eq!(rocm_camera.pixel_count(), 57_600);
    }

    #[test]
    fn default_scene_converts_to_gpu_buffers() {
        let scene = sim_core::Scene::default_sensor_scene();

        let buffers = build_gpu_scene_buffers(&scene).unwrap();

        assert_eq!(buffers.plane_count(), 1);
        assert_eq!(buffers.sphere_count(), 3);
        assert_eq!(buffers.material_count(), 4);
        assert_eq!(buffers.planes[0].object_id, 1);
        assert_eq!(
            buffers
                .spheres
                .iter()
                .map(|sphere| sphere.object_id)
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn material_mapping_uses_entity_order() {
        let mut scene = sim_core::Scene::new();
        scene.add_entity(Entity::new(
            "custom sphere",
            PrimitiveShape::sphere(1.0),
            Transform::from_translation(Vec3::new(1.0, 2.0, 3.0)),
            Material::matte(Vec3::new(0.25, 0.5, 0.75)),
            ObjectId::new(42),
        ));

        let buffers = build_gpu_scene_buffers(&scene).unwrap();

        assert_eq!(buffers.spheres[0].material_id, 0);
        assert_eq!(buffers.spheres[0].object_id, 42);
        assert_eq!(
            buffers.materials[0].base_color,
            GpuVec3::new(0.25, 0.5, 0.75)
        );
    }

    #[test]
    fn unsupported_box_reports_entity_name() {
        let mut scene = sim_core::Scene::new();
        scene.add_entity(Entity::new(
            "box for later",
            PrimitiveShape::box_with_half_extents(Vec3::splat(0.5)),
            Transform::default(),
            Material::default(),
            ObjectId::new(9),
        ));

        let err = build_gpu_scene_buffers(&scene).unwrap_err();

        assert!(err.to_string().contains("box for later"));
        assert!(err.to_string().contains("Box"));
    }

    #[test]
    fn gpu_scene_struct_layouts_match_kernel_abi() {
        assert_eq!(size_of::<GpuVec3>(), 16);
        assert_eq!(size_of::<GpuSphere>(), 32);
        assert_eq!(size_of::<GpuPlane>(), 48);
        assert_eq!(size_of::<GpuMaterial>(), 48);
        assert_eq!(align_of::<GpuVec3>(), 4);
        assert_eq!(align_of::<GpuSphere>(), 4);
        assert_eq!(align_of::<GpuPlane>(), 4);
        assert_eq!(align_of::<GpuMaterial>(), 4);
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn renderer_reports_unavailable_without_rocm_feature() {
        assert!(matches!(
            RocmSensorRenderer::new(),
            Err(RocmRenderError::BackendUnavailable)
        ));
    }

    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "requires ROCm, HIPRTC, and a visible AMD GPU"]
    fn uploads_and_renders_scene_with_rocm_backend() {
        let scene = sim_core::Scene::default_sensor_scene();
        let camera = sim_core::Camera::default_rgb().with_resolution(64, 36);
        let sensor = RgbCameraSensor::new("rgb", camera);
        let metadata = FrameMetadata::new(1, 0.0, sensor.id());
        let renderer = RocmSensorRenderer::new().expect("ROCm renderer should initialize");

        let uploaded = renderer
            .upload_scene(&scene)
            .expect("scene upload should complete");
        assert_eq!(uploaded.sphere_count(), 3);
        assert_eq!(uploaded.plane_count(), 1);

        let output = renderer
            .render_uploaded_scene_to_device(&uploaded, &sensor, metadata)
            .and_then(|device_output| renderer.copy_all_to_host(&device_output))
            .expect("ROCm render should complete");

        assert_eq!(output.rgb.width, 64);
        assert_eq!(output.rgb.height, 36);
        assert_eq!(output.rgb.pixels.len(), 64 * 36);
        assert_eq!(output.depth.pixels.len(), 64 * 36);
        assert_eq!(output.segmentation.pixels.len(), 64 * 36);
        assert!(output.rgb.pixels.iter().any(|&pixel| pixel != 0));
        assert!(output.depth.pixels.iter().any(|&depth| depth > 0.0));
        assert!(
            output
                .segmentation
                .pixels
                .iter()
                .any(|&object_id| object_id > 0)
        );
    }
}
