use clap::Parser;
use sim_core::{Camera, Scene};
use sim_datasets::{
    SensorImageSet, write_depth_f32, write_depth_preview_pgm, write_metadata_json, write_ppm,
    write_segmentation_preview_ppm, write_segmentation_u32,
};
use sim_render_rocm::{RocmSensorRenderer, rocm_feature_enabled};
use sim_sensors::{DepthMetadata, FrameMetadata, FrameOutputMetadata, RgbCameraSensor};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(about = "Render one RGB/depth/segmentation frame from a sim-core scene.")]
struct Args {
    #[arg(long)]
    scene: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let output_dir = PathBuf::from("target/sensor_lab");
    fs::create_dir_all(&output_dir)?;

    let scene = load_scene(args.scene.as_deref())?;
    let camera = Camera::default_rgb();
    let sensor = RgbCameraSensor::new("rgb-main", camera);
    let metadata = metadata_for_sensor_lab(sensor.id(), &scene);

    println!("sensor_lab: scene has {} entities", scene.len());
    println!(
        "sensor_lab: resolution {}x{}",
        sensor.intrinsics().width,
        sensor.intrinsics().height
    );
    println!("sensor_lab: depth is linear camera ray distance in meters; 0.0 means miss");

    let images = match RocmSensorRenderer::new() {
        Ok(renderer) => {
            println!(
                "sensor_lab: ROCm backend active on {}",
                renderer.device_arch()
            );
            let output = renderer.render_all_host(&scene, &sensor, metadata.clone())?;
            SensorImageSet::from_frames(output.rgb, output.depth, output.segmentation)?
        }
        Err(err) if rocm_feature_enabled() => {
            return Err(Box::new(err));
        }
        Err(err) => {
            println!("sensor_lab: ROCm renderer unavailable: {err}");
            println!("sensor_lab: writing deterministic CPU preview outputs instead");
            SensorImageSet::synthetic_preview(camera.width, camera.height, metadata.frame_index)
        }
    };

    let rgb_path = output_dir.join("rgb.ppm");
    let depth_path = output_dir.join("depth.f32");
    let depth_preview_path = output_dir.join("depth_preview.pgm");
    let segmentation_path = output_dir.join("segmentation.u32");
    let segmentation_preview_path = output_dir.join("segmentation_preview.ppm");
    let metadata_path = output_dir.join("metadata.json");
    write_ppm(&rgb_path, &images.rgb)?;
    write_depth_f32(&depth_path, &images.depth)?;
    write_depth_preview_pgm(&depth_preview_path, &images.depth)?;
    write_segmentation_u32(&segmentation_path, &images.segmentation)?;
    write_segmentation_preview_ppm(&segmentation_preview_path, &images.segmentation)?;
    write_metadata_json(&metadata_path, &metadata)?;

    println!("sensor_lab: object IDs");
    for object in &metadata.object_ids {
        println!("  {} = {}", object.id, object.label);
    }
    println!("sensor_lab: wrote outputs");
    print_path(&rgb_path);
    print_path(&depth_path);
    print_path(&depth_preview_path);
    print_path(&segmentation_path);
    print_path(&segmentation_preview_path);
    print_path(&metadata_path);
    Ok(())
}

fn load_scene(path: Option<&Path>) -> Result<Scene, Box<dyn Error>> {
    if let Some(path) = path {
        let json = fs::read_to_string(path)?;
        let scene = serde_json::from_str(&json)?;
        println!("sensor_lab: loaded scene {}", path.display());
        Ok(scene)
    } else {
        Ok(Scene::default_sensor_scene())
    }
}

fn metadata_for_sensor_lab(sensor_id: &str, scene: &Scene) -> FrameMetadata {
    FrameMetadata::new(1, 0.0, sensor_id)
        .with_depth(DepthMetadata::linear_ray_distance_meters())
        .with_output(FrameOutputMetadata::new("rgb", "ppm", "rgb.ppm"))
        .with_output(FrameOutputMetadata::new(
            "depth",
            "raw-f32-little-endian",
            "depth.f32",
        ))
        .with_output(FrameOutputMetadata::new(
            "depth_preview",
            "pgm",
            "depth_preview.pgm",
        ))
        .with_output(FrameOutputMetadata::new(
            "segmentation",
            "raw-u32-little-endian",
            "segmentation.u32",
        ))
        .with_output(FrameOutputMetadata::new(
            "segmentation_preview",
            "ppm",
            "segmentation_preview.ppm",
        ))
        .with_scene_object_ids(scene)
}

fn print_path(path: &Path) {
    println!("  {}", path.display());
}
