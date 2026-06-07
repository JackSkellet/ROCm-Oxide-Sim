use clap::Parser;
use sim_core::{Camera, Scene, Vec3};
use sim_datasets::{DatasetWriter, SensorImageSet};
use sim_render_rocm::{RocmSensorRenderer, rocm_feature_enabled};
use sim_sensors::{DepthMetadata, FrameMetadata, FrameOutputMetadata, RgbCameraSensor};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(about = "Generate a small RGB/depth/segmentation dataset from a simple camera path.")]
struct Args {
    #[arg(short = 'n', long, default_value_t = 8)]
    frames: u64,
    #[arg(long, default_value_t = 640)]
    width: u32,
    #[arg(long, default_value_t = 360)]
    height: u32,
    #[arg(
        short = 'o',
        long = "out",
        alias = "output",
        default_value = "target/dataset_generator"
    )]
    out: PathBuf,
    #[arg(long)]
    scene: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let scene = load_scene(args.scene.as_deref())?;
    let mut writer = DatasetWriter::new(&args.out)?;
    println!("dataset_generator: scene entities={}", scene.len());

    let renderer = match RocmSensorRenderer::new() {
        Ok(renderer) => {
            println!(
                "dataset_generator: using ROCm-Oxide renderer on {}",
                renderer.device_arch()
            );
            Some(renderer)
        }
        Err(err) if rocm_feature_enabled() => return Err(Box::new(err)),
        Err(err) => {
            println!("dataset_generator: ROCm renderer unavailable: {err}");
            println!("dataset_generator: writing deterministic CPU preview outputs");
            None
        }
    };

    for frame_index in 1..=args.frames {
        let timestamp_seconds = (frame_index - 1) as f64 / 30.0;
        let camera = camera_for_frame(frame_index, args.width, args.height);
        let sensor = RgbCameraSensor::new("rgb-main", camera);
        let metadata =
            metadata_for_dataset_frame(frame_index, timestamp_seconds, sensor.id(), &scene);

        let images = if let Some(renderer) = &renderer {
            let output = renderer.render_all_host(&scene, &sensor, metadata.clone())?;
            SensorImageSet::from_frames(output.rgb, output.depth, output.segmentation)?
        } else {
            SensorImageSet::synthetic_preview(args.width, args.height, frame_index)
        };

        writer.write_sensor_outputs(frame_index, &images, &metadata)?;
    }

    let manifest = writer.finish()?;
    println!(
        "dataset_generator: wrote {} frames to {}",
        manifest.frame_count,
        args.out.display()
    );
    Ok(())
}

fn load_scene(path: Option<&Path>) -> Result<Scene, Box<dyn Error>> {
    if let Some(path) = path {
        let json = fs::read_to_string(path)?;
        let scene = serde_json::from_str(&json)?;
        println!("dataset_generator: loaded scene {}", path.display());
        Ok(scene)
    } else {
        Ok(Scene::default_sensor_scene())
    }
}

fn camera_for_frame(frame_index: u64, width: u32, height: u32) -> Camera {
    let t = (frame_index.saturating_sub(1) as f32) * 0.08;
    let aspect_ratio = width.max(1) as f32 / height.max(1) as f32;
    Camera::look_at(
        Vec3::new(t.sin() * 0.65, 1.1, 4.5 + t.cos() * 0.25),
        Vec3::new(0.0, 0.55, -1.45),
        55.0,
        aspect_ratio,
    )
    .with_resolution(width, height)
}

fn metadata_for_dataset_frame(
    frame_index: u64,
    timestamp_seconds: f64,
    sensor_id: &str,
    scene: &Scene,
) -> FrameMetadata {
    let file_stem = format!("frame_{frame_index:06}");
    FrameMetadata::new(frame_index, timestamp_seconds, sensor_id)
        .with_depth(DepthMetadata::linear_ray_distance_meters())
        .with_output(FrameOutputMetadata::new(
            "rgb",
            "ppm",
            format!("rgb/{file_stem}.ppm"),
        ))
        .with_output(FrameOutputMetadata::new(
            "depth",
            "raw-f32-little-endian",
            format!("depth/{file_stem}.f32"),
        ))
        .with_output(FrameOutputMetadata::new(
            "depth_preview",
            "pgm",
            format!("depth_preview/{file_stem}.pgm"),
        ))
        .with_output(FrameOutputMetadata::new(
            "segmentation",
            "raw-u32-little-endian",
            format!("segmentation/{file_stem}.u32"),
        ))
        .with_output(FrameOutputMetadata::new(
            "segmentation_preview",
            "ppm",
            format!("segmentation_preview/{file_stem}.ppm"),
        ))
        .with_scene_object_ids(scene)
}
