use clap::Parser;
use sim_core::{Camera, Scene};
use sim_datasets::{
    ScenarioConfig, SensorImageSet, synthetic_lidar_frame, write_depth_f32,
    write_depth_preview_pgm, write_lidar_object_ids_u32, write_lidar_points_xyz,
    write_lidar_preview_pgm, write_lidar_range_f32, write_metadata_json, write_ppm,
    write_segmentation_preview_ppm, write_segmentation_u32,
};
use sim_render_rocm::{RocmSensorRenderer, rocm_feature_enabled};
use sim_sensors::{
    DepthMetadata, FrameMetadata, FrameOutputMetadata, LidarConfig, RgbCameraSensor,
};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(about = "Render one RGB/depth/segmentation frame from a sim-core scene.")]
struct Args {
    #[arg(long)]
    scene: Option<PathBuf>,
    #[arg(long)]
    scenario: Option<PathBuf>,
    #[arg(long)]
    lidar: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let output_dir = PathBuf::from("target/sensor_lab");
    fs::create_dir_all(&output_dir)?;

    let scenario = args.scenario.as_deref().map(load_scenario).transpose()?;
    let scene_path = args.scene.as_deref().or_else(|| {
        scenario
            .as_ref()
            .map(|scenario| scenario.scene_path.as_path())
    });
    let scene = load_scene(scene_path)?;
    if let Some(scenario) = &scenario {
        println!(
            "sensor_lab: scenario={} rig={} sensors={}",
            scenario.name,
            scenario.rig.name,
            scenario.rig.mounts.len()
        );
    }
    let camera = scenario
        .as_ref()
        .and_then(|scenario| scenario.primary_camera().map(|(_mount, camera)| camera))
        .unwrap_or_else(Camera::default_rgb);
    let sensor_id = scenario
        .as_ref()
        .and_then(|scenario| {
            scenario
                .primary_camera()
                .map(|(mount, _camera)| mount.name.as_str())
        })
        .unwrap_or("rgb-main");
    let sensor = RgbCameraSensor::new(sensor_id, camera);
    let metadata = metadata_for_sensor_lab(sensor.id(), &scene, args.lidar);
    let lidar_config = scenario
        .as_ref()
        .and_then(|scenario| scenario.primary_lidar().map(|(_mount, lidar)| lidar))
        .unwrap_or_else(LidarConfig::default);

    println!("sensor_lab: scene has {} entities", scene.len());
    println!(
        "sensor_lab: resolution {}x{}",
        sensor.intrinsics().width,
        sensor.intrinsics().height
    );
    println!("sensor_lab: depth is linear camera ray distance in meters; 0.0 means miss");

    let (images, lidar_frame) = match RocmSensorRenderer::new() {
        Ok(renderer) => {
            println!(
                "sensor_lab: ROCm backend active on {}",
                renderer.device_arch()
            );
            if args.lidar {
                let uploaded = renderer.upload_scene(&scene)?;
                let output = renderer.render_uploaded_scene_to_device(
                    &uploaded,
                    &sensor,
                    metadata.clone(),
                )?;
                let output = renderer.copy_all_to_host(&output)?;
                let lidar_output = renderer.render_lidar_uploaded_scene_to_device(
                    &uploaded,
                    lidar_config,
                    FrameMetadata::new(1, 0.0, "lidar-main"),
                )?;
                let lidar = renderer.copy_lidar_to_host(&lidar_output)?;
                (
                    SensorImageSet::from_frames(output.rgb, output.depth, output.segmentation)?,
                    Some(lidar),
                )
            } else {
                let output = renderer.render_all_host(&scene, &sensor, metadata.clone())?;
                (
                    SensorImageSet::from_frames(output.rgb, output.depth, output.segmentation)?,
                    None,
                )
            }
        }
        Err(err) if rocm_feature_enabled() => {
            return Err(Box::new(err));
        }
        Err(err) => {
            println!("sensor_lab: ROCm renderer unavailable: {err}");
            println!("sensor_lab: writing deterministic CPU preview outputs instead");
            let lidar = args.lidar.then(|| {
                synthetic_lidar_frame(
                    lidar_config,
                    metadata.frame_index,
                    FrameMetadata::new(1, 0.0, "lidar-main"),
                )
            });
            (
                SensorImageSet::synthetic_preview(
                    camera.width,
                    camera.height,
                    metadata.frame_index,
                ),
                lidar,
            )
        }
    };

    let rgb_path = output_dir.join("rgb.ppm");
    let depth_path = output_dir.join("depth.f32");
    let depth_preview_path = output_dir.join("depth_preview.pgm");
    let segmentation_path = output_dir.join("segmentation.u32");
    let segmentation_preview_path = output_dir.join("segmentation_preview.ppm");
    let lidar_range_path = output_dir.join("lidar_range.f32");
    let lidar_points_path = output_dir.join("lidar_points.xyz");
    let lidar_object_ids_path = output_dir.join("lidar_object_ids.u32");
    let lidar_preview_path = output_dir.join("lidar_preview.pgm");
    let metadata_path = output_dir.join("metadata.json");
    write_ppm(&rgb_path, &images.rgb)?;
    write_depth_f32(&depth_path, &images.depth)?;
    write_depth_preview_pgm(&depth_preview_path, &images.depth)?;
    write_segmentation_u32(&segmentation_path, &images.segmentation)?;
    write_segmentation_preview_ppm(&segmentation_preview_path, &images.segmentation)?;
    if let Some(lidar) = &lidar_frame {
        write_lidar_range_f32(&lidar_range_path, lidar)?;
        write_lidar_points_xyz(&lidar_points_path, lidar)?;
        write_lidar_object_ids_u32(&lidar_object_ids_path, lidar)?;
        write_lidar_preview_pgm(&lidar_preview_path, lidar)?;
    }
    write_metadata_json(&metadata_path, &metadata)?;

    println!("sensor_lab: object IDs");
    for object in &metadata.object_ids {
        let primitive = object.primitive.as_deref().unwrap_or("unknown");
        let material = object.material.as_deref().unwrap_or("unknown");
        println!(
            "  {} = {} primitive={} material={}",
            object.id, object.label, primitive, material
        );
    }
    println!("sensor_lab: wrote outputs");
    print_path(&rgb_path);
    print_path(&depth_path);
    print_path(&depth_preview_path);
    print_path(&segmentation_path);
    print_path(&segmentation_preview_path);
    if lidar_frame.is_some() {
        println!(
            "sensor_lab: LiDAR {}x{} range convention: linear ray distance meters; 0.0 means miss",
            lidar_config.horizontal_samples, lidar_config.vertical_channels
        );
        print_path(&lidar_range_path);
        print_path(&lidar_points_path);
        print_path(&lidar_object_ids_path);
        print_path(&lidar_preview_path);
    }
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

fn load_scenario(path: &Path) -> Result<ScenarioConfig, Box<dyn Error>> {
    let json = fs::read_to_string(path)?;
    let scenario = serde_json::from_str::<ScenarioConfig>(&json)?;
    println!("sensor_lab: loaded scenario {}", path.display());
    Ok(scenario)
}

fn metadata_for_sensor_lab(sensor_id: &str, scene: &Scene, include_lidar: bool) -> FrameMetadata {
    let mut metadata = FrameMetadata::new(1, 0.0, sensor_id)
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
        .with_scene_object_ids(scene);
    if include_lidar {
        metadata = metadata
            .with_output(FrameOutputMetadata::new(
                "lidar_range",
                "raw-f32-little-endian",
                "lidar_range.f32",
            ))
            .with_output(FrameOutputMetadata::new(
                "lidar_points",
                "xyz",
                "lidar_points.xyz",
            ))
            .with_output(FrameOutputMetadata::new(
                "lidar_object_ids",
                "raw-u32-little-endian",
                "lidar_object_ids.u32",
            ))
            .with_output(FrameOutputMetadata::new(
                "lidar_preview",
                "pgm",
                "lidar_preview.pgm",
            ));
    }
    metadata
}

fn print_path(path: &Path) {
    println!("  {}", path.display());
}
