use clap::{Parser, Subcommand, ValueEnum};
use sim_core::Scene;
use sim_datasets::{
    CameraPathConfig, DatasetConfig, DatasetFrameMetadata, DatasetWriter, SensorImageSet,
    camera_for_dataset_frame, frame_output_paths_for_selection, validate_dataset,
};
#[cfg(feature = "rocm")]
use sim_render_rocm::RocmSensorRenderer;
#[cfg(feature = "rocm")]
use sim_sensors::FrameMetadata;
use sim_sensors::{RgbCameraSensor, scene_object_ids};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(about = "Generate a small RGB/depth/segmentation dataset from a simple camera path.")]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    scene: Option<PathBuf>,
    #[arg(short = 'n', long)]
    frames: Option<u32>,
    #[arg(long)]
    width: Option<u32>,
    #[arg(long)]
    height: Option<u32>,
    #[arg(short = 'o', long = "out", alias = "output")]
    out: Option<PathBuf>,
    #[arg(long, value_enum)]
    camera_path: Option<CameraPathKind>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    overwrite: bool,
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    Validate(ValidateArgs),
}

#[derive(Debug, Parser)]
struct ValidateArgs {
    #[arg(long)]
    dataset: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CameraPathKind {
    Static,
    Orbit,
    Line,
    Random,
}

impl CameraPathKind {
    fn to_config(self) -> CameraPathConfig {
        match self {
            Self::Static => CameraPathConfig::static_default(),
            Self::Orbit => CameraPathConfig::orbit_default(),
            Self::Line => CameraPathConfig::line_default(),
            Self::Random => CameraPathConfig::random_default(),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if let Some(Command::Validate(validate)) = &args.command {
        let report = validate_dataset(&validate.dataset)?;
        println!(
            "dataset_generator: validated {} frames in {}",
            report.frame_count,
            validate.dataset.display()
        );
        return Ok(());
    }

    run_generate(args)
}

fn run_generate(args: Args) -> Result<(), Box<dyn Error>> {
    let (config, config_path) = resolve_config(&args)?;
    if args.dry_run {
        print_dry_run(&config)?;
        return Ok(());
    }

    prepare_output_dir(&config.output_dir, args.overwrite)?;

    let scene = load_scene(config.scene_path.as_deref())?;
    let object_ids = scene_object_ids(&scene);
    println!("dataset_generator: scene entities={}", scene.len());
    let render_session = RenderSession::open(&scene)?;

    let mut writer = DatasetWriter::new_with_config(
        &config.output_dir,
        config.clone(),
        config.scene_path.as_ref().map(path_to_string),
        config_path.as_ref().map(path_to_string),
        render_session.backend_label(),
        object_ids.clone(),
    )?;

    render_dataset(&config, &object_ids, &render_session, &mut writer)?;

    let manifest = writer.finish()?;
    println!(
        "dataset_generator: wrote {} frames to {}",
        manifest.frame_count,
        config.output_dir.display()
    );
    Ok(())
}

fn resolve_config(args: &Args) -> Result<(DatasetConfig, Option<PathBuf>), Box<dyn Error>> {
    let mut config = if let Some(path) = &args.config {
        let json = fs::read_to_string(path)?;
        serde_json::from_str::<DatasetConfig>(&json)
            .map_err(|err| format!("failed to parse dataset config {}: {err}", path.display()))?
    } else {
        DatasetConfig::default()
    };

    if let Some(scene) = &args.scene {
        config.scene_path = Some(scene.clone());
    }
    if let Some(out) = &args.out {
        config.output_dir = out.clone();
    }
    if let Some(frames) = args.frames {
        config.frame_count = frames;
    }
    if let Some(width) = args.width {
        config.width = width;
    }
    if let Some(height) = args.height {
        config.height = height;
    }
    if let Some(camera_path) = args.camera_path {
        config.camera_path = camera_path.to_config();
    }
    if let Some(seed) = args.seed {
        config.seed = seed;
    }

    Ok((config.normalized(), args.config.clone()))
}

fn prepare_output_dir(output_dir: &Path, overwrite: bool) -> Result<(), Box<dyn Error>> {
    if output_dir.exists() {
        let is_empty = output_dir.read_dir()?.next().is_none();
        if !is_empty && !overwrite {
            return Err(format!(
                "output directory {} already exists and is not empty; pass --overwrite to replace it",
                output_dir.display()
            )
            .into());
        }
        if overwrite {
            fs::remove_dir_all(output_dir)?;
        }
    }
    Ok(())
}

fn print_dry_run(config: &DatasetConfig) -> Result<(), Box<dyn Error>> {
    println!("dataset_generator: dry run");
    println!("{}", serde_json::to_string_pretty(config)?);
    println!("dataset_generator: planned output files");
    for frame_index in 1..=config.frame_count {
        let paths = frame_output_paths_for_selection(frame_index as u64, config.outputs);
        for path in [
            paths.rgb.as_deref(),
            paths.depth.as_deref(),
            paths.depth_preview.as_deref(),
            paths.segmentation.as_deref(),
            paths.segmentation_preview.as_deref(),
            Some(paths.metadata.as_str()),
        ]
        .into_iter()
        .flatten()
        {
            println!("  {}", config.output_dir.join(path).display());
        }
    }
    println!(
        "  {}",
        config.output_dir.join("dataset_manifest.json").display()
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

fn render_dataset(
    config: &DatasetConfig,
    object_ids: &[sim_sensors::ObjectIdMetadata],
    render_session: &RenderSession,
    writer: &mut DatasetWriter,
) -> Result<(), Box<dyn Error>> {
    for frame_index in 1..=config.frame_count {
        let timestamp_seconds = (frame_index - 1) as f64 / 30.0;
        let camera = camera_for_dataset_frame(
            &config.camera_path,
            frame_index,
            config.frame_count,
            config.width,
            config.height,
            config.seed,
        );
        let sensor = RgbCameraSensor::new("rgb-main", camera);
        let paths = frame_output_paths_for_selection(frame_index as u64, config.outputs);
        let metadata = DatasetFrameMetadata::new(
            frame_index as u64,
            timestamp_seconds,
            sensor.id(),
            config.seed,
            &config.camera_path,
            sensor.camera(),
            &paths,
            config.scene_path.as_ref().map(path_to_string),
            object_ids.to_vec(),
            Some(render_session.backend_label()),
        );

        let images = render_session.render_images(&sensor, frame_index as u64)?;
        writer.write_sensor_outputs(frame_index as u64, &images, &metadata)?;
    }

    Ok(())
}

enum RenderSession {
    #[cfg(feature = "rocm")]
    Rocm(RocmDatasetRenderer),
    #[cfg(not(feature = "rocm"))]
    CpuPreview,
}

impl RenderSession {
    fn open(scene: &Scene) -> Result<Self, Box<dyn Error>> {
        #[cfg(feature = "rocm")]
        {
            match RocmSensorRenderer::new() {
                Ok(renderer) => {
                    println!(
                        "dataset_generator: using ROCm-Oxide renderer on {}",
                        renderer.device_arch()
                    );
                    let uploaded_scene = renderer.upload_scene(scene)?;
                    return Ok(Self::Rocm(RocmDatasetRenderer {
                        backend_label: format!("rocm:{}", renderer.device_arch()),
                        renderer,
                        uploaded_scene,
                    }));
                }
                Err(err) => return Err(Box::new(err)),
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = scene;
            println!("dataset_generator: ROCm renderer unavailable: built without --features rocm");
            println!("dataset_generator: writing deterministic CPU preview outputs");
            Ok(Self::CpuPreview)
        }
    }

    fn backend_label(&self) -> String {
        match self {
            #[cfg(feature = "rocm")]
            Self::Rocm(rocm) => rocm.backend_label.clone(),
            #[cfg(not(feature = "rocm"))]
            Self::CpuPreview => "cpu-preview".to_string(),
        }
    }

    fn render_images(
        &self,
        sensor: &RgbCameraSensor,
        frame_index: u64,
    ) -> Result<SensorImageSet, Box<dyn Error>> {
        match self {
            #[cfg(feature = "rocm")]
            Self::Rocm(rocm) => {
                let render_metadata =
                    FrameMetadata::new(frame_index, frame_index as f64 / 30.0, sensor.id());
                let output = rocm.renderer.render_uploaded_scene_to_device(
                    &rocm.uploaded_scene,
                    sensor,
                    render_metadata,
                )?;
                let host = rocm.renderer.copy_all_to_host(&output)?;
                Ok(SensorImageSet::from_frames(
                    host.rgb,
                    host.depth,
                    host.segmentation,
                )?)
            }
            #[cfg(not(feature = "rocm"))]
            Self::CpuPreview => Ok(SensorImageSet::synthetic_preview(
                sensor.intrinsics().width,
                sensor.intrinsics().height,
                frame_index,
            )),
        }
    }
}

#[cfg(feature = "rocm")]
struct RocmDatasetRenderer {
    backend_label: String,
    renderer: RocmSensorRenderer,
    uploaded_scene: sim_render_rocm::RocmScene,
}

fn path_to_string(path: &PathBuf) -> String {
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_accepts_generation_flags_and_validate_subcommand() {
        let generate = Args::try_parse_from([
            "dataset_generator",
            "--camera-path",
            "random",
            "--seed",
            "1234",
            "--frames",
            "4",
            "--out",
            "target/sim_dataset",
            "--overwrite",
            "--dry-run",
        ])
        .unwrap();

        assert_eq!(generate.frames, Some(4));
        assert_eq!(generate.seed, Some(1234));
        assert_eq!(generate.camera_path, Some(CameraPathKind::Random));
        assert!(generate.overwrite);
        assert!(generate.dry_run);

        let validate = Args::try_parse_from([
            "dataset_generator",
            "validate",
            "--dataset",
            "target/sim_dataset",
        ])
        .unwrap();

        assert!(matches!(
            validate.command,
            Some(Command::Validate(ValidateArgs { .. }))
        ));
    }
}
