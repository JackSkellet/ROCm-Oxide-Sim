use sim_datasets::{
    CameraPathConfig, DatasetConfig, DatasetManifest, DatasetWriter, DepthImage, OutputSelection,
    RgbImage, SegmentationImage, SensorImageSet, ValidationError, camera_for_dataset_frame,
    depth_preview_pixels, frame_output_paths, segmentation_color, validate_dataset,
};
use sim_sensors::{DepthMetadata, FrameMetadata, FrameOutputMetadata, ObjectIdMetadata};

#[test]
fn dataset_writer_writes_all_sensor_outputs_and_manifest() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut writer = DatasetWriter::new(temp_dir.path()).unwrap();
    let metadata = FrameMetadata::new(1, 0.5, "rgb-main")
        .with_depth(DepthMetadata::linear_ray_distance_meters())
        .with_output(FrameOutputMetadata::new(
            "rgb",
            "ppm",
            "rgb/frame_000001.ppm",
        ))
        .with_object_id(ObjectIdMetadata::new(2, "red sphere"));
    let images = SensorImageSet {
        rgb: RgbImage::new(2, 1, vec![0x00ff_0000, 0x0000_ff00]).unwrap(),
        depth: DepthImage::new(2, 1, vec![1.0, 3.0]).unwrap(),
        segmentation: SegmentationImage::new(2, 1, vec![2, 0]).unwrap(),
    };

    writer.write_sensor_outputs(1, &images, &metadata).unwrap();
    writer.finish().unwrap();

    assert!(temp_dir.path().join("rgb/frame_000001.ppm").exists());
    assert!(temp_dir.path().join("depth/frame_000001.f32").exists());
    assert!(
        temp_dir
            .path()
            .join("depth_preview/frame_000001.pgm")
            .exists()
    );
    assert!(
        temp_dir
            .path()
            .join("segmentation/frame_000001.u32")
            .exists()
    );
    assert!(
        temp_dir
            .path()
            .join("segmentation_preview/frame_000001.ppm")
            .exists()
    );
    assert!(temp_dir.path().join("metadata/frame_000001.json").exists());

    let manifest = std::fs::read_to_string(temp_dir.path().join("dataset_manifest.json")).unwrap();
    assert!(manifest.contains("\"frame_count\": 1"));
    assert!(manifest.contains("rgb/frame_000001.ppm"));
    assert!(manifest.contains("depth/frame_000001.f32"));
    assert!(manifest.contains("segmentation/frame_000001.u32"));
}

#[test]
fn depth_preview_maps_misses_to_black_and_normalizes_finite_depths() {
    let depth = DepthImage::new(4, 1, vec![0.0, 1.0, 3.0, f32::INFINITY]).unwrap();

    let preview = depth_preview_pixels(&depth);

    assert_eq!(preview, vec![0, 255, 32, 0]);
}

#[test]
fn segmentation_color_mapping_is_stable_for_builtin_ids() {
    assert_eq!(segmentation_color(0), 0x0000_0000);
    assert_eq!(segmentation_color(1), 0x0080_8080);
    assert_eq!(segmentation_color(2), 0x00e6_1f1a);
    assert_eq!(segmentation_color(3), 0x001a_9e38);
    assert_eq!(segmentation_color(4), 0x001a_47e6);
}

#[test]
fn frame_output_paths_use_six_digit_relative_names() {
    let paths = frame_output_paths(42);

    assert_eq!(paths.rgb.as_deref(), Some("rgb/frame_000042.ppm"));
    assert_eq!(paths.depth.as_deref(), Some("depth/frame_000042.f32"));
    assert_eq!(
        paths.depth_preview.as_deref(),
        Some("depth_preview/frame_000042.pgm")
    );
    assert_eq!(
        paths.segmentation.as_deref(),
        Some("segmentation/frame_000042.u32")
    );
    assert_eq!(
        paths.segmentation_preview.as_deref(),
        Some("segmentation_preview/frame_000042.ppm")
    );
    assert_eq!(paths.metadata, "metadata/frame_000042.json");
}

#[test]
fn dataset_config_parses_camera_path_and_output_selection() {
    let json = r#"{
      "scene_path": "examples/scenes/basic_scene.json",
      "output_dir": "target/config_dataset",
      "frame_count": 6,
      "width": 320,
      "height": 180,
      "camera_path": {
        "kind": "line",
        "start_position": { "x": -1.0, "y": 1.0, "z": 4.0 },
        "end_position": { "x": 1.0, "y": 1.2, "z": 3.0 },
        "target": { "x": 0.0, "y": 0.5, "z": -1.5 },
        "fov_y_degrees": 50.0
      },
      "seed": 99,
      "outputs": {
        "rgb": true,
        "depth": true,
        "depth_preview": false,
        "segmentation": true,
        "segmentation_preview": false,
        "metadata": true
      }
    }"#;

    let config: DatasetConfig = serde_json::from_str(json).unwrap();

    assert_eq!(config.frame_count, 6);
    assert_eq!(config.width, 320);
    assert_eq!(config.height, 180);
    assert_eq!(config.seed, 99);
    assert!(matches!(config.camera_path, CameraPathConfig::Line { .. }));
    assert!(config.outputs.depth);
    assert!(!config.outputs.depth_preview);
}

#[test]
fn random_camera_path_is_reproducible_for_seed() {
    let config = CameraPathConfig::random_default();

    let a = camera_for_dataset_frame(&config, 2, 5, 320, 180, 1234);
    let b = camera_for_dataset_frame(&config, 2, 5, 320, 180, 1234);
    let c = camera_for_dataset_frame(&config, 2, 5, 320, 180, 4321);

    assert_eq!(a, b);
    assert_ne!(a.position, c.position);
}

#[test]
fn orbit_and_line_camera_paths_are_deterministic() {
    let orbit = CameraPathConfig::orbit_default();
    let line = CameraPathConfig::line_default();

    let orbit_first = camera_for_dataset_frame(&orbit, 1, 4, 640, 360, 7);
    let orbit_last = camera_for_dataset_frame(&orbit, 4, 4, 640, 360, 7);
    let line_first = camera_for_dataset_frame(&line, 1, 4, 640, 360, 7);
    let line_last = camera_for_dataset_frame(&line, 4, 4, 640, 360, 7);

    assert_ne!(orbit_first.position, orbit_last.position);
    assert_ne!(line_first.position, line_last.position);
    assert_eq!(line_first.position, line.start_position().unwrap());
    assert_eq!(line_last.position, line.end_position().unwrap());
}

#[test]
fn manifest_contains_reproducibility_and_output_contracts() {
    let manifest = DatasetManifest::new(
        2,
        640,
        360,
        123,
        CameraPathConfig::static_default(),
        OutputSelection::all(),
    )
    .with_scene_path(Some("examples/scenes/basic_scene.json".into()))
    .with_config_path(Some("examples/datasets/basic_orbit.json".into()))
    .with_renderer_backend("rocm:gfx1201")
    .with_object_ids(vec![ObjectIdMetadata::new(0, "background")])
    .with_frames(vec![
        frame_output_paths(1).to_manifest_frame(1),
        frame_output_paths(2).to_manifest_frame(2),
    ]);

    assert_eq!(manifest.dataset_format_version, 1);
    assert_eq!(manifest.generator, "rocm-oxide-sim dataset_generator");
    assert_eq!(manifest.frame_count, 2);
    assert_eq!(manifest.width, 640);
    assert_eq!(manifest.seed, 123);
    assert_eq!(
        manifest.scene_path.as_deref(),
        Some("examples/scenes/basic_scene.json")
    );
    assert_eq!(
        manifest.config_path.as_deref(),
        Some("examples/datasets/basic_orbit.json")
    );
    assert_eq!(manifest.renderer_backend.as_deref(), Some("rocm:gfx1201"));
    assert_eq!(manifest.frames[1].metadata, "metadata/frame_000002.json");
    assert_eq!(
        manifest.depth_convention.miss_value, 0.0,
        "depth miss convention should remain stable"
    );
    assert_eq!(manifest.segmentation_convention.background_id, 0);
}

#[test]
fn manifest_preserves_box_object_metadata() {
    let manifest = DatasetManifest::new(
        1,
        320,
        180,
        11,
        CameraPathConfig::static_default(),
        OutputSelection::all(),
    )
    .with_object_ids(vec![
        ObjectIdMetadata::new(0, "background"),
        ObjectIdMetadata::new(8, "red box")
            .with_primitive("box")
            .with_material("matte"),
    ])
    .with_frames(vec![frame_output_paths(1).to_manifest_frame(1)]);

    let red_box = manifest
        .object_ids
        .iter()
        .find(|entry| entry.id == 8)
        .unwrap();

    assert_eq!(red_box.primitive.as_deref(), Some("box"));
    assert_eq!(red_box.material.as_deref(), Some("matte"));
}

#[test]
fn validation_reports_missing_expected_files() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut writer = DatasetWriter::new_with_config(
        temp_dir.path(),
        DatasetConfig {
            frame_count: 1,
            width: 2,
            height: 1,
            outputs: OutputSelection::all(),
            ..DatasetConfig::default()
        },
        None,
        None,
        "cpu-preview".to_string(),
        vec![ObjectIdMetadata::new(0, "background")],
    )
    .unwrap();
    let metadata = FrameMetadata::new(1, 0.0, "rgb-main")
        .with_depth(DepthMetadata::linear_ray_distance_meters())
        .with_object_id(ObjectIdMetadata::new(0, "background"));
    let images = SensorImageSet {
        rgb: RgbImage::new(2, 1, vec![0, 0]).unwrap(),
        depth: DepthImage::new(2, 1, vec![0.0, 1.0]).unwrap(),
        segmentation: SegmentationImage::new(2, 1, vec![0, 1]).unwrap(),
    };

    writer.write_sensor_outputs(1, &images, &metadata).unwrap();
    writer.finish().unwrap();
    std::fs::remove_file(temp_dir.path().join("rgb/frame_000001.ppm")).unwrap();

    let err = validate_dataset(temp_dir.path()).unwrap_err();

    assert!(matches!(err, ValidationError::MissingFile(_)));
    assert!(err.to_string().contains("rgb/frame_000001.ppm"));
}
