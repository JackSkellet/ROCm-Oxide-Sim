use sim_datasets::{
    DatasetWriter, DepthImage, RgbImage, SegmentationImage, SensorImageSet, depth_preview_pixels,
    segmentation_color,
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
