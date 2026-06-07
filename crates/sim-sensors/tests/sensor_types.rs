use sim_core::{
    Camera, Entity, Material, MaterialKind, ObjectId, PrimitiveShape, Scene, Transform, Vec3,
};
use sim_sensors::{
    CameraIntrinsics, DepthFrame, DepthMetadata, FrameMetadata, FrameOutputMetadata, LidarConfig,
    LidarFrame, LidarSensor, ObjectIdMetadata, RgbCameraSensor, RgbFrame, SegmentationFrame,
    SensorFrame, SensorPose, scene_object_ids,
};

#[test]
fn camera_intrinsics_are_derived_from_camera_fov() {
    let camera = Camera::look_at(
        Vec3::new(0.0, 0.0, 4.0),
        Vec3::new(0.0, 0.0, 0.0),
        60.0,
        4.0 / 3.0,
    )
    .with_resolution(800, 600);

    let intrinsics = CameraIntrinsics::from_camera(&camera);

    assert_eq!(intrinsics.width, 800);
    assert_eq!(intrinsics.height, 600);
    assert!((intrinsics.cx - 399.5).abs() < 1.0e-6);
    assert!((intrinsics.cy - 299.5).abs() < 1.0e-6);
    assert!(intrinsics.fx > 0.0);
    assert!(intrinsics.fy > 0.0);
}

#[test]
fn sensor_frame_keeps_metadata_and_pixel_payload() {
    let metadata = FrameMetadata::new(42, 1.25, "rgb-main");
    let frame = SensorFrame::new(
        2,
        1,
        metadata.clone(),
        vec![0xff00_0000_u32, 0xffff_ffff_u32],
    );

    assert_eq!(frame.width, 2);
    assert_eq!(frame.height, 1);
    assert_eq!(frame.metadata, metadata);
    assert_eq!(frame.pixels.len(), 2);
}

#[test]
fn rgb_sensor_trait_exposes_pose_and_intrinsics() {
    let camera = Camera::default_rgb();
    let sensor = RgbCameraSensor::new("rgb-main", camera);

    assert_eq!(sensor.id(), "rgb-main");
    assert_eq!(sensor.pose(), SensorPose::from_camera(&camera));
    assert_eq!(sensor.intrinsics().width, camera.width);
}

#[test]
fn frame_type_aliases_cover_rgb_depth_and_segmentation_payloads() {
    let metadata = FrameMetadata::new(3, 0.1, "rgb-main");
    let rgb: RgbFrame = SensorFrame::new(1, 1, metadata.clone(), vec![0x0012_3456]);
    let depth: DepthFrame = SensorFrame::new(1, 1, metadata.clone(), vec![2.5]);
    let segmentation: SegmentationFrame = SensorFrame::new(1, 1, metadata, vec![2]);

    assert_eq!(rgb.pixels[0], 0x0012_3456);
    assert_eq!(depth.pixels[0], 2.5);
    assert_eq!(segmentation.pixels[0], 2);
}

#[test]
fn frame_metadata_describes_outputs_depth_and_builtin_object_ids() {
    let metadata = FrameMetadata::new(1, 0.0, "rgb-main")
        .with_depth(DepthMetadata::linear_ray_distance_meters())
        .with_output(FrameOutputMetadata::new(
            "rgb",
            "ppm",
            "rgb/frame_000001.ppm",
        ))
        .with_output(FrameOutputMetadata::new(
            "depth",
            "raw-f32-little-endian",
            "depth/frame_000001.f32",
        ))
        .with_object_id(ObjectIdMetadata::new(2, "red sphere"));

    assert_eq!(metadata.depth.as_ref().unwrap().miss_value, 0.0);
    assert_eq!(metadata.outputs.len(), 2);
    assert_eq!(metadata.object_ids[0].id, 2);
}

#[test]
fn scene_object_id_metadata_comes_from_scene_entities() {
    let scene = Scene::default_sensor_scene();

    let ids = scene_object_ids(&scene);

    assert_eq!(ids[0], ObjectIdMetadata::new(0, "background"));
    assert_eq!(
        ids.iter().map(|entry| entry.id).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );
    assert!(ids.iter().any(|entry| entry.label == "green sphere"));
}

#[test]
fn scene_object_id_metadata_includes_primitive_and_material_kind() {
    let mut scene = Scene::new();
    scene.add_entity(Entity::new(
        "red box",
        PrimitiveShape::box_with_half_extents(Vec3::splat(0.5)),
        Transform::from_translation(Vec3::new(0.0, 0.5, -1.0)),
        Material::new(Vec3::new(0.9, 0.1, 0.1), 0.45, 0.0).with_kind(MaterialKind::Matte),
        ObjectId::new(9),
    ));

    let ids = scene_object_ids(&scene);
    let red_box = ids.iter().find(|entry| entry.id == 9).unwrap();

    assert_eq!(red_box.label, "red box");
    assert_eq!(red_box.primitive.as_deref(), Some("box"));
    assert_eq!(red_box.material.as_deref(), Some("matte"));
}

#[test]
fn lidar_config_defaults_match_milestone_contract() {
    let config = LidarConfig::default();
    let sensor = LidarSensor::new("lidar-main", config);

    assert_eq!(sensor.id(), "lidar-main");
    assert_eq!(config.horizontal_samples, 512);
    assert_eq!(config.vertical_channels, 32);
    assert_eq!(config.horizontal_fov_degrees, 360.0);
    assert_eq!(config.vertical_fov_degrees, 30.0);
    assert_eq!(config.min_range_m, 0.1);
    assert_eq!(config.max_range_m, 50.0);
    assert_eq!(sensor.sample_count(), 512 * 32);
}

#[test]
fn lidar_frame_uses_zero_miss_convention() {
    let metadata = FrameMetadata::new(1, 0.0, "lidar-main");
    let frame = LidarFrame::new(
        2,
        1,
        metadata,
        vec![0.0, 4.5],
        vec![Vec3::ZERO, Vec3::new(0.0, 1.0, -4.5)],
        vec![0, 7],
    );

    assert_eq!(frame.sample_count(), 2);
    assert_eq!(frame.miss_sample_count(), 1);
    assert_eq!(frame.object_ids[0], 0);
    assert_eq!(frame.ranges_m[0], 0.0);
    assert_eq!(frame.points_xyz[0], Vec3::ZERO);
}
