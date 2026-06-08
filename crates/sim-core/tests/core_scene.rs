use sim_core::{
    Camera, Entity, EntityId, EulerRotation, Material, MaterialKind, ObjectId, PrimitiveShape,
    Scene, Transform, Vec3,
};

#[test]
fn transform_applies_scale_rotation_and_translation() {
    let transform = Transform {
        translation: Vec3::new(1.0, 2.0, 3.0),
        rotation: EulerRotation::from_degrees(0.0, 0.0, 90.0),
        scale: Vec3::splat(2.0),
    };

    let transformed = transform.transform_point(Vec3::new(1.0, 0.0, 0.0));

    assert!((transformed.x - 1.0).abs() < 1.0e-5);
    assert!((transformed.y - 4.0).abs() < 1.0e-5);
    assert!((transformed.z - 3.0).abs() < 1.0e-5);
}

#[test]
fn transform_composes_parent_and_child_for_sensor_mounts() {
    let parent = Transform {
        translation: Vec3::new(10.0, 2.0, -3.0),
        rotation: EulerRotation::IDENTITY,
        scale: Vec3::new(2.0, 1.0, 0.5),
    };
    let child = Transform::from_translation(Vec3::new(1.0, 3.0, 4.0));

    let world = parent.compose(child);

    assert_eq!(world.translation, Vec3::new(12.0, 5.0, -1.0));
    assert_eq!(world.scale, Vec3::new(2.0, 1.0, 0.5));
}

#[test]
fn camera_from_look_at_points_toward_target() {
    let camera = Camera::look_at(
        Vec3::new(0.0, 1.0, 5.0),
        Vec3::new(0.0, 1.0, 0.0),
        60.0,
        16.0 / 9.0,
    );

    assert_eq!(camera.position, Vec3::new(0.0, 1.0, 5.0));
    assert!((camera.forward.z + 1.0).abs() < 1.0e-6);
    assert_eq!(camera.width, 640);
    assert_eq!(camera.height, 360);
}

#[test]
fn scene_adds_and_queries_entities_by_id_and_object_id() {
    let mut scene = Scene::new();
    let entity = Entity::new(
        "red sphere",
        PrimitiveShape::sphere(0.5),
        Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
        Material::matte(Vec3::new(1.0, 0.0, 0.0)),
        ObjectId::new(7),
    );

    let id = scene.add_entity(entity);

    assert_eq!(id, EntityId::new(1));
    assert_eq!(scene.entity(id).unwrap().name, "red sphere");
    assert_eq!(scene.by_object_id(ObjectId::new(7)).unwrap().id, id);
    assert_eq!(scene.entities().count(), 1);
}

#[test]
fn scene_json_parses_box_and_material_kind() {
    let json = r#"{
      "entities": [
        {
          "id": 1,
          "name": "emissive box",
          "shape": {
            "Box": {
              "half_extents": { "x": 0.5, "y": 0.25, "z": 0.75 }
            }
          },
          "transform": {
            "translation": { "x": 1.0, "y": 0.25, "z": -2.0 },
            "rotation": { "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
            "scale": { "x": 1.0, "y": 2.0, "z": 1.0 }
          },
          "material": {
            "base_color": { "x": 1.0, "y": 0.7, "z": 0.2 },
            "emission": { "x": 0.2, "y": 0.1, "z": 0.0 },
            "roughness": 0.35,
            "metallic": 0.0,
            "kind": "emissive"
          },
          "object_id": 8
        }
      ],
      "next_id": 2
    }"#;

    let scene: Scene = serde_json::from_str(json).unwrap();
    let entity = scene.by_object_id(ObjectId::new(8)).unwrap();

    assert_eq!(entity.name, "emissive box");
    assert_eq!(
        entity.shape,
        PrimitiveShape::box_with_half_extents(Vec3::new(0.5, 0.25, 0.75))
    );
    assert_eq!(entity.material.kind, MaterialKind::Emissive);
    assert_eq!(entity.material.emission, Vec3::new(0.2, 0.1, 0.0));
}

#[test]
fn boxes_scene_example_parses_expected_primitives() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/scenes/boxes_scene.json");
    let scene: Scene = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

    assert_eq!(scene.len(), 4);
    assert!(matches!(
        scene.by_object_id(ObjectId::new(5)).unwrap().shape,
        PrimitiveShape::Box { .. }
    ));
    assert_eq!(
        scene.by_object_id(ObjectId::new(6)).unwrap().material.kind,
        MaterialKind::MetalPreview
    );
}
