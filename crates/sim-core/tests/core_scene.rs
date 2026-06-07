use sim_core::{
    Camera, Entity, EntityId, EulerRotation, Material, ObjectId, PrimitiveShape, Scene, Transform,
    Vec3,
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
