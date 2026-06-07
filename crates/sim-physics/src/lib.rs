//! Placeholder physics traits.
//!
//! This crate defines the future physics boundary without pulling in Rapier,
//! MuJoCo, Bullet, or any other heavy dependency yet.

use sim_core::Scene;
use std::convert::Infallible;

/// Minimal physics backend interface.
pub trait PhysicsWorld {
    type Error;

    fn step(&mut self, scene: &mut Scene, dt_seconds: f32) -> Result<(), Self::Error>;
}

/// Placeholder backend that leaves the scene unchanged.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopPhysicsWorld;

impl PhysicsWorld for NoopPhysicsWorld {
    type Error = Infallible;

    fn step(&mut self, _scene: &mut Scene, _dt_seconds: f32) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_physics_keeps_scene_unchanged() {
        let mut scene = Scene::default_sensor_scene();
        let before = scene.clone();
        let mut physics = NoopPhysicsWorld;

        physics.step(&mut scene, 1.0 / 60.0).unwrap();

        assert_eq!(scene, before);
    }
}
