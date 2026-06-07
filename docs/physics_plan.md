# Physics Plan

`sim-physics` currently contains only a `PhysicsWorld` trait and a
`NoopPhysicsWorld` implementation. That is intentional: the first milestone is
sensor output, not dynamics.

## Near-Term Boundary

The physics crate should initially define:

- Time-step API.
- Scene synchronization conventions.
- Static/dynamic body descriptors.
- Contact and raycast query traits.
- Error types that do not expose a specific engine.

## Backend Options

Rapier-style backend

Good first Rust-native option. It is relatively easy to integrate, has broad
community usage, and fits simple rigid body simulation. It is the likely first
adapter once object/body ownership is defined.

MuJoCo-style backend

Useful later for robotics-grade articulated bodies and control tasks. It brings
heavier model assumptions and should wait until robot model and actuator needs
are clearer.

Bullet-style backend

Useful for broad compatibility and robotics precedent, but the Rust boundary
may be more FFI-heavy. It is better as a later adapter if Rapier is insufficient.

Custom minimal backend

Useful only for deterministic kinematic tests, static collision queries, and
simple raycasts. It should not grow into a full physics engine.

## Deferred Work

- No full physics dependency in Milestone 0.
- No ROS2 integration yet.
- No URDF import yet.
- No GPU physics path yet.
- No scene editor or OpenUSD physics schema yet.
