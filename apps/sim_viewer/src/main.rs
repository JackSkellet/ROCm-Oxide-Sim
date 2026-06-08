use clap::{Parser, ValueEnum};
use pixels::{Pixels, SurfaceTexture};
use sim_core::{Camera, Scene, Vec3};
use sim_datasets::{
    DepthImage, RgbImage, ScenarioConfig, SegmentationImage, SensorImageSet, depth_preview_pixels,
    segmentation_color,
};
use sim_render_rocm::{RocmSensorRenderer, rocm_feature_enabled};
use sim_sensors::{DepthMetadata, FrameMetadata, RgbCameraSensor};
use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

#[derive(Debug, Parser)]
#[command(about = "Live viewer for an uploaded sim-core scene rendered by ROCm.")]
struct Args {
    #[arg(long, default_value_t = 1280)]
    width: u32,
    #[arg(long, default_value_t = 720)]
    height: u32,
    #[arg(long, value_enum, default_value_t = ViewMode::Rgb)]
    mode: ViewMode,
    #[arg(long)]
    frames: Option<u64>,
    #[arg(long, value_enum, default_value_t = CameraMode::Static)]
    camera: CameraMode,
    #[arg(long)]
    scene: Option<std::path::PathBuf>,
    #[arg(long)]
    scenario: Option<std::path::PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ViewMode {
    Rgb,
    Depth,
    Segmentation,
}

impl fmt::Display for ViewMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rgb => write!(f, "rgb"),
            Self::Depth => write!(f, "depth"),
            Self::Segmentation => write!(f, "segmentation"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CameraMode {
    Static,
    Orbit,
}

impl fmt::Display for CameraMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Static => write!(f, "static"),
            Self::Orbit => write!(f, "orbit"),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.width == 0 || args.height == 0 {
        return Err("viewer width and height must be nonzero".into());
    }

    let scenario = args.scenario.as_deref().map(load_scenario).transpose()?;
    let scene_path = args.scene.as_deref().or_else(|| {
        scenario
            .as_ref()
            .map(|scenario| scenario.scene_path.as_path())
    });
    let scene = load_scene(scene_path)?;
    let initial_camera = scenario
        .as_ref()
        .and_then(|scenario| scenario.primary_camera().map(|(_mount, camera)| camera))
        .unwrap_or_else(Camera::default_rgb)
        .with_resolution(args.width, args.height);
    let renderer = open_renderer()?;
    if let Some(scenario) = &scenario {
        println!(
            "sim_viewer: scenario={} rig={} sensors={}",
            scenario.name,
            scenario.rig.name,
            scenario.rig.mounts.len()
        );
    }

    println!(
        "sim_viewer: {}x{} mode={} camera={}",
        args.width, args.height, args.mode, args.camera
    );
    println!("sim_viewer: scene entities={}", scene.len());
    println!("sim_viewer: presentation path is ROCm render -> host copy -> pixels/winit upload");

    if let Some(frames) = args.frames {
        return run_headless_frames(&args, &scene, renderer.as_ref(), frames, initial_camera);
    }

    run_windowed(args, scene, renderer, initial_camera)
}

fn load_scene(path: Option<&std::path::Path>) -> Result<Scene, Box<dyn Error>> {
    if let Some(path) = path {
        let json = std::fs::read_to_string(path)?;
        let scene = serde_json::from_str(&json)?;
        println!("sim_viewer: loaded scene {}", path.display());
        Ok(scene)
    } else {
        Ok(Scene::default_sensor_scene())
    }
}

fn load_scenario(path: &std::path::Path) -> Result<ScenarioConfig, Box<dyn Error>> {
    let json = std::fs::read_to_string(path)?;
    let scenario = serde_json::from_str::<ScenarioConfig>(&json)?;
    println!("sim_viewer: loaded scenario {}", path.display());
    Ok(scenario)
}

fn open_renderer() -> Result<Option<RocmSensorRenderer>, Box<dyn Error>> {
    match RocmSensorRenderer::new() {
        Ok(renderer) => {
            println!(
                "sim_viewer: ROCm backend active on {}",
                renderer.device_arch()
            );
            Ok(Some(renderer))
        }
        Err(err) if rocm_feature_enabled() => Err(Box::new(err)),
        Err(err) => {
            println!("sim_viewer: ROCm backend unavailable: {err}");
            println!("sim_viewer: using deterministic CPU preview frames");
            Ok(None)
        }
    }
}

fn run_headless_frames(
    args: &Args,
    scene: &Scene,
    renderer: Option<&RocmSensorRenderer>,
    frames: u64,
    initial_camera: Camera,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let mut rgba_len = 0usize;

    for frame_index in 1..=frames {
        let camera = camera_for_mode(
            args.camera,
            args.width,
            args.height,
            frame_index,
            initial_camera,
        );
        let images = render_images(scene, renderer, camera, frame_index)?;
        let rgba = rgba_for_mode(args.mode, &images);
        rgba_len = rgba.len();
        println!(
            "sim_viewer: rendered frame {frame_index}/{frames} mode={} bytes={}",
            args.mode, rgba_len
        );
    }

    let elapsed = started.elapsed();
    let fps = if elapsed.as_secs_f64() > 0.0 {
        frames as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    println!(
        "sim_viewer: completed {frames} frames in {:.2}s ({:.1} fps), last rgba bytes={rgba_len}",
        elapsed.as_secs_f64(),
        fps
    );
    Ok(())
}

fn run_windowed(
    args: Args,
    scene: Scene,
    renderer: Option<RocmSensorRenderer>,
    initial_camera: Camera,
) -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("rocm-oxide-sim viewer")
        .with_inner_size(LogicalSize::new(args.width as f64, args.height as f64))
        .build(&event_loop)?;
    let surface = SurfaceTexture::new(args.width, args.height, &window);
    let mut pixels = Pixels::new(args.width, args.height, surface)?;

    println!("sim_viewer: controls: 1 RGB, 2 depth, 3 segmentation, R reset, Esc quit");
    println!("sim_viewer: controls: W/S forward/back, A/D strafe, arrows look, Shift faster");

    let mut state = ViewerState::new(
        args.mode,
        args.camera,
        args.width,
        args.height,
        initial_camera,
    );
    let mut pressed = HashSet::new();
    let mut last_report = Instant::now();
    let mut frames_since_report = 0u64;
    let mut last_frame = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => {
                    handle_keyboard_input(input, &mut pressed, &mut state, control_flow);
                }
                WindowEvent::Resized(size) => {
                    if let Err(err) = pixels.resize_surface(size.width, size.height) {
                        eprintln!("sim_viewer: failed to resize surface: {err}");
                        *control_flow = ControlFlow::Exit;
                    }
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let now = Instant::now();
                let dt = now.duration_since(last_frame);
                last_frame = now;
                state.update_controls(&pressed, dt);
                state.frame_index += 1;

                let camera = state.camera();
                let render_result =
                    render_images(&scene, renderer.as_ref(), camera, state.frame_index)
                        .map(|images| rgba_for_mode(state.mode, &images));

                match render_result {
                    Ok(rgba) => {
                        pixels.frame_mut().copy_from_slice(&rgba);
                        if let Err(err) = pixels.render() {
                            eprintln!("sim_viewer: pixels render failed: {err}");
                            *control_flow = ControlFlow::Exit;
                        }
                    }
                    Err(err) => {
                        eprintln!("sim_viewer: render failed: {err}");
                        *control_flow = ControlFlow::Exit;
                    }
                }

                frames_since_report += 1;
                if last_report.elapsed() >= Duration::from_secs(1) {
                    let fps = frames_since_report as f64 / last_report.elapsed().as_secs_f64();
                    println!(
                        "sim_viewer: frame={} mode={} camera={} fps={:.1}",
                        state.frame_index, state.mode, state.camera_mode, fps
                    );
                    last_report = Instant::now();
                    frames_since_report = 0;
                }
            }
            _ => {}
        }
    });
}

fn handle_keyboard_input(
    input: KeyboardInput,
    pressed: &mut HashSet<VirtualKeyCode>,
    state: &mut ViewerState,
    control_flow: &mut ControlFlow,
) {
    let Some(key) = input.virtual_keycode else {
        return;
    };

    match input.state {
        ElementState::Pressed => {
            pressed.insert(key);
            match key {
                VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                VirtualKeyCode::Key1 => state.mode = ViewMode::Rgb,
                VirtualKeyCode::Key2 => state.mode = ViewMode::Depth,
                VirtualKeyCode::Key3 => state.mode = ViewMode::Segmentation,
                VirtualKeyCode::R => state.reset_camera(),
                _ => {}
            }
        }
        ElementState::Released => {
            pressed.remove(&key);
        }
    }
}

#[derive(Debug)]
struct ViewerState {
    mode: ViewMode,
    camera_mode: CameraMode,
    width: u32,
    height: u32,
    initial_camera: Camera,
    frame_index: u64,
    position: Vec3,
    yaw: f32,
    pitch: f32,
}

impl ViewerState {
    fn new(
        mode: ViewMode,
        camera_mode: CameraMode,
        width: u32,
        height: u32,
        initial_camera: Camera,
    ) -> Self {
        let mut state = Self {
            mode,
            camera_mode,
            width,
            height,
            initial_camera,
            frame_index: 0,
            position: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
        };
        state.reset_camera();
        state
    }

    fn reset_camera(&mut self) {
        let camera = self.initial_camera.with_resolution(self.width, self.height);
        self.position = camera.position;
        self.yaw = camera.forward.x.atan2(-camera.forward.z);
        self.pitch = camera.forward.y.asin();
        self.frame_index = 0;
    }

    fn update_controls(&mut self, pressed: &HashSet<VirtualKeyCode>, dt: Duration) {
        if self.camera_mode == CameraMode::Orbit {
            return;
        }

        let seconds = dt.as_secs_f32().clamp(0.0, 0.05);
        let speed = if pressed.contains(&VirtualKeyCode::LShift)
            || pressed.contains(&VirtualKeyCode::RShift)
        {
            5.0
        } else {
            1.8
        };
        let look_speed = 1.4;

        if pressed.contains(&VirtualKeyCode::Left) {
            self.yaw -= look_speed * seconds;
        }
        if pressed.contains(&VirtualKeyCode::Right) {
            self.yaw += look_speed * seconds;
        }
        if pressed.contains(&VirtualKeyCode::Up) {
            self.pitch += look_speed * seconds;
        }
        if pressed.contains(&VirtualKeyCode::Down) {
            self.pitch -= look_speed * seconds;
        }
        self.pitch = self.pitch.clamp(-1.2, 1.2);

        let forward = forward_from_angles(self.yaw, self.pitch);
        let right = forward.cross(Vec3::Y).normalized();
        let step = speed * seconds;

        if pressed.contains(&VirtualKeyCode::W) {
            self.position = self.position + forward * step;
        }
        if pressed.contains(&VirtualKeyCode::S) {
            self.position = self.position - forward * step;
        }
        if pressed.contains(&VirtualKeyCode::D) {
            self.position = self.position + right * step;
        }
        if pressed.contains(&VirtualKeyCode::A) {
            self.position = self.position - right * step;
        }
    }

    fn camera(&self) -> Camera {
        match self.camera_mode {
            CameraMode::Static => {
                let forward = forward_from_angles(self.yaw, self.pitch);
                let aspect = self.width as f32 / self.height as f32;
                Camera::look_at(self.position, self.position + forward, 55.0, aspect)
                    .with_resolution(self.width, self.height)
            }
            CameraMode::Orbit => camera_for_mode(
                self.camera_mode,
                self.width,
                self.height,
                self.frame_index,
                self.initial_camera,
            ),
        }
    }
}

fn camera_for_mode(
    mode: CameraMode,
    width: u32,
    height: u32,
    frame_index: u64,
    initial_camera: Camera,
) -> Camera {
    let aspect = width as f32 / height as f32;
    match mode {
        CameraMode::Static => initial_camera.with_resolution(width, height),
        CameraMode::Orbit => {
            let t = frame_index as f32 * 0.035;
            let target = Vec3::new(0.0, 0.55, -1.45);
            let position = Vec3::new(
                t.sin() * 4.1,
                1.35 + (t * 0.5).sin() * 0.25,
                -1.45 + t.cos() * 4.1,
            );
            Camera::look_at(position, target, 55.0, aspect).with_resolution(width, height)
        }
    }
}

fn forward_from_angles(yaw: f32, pitch: f32) -> Vec3 {
    let cos_pitch = pitch.cos();
    Vec3::new(yaw.sin() * cos_pitch, pitch.sin(), -yaw.cos() * cos_pitch).normalized()
}

fn render_images(
    scene: &Scene,
    renderer: Option<&RocmSensorRenderer>,
    camera: Camera,
    frame_index: u64,
) -> Result<SensorImageSet, Box<dyn Error>> {
    if let Some(renderer) = renderer {
        let sensor = RgbCameraSensor::new("rgb-main", camera);
        let metadata = FrameMetadata::new(frame_index, frame_index as f64 / 60.0, sensor.id())
            .with_depth(DepthMetadata::linear_ray_distance_meters())
            .with_scene_object_ids(scene);
        let output = renderer.render_all_host(scene, &sensor, metadata)?;
        Ok(SensorImageSet::from_frames(
            output.rgb,
            output.depth,
            output.segmentation,
        )?)
    } else {
        Ok(SensorImageSet::synthetic_preview(
            camera.width,
            camera.height,
            frame_index,
        ))
    }
}

fn rgba_for_mode(mode: ViewMode, images: &SensorImageSet) -> Vec<u8> {
    match mode {
        ViewMode::Rgb => rgb_to_rgba(&images.rgb),
        ViewMode::Depth => depth_to_rgba(&images.depth),
        ViewMode::Segmentation => segmentation_to_rgba(&images.segmentation),
    }
}

fn rgb_to_rgba(image: &RgbImage) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(image.pixels.len() * 4);
    for &pixel in &image.pixels {
        rgba.push(((pixel >> 16) & 0xff) as u8);
        rgba.push(((pixel >> 8) & 0xff) as u8);
        rgba.push((pixel & 0xff) as u8);
        rgba.push(255);
    }
    rgba
}

fn depth_to_rgba(image: &DepthImage) -> Vec<u8> {
    let preview = depth_preview_pixels(image);
    let mut rgba = Vec::with_capacity(preview.len() * 4);
    for value in preview {
        rgba.extend_from_slice(&[value, value, value, 255]);
    }
    rgba
}

fn segmentation_to_rgba(image: &SegmentationImage) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(image.pixels.len() * 4);
    for &object_id in &image.pixels {
        let color = segmentation_color(object_id);
        rgba.push(((color >> 16) & 0xff) as u8);
        rgba.push(((color >> 8) & 0xff) as u8);
        rgba.push((color & 0xff) as u8);
        rgba.push(255);
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_accepts_viewer_options() {
        Args::command().debug_assert();

        let args = Args::try_parse_from([
            "sim_viewer",
            "--width",
            "320",
            "--height",
            "180",
            "--mode",
            "segmentation",
            "--frames",
            "4",
            "--camera",
            "orbit",
            "--scenario",
            "examples/scenarios/basic_sensor_rig.json",
        ])
        .unwrap();

        assert_eq!(args.width, 320);
        assert_eq!(args.height, 180);
        assert_eq!(args.mode, ViewMode::Segmentation);
        assert_eq!(args.frames, Some(4));
        assert_eq!(args.camera, CameraMode::Orbit);
        assert_eq!(
            args.scenario.as_deref(),
            Some(std::path::Path::new(
                "examples/scenarios/basic_sensor_rig.json"
            ))
        );
    }

    #[test]
    fn rgb_depth_and_segmentation_convert_to_rgba() {
        let images = SensorImageSet {
            rgb: RgbImage::new(1, 1, vec![0x0012_3456]).unwrap(),
            depth: DepthImage::new(2, 1, vec![0.0, 4.0]).unwrap(),
            segmentation: SegmentationImage::new(1, 1, vec![2]).unwrap(),
        };

        assert_eq!(
            rgba_for_mode(ViewMode::Rgb, &images),
            vec![0x12, 0x34, 0x56, 255]
        );
        assert_eq!(
            rgba_for_mode(ViewMode::Depth, &images),
            vec![0, 0, 0, 255, 255, 255, 255, 255]
        );
        assert_eq!(
            rgba_for_mode(ViewMode::Segmentation, &images),
            vec![0xe6, 0x1f, 0x1a, 255]
        );
    }
}
