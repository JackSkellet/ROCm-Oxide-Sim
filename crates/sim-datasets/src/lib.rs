//! Dataset export helpers.
//!
//! The initial dataset layout is intentionally small:
//!
//! ```text
//! dataset_root/
//!   rgb/frame_000001.ppm
//!   depth/frame_000001.f32
//!   depth_preview/frame_000001.pgm
//!   segmentation/frame_000001.u32
//!   segmentation_preview/frame_000001.ppm
//!   metadata/frame_000001.json
//!   dataset_manifest.json
//! ```

use serde::{Deserialize, Serialize};
use sim_sensors::{DepthFrame, FrameMetadata, RgbFrame, SegmentationFrame};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid image dimensions: {0}")]
    InvalidImage(String),
}

pub type Result<T> = std::result::Result<T, DatasetError>;

/// Host RGB image in packed `0x00RRGGBB` format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
}

impl RgbImage {
    pub fn new(width: u32, height: u32, pixels: Vec<u32>) -> Result<Self> {
        let expected = width as usize * height as usize;
        if pixels.len() != expected {
            return Err(DatasetError::InvalidImage(format!(
                "expected {expected} pixels for {width}x{height}, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn from_frame(frame: RgbFrame) -> Result<Self> {
        Self::new(frame.width, frame.height, frame.pixels)
    }

    /// Deterministic CPU preview image used only when the ROCm backend is not
    /// compiled in. This keeps CLI smoke tests useful on machines without ROCm.
    pub fn synthetic_preview(width: u32, height: u32, frame_index: u64) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let mut pixels = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let checker = ((x / 32) + (y / 32) + frame_index as u32) & 1;
                let red = ((x as f32 / width as f32) * 180.0 + 40.0) as u32;
                let green = ((y as f32 / height as f32) * 160.0 + 55.0) as u32;
                let blue = if checker == 0 { 210 } else { 120 };
                pixels.push((red.min(255) << 16) | (green.min(255) << 8) | blue);
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }
}

/// Host depth image. Values are linear camera ray distance in meters.
/// `0.0` means background/miss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<f32>,
}

impl DepthImage {
    pub fn new(width: u32, height: u32, pixels: Vec<f32>) -> Result<Self> {
        let expected = width as usize * height as usize;
        if pixels.len() != expected {
            return Err(DatasetError::InvalidImage(format!(
                "expected {expected} depth samples for {width}x{height}, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn from_frame(frame: DepthFrame) -> Result<Self> {
        Self::new(frame.width, frame.height, frame.pixels)
    }

    pub fn synthetic_preview(width: u32, height: u32, frame_index: u64) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let mut pixels = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let miss = ((x / 80) + (y / 80) + frame_index as u32) % 7 == 0;
                if miss {
                    pixels.push(0.0);
                } else {
                    let nx = x as f32 / width as f32;
                    let ny = y as f32 / height as f32;
                    pixels.push(1.0 + nx * 2.5 + ny * 4.0);
                }
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }
}

/// Host segmentation image with stable `u32` object IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentationImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
}

impl SegmentationImage {
    pub fn new(width: u32, height: u32, pixels: Vec<u32>) -> Result<Self> {
        let expected = width as usize * height as usize;
        if pixels.len() != expected {
            return Err(DatasetError::InvalidImage(format!(
                "expected {expected} segmentation samples for {width}x{height}, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn from_frame(frame: SegmentationFrame) -> Result<Self> {
        Self::new(frame.width, frame.height, frame.pixels)
    }

    pub fn synthetic_preview(width: u32, height: u32, frame_index: u64) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let mut pixels = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let id = match ((x * 4 / width) + (y * 3 / height) + frame_index as u32) % 5 {
                    0 => 0,
                    1 => 1,
                    2 => 2,
                    3 => 3,
                    _ => 4,
                };
                pixels.push(id);
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorImageSet {
    pub rgb: RgbImage,
    pub depth: DepthImage,
    pub segmentation: SegmentationImage,
}

impl SensorImageSet {
    pub fn from_frames(
        rgb: RgbFrame,
        depth: DepthFrame,
        segmentation: SegmentationFrame,
    ) -> Result<Self> {
        Ok(Self {
            rgb: RgbImage::from_frame(rgb)?,
            depth: DepthImage::from_frame(depth)?,
            segmentation: SegmentationImage::from_frame(segmentation)?,
        })
    }

    pub fn synthetic_preview(width: u32, height: u32, frame_index: u64) -> Self {
        Self {
            rgb: RgbImage::synthetic_preview(width, height, frame_index),
            depth: DepthImage::synthetic_preview(width, height, frame_index),
            segmentation: SegmentationImage::synthetic_preview(width, height, frame_index),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestFrame {
    pub frame_index: u64,
    pub rgb: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segmentation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segmentation_preview: Option<String>,
    pub metadata: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub format: String,
    pub version: u32,
    pub frame_count: usize,
    pub frames: Vec<ManifestFrame>,
}

/// Writer for the initial RGB + metadata dataset layout.
#[derive(Debug)]
pub struct DatasetWriter {
    root: PathBuf,
    frames: Vec<ManifestFrame>,
}

impl DatasetWriter {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("rgb"))?;
        fs::create_dir_all(root.join("depth"))?;
        fs::create_dir_all(root.join("depth_preview"))?;
        fs::create_dir_all(root.join("segmentation"))?;
        fs::create_dir_all(root.join("segmentation_preview"))?;
        fs::create_dir_all(root.join("metadata"))?;
        Ok(Self {
            root,
            frames: Vec::new(),
        })
    }

    pub fn write_rgb_frame(
        &mut self,
        frame_index: u64,
        image: &RgbImage,
        metadata: &FrameMetadata,
    ) -> Result<()> {
        let file_stem = format!("frame_{frame_index:06}");
        let rgb_relative = format!("rgb/{file_stem}.ppm");
        let metadata_relative = format!("metadata/{file_stem}.json");

        write_ppm(self.root.join(&rgb_relative), image)?;
        write_metadata_json(self.root.join(&metadata_relative), metadata)?;

        self.frames.push(ManifestFrame {
            frame_index,
            rgb: rgb_relative,
            depth: None,
            depth_preview: None,
            segmentation: None,
            segmentation_preview: None,
            metadata: metadata_relative,
        });
        Ok(())
    }

    pub fn write_sensor_outputs(
        &mut self,
        frame_index: u64,
        images: &SensorImageSet,
        metadata: &FrameMetadata,
    ) -> Result<()> {
        let file_stem = format!("frame_{frame_index:06}");
        let rgb_relative = format!("rgb/{file_stem}.ppm");
        let depth_relative = format!("depth/{file_stem}.f32");
        let depth_preview_relative = format!("depth_preview/{file_stem}.pgm");
        let segmentation_relative = format!("segmentation/{file_stem}.u32");
        let segmentation_preview_relative = format!("segmentation_preview/{file_stem}.ppm");
        let metadata_relative = format!("metadata/{file_stem}.json");

        write_ppm(self.root.join(&rgb_relative), &images.rgb)?;
        write_depth_f32(self.root.join(&depth_relative), &images.depth)?;
        write_depth_preview_pgm(self.root.join(&depth_preview_relative), &images.depth)?;
        write_segmentation_u32(self.root.join(&segmentation_relative), &images.segmentation)?;
        write_segmentation_preview_ppm(
            self.root.join(&segmentation_preview_relative),
            &images.segmentation,
        )?;
        write_metadata_json(self.root.join(&metadata_relative), metadata)?;

        self.frames.push(ManifestFrame {
            frame_index,
            rgb: rgb_relative,
            depth: Some(depth_relative),
            depth_preview: Some(depth_preview_relative),
            segmentation: Some(segmentation_relative),
            segmentation_preview: Some(segmentation_preview_relative),
            metadata: metadata_relative,
        });
        Ok(())
    }

    pub fn finish(&self) -> Result<DatasetManifest> {
        let manifest = DatasetManifest {
            format: "rocm-oxide-sim-rgbd-segmentation".to_string(),
            version: 1,
            frame_count: self.frames.len(),
            frames: self.frames.clone(),
        };
        let path = self.root.join("dataset_manifest.json");
        let json = serde_json::to_string_pretty(&manifest)?;
        fs::write(path, json)?;
        Ok(manifest)
    }
}

pub fn write_ppm(path: impl AsRef<Path>, image: &RgbImage) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    write!(writer, "P6\n{} {}\n255\n", image.width, image.height)?;
    for &pixel in &image.pixels {
        let red = ((pixel >> 16) & 0xff) as u8;
        let green = ((pixel >> 8) & 0xff) as u8;
        let blue = (pixel & 0xff) as u8;
        writer.write_all(&[red, green, blue])?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_depth_f32(path: impl AsRef<Path>, image: &DepthImage) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for &sample in &image.pixels {
        writer.write_all(&sample.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_depth_preview_pgm(path: impl AsRef<Path>, image: &DepthImage) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    write!(writer, "P5\n{} {}\n255\n", image.width, image.height)?;
    writer.write_all(&depth_preview_pixels(image))?;
    writer.flush()?;
    Ok(())
}

pub fn depth_preview_pixels(image: &DepthImage) -> Vec<u8> {
    let finite_positive = image
        .pixels
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0);
    let (mut min_depth, mut max_depth) = (f32::INFINITY, f32::NEG_INFINITY);
    for value in finite_positive {
        min_depth = min_depth.min(value);
        max_depth = max_depth.max(value);
    }

    if !min_depth.is_finite() || !max_depth.is_finite() {
        return vec![0; image.pixels.len()];
    }

    image
        .pixels
        .iter()
        .map(|&value| {
            if !value.is_finite() || value <= 0.0 {
                return 0;
            }
            if (max_depth - min_depth).abs() <= f32::EPSILON {
                return 255;
            }
            let normalized = (value - min_depth) / (max_depth - min_depth);
            (255.0 - normalized * 223.0).round().clamp(32.0, 255.0) as u8
        })
        .collect()
}

pub fn write_segmentation_u32(path: impl AsRef<Path>, image: &SegmentationImage) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for &sample in &image.pixels {
        writer.write_all(&sample.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_segmentation_preview_ppm(
    path: impl AsRef<Path>,
    image: &SegmentationImage,
) -> Result<()> {
    let preview = RgbImage::new(
        image.width,
        image.height,
        image
            .pixels
            .iter()
            .copied()
            .map(segmentation_color)
            .collect(),
    )?;
    write_ppm(path, &preview)
}

pub fn segmentation_color(object_id: u32) -> u32 {
    match object_id {
        0 => 0x0000_0000,
        1 => 0x0080_8080,
        2 => 0x00e6_1f1a,
        3 => 0x001a_9e38,
        4 => 0x001a_47e6,
        other => {
            let hash = other.wrapping_mul(0x45d9_f3b);
            let red = 48 + (hash & 0x7f);
            let green = 48 + ((hash >> 8) & 0x7f);
            let blue = 48 + ((hash >> 16) & 0x7f);
            (red << 16) | (green << 8) | blue
        }
    }
}

pub fn write_metadata_json(path: impl AsRef<Path>, metadata: &FrameMetadata) -> Result<()> {
    let json = serde_json::to_string_pretty(metadata)?;
    fs::write(path, json)?;
    Ok(())
}
