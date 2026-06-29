//! BrainVision Core Data Format reader.
//!
//! The SpanishBCBL EEG recordings ship as BrainVision triplets:
//! `*.vhdr` (text header), `*.vmrk` (text markers), and `*.eeg` (binary data).
//! This module parses all three into rUv Neural's
//! [`MultiChannelTimeSeries`](ruv_neural_core::signal::MultiChannelTimeSeries)
//! plus a list of markers, with no external dependencies.
//!
//! Supported: `BINARY` data format, `MULTIPLEXED` and `VECTORIZED` orientation,
//! `INT_16` and `IEEE_FLOAT_32` binary formats, per-channel resolution scaling.

use std::fs;
use std::path::{Path, PathBuf};

use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::signal::MultiChannelTimeSeries;

fn err(msg: impl Into<String>) -> RuvNeuralError {
    RuvNeuralError::Signal(msg.into())
}

/// Binary sample encoding declared in `[Binary Infos]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinaryFormat {
    Int16,
    Float32,
}

/// Sample layout declared in `[Common Infos]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Orientation {
    /// `ch0[t0], ch1[t0], ... chN[t0], ch0[t1], ...`
    Multiplexed,
    /// `ch0[t0..tT], ch1[t0..tT], ...`
    Vectorized,
}

/// One channel's metadata from `[Channel Infos]`.
#[derive(Debug, Clone)]
pub struct ChannelInfo {
    /// Channel label (e.g. "Fz").
    pub name: String,
    /// Resolution in microvolts per least-significant-bit.
    pub resolution_uv: f64,
}

/// A marker parsed from the `.vmrk` file.
#[derive(Debug, Clone, PartialEq)]
pub struct Marker {
    /// Marker type (e.g. "Stimulus", "Response", "Comment").
    pub kind: String,
    /// Free-text description (e.g. "S  1", or a typed character).
    pub description: String,
    /// Zero-based sample position of the marker onset.
    pub position: usize,
}

impl Marker {
    /// Onset time in seconds given the recording sample rate.
    pub fn onset_s(&self, sample_rate_hz: f64) -> f64 {
        self.position as f64 / sample_rate_hz
    }
}

/// Parsed BrainVision header (`.vhdr`).
#[derive(Debug, Clone)]
struct VHeader {
    data_file: String,
    marker_file: String,
    num_channels: usize,
    sample_rate_hz: f64,
    binary_format: BinaryFormat,
    orientation: Orientation,
    channels: Vec<ChannelInfo>,
}

/// Result of reading a BrainVision recording.
#[derive(Debug, Clone)]
pub struct BrainVisionRecording {
    /// Continuous multi-channel signal in microvolts.
    pub series: MultiChannelTimeSeries,
    /// Per-channel metadata.
    pub channels: Vec<ChannelInfo>,
    /// Markers from the `.vmrk` file (empty if none present).
    pub markers: Vec<Marker>,
}

/// Split a `key=value` INI line, trimming whitespace.
fn split_kv(line: &str) -> Option<(&str, &str)> {
    line.split_once('=').map(|(k, v)| (k.trim(), v.trim()))
}

fn parse_header(text: &str) -> Result<VHeader> {
    let mut data_file = String::new();
    let mut marker_file = String::new();
    let mut num_channels = 0usize;
    let mut sampling_interval_us = 0.0f64;
    let mut binary_format = BinaryFormat::Int16;
    let mut orientation = Orientation::Multiplexed;
    let mut channels: Vec<ChannelInfo> = Vec::new();

    let mut section = String::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with(';') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].to_string();
            continue;
        }
        match section.as_str() {
            "Common Infos" => {
                if let Some((k, v)) = split_kv(line) {
                    match k {
                        "DataFile" => data_file = v.to_string(),
                        "MarkerFile" => marker_file = v.to_string(),
                        "NumberOfChannels" => {
                            num_channels = v.parse().map_err(|_| err("bad NumberOfChannels"))?
                        }
                        "SamplingInterval" => {
                            sampling_interval_us =
                                v.parse().map_err(|_| err("bad SamplingInterval"))?
                        }
                        "DataFormat" => {
                            if !v.eq_ignore_ascii_case("BINARY") {
                                return Err(err(format!("unsupported DataFormat: {v}")));
                            }
                        }
                        "DataOrientation" => {
                            orientation = match v.to_ascii_uppercase().as_str() {
                                "MULTIPLEXED" => Orientation::Multiplexed,
                                "VECTORIZED" => Orientation::Vectorized,
                                other => return Err(err(format!("bad DataOrientation: {other}"))),
                            }
                        }
                        _ => {}
                    }
                }
            }
            "Binary Infos" => {
                if let Some((k, v)) = split_kv(line) {
                    if k == "BinaryFormat" {
                        binary_format = match v.to_ascii_uppercase().as_str() {
                            "INT_16" => BinaryFormat::Int16,
                            "IEEE_FLOAT_32" => BinaryFormat::Float32,
                            other => return Err(err(format!("bad BinaryFormat: {other}"))),
                        };
                    }
                }
            }
            "Channel Infos" => {
                // ChN=name,reference,resolution,unit
                if let Some((_, v)) = split_kv(line) {
                    let parts: Vec<&str> = v.split(',').collect();
                    let name = parts.first().copied().unwrap_or("").to_string();
                    let resolution_uv = parts
                        .get(2)
                        .and_then(|s| s.trim().parse::<f64>().ok())
                        .filter(|r| *r != 0.0)
                        .unwrap_or(1.0);
                    channels.push(ChannelInfo { name, resolution_uv });
                }
            }
            _ => {}
        }
    }

    if sampling_interval_us <= 0.0 {
        return Err(err("missing/invalid SamplingInterval"));
    }
    if num_channels == 0 {
        return Err(err("missing NumberOfChannels"));
    }
    if data_file.is_empty() {
        return Err(err("missing DataFile"));
    }
    // Fill in channel info if the header omitted it.
    while channels.len() < num_channels {
        channels.push(ChannelInfo {
            name: format!("Ch{}", channels.len() + 1),
            resolution_uv: 1.0,
        });
    }

    Ok(VHeader {
        data_file,
        marker_file,
        num_channels,
        sample_rate_hz: 1.0e6 / sampling_interval_us,
        binary_format,
        orientation,
        channels,
    })
}

fn parse_markers(text: &str) -> Vec<Marker> {
    let mut markers = Vec::new();
    let mut in_section = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_section = line.eq_ignore_ascii_case("[Marker Infos]");
            continue;
        }
        if !in_section || line.is_empty() || line.starts_with(';') {
            continue;
        }
        // MkN=type,description,position,size,channel[,date]
        if let Some((_, v)) = split_kv(line) {
            let parts: Vec<&str> = v.split(',').collect();
            if parts.len() >= 3 {
                let position = parts[2].trim().parse::<usize>().unwrap_or(0);
                // BrainVision marker positions are 1-based; convert to 0-based.
                let position = position.saturating_sub(1);
                markers.push(Marker {
                    kind: parts[0].trim().to_string(),
                    description: parts[1].trim().to_string(),
                    position,
                });
            }
        }
    }
    markers
}

fn decode_binary(
    bytes: &[u8],
    header: &VHeader,
) -> Result<Vec<Vec<f64>>> {
    let nch = header.num_channels;
    let sample_bytes = match header.binary_format {
        BinaryFormat::Int16 => 2,
        BinaryFormat::Float32 => 4,
    };
    if bytes.len() % (sample_bytes * nch) != 0 {
        return Err(err("binary data length not a multiple of frame size"));
    }
    let total_samples = bytes.len() / sample_bytes;
    let num_per_channel = total_samples / nch;

    let read_value = |i: usize| -> f64 {
        let off = i * sample_bytes;
        match header.binary_format {
            BinaryFormat::Int16 => {
                i16::from_le_bytes([bytes[off], bytes[off + 1]]) as f64
            }
            BinaryFormat::Float32 => f32::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
            ]) as f64,
        }
    };

    let mut data = vec![vec![0.0f64; num_per_channel]; nch];
    match header.orientation {
        Orientation::Multiplexed => {
            for t in 0..num_per_channel {
                for c in 0..nch {
                    let v = read_value(t * nch + c) * header.channels[c].resolution_uv;
                    data[c][t] = v;
                }
            }
        }
        Orientation::Vectorized => {
            for c in 0..nch {
                for t in 0..num_per_channel {
                    let v = read_value(c * num_per_channel + t) * header.channels[c].resolution_uv;
                    data[c][t] = v;
                }
            }
        }
    }
    Ok(data)
}

/// Read a BrainVision recording given the path to its `.vhdr` file.
///
/// The `.eeg`/`.vmrk` files are resolved relative to the header's directory
/// using the names declared inside the header.
pub fn read_vhdr(vhdr_path: impl AsRef<Path>) -> Result<BrainVisionRecording> {
    let vhdr_path = vhdr_path.as_ref();
    let dir = vhdr_path.parent().unwrap_or_else(|| Path::new("."));

    let header_text = fs::read_to_string(vhdr_path)
        .map_err(|e| err(format!("read vhdr: {e}")))?;
    let header = parse_header(&header_text)?;

    let data_path: PathBuf = dir.join(&header.data_file);
    let bytes = fs::read(&data_path).map_err(|e| err(format!("read eeg data: {e}")))?;
    let data = decode_binary(&bytes, &header)?;

    let markers = if header.marker_file.is_empty() {
        Vec::new()
    } else {
        let mpath = dir.join(&header.marker_file);
        match fs::read_to_string(&mpath) {
            Ok(t) => parse_markers(&t),
            Err(_) => Vec::new(), // markers are optional
        }
    };

    let series = MultiChannelTimeSeries::new(data, header.sample_rate_hz, 0.0)?;
    Ok(BrainVisionRecording {
        series,
        channels: header.channels,
        markers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a tiny INT_16 multiplexed BrainVision triplet and read it back.
    #[test]
    fn round_trip_int16_multiplexed() {
        let dir = std::env::temp_dir().join(format!("bv_test_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let base = dir.join("rec");
        let vhdr = base.with_extension("vhdr");
        let vmrk = base.with_extension("vmrk");
        let eeg = base.with_extension("eeg");

        let vhdr_text = "\
Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile=rec.eeg
MarkerFile=rec.vmrk
DataFormat=BINARY
DataOrientation=MULTIPLEXED
NumberOfChannels=2
SamplingInterval=1000
[Binary Infos]
BinaryFormat=INT_16
[Channel Infos]
Ch1=Fz,,0.5,µV
Ch2=Cz,,0.5,µV
";
        fs::write(&vhdr, vhdr_text).unwrap();

        let vmrk_text = "\
Brain Vision Data Exchange Marker File, Version 1.0
[Marker Infos]
Mk1=New Segment,,1,1,0
Mk2=Response,a,3,1,0
";
        fs::write(&vmrk, vmrk_text).unwrap();

        // 3 time points, 2 channels, multiplexed: t0(c0,c1), t1(c0,c1), t2(c0,c1)
        let samples: [i16; 6] = [10, 20, 30, 40, 50, 60];
        let mut f = fs::File::create(&eeg).unwrap();
        for s in samples {
            f.write_all(&s.to_le_bytes()).unwrap();
        }
        drop(f);

        let rec = read_vhdr(&vhdr).unwrap();
        assert_eq!(rec.series.num_channels, 2);
        assert_eq!(rec.series.num_samples, 3);
        assert_eq!(rec.series.sample_rate_hz, 1000.0); // 1e6 / 1000us
        // resolution scaling: 10 * 0.5 = 5.0 on channel 0, sample 0
        assert!((rec.series.data[0][0] - 5.0).abs() < 1e-9);
        assert!((rec.series.data[1][0] - 10.0).abs() < 1e-9);
        assert!((rec.series.data[0][2] - 25.0).abs() < 1e-9);
        // markers: response 'a' at position 3 (1-based) -> 2 (0-based)
        let resp: Vec<&Marker> = rec.markers.iter().filter(|m| m.kind == "Response").collect();
        assert_eq!(resp.len(), 1);
        assert_eq!(resp[0].description, "a");
        assert_eq!(resp[0].position, 2);
        assert!((resp[0].onset_s(1000.0) - 0.002).abs() < 1e-9);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn float32_vectorized() {
        let dir = std::env::temp_dir().join(format!("bv_test_f32_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let base = dir.join("rec");
        let vhdr = base.with_extension("vhdr");
        let eeg = base.with_extension("eeg");

        let vhdr_text = "\
[Common Infos]
DataFile=rec.eeg
MarkerFile=
DataFormat=BINARY
DataOrientation=VECTORIZED
NumberOfChannels=2
SamplingInterval=2000
[Binary Infos]
BinaryFormat=IEEE_FLOAT_32
[Channel Infos]
Ch1=A,,1.0,µV
Ch2=B,,1.0,µV
";
        fs::write(&vhdr, vhdr_text).unwrap();

        // vectorized: c0[t0,t1,t2], c1[t0,t1,t2]
        let vals: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut f = fs::File::create(&eeg).unwrap();
        for v in vals {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        let rec = read_vhdr(&vhdr).unwrap();
        assert_eq!(rec.series.sample_rate_hz, 500.0); // 1e6/2000
        assert_eq!(rec.series.data[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(rec.series.data[1], vec![4.0, 5.0, 6.0]);
        assert!(rec.markers.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }
}
