//! BIDS-EEG export for cross-tool interoperability (ADR-0016).
//!
//! ADR-0016 keeps the platform non-invasive but commits to *interoperate* with
//! the wider neuroscience tool-chain ("interop, not parity"). This module writes
//! a recording as a minimal, valid **BIDS** dataset using the BIDS-recommended
//! **BrainVision** representation, so a session can be opened in MNE-Python,
//! EEGLAB, FieldTrip, or any BIDS-aware tool.
//!
//! It emits, under `root/`:
//!
//! ```text
//! dataset_description.json
//! sub-<S>/[ses-<E>/]eeg/
//!   sub-<S>[_ses-<E>]_task-<T>_eeg.vhdr   # BrainVision header (text)
//!   sub-<S>[_ses-<E>]_task-<T>_eeg.vmrk   # BrainVision markers (text)
//!   sub-<S>[_ses-<E>]_task-<T>_eeg.eeg    # IEEE float32, multiplexed (binary)
//!   sub-<S>[_ses-<E>]_task-<T>_channels.tsv
//!   sub-<S>[_ses-<E>]_task-<T>_eeg.json   # BIDS sidecar
//! ```
//!
//! NWB (HDF5) and LSL (a live network protocol) need external C runtimes and are
//! intentionally **not** implemented here; they remain ADR-0016 roadmap items.

use std::io::Write;
use std::path::{Path, PathBuf};

use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::sensor::{SensorArray, SensorType};
use ruv_neural_core::signal::MultiChannelTimeSeries;

/// Metadata needed to place a recording in a BIDS dataset.
#[derive(Debug, Clone)]
pub struct BidsMetadata {
    /// Subject label (BIDS `sub-<label>`, alphanumeric).
    pub subject: String,
    /// Optional session label (BIDS `ses-<label>`).
    pub session: Option<String>,
    /// Task label (BIDS `task-<label>`).
    pub task: String,
    /// Human-readable dataset name for `dataset_description.json`.
    pub dataset_name: String,
    /// Power-line frequency in Hz (50 or 60), required by the BIDS-EEG sidecar.
    pub power_line_hz: f64,
    /// EEG reference description (e.g. "average", "Cz").
    pub eeg_reference: String,
}

impl BidsMetadata {
    /// Minimal metadata with sensible defaults (60 Hz line, average reference).
    pub fn new(subject: &str, task: &str) -> Self {
        Self {
            subject: sanitize_label(subject),
            session: None,
            task: sanitize_label(task),
            dataset_name: "rUv Neural export".into(),
            power_line_hz: 60.0,
            eeg_reference: "n/a".into(),
        }
    }

    /// Set the session label.
    pub fn with_session(mut self, session: &str) -> Self {
        self.session = Some(sanitize_label(session));
        self
    }

    /// The BIDS filename stem, e.g. `sub-01_ses-1_task-rest`.
    fn stem(&self) -> String {
        match &self.session {
            Some(s) => format!("sub-{}_ses-{}_task-{}", self.subject, s, self.task),
            None => format!("sub-{}_task-{}", self.subject, self.task),
        }
    }
}

/// Keep only BIDS-legal label characters (alphanumeric).
fn sanitize_label(s: &str) -> String {
    let cleaned: String = s.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if cleaned.is_empty() {
        "x".into()
    } else {
        cleaned
    }
}

/// BIDS channel "type" string for one of our sensor technologies.
fn bids_channel_type(sensor: SensorType) -> &'static str {
    match sensor {
        SensorType::Eeg => "EEG",
        // OPM/SQUID/NV/atom-interferometer are magnetometry channels.
        SensorType::Opm | SensorType::SquidMeg | SensorType::NvDiamond => "MEGMAG",
        SensorType::AtomInterferometer => "MISC",
    }
}

/// Physical unit recorded for a sensor technology.
fn channel_units(sensor: SensorType) -> &'static str {
    match sensor {
        SensorType::Eeg => "µV",
        _ => "fT",
    }
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = std::fs::File::create(path)
        .map_err(|e| RuvNeuralError::Serialization(format!("create {}: {e}", path.display())))?;
    f.write_all(bytes)
        .map_err(|e| RuvNeuralError::Serialization(format!("write {}: {e}", path.display())))?;
    Ok(())
}

/// Export a recording as a minimal valid BIDS-EEG dataset rooted at `root`.
///
/// Returns the path to the `eeg/` directory that was written.
///
/// # Errors
/// Returns an error if the channel count disagrees with the signal, or on I/O.
pub fn export_bids_eeg(
    root: &str,
    signal: &MultiChannelTimeSeries,
    array: &SensorArray,
    meta: &BidsMetadata,
) -> Result<PathBuf> {
    if array.num_channels() != signal.num_channels {
        return Err(RuvNeuralError::DimensionMismatch {
            expected: signal.num_channels,
            got: array.num_channels(),
        });
    }
    if signal.num_channels == 0 || signal.num_samples == 0 {
        return Err(RuvNeuralError::Serialization(
            "BIDS export needs a non-empty recording".into(),
        ));
    }

    let root = Path::new(root);
    let eeg_dir = match &meta.session {
        Some(s) => root.join(format!("sub-{}", meta.subject)).join(format!("ses-{s}")),
        None => root.join(format!("sub-{}", meta.subject)),
    }
    .join("eeg");
    std::fs::create_dir_all(&eeg_dir)
        .map_err(|e| RuvNeuralError::Serialization(format!("mkdir {}: {e}", eeg_dir.display())))?;

    let stem = meta.stem();

    // ── dataset_description.json (BIDS root, required) ──────────────────
    let dd = serde_json::json!({
        "Name": meta.dataset_name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{ "Name": "ruv-neural-viz", "Description": "BIDS-EEG export (ADR-0016)" }],
    });
    write_file(
        &root.join("dataset_description.json"),
        serde_json::to_vec_pretty(&dd)
            .map_err(|e| RuvNeuralError::Serialization(e.to_string()))?
            .as_slice(),
    )?;

    // ── Binary data: IEEE float32, multiplexed (sample-major) ───────────
    let mut eeg_bytes = Vec::with_capacity(signal.num_channels * signal.num_samples * 4);
    for s in 0..signal.num_samples {
        for ch in 0..signal.num_channels {
            let v = signal.data[ch][s] as f32;
            eeg_bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let data_name = format!("{stem}_eeg.eeg");
    let mrk_name = format!("{stem}_eeg.vmrk");
    write_file(&eeg_dir.join(&data_name), &eeg_bytes)?;

    // ── BrainVision header (.vhdr) ──────────────────────────────────────
    let sampling_interval_us = 1.0e6 / signal.sample_rate_hz;
    let mut vhdr = String::new();
    vhdr.push_str("Brain Vision Data Exchange Header File Version 1.0\n");
    vhdr.push_str("; Written by ruv-neural-viz (ADR-0016 BIDS-EEG export)\n\n");
    vhdr.push_str("[Common Infos]\n");
    vhdr.push_str("Codepage=UTF-8\n");
    vhdr.push_str(&format!("DataFile={data_name}\n"));
    vhdr.push_str(&format!("MarkerFile={mrk_name}\n"));
    vhdr.push_str("DataFormat=BINARY\n");
    vhdr.push_str("DataOrientation=MULTIPLEXED\n");
    vhdr.push_str(&format!("NumberOfChannels={}\n", signal.num_channels));
    vhdr.push_str(&format!("SamplingInterval={sampling_interval_us}\n\n"));
    vhdr.push_str("[Binary Infos]\n");
    vhdr.push_str("BinaryFormat=IEEE_FLOAT_32\n\n");
    vhdr.push_str("[Channel Infos]\n");
    for (i, ch) in array.channels.iter().enumerate() {
        // Ch<n>=<name>,<ref>,<resolution>,<unit>
        let unit = channel_units(ch.sensor_type);
        vhdr.push_str(&format!("Ch{}={},,1,{}\n", i + 1, ch.label, unit));
    }
    write_file(&eeg_dir.join(format!("{stem}_eeg.vhdr")), vhdr.as_bytes())?;

    // ── BrainVision markers (.vmrk): a single "New Segment" at t0 ───────
    let mut vmrk = String::new();
    vmrk.push_str("Brain Vision Data Exchange Marker File, Version 1.0\n\n");
    vmrk.push_str("[Common Infos]\n");
    vmrk.push_str("Codepage=UTF-8\n");
    vmrk.push_str(&format!("DataFile={data_name}\n\n"));
    vmrk.push_str("[Marker Infos]\n");
    vmrk.push_str("Mk1=New Segment,,1,1,0,00000000000000000000\n");
    write_file(&eeg_dir.join(&mrk_name), vmrk.as_bytes())?;

    // ── channels.tsv ────────────────────────────────────────────────────
    let mut tsv = String::from("name\ttype\tunits\tsampling_frequency\n");
    for ch in &array.channels {
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            ch.label,
            bids_channel_type(ch.sensor_type),
            channel_units(ch.sensor_type),
            signal.sample_rate_hz,
        ));
    }
    write_file(&eeg_dir.join(format!("{stem}_channels.tsv")), tsv.as_bytes())?;

    // ── BIDS sidecar (_eeg.json) ────────────────────────────────────────
    let eeg_count = array
        .channels
        .iter()
        .filter(|c| c.sensor_type == SensorType::Eeg)
        .count();
    let meg_count = array
        .channels
        .iter()
        .filter(|c| matches!(c.sensor_type, SensorType::Opm | SensorType::SquidMeg | SensorType::NvDiamond))
        .count();
    let sidecar = serde_json::json!({
        "TaskName": meta.task,
        "SamplingFrequency": signal.sample_rate_hz,
        "EEGReference": meta.eeg_reference,
        "PowerLineFrequency": meta.power_line_hz,
        "SoftwareFilters": "n/a",
        "RecordingDuration": signal.duration_s(),
        "RecordingType": "continuous",
        "EEGChannelCount": eeg_count,
        "MEGChannelCount": meg_count,
    });
    write_file(
        &eeg_dir.join(format!("{stem}_eeg.json")),
        serde_json::to_vec_pretty(&sidecar)
            .map_err(|e| RuvNeuralError::Serialization(e.to_string()))?
            .as_slice(),
    )?;

    Ok(eeg_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruv_neural_core::sensor::SensorChannel;

    fn array(n: usize) -> SensorArray {
        let channels = (0..n)
            .map(|i| SensorChannel {
                id: i,
                sensor_type: SensorType::Eeg,
                position: [i as f64 * 0.01, 0.0, 0.1],
                orientation: [0.0, 0.0, 1.0],
                sensitivity_ft_sqrt_hz: 1000.0,
                sample_rate_hz: 256.0,
                label: format!("EEG{:03}", i + 1),
            })
            .collect();
        SensorArray {
            channels,
            sensor_type: SensorType::Eeg,
            name: "test".into(),
        }
    }

    fn signal(n: usize, samples: usize) -> MultiChannelTimeSeries {
        let data: Vec<Vec<f64>> = (0..n)
            .map(|c| (0..samples).map(|s| ((s + c) as f64 * 0.1).sin()).collect())
            .collect();
        MultiChannelTimeSeries::new(data, 256.0, 0.0).unwrap()
    }

    #[test]
    fn writes_a_valid_bids_layout() {
        let dir = std::env::temp_dir().join(format!("bids_test_{}", std::process::id()));
        let root = dir.to_str().unwrap();
        let meta = BidsMetadata::new("01", "rest").with_session("1");
        let eeg_dir = export_bids_eeg(root, &signal(4, 256), &array(4), &meta).unwrap();

        // Required files exist at the expected BIDS paths.
        assert!(dir.join("dataset_description.json").is_file());
        assert!(eeg_dir.join("sub-01_ses-1_task-rest_eeg.vhdr").is_file());
        assert!(eeg_dir.join("sub-01_ses-1_task-rest_eeg.vmrk").is_file());
        assert!(eeg_dir.join("sub-01_ses-1_task-rest_eeg.eeg").is_file());
        assert!(eeg_dir.join("sub-01_ses-1_task-rest_channels.tsv").is_file());
        assert!(eeg_dir.join("sub-01_ses-1_task-rest_eeg.json").is_file());

        // Binary size = channels * samples * 4 bytes (float32, multiplexed).
        let eeg = std::fs::read(eeg_dir.join("sub-01_ses-1_task-rest_eeg.eeg")).unwrap();
        assert_eq!(eeg.len(), 4 * 256 * 4);

        // Header declares the right channel count and float format.
        let vhdr = std::fs::read_to_string(eeg_dir.join("sub-01_ses-1_task-rest_eeg.vhdr")).unwrap();
        assert!(vhdr.contains("NumberOfChannels=4"));
        assert!(vhdr.contains("BinaryFormat=IEEE_FLOAT_32"));
        assert!(vhdr.contains("SamplingInterval=3906.25")); // 1e6/256

        // channels.tsv has a header + one row per channel.
        let tsv = std::fs::read_to_string(eeg_dir.join("sub-01_ses-1_task-rest_channels.tsv")).unwrap();
        assert_eq!(tsv.lines().count(), 5);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn round_trips_sample_values() {
        let dir = std::env::temp_dir().join(format!("bids_rt_{}", std::process::id()));
        let root = dir.to_str().unwrap();
        let sig = signal(2, 8);
        let meta = BidsMetadata::new("02", "task");
        let eeg_dir = export_bids_eeg(root, &sig, &array(2), &meta).unwrap();

        // Read back the multiplexed float32 and compare to the source.
        let bytes = std::fs::read(eeg_dir.join("sub-02_task-task_eeg.eeg")).unwrap();
        for s in 0..sig.num_samples {
            for ch in 0..sig.num_channels {
                let off = (s * sig.num_channels + ch) * 4;
                let v = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
                assert!((v as f64 - sig.data[ch][s]).abs() < 1e-6);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn channel_count_mismatch_errors() {
        let meta = BidsMetadata::new("03", "rest");
        let err = export_bids_eeg("/tmp/should_not_exist_bids", &signal(3, 16), &array(2), &meta);
        assert!(err.is_err());
    }

    #[test]
    fn labels_are_sanitized() {
        let meta = BidsMetadata::new("sub 01!", "rest_state");
        assert_eq!(meta.subject, "sub01");
        assert_eq!(meta.task, "reststate");
    }
}
