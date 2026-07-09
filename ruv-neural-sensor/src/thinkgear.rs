//! NeuroSky ThinkGear Connector (TGC) socket source.
//!
//! TGC runs as a local background process and exposes ThinkGear headset data
//! over TCP, normally at `127.0.0.1:13854`. This module adapts its carriage
//! return-delimited JSON stream into the `SensorSource` trait.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::sensor::SensorType;
use ruv_neural_core::signal::MultiChannelTimeSeries;
use ruv_neural_core::traits::SensorSource;
use serde::{Deserialize, Serialize};

/// Default TGC host.
pub const DEFAULT_TGC_HOST: &str = "127.0.0.1";
/// Default TGC TCP port.
pub const DEFAULT_TGC_PORT: u16 = 13_854;
/// Raw EEG rate documented for TGC JSON streams.
pub const DEFAULT_RAW_EEG_SAMPLE_RATE_HZ: f64 = 512.0;

/// Configuration for a ThinkGear Connector source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkGearConfig {
    /// TGC host, usually localhost.
    pub host: String,
    /// TGC TCP port, usually 13854.
    pub port: u16,
    /// Human-readable app name sent to TGC authorization.
    pub app_name: String,
    /// 40-character hexadecimal application key.
    pub app_key: String,
    /// Whether to request raw EEG output.
    pub enable_raw_output: bool,
    /// Expected raw EEG sample rate.
    pub sample_rate_hz: f64,
    /// Socket read timeout.
    pub read_timeout: Duration,
}

impl Default for ThinkGearConfig {
    fn default() -> Self {
        Self {
            host: DEFAULT_TGC_HOST.to_string(),
            port: DEFAULT_TGC_PORT,
            app_name: "ruv-neural".to_string(),
            app_key: "0000000000000000000000000000000000000000".to_string(),
            enable_raw_output: true,
            sample_rate_hz: DEFAULT_RAW_EEG_SAMPLE_RATE_HZ,
            read_timeout: Duration::from_secs(5),
        }
    }
}

/// Parsed status values from the latest TGC stream packets.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThinkGearStatus {
    /// Signal quality: 0 is good, 200 usually means off-head.
    pub poor_signal_level: Option<u16>,
    /// eSense attention, 0-100.
    pub attention: Option<u8>,
    /// eSense meditation, 0-100.
    pub meditation: Option<u8>,
    /// Last blink strength, 0-255.
    pub blink_strength: Option<u8>,
    /// Latest EEG band powers.
    pub eeg_power: Option<ThinkGearEegPower>,
}

/// TGC EEG power packet.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkGearEegPower {
    /// Delta band power.
    pub delta: Option<f64>,
    /// Theta band power.
    pub theta: Option<f64>,
    /// Low-alpha band power.
    pub low_alpha: Option<f64>,
    /// High-alpha band power.
    pub high_alpha: Option<f64>,
    /// Low-beta band power.
    pub low_beta: Option<f64>,
    /// High-beta band power.
    pub high_beta: Option<f64>,
    /// Low-gamma band power.
    pub low_gamma: Option<f64>,
    /// High-gamma band power.
    pub high_gamma: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ThinkGearPacket {
    raw_eeg: Option<f64>,
    raw_eeg_multi: Option<RawEegMulti>,
    poor_signal_level: Option<u16>,
    #[serde(default)]
    e_sense: Option<ESense>,
    eeg_power: Option<ThinkGearEegPower>,
    blink_strength: Option<u8>,
    is_authorized: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ESense {
    attention: Option<u8>,
    meditation: Option<u8>,
}

#[derive(Debug, Deserialize)]
struct RawEegMulti {
    ch1: Option<f64>,
    ch2: Option<f64>,
    ch3: Option<f64>,
    ch4: Option<f64>,
    ch5: Option<f64>,
    ch6: Option<f64>,
    ch7: Option<f64>,
    ch8: Option<f64>,
}

impl RawEegMulti {
    fn channels(&self) -> Vec<f64> {
        [
            self.ch1, self.ch2, self.ch3, self.ch4, self.ch5, self.ch6, self.ch7, self.ch8,
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

/// Blocking ThinkGear Connector EEG source.
#[derive(Debug)]
pub struct ThinkGearSource {
    config: ThinkGearConfig,
    stream: TcpStream,
    line_buffer: Vec<u8>,
    pending_channels: Vec<VecDeque<f64>>,
    timestamp_start: f64,
    samples_read: usize,
    status: ThinkGearStatus,
}

impl ThinkGearSource {
    /// Connect to TGC, authorize the application, and request JSON output.
    pub fn connect(config: ThinkGearConfig) -> Result<Self> {
        validate_config(&config)?;
        let addr = (config.host.as_str(), config.port)
            .to_socket_addrs()
            .map_err(|e| RuvNeuralError::Sensor(format!("invalid TGC address: {e}")))?
            .next()
            .ok_or_else(|| RuvNeuralError::Sensor("TGC address resolved to no endpoints".into()))?;

        let mut stream = TcpStream::connect_timeout(&addr, config.read_timeout)
            .map_err(|e| RuvNeuralError::Sensor(format!("failed to connect to TGC: {e}")))?;
        stream
            .set_read_timeout(Some(config.read_timeout))
            .map_err(|e| RuvNeuralError::Sensor(format!("failed to configure TGC socket: {e}")))?;

        write_json_line(
            &mut stream,
            &serde_json::json!({
                "appName": config.app_name,
                "appKey": config.app_key,
            }),
        )?;
        write_json_line(
            &mut stream,
            &serde_json::json!({
                "enableRawOutput": config.enable_raw_output,
                "format": "Json",
            }),
        )?;

        Ok(Self {
            config,
            stream,
            line_buffer: Vec::with_capacity(512),
            pending_channels: vec![VecDeque::new()],
            timestamp_start: unix_time_seconds(),
            samples_read: 0,
            status: ThinkGearStatus::default(),
        })
    }

    /// Returns the latest auxiliary TGC status values.
    pub fn status(&self) -> &ThinkGearStatus {
        &self.status
    }

    fn read_until_packet(&mut self) -> Result<ThinkGearPacket> {
        loop {
            let mut byte = [0_u8; 1];
            let read = self
                .stream
                .read(&mut byte)
                .map_err(|e| RuvNeuralError::Sensor(format!("failed to read TGC stream: {e}")))?;
            if read == 0 {
                return Err(RuvNeuralError::Sensor("TGC stream closed".into()));
            }

            if byte[0] == b'\r' || byte[0] == b'\n' {
                if self.line_buffer.is_empty() {
                    continue;
                }
                let line = std::mem::take(&mut self.line_buffer);
                match serde_json::from_slice::<ThinkGearPacket>(&line) {
                    Ok(packet) => return Ok(packet),
                    Err(_) => {
                        // TGC may emit a few binary packets before JSON mode starts.
                        continue;
                    }
                }
            } else {
                self.line_buffer.push(byte[0]);
                if self.line_buffer.len() > 16 * 1024 {
                    self.line_buffer.clear();
                    return Err(RuvNeuralError::Sensor(
                        "discarded oversized TGC packet".into(),
                    ));
                }
            }
        }
    }

    fn ingest_packet(&mut self, packet: ThinkGearPacket) -> Result<()> {
        if packet.is_authorized == Some(false) {
            return Err(RuvNeuralError::Sensor(
                "TGC rejected application authorization".into(),
            ));
        }
        if let Some(level) = packet.poor_signal_level {
            self.status.poor_signal_level = Some(level);
        }
        if let Some(e_sense) = packet.e_sense {
            if let Some(attention) = e_sense.attention {
                self.status.attention = Some(attention);
            }
            if let Some(meditation) = e_sense.meditation {
                self.status.meditation = Some(meditation);
            }
        }
        if let Some(power) = packet.eeg_power {
            self.status.eeg_power = Some(power);
        }
        if let Some(blink) = packet.blink_strength {
            self.status.blink_strength = Some(blink);
        }

        if let Some(multi) = packet.raw_eeg_multi {
            let channels = multi.channels();
            if !channels.is_empty() && channels.iter().all(|v| v.is_finite()) {
                self.ensure_channels(channels.len());
                for (idx, value) in channels.into_iter().enumerate() {
                    self.pending_channels[idx].push_back(value);
                }
            }
        } else if let Some(raw) = packet.raw_eeg {
            if raw.is_finite() {
                self.ensure_channels(1);
                self.pending_channels[0].push_back(raw);
            }
        }
        Ok(())
    }

    fn ensure_channels(&mut self, channels: usize) {
        if self.pending_channels.len() != channels {
            self.pending_channels = (0..channels).map(|_| VecDeque::new()).collect();
        }
    }
}

impl SensorSource for ThinkGearSource {
    fn sensor_type(&self) -> SensorType {
        SensorType::Eeg
    }

    fn num_channels(&self) -> usize {
        self.pending_channels.len().max(1)
    }

    fn sample_rate_hz(&self) -> f64 {
        self.config.sample_rate_hz
    }

    fn read_chunk(&mut self, num_samples: usize) -> Result<MultiChannelTimeSeries> {
        if num_samples == 0 {
            return Err(RuvNeuralError::Sensor(
                "ThinkGear chunk size must be positive".into(),
            ));
        }

        while self
            .pending_channels
            .iter()
            .any(|channel| channel.len() < num_samples)
        {
            let packet = self.read_until_packet()?;
            self.ingest_packet(packet)?;
        }

        let timestamp = self.timestamp_start
            + self.samples_read as f64 / self.config.sample_rate_hz;
        let data = self
            .pending_channels
            .iter_mut()
            .map(|channel| channel.drain(..num_samples).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        self.samples_read += num_samples;
        MultiChannelTimeSeries::new(data, self.config.sample_rate_hz, timestamp)
    }
}

fn validate_config(config: &ThinkGearConfig) -> Result<()> {
    let key_ok = config.app_key.len() == 40
        && config
            .app_key
            .bytes()
            .all(|b| b.is_ascii_hexdigit());
    if !key_ok {
        return Err(RuvNeuralError::Sensor(
            "ThinkGear app_key must be 40 hexadecimal characters".into(),
        ));
    }
    if !config.sample_rate_hz.is_finite() || config.sample_rate_hz <= 0.0 {
        return Err(RuvNeuralError::Sensor(
            "ThinkGear sample_rate_hz must be finite and positive".into(),
        ));
    }
    Ok(())
}

fn write_json_line(stream: &mut TcpStream, value: &serde_json::Value) -> Result<()> {
    let line = serde_json::to_vec(value).map_err(|e| {
        RuvNeuralError::Serialization(format!("failed to encode TGC command: {e}"))
    })?;
    stream
        .write_all(&line)
        .and_then(|_| stream.write_all(b"\r"))
        .map_err(|e| RuvNeuralError::Sensor(format!("failed to write TGC command: {e}")))
}

fn unix_time_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_invalid_app_key() {
        let config = ThinkGearConfig {
            app_key: "not-hex".to_string(),
            ..ThinkGearConfig::default()
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn config_accepts_default_app_key_shape() {
        assert!(validate_config(&ThinkGearConfig::default()).is_ok());
    }

    #[test]
    fn raw_multi_preserves_channel_order() {
        let multi = RawEegMulti {
            ch1: Some(1.0),
            ch2: Some(2.0),
            ch3: None,
            ch4: Some(4.0),
            ch5: None,
            ch6: None,
            ch7: None,
            ch8: Some(8.0),
        };
        assert_eq!(multi.channels(), vec![1.0, 2.0, 4.0, 8.0]);
    }
}
