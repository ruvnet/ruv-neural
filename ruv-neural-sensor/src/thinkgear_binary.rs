//! Direct ThinkGear binary protocol parser and stream source.
//!
//! This is the Rust equivalent of the useful part of `python-mindwave`: it
//! consumes bytes from a serial/RFCOMM stream, reconstructs ThinkGear packets,
//! and exposes raw EEG through `SensorSource`.

use std::collections::VecDeque;
use std::io::Read;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::sensor::SensorType;
use ruv_neural_core::signal::MultiChannelTimeSeries;
use ruv_neural_core::traits::SensorSource;
use serde::{Deserialize, Serialize};

/// Default raw EEG sample rate for MindWave Mobile raw wave output.
pub const DEFAULT_BINARY_SAMPLE_RATE_HZ: f64 = 512.0;
/// Common ThinkGear serial baud rate.
pub const DEFAULT_BAUD_RATE: u32 = 57_600;

const SYNC: u8 = 0xAA;
const MAX_PAYLOAD_LEN: usize = 169;
const CODE_POOR_SIGNAL: u8 = 0x02;
const CODE_ATTENTION: u8 = 0x04;
const CODE_MEDITATION: u8 = 0x05;
const CODE_BLINK: u8 = 0x16;
const CODE_RAW_WAVE: u8 = 0x80;
const CODE_ASIC_EEG_POWER: u8 = 0x83;

/// Parsed binary ThinkGear packet.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThinkGearBinaryPacket {
    /// Raw EEG wave sample.
    pub raw_eeg: Option<i16>,
    /// Poor-signal quality value. 0 is good, 200 usually means off-head.
    pub poor_signal: Option<u8>,
    /// eSense attention value.
    pub attention: Option<u8>,
    /// eSense meditation value.
    pub meditation: Option<u8>,
    /// Blink strength.
    pub blink_strength: Option<u8>,
    /// EEG band powers.
    pub eeg_power: Option<ThinkGearBinaryEegPower>,
}

/// Eight ThinkGear ASIC EEG band powers.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ThinkGearBinaryEegPower {
    /// Delta band power.
    pub delta: u32,
    /// Theta band power.
    pub theta: u32,
    /// Low-alpha band power.
    pub low_alpha: u32,
    /// High-alpha band power.
    pub high_alpha: u32,
    /// Low-beta band power.
    pub low_beta: u32,
    /// High-beta band power.
    pub high_beta: u32,
    /// Low-gamma band power.
    pub low_gamma: u32,
    /// Mid/high-gamma band power.
    pub high_gamma: u32,
}

#[derive(Debug, Clone, Copy)]
enum ParserState {
    Sync1,
    Sync2,
    Length,
    Payload { len: usize },
    Checksum,
}

/// Stateful parser for ThinkGear serial-stream packets.
#[derive(Debug, Clone)]
pub struct ThinkGearBinaryParser {
    state: ParserState,
    payload: Vec<u8>,
}

impl Default for ThinkGearBinaryParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ThinkGearBinaryParser {
    /// Create a new parser.
    pub fn new() -> Self {
        Self {
            state: ParserState::Sync1,
            payload: Vec::with_capacity(MAX_PAYLOAD_LEN),
        }
    }

    /// Feed one byte. Returns a parsed packet when a valid packet completes.
    pub fn feed(&mut self, byte: u8) -> Result<Option<ThinkGearBinaryPacket>> {
        match self.state {
            ParserState::Sync1 => {
                if byte == SYNC {
                    self.state = ParserState::Sync2;
                }
                Ok(None)
            }
            ParserState::Sync2 => {
                self.state = if byte == SYNC {
                    ParserState::Length
                } else {
                    ParserState::Sync1
                };
                Ok(None)
            }
            ParserState::Length => {
                let len = byte as usize;
                if len > MAX_PAYLOAD_LEN {
                    self.reset();
                    return Err(RuvNeuralError::Sensor(format!(
                        "ThinkGear payload length {len} exceeds {MAX_PAYLOAD_LEN}"
                    )));
                }
                self.payload.clear();
                self.state = ParserState::Payload { len };
                Ok(None)
            }
            ParserState::Payload { len } => {
                self.payload.push(byte);
                if self.payload.len() == len {
                    self.state = ParserState::Checksum;
                }
                Ok(None)
            }
            ParserState::Checksum => {
                let valid = checksum(&self.payload) == byte;
                let payload = self.payload.clone();
                self.reset();
                if !valid {
                    return Err(RuvNeuralError::Sensor(
                        "ThinkGear packet checksum mismatch".into(),
                    ));
                }
                parse_payload(&payload).map(Some)
            }
        }
    }

    fn reset(&mut self) {
        self.state = ParserState::Sync1;
        self.payload.clear();
    }
}

/// Source for direct binary ThinkGear streams.
pub struct ThinkGearBinarySource {
    reader: Box<dyn Read + Send>,
    parser: ThinkGearBinaryParser,
    pending: VecDeque<f64>,
    sample_rate_hz: f64,
    timestamp_start: f64,
    samples_read: usize,
    status: ThinkGearBinaryPacket,
}

impl ThinkGearBinarySource {
    /// Create a source from any byte reader.
    pub fn from_reader(reader: Box<dyn Read + Send>, sample_rate_hz: f64) -> Result<Self> {
        if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
            return Err(RuvNeuralError::Sensor(
                "ThinkGear binary sample_rate_hz must be finite and positive".into(),
            ));
        }
        Ok(Self {
            reader,
            parser: ThinkGearBinaryParser::new(),
            pending: VecDeque::new(),
            sample_rate_hz,
            timestamp_start: unix_time_seconds(),
            samples_read: 0,
            status: ThinkGearBinaryPacket::default(),
        })
    }

    /// Open a serial/RFCOMM port for a MindWave-style binary stream.
    pub fn open_serial(port_name: &str, baud_rate: u32, sample_rate_hz: f64) -> Result<Self> {
        let port = serialport::new(port_name, baud_rate)
            .timeout(Duration::from_secs(5))
            .open()
            .map_err(|e| {
                RuvNeuralError::Sensor(format!("failed to open ThinkGear serial port: {e}"))
            })?;
        Self::from_reader(port, sample_rate_hz)
    }

    /// Latest low-rate status values observed in the stream.
    pub fn status(&self) -> &ThinkGearBinaryPacket {
        &self.status
    }

    fn read_next_packet(&mut self) -> Result<ThinkGearBinaryPacket> {
        loop {
            let mut byte = [0_u8; 1];
            self.reader
                .read_exact(&mut byte)
                .map_err(|e| RuvNeuralError::Sensor(format!("failed to read ThinkGear byte: {e}")))?;
            match self.parser.feed(byte[0]) {
                Ok(Some(packet)) => return Ok(packet),
                Ok(None) => {}
                Err(_) => {
                    // Bad packets happen on noisy serial links. Resync and keep reading.
                }
            }
        }
    }

    fn ingest_packet(&mut self, packet: ThinkGearBinaryPacket) {
        if let Some(raw) = packet.raw_eeg {
            self.pending.push_back(raw as f64);
        }
        if packet.poor_signal.is_some() {
            self.status.poor_signal = packet.poor_signal;
        }
        if packet.attention.is_some() {
            self.status.attention = packet.attention;
        }
        if packet.meditation.is_some() {
            self.status.meditation = packet.meditation;
        }
        if packet.blink_strength.is_some() {
            self.status.blink_strength = packet.blink_strength;
        }
        if packet.eeg_power.is_some() {
            self.status.eeg_power = packet.eeg_power;
        }
    }
}

impl SensorSource for ThinkGearBinarySource {
    fn sensor_type(&self) -> SensorType {
        SensorType::Eeg
    }

    fn num_channels(&self) -> usize {
        1
    }

    fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }

    fn read_chunk(&mut self, num_samples: usize) -> Result<MultiChannelTimeSeries> {
        if num_samples == 0 {
            return Err(RuvNeuralError::Sensor(
                "ThinkGear binary chunk size must be positive".into(),
            ));
        }
        while self.pending.len() < num_samples {
            let packet = self.read_next_packet()?;
            self.ingest_packet(packet);
        }

        let timestamp = self.timestamp_start + self.samples_read as f64 / self.sample_rate_hz;
        let channel = self.pending.drain(..num_samples).collect::<Vec<_>>();
        self.samples_read += num_samples;
        MultiChannelTimeSeries::new(vec![channel], self.sample_rate_hz, timestamp)
    }
}

fn checksum(payload: &[u8]) -> u8 {
    let sum = payload
        .iter()
        .fold(0_u8, |acc, byte| acc.wrapping_add(*byte));
    !sum
}

fn parse_payload(payload: &[u8]) -> Result<ThinkGearBinaryPacket> {
    let mut packet = ThinkGearBinaryPacket::default();
    let mut idx = 0;
    while idx < payload.len() {
        let code = payload[idx];
        idx += 1;
        if code >= 0x80 {
            if idx >= payload.len() {
                return Err(RuvNeuralError::Sensor(
                    "ThinkGear extended code missing value length".into(),
                ));
            }
            let len = payload[idx] as usize;
            idx += 1;
            if idx + len > payload.len() {
                return Err(RuvNeuralError::Sensor(
                    "ThinkGear extended value exceeds payload".into(),
                ));
            }
            parse_extended_value(code, &payload[idx..idx + len], &mut packet)?;
            idx += len;
        } else {
            if idx >= payload.len() {
                return Err(RuvNeuralError::Sensor(
                    "ThinkGear one-byte code missing value".into(),
                ));
            }
            parse_single_value(code, payload[idx], &mut packet);
            idx += 1;
        }
    }
    Ok(packet)
}

fn parse_single_value(code: u8, value: u8, packet: &mut ThinkGearBinaryPacket) {
    match code {
        CODE_POOR_SIGNAL => packet.poor_signal = Some(value),
        CODE_ATTENTION => packet.attention = Some(value),
        CODE_MEDITATION => packet.meditation = Some(value),
        CODE_BLINK => packet.blink_strength = Some(value),
        _ => {}
    }
}

fn parse_extended_value(
    code: u8,
    value: &[u8],
    packet: &mut ThinkGearBinaryPacket,
) -> Result<()> {
    match code {
        CODE_RAW_WAVE if value.len() == 2 => {
            packet.raw_eeg = Some(i16::from_be_bytes([value[0], value[1]]));
        }
        CODE_ASIC_EEG_POWER if value.len() == 24 => {
            let mut bands = [0_u32; 8];
            for (idx, chunk) in value.chunks_exact(3).enumerate() {
                bands[idx] =
                    ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | chunk[2] as u32;
            }
            packet.eeg_power = Some(ThinkGearBinaryEegPower {
                delta: bands[0],
                theta: bands[1],
                low_alpha: bands[2],
                high_alpha: bands[3],
                low_beta: bands[4],
                high_beta: bands[5],
                low_gamma: bands[6],
                high_gamma: bands[7],
            });
        }
        CODE_RAW_WAVE | CODE_ASIC_EEG_POWER => {
            return Err(RuvNeuralError::Sensor(format!(
                "ThinkGear code 0x{code:02x} has invalid length {}",
                value.len()
            )));
        }
        _ => {}
    }
    Ok(())
}

fn unix_time_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn framed(payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![SYNC, SYNC, payload.len() as u8];
        bytes.extend_from_slice(payload);
        bytes.push(checksum(payload));
        bytes
    }

    #[test]
    fn parser_reads_raw_attention_and_meditation() {
        let payload = [
            CODE_RAW_WAVE,
            0x02,
            0x01,
            0x23,
            CODE_ATTENTION,
            57,
            CODE_MEDITATION,
            42,
        ];
        let mut parser = ThinkGearBinaryParser::new();
        let mut parsed = None;
        for byte in framed(&payload) {
            parsed = parser.feed(byte).unwrap().or(parsed);
        }
        let packet = parsed.unwrap();
        assert_eq!(packet.raw_eeg, Some(0x0123));
        assert_eq!(packet.attention, Some(57));
        assert_eq!(packet.meditation, Some(42));
    }

    #[test]
    fn parser_rejects_bad_checksum() {
        let mut bytes = framed(&[CODE_ATTENTION, 50]);
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let mut parser = ThinkGearBinaryParser::new();
        let mut saw_error = false;
        for byte in bytes {
            if parser.feed(byte).is_err() {
                saw_error = true;
            }
        }
        assert!(saw_error);
    }

    #[test]
    fn source_reads_chunk_from_byte_stream() {
        let mut bytes = Vec::new();
        bytes.extend(framed(&[CODE_RAW_WAVE, 0x02, 0x00, 0x01]));
        bytes.extend(framed(&[CODE_RAW_WAVE, 0x02, 0x00, 0x02]));
        bytes.extend(framed(&[CODE_RAW_WAVE, 0x02, 0xff, 0xff]));

        let reader = Box::new(Cursor::new(bytes));
        let mut source = ThinkGearBinarySource::from_reader(reader, 512.0).unwrap();
        let chunk = source.read_chunk(3).unwrap();
        assert_eq!(chunk.num_channels, 1);
        assert_eq!(chunk.data[0], vec![1.0, 2.0, -1.0]);
    }
}
