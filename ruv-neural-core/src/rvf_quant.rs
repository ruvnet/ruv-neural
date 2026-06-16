//! Vector quantization codecs for the RVF `VEC` / `QUANT` segments.
//!
//! Mirrors the data-type codes RuVector uses for temperature-tiered vector
//! storage (`fp32` hot, `fp16` warm, `int8`/`binary` cold), plus an `f64`
//! lossless type for this workspace's double-precision embeddings. All codecs
//! are pure Rust with **zero external dependencies** so they build on `no_std`
//! edge targets and `wasm32` alike.
//!
//! | dtype    | code | bytes / dim | notes                              |
//! |----------|------|-------------|------------------------------------|
//! | `F32`    | 0    | 4           | single precision                   |
//! | `F16`    | 1    | 2           | IEEE-754 half, ~3-4 decimal digits |
//! | `I8`     | 2    | 1 (+4/vec)  | symmetric per-vector scalar quant  |
//! | `Binary` | 3    | 1 bit       | sign quantization (Hamming space)  |
//! | `F64`    | 4    | 8           | lossless double precision (default)|

use crate::error::{Result, RuvNeuralError};

/// Vector element data type stored in a `VEC` segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VecDType {
    /// 32-bit IEEE-754 float.
    F32,
    /// 16-bit IEEE-754 half float.
    F16,
    /// 8-bit symmetric scalar quantization with a per-vector `f32` scale.
    I8,
    /// 1-bit sign quantization (binary / Hamming-space).
    Binary,
    /// 64-bit IEEE-754 double (lossless for this workspace's embeddings).
    F64,
}

impl VecDType {
    /// On-wire code, matching RuVector's vector data-type enumeration
    /// (with `F64` as a workspace extension).
    pub fn to_code(self) -> u8 {
        match self {
            VecDType::F32 => 0,
            VecDType::F16 => 1,
            VecDType::I8 => 2,
            VecDType::Binary => 3,
            VecDType::F64 => 4,
        }
    }

    /// Parse an on-wire code.
    pub fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(VecDType::F32),
            1 => Ok(VecDType::F16),
            2 => Ok(VecDType::I8),
            3 => Ok(VecDType::Binary),
            4 => Ok(VecDType::F64),
            other => Err(RuvNeuralError::Serialization(format!(
                "unknown VEC dtype code: {other}"
            ))),
        }
    }

    /// Lower-case name used in the JSON `META` segment.
    pub fn name(self) -> &'static str {
        match self {
            VecDType::F32 => "f32",
            VecDType::F16 => "f16",
            VecDType::I8 => "i8",
            VecDType::Binary => "binary",
            VecDType::F64 => "f64",
        }
    }

    /// Parse a name from the `META` segment.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "f32" => Ok(VecDType::F32),
            "f16" => Ok(VecDType::F16),
            "i8" => Ok(VecDType::I8),
            "binary" => Ok(VecDType::Binary),
            "f64" => Ok(VecDType::F64),
            other => Err(RuvNeuralError::Serialization(format!(
                "unknown VEC dtype name: {other}"
            ))),
        }
    }

    /// Encoded byte length of a single `dim`-element vector under this dtype.
    pub fn encoded_len(self, dim: usize) -> usize {
        match self {
            VecDType::F64 => dim * 8,
            VecDType::F32 => dim * 4,
            VecDType::F16 => dim * 2,
            // 4-byte scale prefix + one byte per element.
            VecDType::I8 => 4 + dim,
            VecDType::Binary => dim.div_ceil(8),
        }
    }

    /// Whether this dtype reconstructs the input exactly.
    pub fn is_lossless(self) -> bool {
        matches!(self, VecDType::F64)
    }
}

// ── IEEE-754 half precision (f16) ───────────────────────────────────────

/// Convert an `f32` to IEEE-754 half precision (round-to-nearest-even).
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf / NaN: preserve a non-zero mantissa so NaN stays NaN.
        let m = if mant != 0 { 0x0200 } else { 0 };
        return sign | 0x7c00 | m;
    }

    // Unbias from f32 (127) and rebias to f16 (15).
    let new_exp = exp - 127 + 15;

    if new_exp >= 0x1f {
        // Overflow → signed infinity.
        return sign | 0x7c00;
    }

    if new_exp <= 0 {
        // Subnormal or underflow to zero.
        if new_exp < -10 {
            return sign;
        }
        // Restore the implicit leading 1 then shift into subnormal position.
        let mant_full = mant | 0x0080_0000;
        let shift = (14 - new_exp) as u32;
        let half_mant = (mant_full >> shift) as u16;
        // Round-to-nearest-even using the bits shifted out.
        let round_bits = mant_full & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let rounded = if round_bits > halfway || (round_bits == halfway && (half_mant & 1) == 1) {
            half_mant + 1
        } else {
            half_mant
        };
        return sign | rounded;
    }

    // Normalized number.
    let half_exp = (new_exp as u16) << 10;
    let half_mant = (mant >> 13) as u16;
    let round_bits = mant & 0x1fff;
    let halfway = 0x1000;
    let base = sign | half_exp | half_mant;
    if round_bits > halfway || (round_bits == halfway && (half_mant & 1) == 1) {
        // Carry naturally rolls mantissa into exponent when it overflows.
        base + 1
    } else {
        base
    }
}

/// Convert an IEEE-754 half-precision value back to `f32`.
pub fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half & 0x8000) as u32) << 16;
    let exp = ((half >> 10) & 0x1f) as u32;
    let mant = (half & 0x03ff) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign // signed zero
        } else {
            // Subnormal: normalize.
            let mut e = -1i32;
            let mut m = mant;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            let new_exp = (e + 1 + 127 - 15) as u32;
            sign | (new_exp << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        // Inf / NaN.
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        let new_exp = exp + 127 - 15;
        sign | (new_exp << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// ── Symmetric int8 scalar quantization ──────────────────────────────────

/// Quantize a vector to symmetric int8 with a single per-vector scale.
///
/// Returns `(scale, codes)` where `x ≈ scale * code`. A zero vector yields a
/// scale of `1.0` and all-zero codes.
pub fn quantize_int8(values: &[f64]) -> (f32, Vec<i8>) {
    let max_abs = values.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if max_abs == 0.0 || !max_abs.is_finite() {
        return (1.0, vec![0; values.len()]);
    }
    let scale = (max_abs / 127.0) as f32;
    let scale_f64 = scale as f64;
    let codes = values
        .iter()
        .map(|&v| (v / scale_f64).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (scale, codes)
}

/// Reconstruct an int8-quantized vector.
pub fn dequantize_int8(scale: f32, codes: &[i8]) -> Vec<f64> {
    codes.iter().map(|&c| c as f64 * scale as f64).collect()
}

// ── Binary (sign) quantization ──────────────────────────────────────────

/// Sign-quantize a vector into packed bits (MSB-first, 1 = non-negative).
pub fn quantize_binary(values: &[f64]) -> Vec<u8> {
    let mut out = vec![0u8; values.len().div_ceil(8)];
    for (i, &v) in values.iter().enumerate() {
        if v >= 0.0 {
            out[i / 8] |= 0x80 >> (i % 8);
        }
    }
    out
}

/// Reconstruct a binary-quantized vector to `±1.0` for `dim` elements.
pub fn dequantize_binary(packed: &[u8], dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|i| {
            let bit = packed[i / 8] & (0x80 >> (i % 8));
            if bit != 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

// ── Encode / decode one vector under a chosen dtype ─────────────────────

/// Encode a single vector to bytes under `dtype`.
pub fn encode_vector(values: &[f64], dtype: VecDType) -> Vec<u8> {
    match dtype {
        VecDType::F64 => {
            let mut out = Vec::with_capacity(values.len() * 8);
            for &v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }
        VecDType::F32 => {
            let mut out = Vec::with_capacity(values.len() * 4);
            for &v in values {
                out.extend_from_slice(&(v as f32).to_le_bytes());
            }
            out
        }
        VecDType::F16 => {
            let mut out = Vec::with_capacity(values.len() * 2);
            for &v in values {
                out.extend_from_slice(&f32_to_f16(v as f32).to_le_bytes());
            }
            out
        }
        VecDType::I8 => {
            let (scale, codes) = quantize_int8(values);
            let mut out = Vec::with_capacity(4 + codes.len());
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend(codes.iter().map(|&c| c as u8));
            out
        }
        VecDType::Binary => quantize_binary(values),
    }
}

/// Decode a single `dim`-element vector from `bytes` under `dtype`.
pub fn decode_vector(bytes: &[u8], dim: usize, dtype: VecDType) -> Result<Vec<f64>> {
    let need = dtype.encoded_len(dim);
    if bytes.len() < need {
        return Err(RuvNeuralError::Serialization(format!(
            "VEC record too short: {} bytes, need {need}",
            bytes.len()
        )));
    }
    Ok(match dtype {
        VecDType::F64 => bytes[..dim * 8]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        VecDType::F32 => bytes[..dim * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        VecDType::F16 => bytes[..dim * 2]
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
            .collect(),
        VecDType::I8 => {
            let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let codes: Vec<i8> = bytes[4..4 + dim].iter().map(|&b| b as i8).collect();
            dequantize_int8(scale, &codes)
        }
        VecDType::Binary => dequantize_binary(&bytes[..dim.div_ceil(8)], dim),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_encodings() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(2.0), 0x4000);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn f16_roundtrip_error_bounded() {
        // Half precision keeps ~3 significant digits across a wide range.
        for &x in &[0.5f32, -0.5, 6.0625, -42.0, 100.25, 0.001, 1234.0] {
            let back = f16_to_f32(f32_to_f16(x));
            let rel = (back - x).abs() / x.abs().max(1e-6);
            assert!(rel < 1e-2, "f16 round-trip of {x} gave {back} (rel {rel})");
        }
    }

    #[test]
    fn f64_lossless_roundtrip() {
        let v: Vec<f64> = vec![1.0, -2.5, 1.2345678901234, 1e-12, -9.87654321e8];
        let enc = encode_vector(&v, VecDType::F64);
        let dec = decode_vector(&enc, v.len(), VecDType::F64).unwrap();
        assert_eq!(v, dec);
    }

    #[test]
    fn int8_roundtrip_error_bounded() {
        let v: Vec<f64> = (0..64).map(|i| (i as f64 - 32.0) * 0.5).collect();
        let (scale, codes) = quantize_int8(&v);
        let back = dequantize_int8(scale, &codes);
        for (a, b) in v.iter().zip(back.iter()) {
            // Error is at most half a quantization step.
            assert!((a - b).abs() <= scale as f64 * 0.5 + 1e-6);
        }
    }

    #[test]
    fn int8_zero_vector_is_safe() {
        let (scale, codes) = quantize_int8(&[0.0, 0.0, 0.0]);
        assert_eq!(scale, 1.0);
        assert_eq!(codes, vec![0, 0, 0]);
    }

    #[test]
    fn binary_preserves_sign() {
        let v = vec![1.0, -1.0, 0.5, -0.5, 0.0, -3.0, 2.0, -7.0, 9.0];
        let packed = quantize_binary(&v);
        let back = dequantize_binary(&packed, v.len());
        for (a, b) in v.iter().zip(back.iter()) {
            let expected = if *a >= 0.0 { 1.0 } else { -1.0 };
            assert_eq!(expected, *b);
        }
    }

    #[test]
    fn encode_decode_all_dtypes() {
        let v: Vec<f64> = (0..17).map(|i| (i as f64 - 8.0) * 0.25).collect();
        for dtype in [
            VecDType::F64,
            VecDType::F32,
            VecDType::F16,
            VecDType::I8,
            VecDType::Binary,
        ] {
            let enc = encode_vector(&v, dtype);
            assert_eq!(enc.len(), dtype.encoded_len(v.len()));
            let dec = decode_vector(&enc, v.len(), dtype).unwrap();
            assert_eq!(dec.len(), v.len());
        }
    }

    #[test]
    fn dtype_code_roundtrip() {
        for dt in [
            VecDType::F64,
            VecDType::F32,
            VecDType::F16,
            VecDType::I8,
            VecDType::Binary,
        ] {
            assert_eq!(VecDType::from_code(dt.to_code()).unwrap(), dt);
            assert_eq!(VecDType::from_name(dt.name()).unwrap(), dt);
        }
        assert!(VecDType::from_code(99).is_err());
    }
}
