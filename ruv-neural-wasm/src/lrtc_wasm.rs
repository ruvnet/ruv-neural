//! WASM bindings for long-range temporal correlation (DFA/Hurst) analysis
//! and the mental-privacy decode gate.
//!
//! Exposes `ruv_neural_signal::lrtc` (detrended fluctuation analysis) and
//! `ruv_neural_core::gate` (decode-armed / decode-locked "thought password"
//! state machine) to the browser.

use ruv_neural_core::gate::{DecodeGate, DecodeGateConfig};
use ruv_neural_signal::lrtc::{dfa, DfaConfig};
use wasm_bindgen::prelude::*;

/// Compute the DFA scaling exponent of a time series (DFA-1).
///
/// # Arguments
/// * `signal` - Input samples (Float64Array).
/// * `min_scale` - Smallest window scale in samples (>= 4). Pass 0 for the
///   default (4).
/// * `max_scale` - Largest window scale in samples. Pass 0 for the default
///   (signal length / 4).
/// * `num_scales` - Number of log-spaced scales (>= 2). Pass 0 for the
///   default (16).
///
/// # Returns
/// A JS object `{ alpha, hurst, rSquared, scales, fluctuations }`.
#[wasm_bindgen]
pub fn compute_dfa(
    signal: &[f64],
    min_scale: usize,
    max_scale: usize,
    num_scales: usize,
) -> Result<JsValue, JsError> {
    let defaults = DfaConfig::default();
    let config = DfaConfig {
        min_scale: if min_scale == 0 {
            defaults.min_scale
        } else {
            min_scale
        },
        max_scale: if max_scale == 0 {
            None
        } else {
            Some(max_scale)
        },
        num_scales: if num_scales == 0 {
            defaults.num_scales
        } else {
            num_scales
        },
    };
    let result = dfa(signal, &config).ok_or_else(|| {
        JsError::new(
            "DFA failed: signal too short, non-finite, (near-)constant, or invalid config",
        )
    })?;

    let out = js_sys::Object::new();
    js_sys::Reflect::set(&out, &"alpha".into(), &result.alpha.into())
        .map_err(|_| JsError::new("failed to build result object"))?;
    js_sys::Reflect::set(&out, &"hurst".into(), &result.hurst().into())
        .map_err(|_| JsError::new("failed to build result object"))?;
    js_sys::Reflect::set(&out, &"rSquared".into(), &result.r_squared.into())
        .map_err(|_| JsError::new("failed to build result object"))?;
    let scales: Vec<f64> = result.scales.iter().map(|&s| s as f64).collect();
    js_sys::Reflect::set(
        &out,
        &"scales".into(),
        &js_sys::Float64Array::from(scales.as_slice()).into(),
    )
    .map_err(|_| JsError::new("failed to build result object"))?;
    js_sys::Reflect::set(
        &out,
        &"fluctuations".into(),
        &js_sys::Float64Array::from(result.fluctuations.as_slice()).into(),
    )
    .map_err(|_| JsError::new("failed to build result object"))?;
    Ok(out.into())
}

/// Mental-privacy decode gate: decoded output stays locked until an imagined
/// password is detected with high confidence, and re-locks after a bounded
/// armed window (Kunz et al., Cell 2025 pattern; ADR-0007/0022).
#[wasm_bindgen]
pub struct WasmDecodeGate {
    inner: DecodeGate,
}

#[wasm_bindgen]
impl WasmDecodeGate {
    /// Create a gate. Pass `0` (or a negative timeout) for any parameter to
    /// use its default (threshold 0.98, 3 frames, 300 s timeout).
    #[wasm_bindgen(constructor)]
    pub fn new(
        arm_threshold: f64,
        arm_frames: u32,
        armed_timeout_s: f64,
    ) -> Result<WasmDecodeGate, JsError> {
        let defaults = DecodeGateConfig::default();
        let config = DecodeGateConfig {
            arm_threshold: if arm_threshold == 0.0 {
                defaults.arm_threshold
            } else {
                arm_threshold
            },
            arm_frames: if arm_frames == 0 {
                defaults.arm_frames
            } else {
                arm_frames
            },
            armed_timeout_s: if armed_timeout_s <= 0.0 {
                defaults.armed_timeout_s
            } else {
                armed_timeout_s
            },
        };
        let inner = DecodeGate::new(config).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmDecodeGate { inner })
    }

    /// Advance the gate by one decode frame; returns the state name
    /// (`"locked"`, `"arming"`, or `"armed"`).
    pub fn update(&mut self, password_confidence: f64, now_s: f64) -> String {
        self.inner.update(password_confidence, now_s).name().to_string()
    }

    /// Whether decoded output may currently be released.
    #[wasm_bindgen(js_name = isArmed)]
    pub fn is_armed(&self) -> bool {
        self.inner.is_armed()
    }

    /// Explicitly re-lock the gate; returns the state name.
    pub fn lock(&mut self, now_s: f64) -> String {
        self.inner.lock(now_s).name().to_string()
    }

    /// Pass decoded output through the gate: the string while armed,
    /// `undefined` while locked.
    pub fn release(&self, decoded: &str) -> Option<String> {
        self.inner.release(decoded).map(|s| s.to_string())
    }

    /// The bounded audit log of state transitions as a JS array.
    pub fn transitions(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(self.inner.transitions())
            .map_err(|e| JsError::new(&e.to_string()))
    }
}
