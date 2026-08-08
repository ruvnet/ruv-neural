/**
 * Long-range temporal correlations via detrended fluctuation analysis (DFA-1).
 *
 * TypeScript reference implementation of `ruv-neural-signal/src/lrtc.rs`,
 * mirrored operation-for-operation so both produce bit-identical IEEE-754
 * results on the same input (verified by the shared parity fixture in
 * `dfa.test.ts`). The WASM export `compute_dfa` from `ruv-neural-wasm` wraps
 * the Rust implementation; this module is the dependency-free fallback and
 * the parity oracle.
 *
 * Interpretation of alpha:
 * - ~0.5  white (uncorrelated) noise
 * - 0.5-1 persistent long-range correlations (healthy resting EEG envelopes)
 * - ~1.0  1/f (pink) noise
 * - ~1.5  Brownian motion
 */

export interface DfaConfig {
  /** Smallest window scale in samples (>= 4). */
  minScale: number;
  /** Largest window scale in samples; null selects floor(n / 4). */
  maxScale: number | null;
  /** Number of logarithmically spaced scales (>= 2). */
  numScales: number;
}

export const DEFAULT_DFA_CONFIG: DfaConfig = {
  minScale: 4,
  maxScale: null,
  numScales: 16,
};

export interface DfaResult {
  /** DFA scaling exponent (slope of log2 F(s) vs log2 s). */
  alpha: number;
  /** Hurst exponent: alpha for alpha <= 1, alpha - 1 otherwise. */
  hurst: number;
  /** Coefficient of determination of the log-log fit. */
  rSquared: number;
  /** Window scales used, in samples. */
  scales: number[];
  /** RMS fluctuation F(s) at each scale. */
  fluctuations: number[];
}

/** Logarithmically spaced integer scales in [min, max], deduplicated. */
function logSpacedScales(min: number, max: number, count: number): number[] {
  const logMin = Math.log(min);
  const logMax = Math.log(max);
  const scales: number[] = [];
  for (let i = 0; i < count; i++) {
    const t = i / (count - 1);
    let s = Math.round(Math.exp(logMin + t * (logMax - logMin)));
    s = Math.min(Math.max(s, min), max);
    if (scales[scales.length - 1] !== s) {
      scales.push(s);
    }
  }
  return scales;
}

/**
 * RMS fluctuation F(s) over non-overlapping windows of length `s`, taken from
 * both the start and the end of the profile, with closed-form linear
 * detrending (same accumulation order as the Rust implementation).
 */
function fluctuationAtScale(profile: Float64Array, s: number): number {
  const n = profile.length;
  const windows = Math.floor(n / s);

  const sf = s;
  const st = (sf * (sf - 1)) / 2;
  const stt = ((sf - 1) * sf * (2 * sf - 1)) / 6;
  const denom = sf * stt - st * st;

  const windowRss = (start: number): number => {
    let sy = 0;
    let sty = 0;
    let syy = 0;
    for (let t = 0; t < s; t++) {
      const y = profile[start + t];
      sy += y;
      sty += t * y;
      syy += y * y;
    }
    const b = (sf * sty - st * sy) / denom;
    const a = (sy - b * st) / sf;
    return Math.max(syy - a * sy - b * sty, 0);
  };

  let total = 0;
  for (let k = 0; k < windows; k++) {
    total += windowRss(k * s);
  }
  const tailOffset = n - windows * s;
  if (tailOffset > 0) {
    for (let k = 0; k < windows; k++) {
      total += windowRss(tailOffset + k * s);
    }
    return Math.sqrt(total / (2 * windows) / sf);
  }
  return Math.sqrt(total / windows / sf);
}

/** Least-squares slope and R² of y against x. */
function linearFit(x: number[], y: number[]): [number, number] | null {
  const n = x.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += x[i];
    my += y[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  if (sxx <= 0) {
    return null;
  }
  const slope = sxy / sxx;
  const rSquared = syy > 0 ? (sxy * sxy) / (sxx * syy) : 1;
  return [slope, rSquared];
}

/**
 * Compute the DFA scaling exponent of a time series (DFA-1).
 *
 * Returns null when the input cannot support the analysis (mirrors the Rust
 * implementation's `None` cases).
 */
export function dfa(
  signal: ArrayLike<number>,
  config: DfaConfig = DEFAULT_DFA_CONFIG,
): DfaResult | null {
  const n = signal.length;
  if (config.minScale < 4 || config.numScales < 2 || n < 4 * config.minScale) {
    return null;
  }
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(signal[i])) {
      return null;
    }
  }
  const maxScale = Math.min(config.maxScale ?? Math.floor(n / 4), Math.floor(n / 4));
  if (maxScale <= config.minScale) {
    return null;
  }

  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += signal[i];
  }
  const mean = sum / n;
  const profile = new Float64Array(n);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    acc += signal[i] - mean;
    profile[i] = acc;
  }

  const scales = logSpacedScales(config.minScale, maxScale, config.numScales);
  if (scales.length < 2) {
    return null;
  }

  const logS: number[] = [];
  const logF: number[] = [];
  const fluctuations: number[] = [];
  for (const s of scales) {
    const f = fluctuationAtScale(profile, s);
    if (!Number.isFinite(f) || f <= 0) {
      return null;
    }
    fluctuations.push(f);
    logS.push(Math.log2(s));
    logF.push(Math.log2(f));
  }

  const fit = linearFit(logS, logF);
  if (fit === null) {
    return null;
  }
  const [alpha, rSquared] = fit;
  return {
    alpha,
    hurst: alpha > 1 ? alpha - 1 : alpha,
    rSquared,
    scales,
    fluctuations,
  };
}
