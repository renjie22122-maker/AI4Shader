/**
 * Reference-image scoring
 * =======================
 *
 * Pure functions over `Uint8ClampedArray` RGBA buffers (top-left origin, exactly what
 * `ImageData` and `WebGL2Renderer.captureFrame()` hand over). No DOM, no GPU: this file
 * is importable from a worker, from a test, and from `node:test`.
 *
 * Three signals are computed, and they answer three *different* questions:
 *
 *   SSIM       is the structure in the right place?          (shape / layout)
 *   直方图      is the tonal and colour distribution right?   (exposure / palette)
 *   频带        is the detail at the right frequency?         (grain / softness)
 *
 * The most valuable export is {@link attribute}: it turns "0.61, not good enough" into a
 * concrete `{node, param, direction}` hint (DESIGN.md §4.3). A score without an
 * attribution is a number nobody can act on.
 */

/* -------------------------------------------------------------------------- */
/* Types                                                                       */
/* -------------------------------------------------------------------------- */

export type Bands = { low: number; mid: number; high: number };

/**
 * Everything the attribution rules look at.
 *
 * The first four fields are the contract; the rest are optional signals that
 * {@link compareFrames} fills in and that make the attribution specific. A caller that
 * only has SSIM/histogram/bands can still call {@link attribute} — it will simply
 * produce fewer, coarser hints.
 */
export type ScoreBreakdown = {
  /** Global SSIM over luma, 8×8 windows. 1 = identical. */
  ssim: number;
  /** 1 − mean per-channel histogram intersection. 0 = identical. */
  hist: number;
  bands: Bands;
  bandsRef: Bands;
  /** SSIM computed on the 1/4-scale luma — isolates "wrong layout" from "wrong grain". */
  ssimLow?: number;
  /** Mean luma of each image, 0..1. */
  meanLuma?: { ref: number; cur: number };
  /** Mean HSV saturation of each image, 0..1. */
  saturation?: { ref: number; cur: number };
  /** RMS gradient magnitude (Sobel-free central difference) of each image, 0..1. */
  gradient?: { ref: number; cur: number };
  /** Circular mean of the hue histogram, 0..1 turns. */
  hueMean?: { ref: number; cur: number };
};

/** A tunable the attribution is allowed to point at. */
export type ParamRef = {
  node: string;
  param: string;
  label?: string;
  min?: number;
  max?: number;
  value?: number;
};

export type ParamRole =
  | 'warpStrength'
  | 'warpScale'
  | 'detailScale'
  | 'octaves'
  | 'lacunarity'
  | 'gain'
  | 'ridged'
  | 'contrast'
  | 'exposure'
  | 'saturation'
  | 'hue'
  | 'gamma'
  | 'speed';

export type AttributionHint = {
  node: string;
  param: string;
  direction: 'increase' | 'decrease';
  /** Why, in the user's language: names the measured quantity, not a vibe. */
  why: string;
  /** Optional concrete target value (midpoint of the remaining range in that direction). */
  suggest?: number;
};

export type Attribution = {
  /** One line the UI can show next to the score. */
  verdict: string;
  hints: AttributionHint[];
};

export type AttributionContext = {
  roles?: Partial<Record<ParamRole, ParamRef>>;
  /** SSIM between two frames of the *current* render at different times (animation check). */
  temporalSsim?: number;
};

/* -------------------------------------------------------------------------- */
/* Small helpers                                                               */
/* -------------------------------------------------------------------------- */

const LUMA_R = 0.2126;
const LUMA_G = 0.7152;
const LUMA_B = 0.0722;

/** Rec.709 luma, 0..255. */
function toLuma(a: Uint8ClampedArray, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = LUMA_R * (a[p] ?? 0) + LUMA_G * (a[p + 1] ?? 0) + LUMA_B * (a[p + 2] ?? 0);
  }
  return out;
}

function mean(x: Float32Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] ?? 0;
  return x.length ? s / x.length : 0;
}

/** RMS about the mean — the "energy" of a band, with the DC term removed. */
function rms(x: Float32Array): number {
  const m = mean(x);
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    const d = (x[i] ?? 0) - m;
    s += d * d;
  }
  return x.length ? Math.sqrt(s / x.length) : 0;
}

function rmsOfDifference(x: Float32Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += (x[i] ?? 0) * (x[i] ?? 0);
  return x.length ? Math.sqrt(s / x.length) : 0;
}

/** 2×2 box downsample. Odd dimensions drop the last row/column. */
function downsample(x: Float32Array, w: number, h: number): { data: Float32Array; w: number; h: number } {
  const nw = Math.max(1, w >> 1);
  const nh = Math.max(1, h >> 1);
  const out = new Float32Array(nw * nh);
  for (let y = 0; y < nh; y++) {
    for (let x2 = 0; x2 < nw; x2++) {
      const x0 = x2 * 2;
      const y0 = y * 2;
      const a = x[y0 * w + x0] ?? 0;
      const b = x0 + 1 < w ? x[y0 * w + x0 + 1] ?? 0 : a;
      const c = y0 + 1 < h ? x[(y0 + 1) * w + x0] ?? 0 : a;
      const d = x0 + 1 < w && y0 + 1 < h ? x[(y0 + 1) * w + x0 + 1] ?? 0 : a;
      out[y * nw + x2] = (a + b + c + d) * 0.25;
    }
  }
  return { data: out, w: nw, h: nh };
}

/**
 * Nearest-neighbour 2× upsample from `(sw,sh)` to `(w,h)`.
 *
 * **One doubling only**: `w` must be `2*sw` and `h` must be `2*sh`. The loop is driven by
 * `w`/`h`, so passing a source that is *smaller* than half the target does not throw — it
 * silently clamps three-quarters of the output to a single source pixel, which turns the
 * residual band that consumes it into a measure of global contrast instead of fine detail.
 * That is not hypothetical: it is the bug this function's callers had, and it went unnoticed
 * because the wrong number is still a plausible-looking number.
 *
 * The clamp below is therefore a guard rail rather than the mechanism: it keeps the indexing
 * in bounds when a dimension is odd (`(h-1) >> 1` can reach `sh`), where reading one row past
 * the end would otherwise yield `undefined` coerced to 0 by the `?? 0`.
 */
function upsample(x: Float32Array, sw: number, sh: number, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(sh - 1, y >> 1);
    for (let x2 = 0; x2 < w; x2++) {
      const sx = Math.min(sw - 1, x2 >> 1);
      out[y * w + x2] = x[sy * sw + sx] ?? 0;
    }
  }
  return out;
}

function difference(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] ?? 0) - (b[i] ?? 0);
  return out;
}

/* -------------------------------------------------------------------------- */
/* SSIM                                                                        */
/* -------------------------------------------------------------------------- */

/**
 * Global SSIM over the luma channel using **non-overlapping 8×8 windows**.
 *
 * Standard constants (Wang et al. 2004), with the dynamic range of 8-bit data:
 *
 *   C1 = (K1 · L)² = (0.01 · 255)² = 6.5025
 *   C2 = (K2 · L)² = (0.03 · 255)² = 58.5225
 *
 * C1 keeps the luminance term stable where both means are ~0, C2 does the same for the
 * contrast/structure term where both variances are ~0. Windows are clamped at the image
 * border rather than skipped, so a 250×250 buffer is fully covered.
 *
 * @returns mean SSIM in [-1, 1] (1 = identical, 0 = unrelated).
 */
export function ssim(a: Uint8ClampedArray, b: Uint8ClampedArray, w: number, h: number): number {
  if (w <= 0 || h <= 0) return 0;
  const la = toLuma(a, w, h);
  const lb = toLuma(b, w, h);
  return ssimLuma(la, lb, w, h);
}

/** SSIM on two pre-computed luma planes (used internally and by {@link compareFrames}). */
export function ssimLuma(la: Float32Array, lb: Float32Array, w: number, h: number): number {
  const L = 255;
  const C1 = (0.01 * L) ** 2;
  const C2 = (0.03 * L) ** 2;
  const win = 8;
  let total = 0;
  let count = 0;

  for (let y0 = 0; y0 < h; y0 += win) {
    const y1 = Math.min(h, y0 + win);
    for (let x0 = 0; x0 < w; x0 += win) {
      const x1 = Math.min(w, x0 + win);
      const n = (x1 - x0) * (y1 - y0);
      let sa = 0;
      let sb = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          sa += la[y * w + x] ?? 0;
          sb += lb[y * w + x] ?? 0;
        }
      }
      const ma = sa / n;
      const mb = sb / n;
      let va = 0;
      let vb = 0;
      let cov = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const da = (la[y * w + x] ?? 0) - ma;
          const db = (lb[y * w + x] ?? 0) - mb;
          va += da * da;
          vb += db * db;
          cov += da * db;
        }
      }
      va /= n;
      vb /= n;
      cov /= n;
      const num = (2 * ma * mb + C1) * (2 * cov + C2);
      const den = (ma * ma + mb * mb + C1) * (va + vb + C2);
      total += den === 0 ? 1 : num / den;
      count += 1;
    }
  }
  return count ? total / count : 0;
}

/* -------------------------------------------------------------------------- */
/* Histogram                                                                   */
/* -------------------------------------------------------------------------- */

export const HIST_BINS = 64;

/** Per-channel normalized histogram, `HIST_BINS` bins each. */
export function histogram(a: Uint8ClampedArray): { r: Float32Array; g: Float32Array; b: Float32Array } {
  const r = new Float32Array(HIST_BINS);
  const g = new Float32Array(HIST_BINS);
  const b = new Float32Array(HIST_BINS);
  const n = a.length / 4;
  if (n === 0) return { r, g, b };
  const scale = HIST_BINS / 256;
  for (let p = 0; p < a.length; p += 4) {
    const ri = Math.min(HIST_BINS - 1, ((a[p] ?? 0) * scale) | 0);
    const gi = Math.min(HIST_BINS - 1, ((a[p + 1] ?? 0) * scale) | 0);
    const bi = Math.min(HIST_BINS - 1, ((a[p + 2] ?? 0) * scale) | 0);
    r[ri] = (r[ri] ?? 0) + 1;
    g[gi] = (g[gi] ?? 0) + 1;
    b[bi] = (b[bi] ?? 0) + 1;
  }
  for (let i = 0; i < HIST_BINS; i++) {
    r[i] = (r[i] ?? 0) / n;
    g[i] = (g[i] ?? 0) / n;
    b[i] = (b[i] ?? 0) / n;
  }
  return { r, g, b };
}

function intersection(p: Float32Array, q: Float32Array): number {
  let s = 0;
  for (let i = 0; i < HIST_BINS; i++) s += Math.min(p[i] ?? 0, q[i] ?? 0);
  return s;
}

/**
 * Normalized per-channel 64-bin histogram **intersection distance**, averaged over RGB.
 *
 *   距离 = 1 − mean_RGB( Σ_i min(p_i, q_i) )      0 = identical distributions
 *
 * Intersection (rather than χ² or EMD) is used because it is bounded, cheap, and reads
 * directly as "fraction of the tonal mass that lands in the same bin" — a number that is
 * easy to explain next to a slider.
 */
export function histogramDistance(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
  const ha = histogram(a);
  const hb = histogram(b);
  const d = (intersection(ha.r, hb.r) + intersection(ha.g, hb.g) + intersection(ha.b, hb.b)) / 3;
  return clamp01(1 - d);
}

/* -------------------------------------------------------------------------- */
/* Frequency bands                                                             */
/* -------------------------------------------------------------------------- */

/**
 * Crude but honest 3-band frequency analysis from a box-downsample pyramid.
 *
 *   P0 = image, P1 = ↓2(P0), P2 = ↓2(P1) = 1/4 scale, P3 = ↓2(P2) = 1/8 scale
 *
 *   low  = RMS about the mean of P3   — coarse structure / composition
 *   mid  = RMS of P2 − ↑2(P3)         — medium features (blobs, streaks)
 *   high = RMS of P0 − ↑2(P1)         — fine detail (grain, thin filaments)
 *
 * A box filter is a poor low-pass (it aliases a little) and this is *not* an FFT, so the
 * numbers are not calibrated in cycles/pixel. They are used as **ratios between two
 * images**, where the filter's exact response cancels out: "the render's high band is
 * 3.2× the reference's" is a statement that survives any reasonable filter choice, and
 * that is all the attribution needs.
 *
 * All values are on the 0..1 luma scale.
 */
export function bandEnergy(a: Uint8ClampedArray, w: number, h: number): Bands {
  const p0 = toLuma(a, w, h);
  return bandEnergyLuma(p0, w, h);
}

export function bandEnergyLuma(p0: Float32Array, w: number, h: number): Bands {
  const d1 = downsample(p0, w, h);
  const d2 = downsample(d1.data, d1.w, d1.h);
  const d3 = downsample(d2.data, d2.w, d2.h);
  /**
   * Each residual is taken at **one scale above** the plane being reconstructed, because
   * `upsample` doubles exactly once:
   *
   *   `high` = p0 − upsample(d1)   d1 is 1/2 scale  -> one doubling reaches full scale
   *   `mid`  = d2 − upsample(d3)   d3 is 1/8 scale  -> one doubling reaches 1/4 scale
   *
   * `high` used to read `upsample(d2.data, …)`: `d2` is 1/4 scale, so one doubling covers
   * only half the target and the remaining three quadrants clamped to a single value.
   * Measured on a 256×256 image with luma std 0.29 the bottom-right quadrant had a standard
   * deviation of exactly 0, and the band read ≈0.294 — i.e. it was tracking global contrast.
   * The consequence was not cosmetic: `attribution`'s noise-frequency rule keys off the
   * high-band *ratio*, so it reported "high-frequency energy matches" for a 4-pixel blur
   * (0.985 where the true ratio is 0.19) and for σ=0.16 noise (1.010 where the truth is 3.36),
   * and the parameter optimiser's λ₃ term then pushed on contrast instead of noise frequency.
   */
  const up2 = upsample(d3.data, d3.w, d3.h, d2.w, d2.h);
  const up1 = upsample(d1.data, d1.w, d1.h, w, h);
  return {
    low: rms(d3.data) / 255,
    mid: rmsOfDifference(difference(d2.data, up2)) / 255,
    high: rmsOfDifference(difference(p0, up1)) / 255,
  };
}

/* -------------------------------------------------------------------------- */
/* Colour / edge statistics                                                    */
/* -------------------------------------------------------------------------- */

/** Mean HSV saturation and circular mean hue, both 0..1. */
export function colorStats(a: Uint8ClampedArray): { saturation: number; hue: number } {
  let sSum = 0;
  let hx = 0;
  let hy = 0;
  const n = a.length / 4;
  if (n === 0) return { saturation: 0, hue: 0 };
  for (let p = 0; p < a.length; p += 4) {
    const r = (a[p] ?? 0) / 255;
    const g = (a[p + 1] ?? 0) / 255;
    const b = (a[p + 2] ?? 0) / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    sSum += max <= 0 ? 0 : d / max;
    if (d > 1e-4) {
      let hue: number;
      if (max === r) hue = ((g - b) / d) % 6;
      else if (max === g) hue = (b - r) / d + 2;
      else hue = (r - g) / d + 4;
      hue = (hue + 6) % 6;
      const angle = (hue / 6) * Math.PI * 2;
      // Weight by saturation: grey pixels have no meaningful hue.
      const weight = d / max;
      hx += Math.cos(angle) * weight;
      hy += Math.sin(angle) * weight;
    }
  }
  const hue = (Math.atan2(hy, hx) / (Math.PI * 2) + 1) % 1;
  return { saturation: sSum / n, hue };
}

/** RMS of the central-difference gradient magnitude, 0..1. A proxy for edge hardness. */
export function gradientEnergy(a: Uint8ClampedArray, w: number, h: number): number {
  const l = toLuma(a, w, h);
  let s = 0;
  let n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const gx = ((l[y * w + x + 1] ?? 0) - (l[y * w + x - 1] ?? 0)) * 0.5;
      const gy = ((l[(y + 1) * w + x] ?? 0) - (l[(y - 1) * w + x] ?? 0)) * 0.5;
      s += gx * gx + gy * gy;
      n += 1;
    }
  }
  return n ? Math.sqrt(s / n) / 255 : 0;
}

/* -------------------------------------------------------------------------- */
/* Composite                                                                   */
/* -------------------------------------------------------------------------- */

/** Everything the attribution needs, from two frames of the same dimensions. */
export function compareFrames(
  ref: Uint8ClampedArray,
  cur: Uint8ClampedArray,
  w: number,
  h: number,
): ScoreBreakdown {
  const refLuma = toLuma(ref, w, h);
  const curLuma = toLuma(cur, w, h);
  const refQ = downsample(downsample(refLuma, w, h).data, w >> 1, h >> 1);
  const curQ = downsample(downsample(curLuma, w, h).data, w >> 1, h >> 1);
  const refStats = colorStats(ref);
  const curStats = colorStats(cur);

  return {
    ssim: ssimLuma(refLuma, curLuma, w, h),
    hist: histogramDistance(ref, cur),
    bands: bandEnergyLuma(curLuma, w, h),
    bandsRef: bandEnergyLuma(refLuma, w, h),
    ssimLow: ssimLuma(refQ.data, curQ.data, refQ.w, refQ.h),
    meanLuma: { ref: mean(refLuma) / 255, cur: mean(curLuma) / 255 },
    saturation: { ref: refStats.saturation, cur: curStats.saturation },
    hueMean: { ref: refStats.hue, cur: curStats.hue },
    gradient: { ref: gradientEnergy(ref, w, h), cur: gradientEnergy(cur, w, h) },
  };
}

/** Scalar loss for the optimizer: lower is better. Weights follow DESIGN.md §6.6. */
export function lossFrom(breakdown: ScoreBreakdown): number {
  const highRatio = breakdown.bandsRef.high > 1e-6 ? breakdown.bands.high / breakdown.bandsRef.high : 1;
  const lowRatio = breakdown.bandsRef.low > 1e-6 ? breakdown.bands.low / breakdown.bandsRef.low : 1;
  return (
    1 - breakdown.ssim +
    0.5 * breakdown.hist +
    0.25 * Math.abs(highRatio - 1) +
    0.15 * Math.abs(lowRatio - 1)
  );
}

/* -------------------------------------------------------------------------- */
/* Attribution — DESIGN.md §4.3 as literal rules                               */
/* -------------------------------------------------------------------------- */

/** Ratio helper that never divides by zero and is symmetric around 1. */
function ratio(cur: number, ref: number): number {
  if (ref <= 1e-9) return cur <= 1e-9 ? 1 : Number.POSITIVE_INFINITY;
  return cur / ref;
}

/**
 * Map the document's tunable uniforms onto the roles the rules talk about.
 *
 * This is name-based on purpose: uniform names are the stable contract (`u_detailScale`
 * is `u_detailScale` in every target and in the CLI's `--set`), whereas node ids vary per
 * document. Anything unrecognised falls back to the closest semantic match.
 */
export function buildRoleTable(
  uniforms: { name: string; origin: { node: string; param: string }; label?: string; min?: number; max?: number; value: number | boolean | number[] }[],
): Partial<Record<ParamRole, ParamRef>> {
  const table: Partial<Record<ParamRole, ParamRef>> = {};
  const rules: [ParamRole, RegExp][] = [
    ['warpStrength', /warp.?strength|strength|distort/i],
    ['warpScale', /warp.?scale/i],
    ['detailScale', /detail.?scale|noise.?scale|texture.?scale|density/i],
    ['octaves', /octave|detail.?level|层数/i],
    ['lacunarity', /lacunar/i],
    ['gain', /gain|decay|衰减/i],
    ['ridged', /ridged|crease|折痕/i],
    ['contrast', /contrast|对比/i],
    ['exposure', /exposure|bright|曝光/i],
    ['saturation', /saturat|饱和/i],
    ['hue', /hue|tint|hue.?shift|色相/i],
    ['gamma', /gamma/i],
    ['speed', /speed|time|flow|速度|相位/i],
  ];
  for (const [role, pattern] of rules) {
    const hit = uniforms.find((u) => pattern.test(u.name) || (u.label ? pattern.test(u.label) : false));
    if (hit && !table[role]) {
      table[role] = {
        node: hit.origin.node,
        param: hit.origin.param,
        label: hit.label,
        min: hit.min,
        max: hit.max,
        value: typeof hit.value === 'number' ? hit.value : undefined,
      };
    }
  }
  // Last resort: a generic "scale-like" knob so hints are never empty when a document
  // has tunables at all.
  if (!table.detailScale) {
    const first = uniforms.find((u) => typeof u.value === 'number');
    if (first) {
      table.detailScale = {
        node: first.origin.node,
        param: first.origin.param,
        label: first.label,
        min: first.min,
        max: first.max,
        value: typeof first.value === 'number' ? first.value : undefined,
      };
    }
  }
  return table;
}

/** Step a parameter one notch in the requested direction, staying inside its range. */
function nudge(ref: ParamRef, direction: 'increase' | 'decrease', steps = 1): number | undefined {
  if (typeof ref.value !== 'number') return undefined;
  const span = typeof ref.min === 'number' && typeof ref.max === 'number' ? ref.max - ref.min : Math.abs(ref.value) || 1;
  const step = span / 12;
  const next = direction === 'increase' ? ref.value + step * steps : ref.value - step * steps;
  const lo = ref.min ?? -Infinity;
  const hi = ref.max ?? Infinity;
  const clamped = Math.min(hi, Math.max(lo, next));
  return Math.round(clamped * 1000) / 1000;
}

function hint(
  roles: Partial<Record<ParamRole, ParamRef>>,
  role: ParamRole,
  direction: 'increase' | 'decrease',
  why: string,
  steps = 1,
): AttributionHint | null {
  const ref = roles[role];
  if (!ref) return null;
  const out: AttributionHint = { node: ref.node, param: ref.param, direction, why };
  const suggest = nudge(ref, direction, steps);
  if (suggest !== undefined) out.suggest = suggest;
  return out;
}

/** Two decimals, with `-0.00` normalised away — it reads as a bug in a verdict line. */
const fmt = (n: number): string => {
  if (!Number.isFinite(n)) return '∞';
  return Math.abs(n) < 0.005 ? '0.00' : n.toFixed(2);
};

/**
 * Turn a score breakdown into an actionable patch suggestion (DESIGN.md §4.3).
 *
 * Every hint names a real node input and a direction; the rules are literal thresholds,
 * deliberately conservative, and they are ordered by how much of the error they explain.
 * The verdict is the single sentence the UI shows next to the score.
 *
 * @param diff  breakdown from {@link compareFrames}
 * @param ctx   role table (see {@link buildRoleTable}) + optional temporal SSIM
 */
export function attribute(diff: ScoreBreakdown, ctx: AttributionContext = {}): Attribution {
  const roles = ctx.roles ?? {};
  const hints: AttributionHint[] = [];
  const reasons: string[] = [];

  const { bands, bandsRef } = diff;
  const highRatio = ratio(bands.high, bandsRef.high);
  const lowRatio = ratio(bands.low, bandsRef.low);

  /* Rule 1 — is the anatomy in the wrong place? ---------------------------- */
  const structureWeak = diff.ssimLow !== undefined && diff.ssimLow < 0.75;
  if (structureWeak) {
    reasons.push(`低频结构偏差大 (1/4 尺度 SSIM ${fmt(diff.ssimLow ?? 0)})`);
    // Direction comes from band energy, not from SSIM: the coarse contrast tells us
    // whether the render is too flat or too busy.
    if (lowRatio < 0.85) {
      hints.push(
        hint(roles, 'warpStrength', 'increase', `低频结构能量只有参考的 ${fmt(lowRatio)}×，画面偏平`, 2) ??
          hint(roles, 'warpScale', 'decrease', '低频结构偏平，先把变形尺度调大再谈细节')!,
      );
    } else if (lowRatio > 1.25) {
      hints.push(
        hint(roles, 'warpStrength', 'decrease', `低频结构能量是参考的 ${fmt(lowRatio)}×，画面过碎`) ??
          hint(roles, 'warpScale', 'increase', '低频结构过碎，先减小变形强度')!,
      );
    } else {
      hints.push(
        hint(roles, 'warpScale', 'increase', '结构位置不匹配，改变变形的空间频率可以整体重排') ??
          hint(roles, 'detailScale', 'decrease', '结构位置不匹配，先降低细节频率看主体')!,
      );
    }
  }

  /* Rule 2 — detail frequency (FFT band ratio > 2×) ------------------------ */
  if (Number.isFinite(highRatio) && (highRatio > 2 || highRatio < 0.5)) {
    if (highRatio > 2) {
      reasons.push(`高频细节能量是参考的 ${fmt(highRatio)}×`);
      hints.push(
        hint(roles, 'detailScale', 'decrease', `高频细节能量过高 (${fmt(highRatio)}×)，纹理比参考更细碎`, 2) ??
          hint(roles, 'octaves', 'decrease', '高频能量过高，减少细节层数')!,
      );
    } else {
      reasons.push(`高频细节能量只有参考的 ${fmt(highRatio)}×`);
      hints.push(
        hint(roles, 'detailScale', 'increase', `高频细节能量不足 (${fmt(highRatio)}×)，纹理比参考更平滑`, 2) ??
          hint(roles, 'octaves', 'increase', '高频能量不足，增加细节层数')!,
      );
    }
  }

  /* Rule 3 — exposure / tonemapping --------------------------------------- */
  if (diff.meanLuma) {
    const delta = diff.meanLuma.cur - diff.meanLuma.ref;
    if (Math.abs(delta) > 0.06) {
      reasons.push(`平均亮度偏差 ${delta > 0 ? '+' : ''}${fmt(delta)}`);
      const direction = delta < 0 ? 'increase' : 'decrease';
      hints.push(
        hint(roles, 'exposure', direction, `平均亮度${delta < 0 ? '偏低' : '偏高'} ${fmt(Math.abs(delta))}（0..1 亮度）`, 2) ??
          hint(roles, 'contrast', direction, `整体明暗偏移 ${fmt(Math.abs(delta))}，先校正曝光/对比`)!,
      );
    }
  }

  /* Rule 4 — saturation ---------------------------------------------------- */
  if (diff.saturation) {
    const delta = diff.saturation.cur - diff.saturation.ref;
    if (Math.abs(delta) > 0.08) {
      reasons.push(`饱和度偏差 ${delta > 0 ? '+' : ''}${fmt(delta)}`);
      hints.push(
        hint(roles, 'saturation', delta < 0 ? 'increase' : 'decrease', delta < 0 ? '画面比参考更灰（HSV S 偏低）' : '画面比参考更艳（HSV S 偏高）')!,
      );
    }
  }

  /* Rule 5 — hue / palette ------------------------------------------------- */
  if (diff.hueMean) {
    // Hue is circular: 0.5 is the farthest apart two hues can be.
    let delta = diff.hueMean.cur - diff.hueMean.ref;
    if (delta > 0.5) delta -= 1;
    if (delta < -0.5) delta += 1;
    if (Math.abs(delta) > 0.04) {
      reasons.push(`主色相偏移 ${fmt(delta)} 圈`);
      hints.push(
        hint(roles, 'hue', delta < 0 ? 'increase' : 'decrease', `主色相与参考相差 ${fmt(Math.abs(delta) * 360)}°`)!,
      );
    }
  }

  /* Rule 6 — edge hardness ------------------------------------------------- */
  if (diff.gradient) {
    const g = ratio(diff.gradient.cur, diff.gradient.ref);
    if (Number.isFinite(g) && (g > 1.5 || g < 0.66)) {
      reasons.push(`边缘梯度是参考的 ${fmt(g)}×`);
      if (g > 1.5) {
        hints.push(
          hint(roles, 'contrast', 'decrease', `边缘过锐 (梯度 ${fmt(g)}×)，先降低对比度`) ??
            hint(roles, 'detailScale', 'decrease', `边缘过锐 (梯度 ${fmt(g)}×)，降低高频细节`)!,
        );
      } else {
        hints.push(
          hint(roles, 'contrast', 'increase', `边缘过软 (梯度 ${fmt(g)}×)，提高对比度让结构立起来`) ??
            hint(roles, 'detailScale', 'increase', `边缘过软 (梯度 ${fmt(g)}×)，提高高频细节`)!,
        );
      }
    }
  }

  /* Rule 7 — is it even animating? ----------------------------------------- */
  if (ctx.temporalSsim !== undefined && ctx.temporalSsim > 0.995) {
    reasons.push('两帧之间几乎完全相同');
    const speed = roles.speed;
    if (speed) {
      hints.push({
        node: speed.node,
        param: speed.param,
        direction: 'increase',
        why: `时间维度 SSIM ≈ ${fmt(ctx.temporalSsim)}，动画没有生效：确认 time 是否接进了域变形/噪声`,
        suggest: nudge(speed, 'increase', 2),
      });
    }
  }

  /* Verdict ---------------------------------------------------------------- */
  const head = `SSIM ${diff.ssim.toFixed(3)} · 直方图距离 ${diff.hist.toFixed(3)}`;
  const verdict = reasons.length ? `${head} — 主要问题：${reasons.join('；')}` : `${head} — 各维度与参考基本一致`;

  const deduped: AttributionHint[] = [];
  for (const h of hints) {
    if (!h) continue;
    if (deduped.some((d) => d.node === h.node && d.param === h.param)) continue;
    deduped.push(h);
    if (deduped.length >= 4) break;
  }

  return { verdict, hints: deduped };
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
