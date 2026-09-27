"""Threshold-calibration experiment (NOT part of the test suite).

Builds a shader-like synthetic image, applies exactly one perturbation at a
time, and prints every observable the attribution rules read.  The thresholds in
attribute.py were chosen from this table rather than guessed.
"""
from __future__ import annotations

import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from ai4shader import imageio as iio
from ai4shader import score as S


# --------------------------------------------------------------------------- #
# A shader-like image: soft SDF blob + fbm detail + a colour ramp.
# --------------------------------------------------------------------------- #

def value_noise(shape, freq, seed):
    rng = np.random.default_rng(seed)
    h, w = shape
    gh, gw = max(2, int(h * freq) + 2), max(2, int(w * freq) + 2)
    g = rng.random((gh, gw))
    ys = np.linspace(0, gh - 1, h)
    xs = np.linspace(0, gw - 1, w)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
    y1 = np.minimum(y0 + 1, gh - 1); x1 = np.minimum(x0 + 1, gw - 1)
    a = g[np.ix_(y0, x0)]; b = g[np.ix_(y0, x1)]
    c = g[np.ix_(y1, x0)]; d = g[np.ix_(y1, x1)]
    fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
    top = a + (b - a) * fx
    bot = c + (d - c) * fx
    return top + (bot - top) * fy


def fbm(shape, base=3.0, octaves=5, gain=0.5, lac=2.0, seed=7):
    out = np.zeros(shape); amp = 0.5; f = base
    for i in range(octaves):
        out += amp * value_noise(shape, f, seed + i * 13)
        amp *= gain; f *= lac
    return out / max(out.max(), 1e-9)


def make_base(n=256, cx=0.5, cy=0.5, radius=0.30, sharp=0.06, seed=7):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64) / (n - 1)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    shape = 1.0 - np.clip((d - radius) / sharp + 0.5, 0.0, 1.0)  # soft SDF disc
    detail = fbm((n, n), seed=seed)
    v = np.clip(shape * 0.72 + detail * 0.42 - 0.06, 0.0, 1.0)
    # colour ramp: dark blue -> warm orange, then a saturation/tint push
    lo = np.array([0.05, 0.07, 0.16]); hi = np.array([0.95, 0.62, 0.24])
    img = lo + (hi - lo) * v[:, :, None]
    img *= (0.85 + 0.30 * shape)[:, :, None]
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Perturbations -- each changes exactly ONE thing.
# --------------------------------------------------------------------------- #

def box_blur(img, radius):
    if radius <= 0:
        return img.copy()
    k = 2 * radius + 1
    pad = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(img)
    for dy in range(k):
        for dx in range(k):
            out += pad[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return (out / (k * k)).astype(np.float32)


def brighten(img, delta):
    return np.clip(img + delta, 0.0, 1.0).astype(np.float32)


def saturate(img, factor):
    l = (img @ np.array([0.2126, 0.7152, 0.0722])).astype(np.float32)
    return np.clip(l[:, :, None] + (img - l[:, :, None]) * factor, 0.0, 1.0).astype(np.float32)


def add_hf_noise(img, sigma, seed=3):
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0.0, sigma, img.shape).astype(np.float32), 0.0, 1.0).astype(np.float32)


def shift(img, dx, dy):
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1).astype(np.float32)


def shift_hue(img, turns):
    """Rotate hue in HSV, preserving S and V -- a pure palette change.

    HSV->RGB is `(C,X,0)+m` with chroma `C = V*S = max-min = d`, so the chroma
    term is `d`, NOT the normalised `S`.  Getting that wrong silently shrinks
    saturation and makes a hue change look like a saturation change.
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn                      # chroma C
    safe = np.where(d > 1e-9, d, 1.0)
    hue = np.where(mx == r, np.mod((g - b) / safe, 6.0),
                   np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    hue = np.mod(hue + turns * 6.0, 6.0)
    x = d * (1.0 - np.abs(np.mod(hue, 2.0) - 1.0))
    m = mx - d                       # value minus chroma
    z = np.zeros_like(d)
    conds = [hue < 1, hue < 2, hue < 3, hue < 4, hue < 5, hue < 6]
    out = np.stack([
        np.select(conds, [d, x, z, z, x, d]),
        np.select(conds, [x, d, d, x, z, z]),
        np.select(conds, [z, z, x, d, d, x]),
    ], axis=2) + m[:, :, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def row(label, ref, cur, bd):
    hr = S.band_ratio(bd.bands, bd.bands_ref, "high")
    mr = S.band_ratio(bd.bands, bd.bands_ref, "mid")
    lr = S.band_ratio(bd.bands, bd.bands_ref, "low")
    gr = bd.gradient["cur"] / max(bd.gradient["ref"], 1e-12)
    dh = bd.hue_mean["cur"] - bd.hue_mean["ref"]
    if dh > 0.5: dh -= 1.0
    if dh < -0.5: dh += 1.0
    print(f"{label:<22} ssim={bd.ssim:6.3f} ssimLow={bd.ssim_low:6.3f} hist={bd.hist:5.3f} "
          f"dLuma={bd.luma_offset:+6.3f} dSat={bd.saturation['cur'] - bd.saturation['ref']:+6.3f} "
          f"dHue={dh:+6.3f} grad={gr:6.3f} | lo={lr:5.2f} mid={mr:5.2f} high={hr:6.2f}")


def main():
    base = make_base()
    N = base.shape[0]
    print(f"base image {base.shape}  luma mean={S.to_luma(base).mean():.3f} "
          f"sat={S.color_stats(base)['saturation']:.3f} grad={S.gradient_energy(base):.4f}")

    cases = [
        ("IDENTICAL", base),
        ("blur r=1", box_blur(base, 1)),
        ("blur r=2", box_blur(base, 2)),
        ("blur r=4", box_blur(base, 4)),
        ("bright +0.04", brighten(base, 0.04)),
        ("bright +0.10", brighten(base, 0.10)),
        ("bright +0.22", brighten(base, 0.22)),
        ("sat x1.35", saturate(base, 1.35)),
        ("sat x1.9", saturate(base, 1.9)),
        ("sat x0.35", saturate(base, 0.35)),
        ("noise s=0.03", add_hf_noise(base, 0.03)),
        ("noise s=0.08", add_hf_noise(base, 0.08)),
        ("noise s=0.16", add_hf_noise(base, 0.16)),
        ("shift 6px", shift(base, 6, 0)),
        ("shift 14px", shift(base, 14, 0)),
        ("shift 28px", shift(base, 28, 0)),
        ("hue +0.04 turn", shift_hue(base, 0.04)),
        ("hue +0.12 turn", shift_hue(base, 0.12)),
        ("blur r=3 only", box_blur(base, 3)),
        ("noise 0.05 only", add_hf_noise(base, 0.05)),
    ]
    print()
    for label, cur in cases:
        row(label, base, cur, S.score(cur, base))

    print("\n--- unrelated control (different seed/structure) ---")
    other = make_base(cx=0.28, cy=0.72, radius=0.18, seed=99)
    row("unrelated", base, other, S.score(other, base))
    noise_only = np.random.default_rng(5).random(base.shape).astype(np.float32)
    row("pure noise", base, noise_only, S.score(noise_only, base))


if __name__ == "__main__":
    main()
