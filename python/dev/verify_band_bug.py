"""Verify the `high` band bug in apps/web/src/compare/score.ts.

score.ts computes:
    up1 = upsample(d2.data, d2.w, d2.h, w, h)     // d2 is 1/4 scale, target is full scale
    high = rms(p0 - up1)

but `upsample` maps output index y -> min(sh-1, y>>1).  Going from 1/4 scale to
full scale needs TWO doublings, and y>>1 only performs one: for y >= 2*sh the
index clamps to sh-1.  So the bottom half / right half of `up1` is a constant,
and `high` degenerates into "how far is the image from one constant", not "how
much energy is above the 2-pixel scale".

The intended expression (matching the documented semantics "high = RMS of
P0 - up2(P1) -- fine detail") is to upsample d1 (1/2 scale), not d2.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from ai4shader.score import _downsample2, _upsample2, to_luma
from dev.measure_thresholds import add_hf_noise, box_blur, make_base


def ts_high(p0):
    """Exactly what score.ts does today (upsample d2 to full scale)."""
    d1 = _downsample2(p0)
    d2 = _downsample2(d1)
    return float(np.sqrt(np.mean((p0 - _upsample2(d2, p0.shape[1], p0.shape[0])) ** 2)))


def fixed_high(p0):
    """What the docstring says it should do (upsample d1 to full scale)."""
    d1 = _downsample2(p0)
    return float(np.sqrt(np.mean((p0 - _upsample2(d1, p0.shape[1], p0.shape[0])) ** 2)))


def main():
    base = make_base()
    l0 = to_luma(base)
    d2 = _downsample2(_downsample2(l0))
    up_ts = _upsample2(d2, l0.shape[1], l0.shape[0])
    half = l0.shape[1] // 2
    # How much of the "upsampled" image is a flat clamp?
    flat_region = up_ts[half:, half:]
    print(f"up1 (score.ts) shape={up_ts.shape}  bottom-right quadrant std={flat_region.std():.6f}")
    print(f"  distinct values in bottom-right quadrant: {len(np.unique(flat_region))}")
    print(f"  distinct values in full up1: {len(np.unique(up_ts))} (should be ~{d2.size} if correct)")
    print(f"  d1.size={_downsample2(l0).size}  d2.size={d2.size}")

    print(f"\n{'case':<18} {'ts_high':>10} {'ratio':>7} | {'fix_high':>10} {'ratio':>7}")
    cases = [("base", base), ("blur r=2", box_blur(base, 2)), ("blur r=4", box_blur(base, 4)),
             ("noise 0.05", add_hf_noise(base, 0.05)), ("noise 0.16", add_hf_noise(base, 0.16)),
             ("sat x1.9", None)]
    ref_ts, ref_fix = ts_high(l0), fixed_high(l0)
    for name, img in cases:
        if img is None:
            continue
        l = to_luma(img)
        a, b = ts_high(l), fixed_high(l)
        print(f"{name:<18} {a:10.5f} {a/ref_ts:7.3f} | {b:10.5f} {b/ref_fix:7.3f}")
    print(f"\nreference (base):  ts_high={ref_ts:.5f}  fixed_high={ref_fix:.5f}")


if __name__ == "__main__":
    main()
