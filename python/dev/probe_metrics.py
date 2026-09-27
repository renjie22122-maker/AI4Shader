"""Probe: do the real LPIPS and CLIP metrics actually run offline on this machine?"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from ai4shader import score as S
from dev.measure_thresholds import add_hf_noise, box_blur, brighten, make_base

print("=== capabilities(probe=False) : cheap presence check ===")
t0 = time.perf_counter()
caps = S.capabilities()
print(f"  took {(time.perf_counter()-t0)*1000:.1f}ms")
print(json.dumps({k: v for k, v in caps.items() if k in ("lpips", "clip_similarity", "environment")},
                 indent=2, default=str))

print("\n=== capabilities(probe=True) : actually load the models (inside no_network) ===")
t0 = time.perf_counter()
caps2 = S.capabilities(probe=True)
dt = (time.perf_counter() - t0) * 1000
print(f"  probe took {dt:.0f}ms")
print(json.dumps({k: caps2[k] for k in ("lpips", "clip_similarity")}, indent=2, default=str))

base = make_base(256)
small = make_base(128, seed=11)
rng = np.random.default_rng(0)
noisy = np.clip(base + rng.normal(0, 0.05, base.shape).astype(np.float32), 0, 1)

print("\n=== real LPIPS ===")
if caps2["lpips"]["available"]:
    t0 = time.perf_counter()
    d_same = S.lpips(base, base)
    t_first = (time.perf_counter() - t0) * 1000
    assert d_same is not None
    t0 = time.perf_counter()
    for _ in range(5):
        d_mild = S.lpips(box_blur(base, 1), base)
    t_each = (time.perf_counter() - t0) / 5 * 1000
    d_heavy = S.lpips(box_blur(base, 6), base)
    d_noise = S.lpips(noisy, base)
    d_other = S.lpips(make_base(256, cx=0.25, cy=0.7, seed=42), base)
    print(f"  lpips(a,a)        = {d_same:.6f}   (must be ~0)")
    print(f"  lpips(blur1, base)= {d_mild:.6f}")
    print(f"  lpips(blur6, base)= {d_heavy:.6f}   (must be > blur1)")
    print(f"  lpips(noise, base)= {d_noise:.6f}")
    print(f"  lpips(other, base)= {d_other:.6f}   (must be largest)")
    mono = d_same < d_mild < d_heavy < d_other
    print(f"  MONOTONIC: {mono}")
    print(f"  timing: first call (incl. 244MB checkpoint load) {t_first:.0f}ms, "
          f"subsequent {t_each:.0f}ms/call")
    print(f"  ssim for the same pair (must differ from lpips): "
          f"ssim={S.ssim(noisy, base):.6f} vs lpips={d_noise:.6f}")
else:
    print("  UNAVAILABLE:", caps2["lpips"]["reason"])

print("\n=== real CLIP ===")
if caps2["clip_similarity"]["available"]:
    t0 = time.perf_counter()
    c_same = S.clip_similarity(base, base)
    t_first = (time.perf_counter() - t0) * 1000
    assert c_same is not None
    t0 = time.perf_counter()
    c_bright = S.clip_similarity(brighten(base, 0.25), base)
    t_each = (time.perf_counter() - t0) * 1000
    c_other = S.clip_similarity(make_base(256, cx=0.25, cy=0.7, seed=42), base)
    print(f"  clip(a,a)          = {c_same:.6f}   (must be ~1)")
    print(f"  clip(bright0.25,a) = {c_bright:.6f}")
    print(f"  clip(other, a)     = {c_other:.6f}")
    print(f"  timing: first call {t_first:.0f}ms, second {t_each:.0f}ms")
else:
    print("  UNAVAILABLE:", caps2["clip_similarity"]["reason"])

print("\n=== fallback path honesty ===")
print(f"  perceptual_fallback(noisy, base) = {S.perceptual_fallback(noisy, base):.6f}")
print(f"  unavailable_metrics()            = {S.unavailable_metrics()}")
print(f"  loss() with lpips                = {S.loss(S.score(noisy, base), lpips_value=d_noise):.6f}")
print(f"  loss() without lpips             = {S.loss(S.score(noisy, base)):.6f}")

print("\n=== offline enforcement really works ===")
try:
    import socket
    with S.no_network():
        socket.create_connection(("pypi.org", 443), timeout=2)
    print("  FAIL: a connection succeeded inside no_network()")
except RuntimeError as exc:
    print(f"  OK: no_network() blocked a real connect attempt: {str(exc)[:80]}...")
except Exception as exc:  # noqa: BLE001
    print(f"  unexpected: {type(exc).__name__}: {exc}")

print("\n=== breakdown JSON keys (must match score.ts) ===")
bd = S.score(noisy, base)
print(" ", json.dumps(bd.to_dict())[:400])
