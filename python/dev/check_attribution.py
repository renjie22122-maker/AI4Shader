"""Check the attribution rules against the measured perturbations."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from ai4shader import attribute as A
from ai4shader import score as S
from dev.measure_thresholds import (add_hf_noise, box_blur, brighten, make_base,
                                    saturate, shift, shift_hue)

base = make_base()

CASES = [
    ("blur r=2", box_blur(base, 2), "edge.hardness"),
    ("blur r=4", box_blur(base, 4), "edge.hardness"),
    ("bright +0.12", brighten(base, 0.12), "tone.exposure"),
    ("bright +0.22", brighten(base, 0.22), "tone.exposure"),
    ("sat x1.9", saturate(base, 1.9), "color.saturation"),
    ("sat x0.35", saturate(base, 0.35), "color.saturation"),
    ("noise 0.10", add_hf_noise(base, 0.10), "detail.frequency"),
    ("noise 0.16", add_hf_noise(base, 0.16), "detail.frequency"),
    ("shift 16px", shift(base, 16, 0), "structure.mismatch"),
    ("shift 28px", shift(base, 28, 0), "structure.mismatch"),
    ("hue +0.12", shift_hue(base, 0.12), "color.hue"),
    ("hue +0.20", shift_hue(base, 0.20), "color.hue"),
]

print(f"{'case':<14} {'expect':<20} {'top rule':<20} {'ok':<4} conf  direction  param")
bad = 0
for label, img, expect in CASES:
    bd = S.score(img, base)
    hints = A.attribute(bd)
    top = hints[0] if hints else None
    ok = top is not None and top["rule"] == expect
    bad += not ok
    print(f"{label:<14} {expect:<20} {(top['rule'] if top else '-'):<20} "
          f"{'OK' if ok else 'FAIL':<4} {top['confidence'] if top else 0:.2f}  "
          f"{str(top['direction']) if top else '-':<10} {top['param'] if top else '-'}")

print("\n--- full hint lists ---")
for label, img, _ in CASES:
    bd = S.score(img, base)
    print(f"\n{label}: ssim={bd.ssim:.3f} ssimLow={bd.ssim_low:.3f}")
    for h in A.attribute(bd):
        print(f"   [{h['confidence']:.2f}] {h['rule']:<20} {str(h['node']):<6}.{str(h['param']):<12} "
              f"{str(h['direction']):<9} {h['why'][:110]}")

print("\n--- temporal / animation rule ---")
bd = S.score(base, base)
h = A.attribute(bd, temporal_ssim=1.0)
print("no catalog:", h[0]["rule"], h[0]["node"], h[0]["direction"], "|", h[0]["why"][:130])

doc_catalog = A.catalog_from_document({
    "graph": {"nodes": [
        {"id": "n_warp", "kind": "domain_warp", "params": {
            "scale": {"kind": "param", "name": "u_warpScale", "value": 0.9, "min": 0.2, "max": 4},
            "strength": {"kind": "param", "name": "u_warpStrength", "value": 0.45, "min": 0, "max": 0.5},
            "time": {"kind": "param", "name": "u_speed", "value": 0.0, "min": 0, "max": 2}}},
        {"id": "n_detail", "kind": "fbm", "params": {
            "scale": {"kind": "param", "name": "u_detailScale", "value": 6.0, "min": 0.1, "max": 64},
            "octaves": {"kind": "param", "name": "u_octaves", "value": 4, "min": 1, "max": 8}}},
        {"id": "n_sdf", "kind": "sdf", "params": {
            "size": {"kind": "vec2", "value": [0.3, 0.3]},
            "smoothness": {"kind": "param", "name": "u_smoothness", "value": 0.01, "min": 0.001, "max": 1}}},
        {"id": "n_grade", "kind": "grade", "params": {
            "exposure": {"kind": "param", "name": "u_exposure", "value": 0.0, "min": -2, "max": 2},
            "saturation": {"kind": "param", "name": "u_saturation", "value": 1.0, "min": 0, "max": 3}}},
    ]}
})
print("\n--- with catalog (resolved node ids + uniform names) ---")
for label, img, _ in CASES:
    bd = S.score(img, base)
    for h in A.attribute(bd, catalog=doc_catalog)[:2]:
        print(f"{label:<12} [{h['confidence']:.2f}] {h['rule']:<20} {h['node']}.{h['param']:<12} "
              f"uniform={str(h['uniform']):<16} {str(h['direction']):<9} suggest={h['suggest']}")
    print()

print("--- temporal with catalog (time stuck at 0) ---")
h = A.attribute(S.score(base, base), catalog=doc_catalog, temporal_ssim=1.0)
print(h[0]["rule"], "|", h[0]["node"], ".", h[0]["param"], "|", h[0]["direction"], "|", h[0]["uniform"],
      "| suggest=", h[0]["suggest"])
print("|", h[0]["why"][:180])

print(f"\n{len(CASES) - bad}/{len(CASES)} discrimination cases routed to the expected rule")
