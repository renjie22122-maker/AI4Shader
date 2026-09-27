"""Probe: real compiler availability + validation of the real emitted artefacts."""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ai4shader import validate as V

print("=== available() ===")
for k, v in V.available().items():
    print(f"  {k:20s} -> {v}")

print("\n=== tool version banners ===")
for tool, path in V.available().items():
    if path:
        print(f"  {tool:20s} {V._tool_version(tool, path)}")

ROOT = pathlib.Path(r"D:\Desktop\Project\AI4Shader\dist\_probe\shader")
CASES = [
    ("shader.frag", "glsl"),
    ("shader.wgsl", "wgsl"),
    ("shader.hlsl", "hlsl"),
    ("shader.metal", "msl"),
    ("Shader.shader", "shaderlab"),
    ("shader.ue.hlsl", "ue"),
]
print("\n=== validate the real emitted artefacts ===")
for name, target in CASES:
    p = ROOT / name
    if not p.exists():
        cands = list(ROOT.glob("*"))
        print(f"  {name:18s} MISSING (dir has: {[c.name for c in cands]})")
        continue
    t0 = time.perf_counter()
    r = V.validate_file(p, target)
    dt = (time.perf_counter() - t0) * 1000
    print(f"\n  {name} (target={target}) -> ok={r.ok} in {dt:.0f}ms")
    print(f"    tool      : {r.tool} @ {r.tool_path}")
    print(f"    entry     : {V.discover_entry(p.read_text(encoding='utf-8-sig'), target)!r}")
    print(f"    command   : {' '.join(r.command)}")
    print(f"    reason    : {r.reason}")
    if r.notes:
        for n in r.notes:
            print(f"    note      : {n}")
    for d in r.diagnostics[:8]:
        print(f"    diag      : {d.severity} {d.line}:{d.column} {d.message}")
    if r.stderr.strip():
        print(f"    stderr    : {r.stderr.strip()[:400]!r}")

print("\n=== negative control: deliberately broken shaders ===")
BROKEN_GLSL = """#version 300 es
precision highp float;
out vec4 fragColor;
void main() {
  fragColor = vec4(undeclared_symbol, 0.0, 0.0, 1.0);
}
"""
BROKEN_HLSL = """float4 PSMain(float2 uv : TEXCOORD0) : SV_Target { return nosuchvar; }
"""
BROKEN_WGSL = """@fragment
fn fs_main() -> @location(0) vec4f { return vec4f(undeclared, 0.0, 0.0, 1.0); }
"""
for src, tgt in ((BROKEN_GLSL, "glsl"), (BROKEN_HLSL, "hlsl"), (BROKEN_WGSL, "wgsl")):
    r = V.validate_source(src, tgt)
    print(f"  {tgt:6s} ok={str(r.ok):5s} errors={r.error_count} reason={r.reason[:90]}")
    for d in r.diagnostics[:3]:
        print(f"        {d.severity} {d.line}:{d.column} {d.message}")

print("\n=== require_tool mode ===")
for tool in ("dxc", "naga", "glslangValidator"):
    r = V.validate_source(BROKEN_GLSL, "glsl", require_tool=tool)
    print(f"  require_tool={tool:18s} ok={str(r.ok):5s} tool={r.tool!r} reason={r.reason[:95]}")

print("\n=== target with no compiler ===")
for tgt in ("ue", "shaderlab", "bogus"):
    r = V.validate_source("float4 x;", tgt)
    print(f"  {tgt:11s} ok={str(r.ok):5s} reason={r.reason[:110]}")
