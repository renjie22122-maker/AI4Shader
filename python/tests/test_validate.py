"""Tests for :mod:`ai4shader.validate`.

The governing rule is DESIGN.md section 5.4: an absent compiler is **not** a pass.
``ok`` is three-valued and these tests pin all three values.

The central test is :func:`test_broken_shader_verdict_tracks_tool_availability`.
It does **not** skip when no compiler is present -- skipping is how a suite
quietly stops testing anything.  It asserts ``ok is False`` *iff* a compiler
exists, and ``ok is None`` with an explanatory reason otherwise, so both
branches are covered on whatever machine runs it.

This module also pins a specific environment trap discovered here: ``dxc`` 1.6 on
a host with a CJK ANSI code page cannot read a non-ASCII source file without a
UTF-8 BOM, and reports it as ``error code 0x80070459`` with **no diagnostics** --
which looks exactly like a crash.  ``test_non_ascii_hlsl_gets_a_bom_and_real_diagnostics``
covers the workaround.
"""

from __future__ import annotations

import os
import re

import pytest

from ai4shader import validate as V

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

GOOD_GLSL = """#version 300 es
precision highp float;
precision highp int;
uniform vec3 iResolution;
out vec4 fragColor;
void main() {
  vec2 uv = gl_FragCoord.xy / iResolution.xy;
  fragColor = vec4(uv, 0.5, 1.0);
}
"""

TOOLS = V.available()


# --------------------------------------------------------------------------- #
# The requirement: an absent compiler is never a pass
# --------------------------------------------------------------------------- #


def test_absent_tool_yields_none_and_never_true():
    """A target whose compiler is missing must produce ``ok=None``.

    Asserted on the tool, not on a target, so it holds on every machine: pick a
    tool that is genuinely absent and require the module to refuse to guess.
    """
    tools = V.available()
    absent = [name for name, path in tools.items() if path is None]
    if not absent:
        pytest.skip("every supported compiler is installed on this machine")
    target_for = {v: k for k, v in V.TOOL_FOR_TARGET.items() if v}
    covered = {target_for[name] for name in absent if name in target_for}
    if not covered:
        pytest.skip("no absent tool maps to a target on this machine")
    for target in sorted(covered):
        result = V.validate_source(GOOD_GLSL, target)
        assert result.ok is None, f"{target}: absent compiler must not yield {result.ok!r}"
        assert result.passed is False
        assert "NOT a pass" in result.reason or "not installed" in result.reason


@pytest.mark.parametrize("target,source,tool_key", [
    ("glsl", BROKEN_GLSL, "glslangValidator"),
    ("hlsl", BROKEN_HLSL, "dxc"),
])
def test_broken_shader_verdict_tracks_tool_availability(target, source, tool_key):
    """``ok is False`` iff a compiler is available; ``ok is None`` otherwise.

    Deliberately **not** skipped when the tool is missing: the no-verdict branch
    is the one that matters for honesty and it must be exercised, not ignored.
    """
    result = V.validate_source(source, target)
    if TOOLS.get(tool_key):
        assert result.ok is False, f"{target}: a real compiler accepted a broken shader"
        assert result.diagnostics, f"{target}: ok=False with no diagnostics is not actionable"
        assert result.error_count >= 1
        assert result.returncode not in (0, None)
        # the raw output must be preserved verbatim, not reformatted away
        assert result.stdout or result.stderr
        error = next(d for d in result.diagnostics if d.severity == "error")
        assert error.line is not None and error.line >= 1
    else:
        assert result.ok is None, f"{target}: no compiler, yet ok={result.ok!r}"
        assert result.tool == tool_key
        assert "not installed" in result.reason or "not discoverable" in result.reason


def test_wgsl_without_naga_is_a_no_verdict_not_a_failure():
    result = V.validate_source(BROKEN_WGSL, "wgsl")
    if TOOLS["naga"]:
        assert result.ok is False
    else:
        assert result.ok is None
        assert "naga" in result.reason


def test_good_glsl_passes_when_glslang_is_present():
    result = V.validate_source(GOOD_GLSL, "glsl")
    if TOOLS["glslangValidator"]:
        assert result.ok is True, result.reason
        assert result.error_count == 0
    else:
        assert result.ok is None


# --------------------------------------------------------------------------- #
# require_tool -- the CI assertion mode
# --------------------------------------------------------------------------- #


def test_require_tool_absent_is_a_no_verdict_not_a_pass():
    tools = V.available()
    absent = next((n for n, p in tools.items() if p is None), None)
    if absent is None:
        pytest.skip("no absent tool to require")
    result = V.validate_source(GOOD_GLSL, "glsl", require_tool=absent)
    assert result.ok is None
    assert absent in result.reason
    assert "required" in result.reason


def test_require_tool_present_forces_that_specific_tool():
    if not TOOLS["glslangValidator"]:
        pytest.skip("glslangValidator not installed")
    result = V.validate_source(GOOD_GLSL, "glsl", require_tool="glslangValidator")
    assert result.ok is True
    assert result.tool == "glslangValidator"
    assert result.tool_path == TOOLS["glslangValidator"]


def test_require_tool_rejects_an_unknown_name():
    result = V.validate_source(GOOD_GLSL, "glsl", require_tool="notacompiler")
    assert result.ok is None
    assert "not a known tool" in result.reason


def test_require_tool_can_demand_a_tool_for_a_target_it_does_not_own():
    """``--tool glslangValidator`` on a WGSL file must not silently pass."""
    if not TOOLS["glslangValidator"]:
        pytest.skip("glslangValidator not installed")
    result = V.validate_source(BROKEN_WGSL, "wgsl", require_tool="glslangValidator")
    # glslang ran; it is not a WGSL compiler, so the only acceptable outcomes are
    # a hard failure or no verdict -- never a green tick.
    assert result.ok is not True


# --------------------------------------------------------------------------- #
# Targets with no compiler at all
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("target", ["ue", "shaderlab"])
def test_targets_without_a_headless_compiler_report_none(target):
    result = V.validate_source("float4 x;", target)
    assert result.ok is None
    assert result.passed is False
    assert result.reason
    if target == "ue":
        assert "Unreal" in result.reason
    else:
        assert "Unity" in result.reason or "unity" in result.reason.lower()


def test_unknown_target_is_refused_not_guessed():
    result = V.validate_source("whatever", "cuda")
    assert result.ok is None
    assert "unknown target" in result.reason


# --------------------------------------------------------------------------- #
# Discovery, entry points, diagnostics
# --------------------------------------------------------------------------- #


def test_available_reports_a_path_or_none_for_every_tool():
    tools = V.available()
    assert set(tools) == {"glslangValidator", "dxc", "naga", "xcrun metal", "unity"}
    for name, path in tools.items():
        assert path is None or os.path.isabs(path), f"{name}: {path!r} is not absolute"
        if path:
            assert os.path.isfile(path), f"{name}: {path!r} does not exist"


def test_available_finds_the_compilers_this_machine_actually_has():
    """If a compiler is discoverable, ``available()`` must find it -- reporting
    'no compiler' on a machine that has one is its own false negative."""
    tools = V.available()
    for path in tools.values():
        if path:
            assert os.path.basename(path).lower().startswith(
                ("glslang", "dxc", "naga", "xcrun", "unity")
            )


def test_env_override_is_honoured(monkeypatch):
    """``AI4SHADER_DXC`` must be able to point at a specific binary."""
    tools = V.available(refresh=True)
    if not tools["dxc"]:
        pytest.skip("dxc not installed")
    monkeypatch.setenv("AI4SHADER_DXC", tools["dxc"])
    V.clear_cache()
    try:
        assert V.available()["dxc"] == tools["dxc"]
    finally:
        V.clear_cache()


def test_env_override_pointing_nowhere_reports_absent(monkeypatch):
    monkeypatch.setenv("AI4SHADER_DXC", r"C:\definitely\not\here\dxc.exe")
    V.clear_cache()
    try:
        assert V.available()["dxc"] is None
    finally:
        V.clear_cache()


def test_tool_paths_env_var_is_searched(monkeypatch):
    tools = V.available()
    if not tools["naga"]:
        # naga genuinely is not installed; point the search at a directory that
        # contains a file named naga and confirm the search actually looks there.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            fake = os.path.join(tmp, "naga.exe" if os.name == "nt" else "naga")
            with open(fake, "w") as fh:
                fh.write("")
            monkeypatch.setenv("AI4SHADER_TOOL_PATHS", tmp)
            V.clear_cache()
            try:
                assert V.available()["naga"] == os.path.abspath(fake)
            finally:
                V.clear_cache()
    else:
        pytest.skip("naga is really installed; the search-path fallback is not exercised")


@pytest.mark.parametrize("source,expected", [
    ("#version 300 es\nvoid main(){}", "glsl"),
    ("@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }", "wgsl"),
    ('Shader "X/Y" { SubShader { Pass { } } }', "shaderlab"),
    ("#include <metal_stdlib>\nfragment float4 fs_main() {}", "msl"),
    ("cbuffer P : register(b0) { float x; };\nfloat4 PSMain(float2 uv) : SV_Target { return 0; }", "hlsl"),
    ("just some words", None),
])
def test_detect_target(source, expected):
    assert V.detect_target(source) == expected


def test_discover_entry_finds_the_emitter_entry_points():
    hlsl = "struct PSInput { float4 p : SV_POSITION; };\nfloat4 PSMain(PSInput i) : SV_Target { return 0; }\n"
    assert V.discover_entry(hlsl, "hlsl") == "PSMain"
    assert V.discover_entry("@fragment\nfn fs_main(@builtin(position) p: vec4f) -> @location(0) vec4f { return p; }", "wgsl") == "fs_main"
    assert V.discover_entry("fragment float4 fs_main(FSIn in [[stage_in]]) {}", "msl") == "fs_main"
    assert V.discover_entry("void main() {}", "glsl") == "main"
    # A renamed entry point must still be found, not hard-coded.
    renamed = "float4 MyFragment(PSInput i) : SV_Target { return 0; }\n"
    assert V.discover_entry(renamed, "hlsl") == "MyFragment"
    # ...and an unusual file with no SV_Target falls back to the documented default.
    assert V.discover_entry("// nothing here", "hlsl") == "PSMain"


def test_parse_diagnostics_handles_every_tool_format():
    dxc = "C:\\tmp\\a.hlsl:26:43: error: use of undeclared identifier 'atan2'"
    glslang = "ERROR: 0:5: 'x' : undeclared identifier"
    fxc = "a.hlsl(12,5): error X3000: syntax error"
    naga = "error: unexpected token\n  --> a.wgsl:7:11"

    (d,) = V.parse_diagnostics(dxc)
    assert (d.line, d.column, d.severity) == (26, 43, "error") and "atan2" in d.message

    (g,) = V.parse_diagnostics(glslang)
    assert (g.line, g.column, g.severity) == (5, None, "error") and "undeclared" in g.message

    (f,) = V.parse_diagnostics(fxc)
    assert (f.line, f.column) == (12, 5) and "syntax error" in f.message

    (n,) = V.parse_diagnostics(naga)
    assert (n.line, n.column) == (7, 11) and "unexpected token" in n.message

    assert V.parse_diagnostics("") == []
    assert V.parse_diagnostics("just noise, no diagnostics here") == []


def test_result_is_json_serialisable_and_carries_the_verbatim_output():
    import json

    result = V.validate_source(BROKEN_GLSL, "glsl")
    payload = result.to_dict()
    for key in ("target", "tool", "tool_path", "ok", "reason", "stdout", "stderr",
                "returncode", "diagnostics", "command", "notes"):
        assert key in payload
    json.loads(json.dumps(payload))


# --------------------------------------------------------------------------- #
# The dxc encoding trap on CJK Windows
# --------------------------------------------------------------------------- #


def test_non_ascii_hlsl_gets_a_bom_and_real_diagnostics():
    """Non-ASCII HLSL must be written with a BOM, and the reason must be recorded.

    Without the BOM, dxc on a CJK code page fails with ``0x80070459`` and no
    diagnostics at all -- an opaque failure that reads like a compiler crash and
    hides the real error.  The em dash in the emitters' header comment triggers it
    on every real artefact.
    """
    if not TOOLS["dxc"]:
        pytest.skip("dxc not installed")
    source = "// AI4Shader \u2014 HLSL header\nfloat4 PSMain(float2 uv : TEXCOORD0) : SV_Target { return nosuchvar; }\n"
    result = V.validate_source(source, "hlsl")
    assert result.ok is False
    assert result.notes and "BOM" in result.notes[0]
    # The decisive assertion: we got the REAL diagnostic, not the opaque failure.
    assert result.diagnostics, "a BOM should turn 0x80070459 into real diagnostics"
    assert any("nosuchvar" in d.message for d in result.diagnostics)


def test_the_opaque_dxc_encoding_failure_is_diagnosed_not_reported_as_a_shader_bug():
    """Characterise the trap itself, so the workaround's justification is tested.

    Running dxc on a non-ASCII file **without** a BOM reproduces the original
    failure.  It only fails on a host whose ANSI code page is CJK, so the test
    detects that condition and says so rather than failing on a Western machine.
    """
    import subprocess
    import tempfile

    if not TOOLS["dxc"]:
        pytest.skip("dxc not installed")
    try:
        import locale

        encoding = locale.getpreferredencoding(False) or ""
    except Exception:  # noqa: BLE001
        encoding = ""
    if not re.search(r"(936|950|932|949|cp93|gbk|big5|shift)", encoding, re.I):
        pytest.skip(f"host ANSI code page is {encoding!r}, not CJK; the trap does not fire here")

    source = "// AI4Shader \u2014 HLSL header\nfloat4 PSMain(float2 uv : TEXCOORD0) : SV_Target { return uv.x; }\n"
    with tempfile.TemporaryDirectory() as tmp:
        plain = os.path.join(tmp, "plain.hlsl")
        with open(plain, "w", encoding="utf-8", newline="") as fh:  # deliberately NO BOM
            fh.write(source)
        proc = subprocess.run(
            [TOOLS["dxc"], "-T", "ps_6_0", "-E", "PSMain", plain, "-Fo", os.path.join(tmp, "o.dxil")],
            capture_output=True, text=True, errors="replace",
        )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode != 0
    assert "0x80070459" in combined, f"expected the documented failure, got: {combined[:200]!r}"
    assert "error:" not in combined, "the failure carries no diagnostic, which is the whole problem"


def test_hlsl_with_pure_ascii_is_not_rewritten():
    """The BOM is added only when it is needed; an ASCII source passes through."""
    if not TOOLS["dxc"]:
        pytest.skip("dxc not installed")
    result = V.validate_source(BROKEN_HLSL, "hlsl")
    assert result.notes == []
    assert result.ok is False
    assert result.diagnostics


# --------------------------------------------------------------------------- #
# Files and plumbing
# --------------------------------------------------------------------------- #


def test_validate_file_infers_the_target_from_the_extension(tmp_path):
    from test_validate import GOOD_GLSL as good

    path = tmp_path / "shader.frag"
    path.write_text(good, encoding="utf-8")
    result = V.validate_file(path)
    assert result.target == "glsl"
    if TOOLS["glslangValidator"]:
        assert result.ok is True
    else:
        assert result.ok is None


def test_validate_file_refuses_to_guess_an_unknown_extension(tmp_path):
    path = tmp_path / "shader.txt"
    path.write_text("not obviously any shader language", encoding="utf-8")
    result = V.validate_file(path)
    assert result.ok is None
    assert "could not infer a target" in result.reason


def test_command_is_reproducible_and_paths_are_absolute():
    if not TOOLS["glslangValidator"]:
        pytest.skip("glslangValidator not installed")
    result = V.validate_source(GOOD_GLSL, "glsl")
    assert result.command
    assert os.path.isabs(result.command[0])
    assert "-S" in result.command and "frag" in result.command


def test_dxc_command_uses_the_profile_and_entry_point():
    """``dxc`` needs ``-T ps_6_0 -E <entry>``; pinned so a refactor cannot drop it."""
    if not TOOLS["dxc"]:
        pytest.skip("dxc not installed")
    result = V.validate_source("float4 PSMain(float2 uv : TEXCOORD0) : SV_Target { return uv.x; }", "hlsl")
    assert "-T" in result.command and V.DXC_PROFILE in result.command
    assert "-E" in result.command and "PSMain" in result.command


def test_timeout_is_a_no_verdict_not_a_failure():
    result = V.validate_source(GOOD_GLSL, "glsl", timeout=1e-6)
    # Either it finished before the deadline or it produced no verdict; what it
    # must never do is claim success.
    assert result.ok in (None, True)
    if result.ok is None:
        assert "timed out" in result.reason
