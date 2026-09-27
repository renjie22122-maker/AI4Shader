"""Compile the emitters' **real output** with a **real compiler**.

This is the test that closes the gap every structural check leaves open. The Node suites
assert the emitted source is *shaped* like GLSL/WGSL/HLSL — balanced, no leaked type names,
no surviving aliases — and `packages/ir/test/wgsl.test.ts` goes further with a hand-written
scanner. None of that proves the output *is* the target language. Only a compiler does.

The chain under test is therefore the whole pipeline:

    examples/liquid-metal.shader.json
        -> apps/cli emit            (IR -> six targets)
        -> validate.validate_file   (target -> real compiler)
        -> a verdict, attributed to a named tool

Why it lives in the Python suite rather than the Node one: `validate.py` is where the
compiler discovery and argv construction live, and it is where a future `naga`/`metal`
integration will land. A Node test would have to duplicate that logic, and the duplicate is
what would rot.

## What is asserted, and what is deliberately not

`glsl` is asserted to **compile**, because `glslangValidator` is present on this machine and
a pass is a real verdict. The other targets are asserted to produce an **honest** verdict:
either a real pass, or `ok is None` with a reason naming the missing tool. What is forbidden
in every case is `ok is True` without a compiler having said so — a fabricated green is worse
than a documented gap, because it silently removes the reason to look.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai4shader import validate  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "apps" / "cli" / "dist" / "cli.js"
EXAMPLE = REPO / "examples" / "liquid-metal.shader.json"
#: A **surface** document: its graph reads `iNormal` / `iLightDir` / `iViewDir` / `iWorldPos`,
#: so every target emits a different entry point than the quad example above (a vertex stage
#: appears, and `uv` comes from the mesh instead of `gl_FragCoord`). Same pipeline, different
#: shape — which is the point of running both through it.
TOON = REPO / "examples" / "toon-surface.shader.json"
#: A **feedback** document: it reads the image channel as its own previous frame
#: (`render.feedback: true`), which is how a "wipe the fog off the glass" effect remembers where
#: the finger has been. Nothing about it changes the emitted *source* — the host binding is the
#: feature — which is exactly why the requirement is asserted at the file level below.
FOG = REPO / "examples" / "fog-wipe.shader.json"
#: The **scene** example: 39 nodes over five layers (water, reflection, lily pads, flower, haze).
#: Nothing in it is a new primitive — it is the composition test. If the catalog stops composing,
#: this document stops emitting, and the failure is a specific node rather than a general feeling
#: that the tool is weak.
POND = REPO / "examples" / "lotus-pond.shader.json"

#: Entry points the emitters declare. `discover_entry` should find these itself; the test
#: asserts it does, which is why they are listed here rather than passed in.
EXPECTED_ENTRY = {"glsl": "main", "hlsl": "PSMain", "msl": "fs_main", "wgsl": "fs_main"}


def _emit_to(tmp: Path, target: str, example: Path = EXAMPLE) -> Path:
    """Run the real CLI and return the file it wrote.

    The CLI is invoked as a subprocess rather than imported: emitting is the interface under
    test, and `apps/cli` is the thing users run.
    """
    # The bundle directory is named after the example, so two examples in one `tmp_path`
    # cannot collide and the "exactly one bundle" assertion below stays meaningful.
    out = tmp / f"out-{example.stem}-{target}"
    proc = subprocess.run(
        ["node", str(CLI), "emit", str(example), "--target", target, "--out", str(out), "--force"],
        capture_output=True,
        # Decode explicitly. `text=True` alone uses the platform locale codec, which on a
        # Windows machine with a CJK locale is GBK — and the CLI's report contains box-drawing
        # characters and node labels like "液态金属", so the reader thread dies with
        # UnicodeDecodeError and pytest reports a thread-exception warning instead of the real
        # result. Node always writes UTF-8; say so.
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO),
        timeout=180,
    )
    assert proc.returncode == 0, f"emit --target {target} failed:\n{proc.stdout}\n{proc.stderr}"

    slug = json.loads(example.read_text(encoding="utf-8"))["title"]
    # The CLI slugifies the title; find the produced directory rather than reimplementing it.
    dirs = [d for d in out.iterdir() if d.is_dir()]
    assert len(dirs) == 1, f"expected one bundle directory, got {[d.name for d in dirs]} (title={slug})"
    files = [f for f in dirs[0].iterdir() if f.is_file()]
    assert files, f"bundle {dirs[0]} is empty"
    return dirs[0]


#: Extension per target, matching what `validate_file` infers.
EXT = {"glsl": "shader.frag", "wgsl": "shader.wgsl", "hlsl": "shader.hlsl", "msl": "shader.metal"}


def _emit_doc(tmp: Path, doc: dict, target: str, name: str = "inline") -> Path:
    """Emit a document written by the test itself, through the real CLI.

    Same pipeline as `_emit_to`, for documents that do not belong in `examples/`: a fixture whose
    whole point is that its text arrives at runtime.
    """
    doc_path = tmp / f"{name}.shader.json"
    doc_path.write_text(json.dumps(doc), encoding="utf-8")
    out = tmp / f"out-{name}-{target}"
    proc = subprocess.run(
        ["node", str(CLI), "emit", str(doc_path), "--target", target, "--out", str(out), "--force"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO),
        timeout=180,
    )
    assert proc.returncode == 0, f"emit --target {target} failed:\n{proc.stdout}\n{proc.stderr}"
    dirs = [d for d in out.iterdir() if d.is_dir()]
    assert len(dirs) == 1, f"expected one bundle directory, got {[d.name for d in dirs]}"
    return dirs[0]


def _escape_hatch_doc() -> dict:
    """A document whose effect lives in a `custom_glsl` body rather than in the catalog.

    Every alias the rewriter has to lower appears in this snippet: `mod_` (floor semantics, and
    `fmod` truncates), `mix_`, `fract_`, `atan2_` (scalar-only in HLSL), `tex2D_`, and
    `mat2_mul_` — which is lowered **vector-first** on HLSL, because HLSL's matrix constructors are
    row-major and `matrix * vector` does not compile there at all.

    That last one is the reason this test exists. The escape hatch had never been given to a real
    compiler: every catalog body is compiled by the other tests, but a `custom_glsl` body is
    written at runtime, and a body that writes `rot2(a) * v` emits cleanly on four targets and
    fails on two.
    """
    return {
        "irVersion": "1.0.0",
        "id": "escape-hatch",
        "title": "escape hatch",
        "spec": {
            "intent": "exercise every rewriter alias from a runtime body",
            "category": "test",
            "tags": ["test"],
            "features": [],
        },
        "stage": "fragment",
        "graph": {
            "nodes": [
                {"id": "n_uv", "kind": "uv"},
                {
                    "id": "n_custom",
                    "kind": "custom_glsl",
                    "body": (
                        "vec2 q_${node_id} = fract_(${uv} * 6.0);"
                        " vec2 r_${node_id} = mat2_mul_(rot2(0.5), q_${node_id} - vec2(0.5, 0.5));"
                        " float a_${node_id} = atan2_(q_${node_id}.y, q_${node_id}.x);"
                        " return vec4(mix_(q_${node_id}.x, mod_(q_${node_id}.x, 0.5), a_${node_id}), r_${node_id}, 1.0);"
                    ),
                    "inputs": {"uv": {"node": "n_uv", "port": "uv"}},
                },
                {"id": "n_out", "kind": "output", "inputs": {"color": {"node": "n_custom", "port": "value"}}},
            ],
            "edges": [],
        },
        "render": {
            "resolution": [512, 512],
            "qualityTier": "medium",
            "maxIterations": 8,
            "transparent": False,
        },
        "targets": ["glsl", "wgsl", "hlsl", "msl", "shaderlab", "ue"],
    }


@pytest.mark.parametrize("target,compiler", [("glsl", "glslangValidator"), ("hlsl", "dxc")])
def test_independent_custom_layers_compile_with_reused_local_names(tmp_path: Path, target: str, compiler: str) -> None:
    if not validate.available(refresh=True).get(compiler):
        pytest.skip(f"{compiler} is not installed")
    doc = _escape_hatch_doc()
    doc["graph"]["nodes"] = [
        {"id": "drops", "kind": "custom_glsl", "body":
         "float hi = 0.3; vec2 nrm = vec2(0.1, 0.2); return vec4(nrm, hi, 0.4);"},
        {"id": "trails", "kind": "custom_glsl", "inputs": {"a": {"node": "drops", "port": "value"}},
         "body": "float hi = 0.6; vec2 nrm = vec2(0.3, 0.4); return ${a} + vec4(nrm, hi, 0.1);"},
        {"id": "out", "kind": "output", "inputs": {"color": {"node": "trails", "port": "value"}}},
    ]
    bundle = _emit_doc(tmp_path, doc, target, name="custom-layers")
    src = (bundle / EXT[target]).read_text(encoding="utf-8-sig")
    for entry in validate.entry_points(src, target):
        result = validate.validate_source(src, target, entry=entry)
        if result.ok is None:
            pytest.skip(result.reason)
        assert result.ok is True, f"{compiler}: {result.stdout}{result.stderr}"


def _escape_hatch_surface_doc() -> dict:
    """The same escape hatch, reading a **surface** host symbol.

    A `custom_glsl` body is free text, so it can name `iNormal` even though no node in the graph
    asked for a surface input. Until this was handled, that document emitted the quad header — which
    has no `iNormal` — with zero diagnostics, and failed to compile. `needs.surface` is now computed
    from the body as well as from the schemas, so the surface entry point is emitted and the symbol
    exists.
    """
    doc = _escape_hatch_doc()
    doc["id"] = "escape-hatch-surface"
    for node in doc["graph"]["nodes"]:
        if node["id"] == "n_custom":
            node["body"] = (
                "float ndl_${node_id} = max(dot(normalize(iNormal), normalize(iLightDir)), 0.0);"
                " vec2 q_${node_id} = fract_(${uv} * 4.0);"
                " return vec4(mix_(vec3(0.1, 0.1, 0.2), vec3(1.0, 0.9, 0.7), ndl_${node_id})"
                " * (0.5 + 0.5 * q_${node_id}.x), 1.0);"
            )
    return doc


def test_a_custom_body_can_read_surface_symbols(tmp_path: Path) -> None:
    """The body's host symbols count towards the document's capabilities.

    Emitted on every target, then compiled where a compiler exists: the point is not that the
    source *looks* right but that `iNormal` is a declared identifier on the quad path too.
    """
    doc = _escape_hatch_surface_doc()
    for target in SURFACE_TARGETS:
        bundle = _emit_doc(tmp_path, doc, target, name="escape-hatch-surface")
        src = (bundle / HOST_SOURCE_FILE[target]).read_text(encoding="utf-8-sig")
        assert "iNormal" in src, f"{target}: the body reads iNormal but the source never mentions it"

    if validate.available(refresh=True).get("glslangValidator"):
        bundle = _emit_doc(tmp_path, doc, "glsl", name="escape-hatch-surface")
        result = validate.validate_file(bundle / "shader.frag")
        assert result.ok is True, (
            "glslangValidator rejected a document whose custom body reads iNormal — that means the "
            "surface entry point was not emitted for it.\n"
            f"  output: {result.stdout}{result.stderr}"
        )


@pytest.mark.parametrize("target", ["glsl", "wgsl", "hlsl", "msl", "shaderlab", "ue"])
def test_the_escape_hatch_emits_on_every_target(target: str, tmp_path: Path) -> None:
    """A runtime-authored body goes through the same pipeline as a catalog body."""
    bundle = _emit_doc(tmp_path, _escape_hatch_doc(), target, name="escape-hatch")
    src = (bundle / HOST_SOURCE_FILE[target]).read_text(encoding="utf-8-sig")
    code = _strip_comments(src)
    # The aliases must not survive into any target: each one is a name that means something else
    # (or nothing) in the language being emitted. Comments are stripped first because the shared
    # library's prose names the aliases it is explaining, and a check that fails on documentation
    # is a check somebody deletes.
    for alias in ("mod_", "mix_", "fract_", "atan2_", "mat2_mul_"):
        assert alias not in code, f"{target}: the alias {alias} survived into the emitted code"


def _strip_comments(src: str) -> str:
    """Remove `//` and `/* */` comments.

    Regex rather than a parser, and that is the honest description: it is used only to ask "does
    this *identifier* still appear in code", where the only realistic false positive is a comment,
    which is what it removes. A string literal containing `//` would be stripped too, and the
    emitters do not produce one.
    """
    src = re.sub(r"/\*[\s\S]*?\*/", " ", src)
    return re.sub(r"//[^\n]*", " ", src)


def test_the_escape_hatch_compiles_with_both_real_compilers(tmp_path: Path) -> None:
    """GLSL and HLSL, both stages — the check that was missing until the transpose bug.

    `dxc` is what caught `matrix * vector` (it does not compile), and what proved the argument
    order of `mul` (disassembly: `mul(m, v)` is the transpose of the GLSL result). Neither is
    visible to a structural check.
    """
    doc = _escape_hatch_doc()

    if validate.available(refresh=True).get("glslangValidator"):
        bundle = _emit_doc(tmp_path, doc, "glsl", name="escape-hatch")
        result = validate.validate_file(bundle / "shader.frag")
        assert result.ok is True, (
            f"glslangValidator rejected the escape-hatch GLSL.\n"
            f"  output: {result.stdout}{result.stderr}"
        )

    if not validate.available(refresh=True).get("dxc"):
        pytest.skip("dxc is not installed here")
    bundle = _emit_doc(tmp_path, doc, "hlsl", name="escape-hatch")
    src = (bundle / "shader.hlsl").read_text(encoding="utf-8-sig")
    for entry in validate.entry_points(src, "hlsl"):
        result = validate.validate_source(src, "hlsl", entry=entry)
        if result.ok is None:
            pytest.skip(f"dxc could not be used here: {result.reason}")
        assert result.ok is True, (
            f"dxc rejected {entry} of the escape-hatch HLSL.\n"
            f"  command: {' '.join(result.command)}\n  stdout: {result.stdout}\n  stderr: {result.stderr}"
        )
        assert result.warning_count == 0, (
            f"{entry}: dxc emitted {result.warning_count} warning(s): "
            + "; ".join(f"{d.line}: {d.message}" for d in result.diagnostics if d.severity == "warning")
        )


@pytest.mark.parametrize("target", ["glsl", "wgsl", "hlsl", "msl"])
def test_emitted_bundle_validates_honestly(target: str, tmp_path: Path) -> None:
    bundle = _emit_to(tmp_path, target)
    src = bundle / EXT[target]
    assert src.is_file(), f"{target}: expected {EXT[target]} in {bundle}"

    result = validate.validate_file(src)

    # Invariant 1: a verdict of True must name the tool that produced it.
    if result.ok is True:
        assert result.tool_path and os.path.isfile(result.tool_path), (
            f"{target}: ok=True but no compiler was actually run ({result.reason})"
        )
        assert result.returncode == 0, f"{target}: ok=True with returncode {result.returncode}"

    # Invariant 2: no verdict must carry a reason that names the missing tool. An unexplained
    # `None` is indistinguishable from a bug in the discovery logic.
    if result.ok is None:
        assert result.reason, f"{target}: ok=None with no reason"
        assert "no " in result.reason.lower() or "not found" in result.reason.lower(), (
            f"{target}: ok=None but the reason does not name a missing tool: {result.reason!r}"
        )

    # Invariant 3: a failure must carry the compiler's own words, not a paraphrase.
    if result.ok is False:
        assert result.stderr or result.stdout, f"{target}: ok=False with no compiler output"
        assert result.returncode not in (0, None), f"{target}: ok=False with returncode {result.returncode}"

    # And it must always be honest about which target it decided it was looking at. No escape
    # hatch here: a mismatch on `glsl` would be the most serious kind, since that is the target
    # with a real compiler attached.
    assert result.target == target, f"{target}: validate_file decided the file was {result.target!r}"


def test_glsl_compiles_with_glslang(tmp_path: Path) -> None:
    """The one target where a pass is available on this machine.

    If this fails, either the emitter regressed or `glslangValidator` disappeared — and the
    assertion message distinguishes the two by reporting what `available()` found.
    """
    tools = validate.available(refresh=True)
    if not tools.get("glslangValidator"):
        pytest.skip("glslangValidator is not installed here")

    bundle = _emit_to(tmp_path, "glsl")
    result = validate.validate_file(bundle / "shader.frag")

    assert result.ok is True, (
        f"glslangValidator rejected the emitted GLSL.\n"
        f"  tool      : {result.tool_path}\n"
        f"  command   : {' '.join(result.command)}\n"
        f"  returncode: {result.returncode}\n"
        f"  stdout    : {result.stdout}\n"
        f"  stderr    : {result.stderr}\n"
        f"  diagnostics: {[d.to_dict() for d in result.diagnostics]}"
    )
    assert result.error_count == 0


def test_a_broken_glsl_shader_is_rejected_by_the_same_path(tmp_path: Path) -> None:
    """The control for the test above.

    Without it, `test_glsl_compiles_with_glslang` would still pass if `validate_file` always
    returned `ok=True` — which is exactly the failure mode this whole module exists to
    prevent. A validator that cannot reject is not a validator.
    """
    if not validate.available(refresh=True).get("glslangValidator"):
        pytest.skip("glslangValidator is not installed here")

    bad = tmp_path / "bad.frag"
    bad.write_text(
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 fragColor;\n"
        "void main() { fragColor = vec4(undefinedSymbol, 1.0); }\n",
        encoding="utf-8",
    )
    result = validate.validate_file(bad)

    assert result.ok is False, f"a shader calling an undefined function must not pass: {result.reason}"
    assert result.error_count >= 1
    assert result.returncode not in (0, None)


def test_entry_point_discovery_matches_the_emitters(tmp_path: Path) -> None:
    """`discover_entry` must find the real entry names, not just plausible ones.

    Passing the wrong `-E` to a compiler produces an error that reads like a shader bug
    ("entry point not found"), so this is worth pinning to the actual emitted sources rather
    than to a hard-coded list.
    """
    for target, expected in EXPECTED_ENTRY.items():
        bundle = _emit_to(tmp_path, target)
        src = (bundle / EXT[target]).read_text(encoding="utf-8-sig")
        got = validate.discover_entry(src, target)
        assert got == expected, f"{target}: discover_entry found {got!r}, emitters declare {expected!r}"


def test_hlsl_has_no_scalar_only_shim_taking_a_vector(tmp_path: Path) -> None:
    """Catch the class of bug dxc found, without needing dxc to find it again.

    The emitted HLSL declares a shim family because GLSL's `mod`/`mix`/`atan2` do not lower
    directly onto HLSL's:

      * `mod` -> `fmod` truncates instead of flooring (a real colour bug), so `wf_mod` exists;
      * HLSL's `atan2` is **scalar-only**, so `atan2(float2(y, x))` — which is valid GLSL and
        valid WGSL — is a compile error here.

    The atan2 variant shipped and was only caught once dxc could actually read the file (see
    the BOM note in `validate.py`). This test is deliberately a *source* check rather than a
    compile: it names the exact mistake, runs without a compiler, and does not depend on dxc's
    encoding behaviour. `test_emitted_bundle_validates_honestly` above remains the end-to-end
    check for when a compiler is available.
    """
    bundle = _emit_to(tmp_path, "hlsl")
    src = (bundle / "shader.hlsl").read_text(encoding="utf-8-sig")

    # `atan2` must be called with two scalars. A `float2`/`float3`/`float4` argument is the bug.
    bad_atan2 = re.findall(r"atan2\s*\(\s*(?:float[234]|[A-Za-z_]\w*\s*\()", src)
    assert not bad_atan2, (
        "atan2 is scalar-only in HLSL, but the emitted source passes something vector-shaped:\n"
        + "\n".join(line for line in src.splitlines() if "atan2" in line)
    )

    # And no shim may be scalar-only while the library calls it with vectors: that compiles by
    # silently truncating, which is worse than failing.
    #
    # This is a *weak* check and is kept only as a cheap early signal. It cannot be made strong
    # without a type checker: the call that actually broke was
    # `wf_mix(lo, hi, step(float3(0.04045), c))`, whose third argument is a *call* returning
    # float3, so nothing here can tell what it evaluates to. The authoritative check is the
    # `warning_count == 0` assertion in `test_hlsl_compiles_when_a_bom_is_available` — dxc knows
    # the types, and it had been reporting the problem in every emitted file.
    for shim in ("wf_mod", "wf_mix"):
        scalar_only = re.search(rf"^\s*float\s+{shim}\s*\(\s*float\s", src, re.M)
        if not scalar_only:
            continue  # not emitted for this target at all
        assert re.search(rf"^\s*float[234]\s+{shim}\s*\(", src, re.M), (
            f"{shim} is declared scalar-only, so every vector call site silently truncates "
            f"(dxc: 'implicit truncation of vector type')"
        )


def test_hlsl_compiles_when_a_bom_is_available(tmp_path: Path) -> None:
    """End-to-end HLSL check, with the encoding caveat made explicit.

    On a host whose ANSI code page cannot represent the UTF-8 the emitters write (this one is
    code page 936), dxc fails with `0x80070459` and **no diagnostics** unless the source has a
    BOM. `validate.py` adds one, so a `False` here is a real shader error rather than an
    encoding artefact — which is precisely the distinction that hid the `atan2` bug.
    """
    if not validate.available(refresh=True).get("dxc"):
        pytest.skip("dxc is not installed here")

    bundle = _emit_to(tmp_path, "hlsl")
    result = validate.validate_file(bundle / "shader.hlsl")

    if result.ok is None:
        pytest.skip(f"dxc could not be used here: {result.reason}")

    assert result.ok is True, (
        f"dxc rejected the emitted HLSL.\n"
        f"  command   : {' '.join(result.command)}\n"
        f"  returncode: {result.returncode}\n"
        f"  stdout    : {result.stdout}\n"
        f"  stderr    : {result.stderr}\n"
        f"  diagnostics: {[d.to_dict() for d in result.diagnostics]}"
    )
    assert result.error_count == 0

    # Warnings are not noise on this target, and asserting only on errors is how a real bug
    # survived here for a while.
    #
    # dxc's `implicit truncation of vector type` is a *semantic* warning: it fires when a
    # vector is passed where a scalar is expected, and the compiler then uses the first
    # component. GLSL's `mix(a, b, t)` is declared for vector `t` and mixes per component, so
    # `srgb2linear`'s `mix_(lo, hi, step(vec3(0.04045), c))` converted each channel by its own
    # threshold in GLSL and all three by the *red* channel's threshold in HLSL — a colour shift
    # on two of the six targets, with no error to point at it.
    #
    # The check that should have caught it (`test_no_shim_is_declared_scalar_only`) looked for a
    # vector overload *existing*, which one did; it never looked at the arity of `t`. The
    # compiler had been saying so in every emitted file, and this assertion is what stops the
    # next one from being ignored.
    assert result.warning_count == 0, (
        f"dxc emitted {result.warning_count} warning(s) for the emitted HLSL. On this target a "
        f"warning is a correctness signal (implicit truncation changes the image), not noise:\n"
        + "\n".join(f"  {d.line}: {d.message}" for d in result.diagnostics if d.severity == "warning")
    )


def test_emitted_source_is_self_describing_about_its_target(tmp_path: Path) -> None:
    """`detect_target` must agree with the extension for every emitted file.

    A disagreement means the emitter writes something that does not look like its own target,
    which is the earliest possible signal of a lowering mistake.
    """
    for target in EXT:
        bundle = _emit_to(tmp_path, target)
        src = (bundle / EXT[target]).read_text(encoding="utf-8-sig")
        detected = validate.detect_target(src)
        assert detected == target, f"{EXT[target]}: content looks like {detected!r}, expected {target!r}"


# --------------------------------------------------------------------------- #
# Surface documents (a graph that reads the surface inputs, not a quad)
#       and feedback documents (a graph that reads its own previous frame)
# --------------------------------------------------------------------------- #

SURFACE_TARGETS = ["glsl", "wgsl", "hlsl", "msl", "shaderlab", "ue"]

#: The two examples that ask something of the *host*, not just of the language. Both are
#: emitted on every target here because "it works everywhere" is exactly the claim a user
#: makes when they ask for 三渲二 or for glass they can wipe.
HOST_FEATURE_EXAMPLES = [
    pytest.param(TOON, id="surface"),
    pytest.param(FOG, id="feedback"),
    pytest.param(POND, id="scene"),
]


@pytest.mark.parametrize("example", HOST_FEATURE_EXAMPLES)
@pytest.mark.parametrize("target", SURFACE_TARGETS)
def test_a_host_feature_document_emits_on_every_target(
    target: str, example: Path, tmp_path: Path
) -> None:
    """Both host-feature examples have to *emit* on all six targets, with zero errors.

    Cheap and worth having as its own test: these are the documents a user asks for by name
    (三渲二, 可擦的水雾玻璃), and a target that refuses one — or grows a diagnostic — should fail
    here rather than inside a compiler's stderr three tests later.
    """
    bundle = _emit_to(tmp_path, target, example)
    files = sorted(f.name for f in bundle.iterdir() if f.is_file())
    assert files, f"{target}: the bundle is empty"
    # The render spec is written next to every target, and a bundle that lost its params is a
    # document nobody can tune — which is half the product.
    assert "params.json" in files, f"{target}: no params.json in {files}"


@pytest.mark.parametrize("target", SURFACE_TARGETS)
def test_a_feedback_document_states_the_host_requirement_on_every_target(
    target: str, tmp_path: Path
) -> None:
    """`render.feedback` is a request to the *host*, and the host is not always this repo.

    The source is identical whether the image channel holds the previous frame or a static
    image; bind the wrong one and the shader compiles, runs at full speed, and produces a still
    frame with nothing to point at. So the requirement travels **inside every emitted file** —
    Unity, UE and Metal hosts never read our docs, they read the file they were handed.
    """
    bundle = _emit_to(tmp_path, target, FOG)
    src = _host_source(bundle, target)
    assert "FEEDBACK" in src, f"{target}: the emitted file does not state the feedback requirement"
    assert "ping-pong" in src, f"{target}: the requirement does not say what the host must do"

    # And the quad example must NOT carry it: a note that appears unconditionally is noise, and
    # noise is how a real requirement gets ignored.
    plain = _emit_to(tmp_path, target)  # liquid-metal
    plain_src = _host_source(plain, target)
    assert "FEEDBACK" not in plain_src, f"{target}: a document without feedback carries its note"


#: Where each target's *shader text* lives inside a bundle. ShaderLab and UE do not write
#: `shader.<ext>`; they write the files their own tool wants, which is the whole shape of those
#: two targets.
HOST_SOURCE_FILE = {
    "glsl": "shader.frag",
    "wgsl": "shader.wgsl",
    "hlsl": "shader.hlsl",
    "msl": "shader.metal",
    "shaderlab": "Shader.shader",
    "ue": "ue/CustomNode.hlsl",
}


def _host_source(bundle: Path, target: str) -> str:
    path = bundle / HOST_SOURCE_FILE[target]
    assert path.is_file(), f"{target}: expected {HOST_SOURCE_FILE[target]} in {bundle}"
    return path.read_text(encoding="utf-8-sig")


def test_surface_entry_points_are_found_per_stage(tmp_path: Path) -> None:
    """Both stages of the emitted HLSL pair are named, and each gets its own profile.

    `discover_entry` answers "which fragment entry point" and returns `PSMain`, which is the
    right answer to that question and the wrong one to "what has to compile". Before this, the
    surface HLSL's vertex stage had never been given to a compiler by any test — and the vertex
    stage is where the surface contract (interpolated world position / normal / uv) lives.
    """
    bundle = _emit_to(tmp_path, "hlsl", TOON)
    src = (bundle / "shader.hlsl").read_text(encoding="utf-8-sig")

    entries = validate.entry_points(src, "hlsl")
    assert entries == ["PSMain", "VSMain"], f"expected both stages, got {entries}"

    # And the profile has to follow the entry point: `-T ps_6_0 -E VSMain` is a dxc error, not
    # a verification of the vertex stage.
    assert validate.dxc_profile("VSMain") == "vs_6_0"
    assert validate.dxc_profile("PSMain") == "ps_6_0"

    # The quad already had a `VSMain` — it draws the fullscreen triangle from `SV_VertexID`, so
    # "there are two entry points" is not what makes this document a surface shader. What does
    # is that the vertex stage now consumes a **mesh**: attributes in, world position / normal
    # out. Asserting the distinction rather than the count is what keeps this test from passing
    # on a bundle that quietly fell back to the quad path.
    assert "POSITION" in src and "NORMAL" in src, "the surface vertex stage takes no mesh attributes"
    assert re.search(r"VSOutput\s+VSMain\s*\(\s*float3\s+\w+\s*:\s*POSITION", src), (
        "VSMain does not take a POSITION attribute; the surface vertex contract changed"
    )
    assert re.search(r"float3\s+worldPos\s*:\s*TEXCOORD\d", src), "worldPos is no longer interpolated"
    assert re.search(r"float3\s+normal\s*:\s*TEXCOORD\d", src), "the normal is no longer interpolated"


@pytest.mark.parametrize("example", HOST_FEATURE_EXAMPLES)
def test_glsl_compiles_with_glslang(example: Path, tmp_path: Path) -> None:
    """The emitted GLSL fragment shader, compiled by the real compiler.

    Parametrised over the two host-feature examples because they exercise different halves of the
    emitter: the surface one adds interpolated inputs and a camera uniform, the feedback one adds
    the channel readback and the `make_vec4`-as-state idiom.
    """
    if not validate.available(refresh=True).get("glslangValidator"):
        pytest.skip("glslangValidator is not installed here")

    bundle = _emit_to(tmp_path, "glsl", example)
    result = validate.validate_file(bundle / "shader.frag")

    if result.ok is None:
        pytest.skip(f"glslangValidator could not be used here: {result.reason}")

    assert result.ok is True, (
        f"glslangValidator rejected the emitted GLSL ({example.name}).\n"
        f"  command   : {' '.join(result.command)}\n"
        f"  returncode: {result.returncode}\n"
        f"  output    : {result.stdout}{result.stderr}"
    )
    assert result.error_count == 0


@pytest.mark.parametrize("example", HOST_FEATURE_EXAMPLES)
def test_hlsl_compiles_both_stages_with_dxc(example: Path, tmp_path: Path) -> None:
    """`VSMain` **and** `PSMain`, each against its own profile.

    This is the strongest automated check the emitters have: dxc type-checks the vertex output
    struct against the pixel input struct, so a shader whose interpolation contract is internally
    inconsistent cannot pass. Zero *warnings* is asserted as well — dxc's `implicit truncation of
    vector type` is a semantic change, not noise.

    It earns its keep: a `vec4(0.0)` scalar splat — valid GLSL, valid WGSL, valid MSL, a hard
    error in HLSL — reached a shipped catalog body, passed every structural test, and was caught
    by exactly this check the first time the feedback example was compiled.
    """
    if not validate.available(refresh=True).get("dxc"):
        pytest.skip("dxc is not installed here")

    bundle = _emit_to(tmp_path, "hlsl", example)
    src = (bundle / "shader.hlsl").read_text(encoding="utf-8-sig")
    entries = validate.entry_points(src, "hlsl")
    assert entries, f"{example.name}: no entry points found in the emitted HLSL"

    for entry in entries:
        result = validate.validate_source(src, "hlsl", entry=entry)
        if result.ok is None:
            pytest.skip(f"dxc could not be used here: {result.reason}")
        assert result.ok is True, (
            f"dxc rejected {entry} of {example.name}.\n"
            f"  command   : {' '.join(result.command)}\n"
            f"  returncode: {result.returncode}\n"
            f"  stdout    : {result.stdout}\n"
            f"  stderr    : {result.stderr}"
        )
        assert result.error_count == 0, f"{entry}: {[d.to_dict() for d in result.diagnostics]}"
        assert result.warning_count == 0, (
            f"{entry}: dxc emitted {result.warning_count} warning(s); on this target a warning "
            f"is a correctness signal:\n"
            + "\n".join(f"  {d.line}: {d.message}" for d in result.diagnostics if d.severity == "warning")
        )


def test_ue_surface_pins_match_the_ir_symbols(tmp_path: Path) -> None:
    """UE has no headless compiler, so its surface contract is checked where it *is* checkable.

    The Custom node's real interface is its pin list, and for a surface document that list has
    to carry the four host inputs the IR promises — with labels naming the UE node that
    supplies each. This is a sidecar check, not a compile: it proves the wiring instructions a
    user follows are present and complete, and claims nothing about UE accepting the HLSL.
    """
    bundle = _emit_to(tmp_path, "ue", TOON)
    pins = json.loads((bundle / "ue" / "CustomNode.inputs.json").read_text(encoding="utf-8"))
    by_name = {p["name"]: p for p in pins}

    for name in ("InWorldPos", "InNormal", "InCameraPos", "InLightDir"):
        assert name in by_name, f"the surface pin {name} is missing from {sorted(by_name)}"
        assert by_name[name]["type"] == "float3", f"{name} should be a float3 pin"
        assert by_name[name]["label"], f"{name} has no label telling the user what to plug in"

    # The tunable uniforms have to reach the pins as well, or the material cannot be tuned in
    # UE at all — the Custom node is the only place they exist on this target.
    doc = json.loads(TOON.read_text(encoding="utf-8"))
    exposed = [
        p["name"]
        for node in doc["graph"]["nodes"]
        for p in (node.get("params") or {}).values()
        if isinstance(p, dict) and p.get("kind") == "param"
    ]
    assert exposed, "the example no longer exposes any uniform"
    missing = [n for n in exposed if n not in by_name]
    assert not missing, f"uniforms with no UE pin: {missing}"
