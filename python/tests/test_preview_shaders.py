"""Compile the **preview's own** hand-written shaders with a real compiler.

Every other compile check in this repository covers what the *emitter* produced. The workbench's
3D pass is different: its material, sky and vertex stages are GLSL written by hand inside
`apps/web/src/preview/surface.ts`, handed to the driver as JavaScript template literals. Nothing
in the Node suites compiles them — `RecordingGl` records `shaderSource` and answers
`COMPILE_STATUS` from a marker string, which is exactly the right thing for testing the *host
contract* and exactly the wrong thing for testing the *GLSL*.

That gap has teeth, because this file has two ways to be broken that no structural check sees:

  * the shader is inside a JavaScript template literal, so a stray backtick in a GLSL comment
    terminates the literal and the projector fails with a parse error pointing somewhere else
    entirely (a non-ASCII character two lines later). That happened twice while the pattern
    mapping was being fixed;
  * `CLAMP_TO_EDGE` versus `REPEAT`, an offset, or a mistyped `.rgb` swizzle all compile perfectly
    and render a plausible picture. Those are covered by `apps/web/test/pattern-mapping.test.ts`,
    not here. This file answers one question only: **is it valid GLSL ES 3.00?**

The sources are read out of `apps/web/dist/preview/surface.js` — the projected file the browser
actually loads — rather than out of the TypeScript, so what is compiled is what runs.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai4shader import validate  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SURFACE = REPO / "apps" / "web" / "dist" / "preview" / "surface.js"

#: Template-literal name -> the glslangValidator stage flag for it.
STAGES = {
    "VERT": "vert",
    "GRAPH_VERT": "vert",
    "SKY_VERT": "vert",
    "SKY_FRAG": "frag",
    "MATERIAL_FRAG": "frag",
}


def _template(name: str, text: str) -> str:
    """The body of `const NAME = \\`...\\`` from the projected module.

    A plain scan for the next backtick, which is correct *because* these shaders contain no
    `${}` interpolation and no escaped backticks — and that property is asserted below rather
    than assumed, since it is the exact thing that broke twice.
    """
    start = text.find(f"const {name} = `")
    assert start != -1, f"{name} is no longer a template literal in {SURFACE.name}"
    body_start = text.index("`", start) + 1
    body_end = text.index("`", body_start)
    return text[body_start:body_end]


@pytest.fixture(scope="module")
def shaders() -> dict[str, str]:
    if not SURFACE.exists():
        pytest.skip(f"{SURFACE} is missing; run `node scripts/project-ts.ts` first")
    text = SURFACE.read_text(encoding="utf-8")
    return {name: _template(name, text) for name in STAGES}


def test_every_preview_shader_is_present_and_self_contained(shaders: dict[str, str]) -> None:
    """Five stages, each a complete ES 3.00 translation unit with no interpolation left in it."""
    assert set(shaders) == set(STAGES)
    for name, src in shaders.items():
        assert src.startswith("#version 300 es"), f"{name} lost its version directive"
        assert "void main()" in src, f"{name} has no entry point"
        # A surviving `${` would mean the template literal was assembled from parts and this
        # extraction is reading half a shader; glslang would then fail for the wrong reason.
        assert "${" not in src, f"{name} still contains a template interpolation"
        assert re.search(r"\bprecision\s+\w+\s+float\s*;", src), f"{name} has no float precision"


def test_hand_written_shaders_compile_with_glslang(shaders: dict[str, str], tmp_path: Path) -> None:
    """The verdict this file exists for: a real compiler accepts the real sources."""
    glslang = validate.available(refresh=True).get("glslangValidator")
    if not glslang:
        pytest.skip("glslangValidator is not installed here")

    for name, src in shaders.items():
        stage = STAGES[name]
        path = tmp_path / f"{name.lower()}.{stage}"
        path.write_text(src, encoding="utf-8", newline="")
        result = subprocess.run(
            [glslang, "-S", stage, str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        combined = f"{result.stdout}\n{result.stderr}"
        assert result.returncode == 0 and "ERROR:" not in combined, (
            f"glslangValidator rejected the preview's {name} stage.\n"
            f"This shader is hand-written GLSL in apps/web/src/preview/surface.ts and is what the\n"
            f"3D preview runs; a rejection here means the panel's 3D view cannot compile at all.\n"
            f"{combined.strip()}"
        )
