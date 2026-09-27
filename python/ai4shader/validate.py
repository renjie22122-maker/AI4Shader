"""Real compiler validation -- subprocess wrappers around real shader compilers.

The whole point of this module is to replace structural guessing with a real
compiler's verdict.  ``docs/IMPLEMENTATION.md`` section 4.2 lists what could not
be verified in this repository; several of those items are exactly what this
module fixes.  It follows DESIGN.md section 5.4 ("honest degradation"): **an
absent compiler is not a pass.**  When a tool is missing,
:meth:`ValidationResult.ok` is ``None`` and ``reason`` says why -- never ``True``.
A false green would defeat the module's entire purpose.

Tools, and what each target needs
---------------------------------
=============  ====================  ==========================  =====================
target         tool                  invocation                  notes
=============  ====================  ==========================  =====================
``glsl``       ``glslangValidator``  ``-S frag <file>``          GLSL ES 3.00
``hlsl``       ``dxc``               ``-T ps_6_0 -E <entry>``    entry from the source
``wgsl``       ``naga``              ``<file>``                  file argument only
``msl``        ``xcrun metal``       ``-sdk macosx metal -c``    macOS only
``shaderlab``  Unity                 ``-batchmode -executeMethod``  needs Unity
``ue``         --                    --                          cannot be compiled
=============  ====================  ==========================  =====================

Tool discovery
--------------
``shutil.which`` alone is not enough: on Windows, ``glslangValidator`` and
``dxc`` ship inside SDKs that are frequently *not* on ``PATH``.  This module also
probes the conventional install roots (Vulkan SDK, Windows Kits, the Android SDK
emulator bundle, ``~/.cargo/bin``) and honours per-tool environment overrides
(``AI4SHADER_DXC``, ``AI4SHADER_GLSLANG``, ``AI4SHADER_NAGA``,
``AI4SHADER_METAL``) plus ``AI4SHADER_TOOL_PATHS`` (a ``os.pathsep``-separated
list of directories).  Reporting "no compiler" on a machine that has one is its
own kind of false negative, so the search is thorough and its result is
reported with the resolved path.

The dxc encoding trap on CJK Windows
------------------------------------
``dxc`` 1.6 on a host with a CJK ANSI code page (code page 936, i.e. a Chinese
Windows install) **cannot read a source file that contains a non-ASCII byte and
no byte-order mark**.  It fails with ``dxc failed : error code 0x80070459`` and
*no diagnostics at all*, which reads exactly like a compiler crash.  The emitted
artefacts contain a UTF-8 em dash in their header comment, so this fires on every
real file.  Two consequences:

* This module writes HLSL temp files **with a UTF-8 BOM** whenever the source is
  not pure ASCII, and records that in ``notes``.  With the BOM, dxc emits real
  ``line:column`` diagnostics.
* ``ok=False`` plus an empty diagnostic list is a *diagnosable* condition, so
  :attr:`ValidationResult.reason` names this trap instead of leaving the caller
  with a bare error code.

Nothing here downloads anything, and nothing here is a network call.  Every
tool is a local executable invoked through :mod:`subprocess`; the exact argv is
returned in :attr:`ValidationResult.command` so a failure is reproducible by
hand.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

__all__ = [
    "ValidationResult",
    "Diagnostic",
    "available",
    "clear_cache",
    "validate_source",
    "validate_file",
    "detect_target",
    "discover_entry",
    "TOOL_FOR_TARGET",
    "EXTENSION_FOR_TARGET",
]

Target = Literal["glsl", "wgsl", "hlsl", "msl", "shaderlab", "ue"]

#: Which tool each target is validated with.  ``shaderlab`` and ``ue`` have no
#: usable command-line compiler; they are listed so the reason can say *why*.
TOOL_FOR_TARGET: dict[str, str | None] = {
    "glsl": "glslangValidator",
    "wgsl": "naga",
    "hlsl": "dxc",
    "msl": "xcrun metal",
    "shaderlab": "unity",
    "ue": None,
}

EXTENSION_FOR_TARGET: dict[str, str] = {
    "glsl": ".frag",
    "wgsl": ".wgsl",
    "hlsl": ".hlsl",
    "msl": ".metal",
    "shaderlab": ".shader",
    "ue": ".hlsl",
}

#: Default entry points, matching ``packages/ir/src/targets/README.md`` and the
#: emitter sources: ``main`` for GLSL, ``fs_main`` for WGSL/MSL, ``PSMain`` for
#: HLSL (``VSMain`` exists too but the fragment stage is what gets validated).
DEFAULT_ENTRY: dict[str, str] = {
    "glsl": "main",
    "wgsl": "fs_main",
    "hlsl": "PSMain",
    "msl": "fs_main",
}

#: Shader profile per target, for the tools that take one.
DXC_PROFILE = "ps_6_0"

#: ``-T`` profile per **entry point**, which is not the same question.
#:
#: A quad document has one entry point (``PSMain``) and one profile answers it. A *surface*
#: document emits a pair — ``VSMain`` transforming the mesh and ``PSMain`` shading it — and
#: ``-T ps_6_0 -E VSMain`` is not a check of the vertex stage, it is a dxc error: the profile
#: and the entry point disagree. Compiling only the fragment half would leave the vertex stage
#: unverified, and the vertex stage is exactly where the surface contract lives (the
#: interpolated world position / normal / uv a host mesh has to supply).
DXC_PROFILE_BY_ENTRY: dict[str, str] = {
    "VSMain": "vs_6_0",
    "VS_Main": "vs_6_0",
    "vs_main": "vs_6_0",
    "PSMain": "ps_6_0",
    "PS_Main": "ps_6_0",
    "ps_main": "ps_6_0",
}


def dxc_profile(entry: str) -> str:
    """The dxc ``-T`` value for ``entry``, falling back to the fragment profile."""
    return DXC_PROFILE_BY_ENTRY.get(entry, DXC_PROFILE)

_ENV_OVERRIDE = {
    "glslangValidator": "AI4SHADER_GLSLANG",
    "dxc": "AI4SHADER_DXC",
    "naga": "AI4SHADER_NAGA",
    "metal": "AI4SHADER_METAL",
}

_DEFAULT_TIMEOUT = 60.0


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass
class Diagnostic:
    """One parsed compiler message.

    ``line`` and ``column`` are **1-based** and are ``None`` when the compiler's
    output did not carry them (``glslangValidator``'s ``ERROR: 0:12:`` form has a
    line but no column).  ``severity`` is one of ``"error"``, ``"warning"``,
    ``"note"``, ``"info"``.  ``message`` is the compiler's text, unaltered.
    """

    line: int | None
    column: int | None
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """The verdict of one real compiler run.  JSON-serialisable via :meth:`to_dict`.

    ``ok`` is a three-valued field, on purpose:

    * ``True``  -- a real compiler ran and accepted the source.
    * ``False`` -- a real compiler ran and rejected it (see ``diagnostics``).
    * ``None``  -- **no verdict exists**: the tool is absent, timed out, or
      crashed without producing a diagnosable message.  ``reason`` always
      explains which.  ``None`` must never be treated as a pass.
    """

    target: str
    #: Resolved tool name actually used, or the name that was missing.
    tool: str | None
    #: Absolute path to the executable, when one was found.
    tool_path: str | None
    ok: bool | None
    #: Human sentence: why ``ok`` is ``None``, or a one-line summary of the verdict.
    reason: str
    #: Compiler **stdout, verbatim** -- never reformatted, never truncated.
    stdout: str = ""
    #: Compiler **stderr, verbatim**.
    stderr: str = ""
    returncode: int | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    #: The exact argv that was executed, so a failure can be reproduced by hand.
    command: list[str] = field(default_factory=list)
    #: Anything this module did to the input before handing it over (e.g. "added
    #: a UTF-8 BOM").  Empty means the source was passed through byte-for-byte.
    notes: list[str] = field(default_factory=list)
    #: Tool version banner, when the tool reports one cheaply.
    tool_version: str | None = None

    @property
    def passed(self) -> bool:
        """``True`` only for a real, positive verdict.  ``None`` is not a pass."""
        return self.ok is True

    @property
    def error_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["diagnostics"] = [d.to_dict() for d in self.diagnostics]
        data["passed"] = self.passed
        data["error_count"] = self.error_count
        data["warning_count"] = self.warning_count
        return data


# --------------------------------------------------------------------------- #
# Tool discovery
# --------------------------------------------------------------------------- #


_TOOL_CACHE: dict[str, str | None] = {}
_VERSION_CACHE: dict[str, str | None] = {}


def _windows_kit_dirs() -> list[str]:
    """``dxc.exe`` locations inside the Windows SDK, newest SDK first.

    DXC ships in the Windows Kits under a version-numbered directory, so the
    version is parsed and sorted rather than globbed in filesystem order.
    """
    roots = [
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Windows Kits", "10", "bin"),
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Windows Kits", "10", "bin"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Windows Kits", "10", "Redist", "D3D"),
    ]
    out: list[tuple[tuple[int, ...], str]] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        for entry in entries:
            base = os.path.join(root, entry)
            for arch in ("x64", "x86", "arm64"):
                cand = os.path.join(base, arch)
                if os.path.isfile(os.path.join(cand, "dxc.exe")):
                    try:
                        key = tuple(int(p) for p in entry.split("."))
                    except ValueError:
                        key = (0,)
                    out.append((key, cand))
        # Redist/D3D/x64 has no version directory
        for arch in ("x64", "x86", "arm64"):
            cand = os.path.join(root, arch)
            if os.path.isfile(os.path.join(cand, "dxc.exe")):
                out.append(((0,), cand))
    out.sort(key=lambda kv: kv[0], reverse=True)
    return [d for _, d in out]


def _android_sdk_roots() -> list[str]:
    roots = []
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        value = os.environ.get(var)
        if value:
            roots.append(value)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(os.path.join(local, "Android", "Sdk"))
    roots.append(os.path.join(os.path.expanduser("~"), "AppData", "Local", "Android", "Sdk"))
    return roots


def _candidate_dirs(tool: str) -> list[str]:
    """Directories worth searching for ``tool``, in preference order."""
    dirs: list[str] = []
    # 1. Explicit directory list, highest priority.
    extra = os.environ.get("AI4SHADER_TOOL_PATHS")
    if extra:
        dirs.extend(p for p in extra.split(os.pathsep) if p)

    # 2. Vulkan SDK -- the canonical home of both glslangValidator and dxc.
    vulkan = os.environ.get("VULKAN_SDK")
    for root in (vulkan, r"C:\VulkanSDK"):
        if root and os.path.isdir(root):
            dirs.append(os.path.join(root, "Bin"))
            dirs.append(os.path.join(root, "bin"))
            dirs.append(root)
            try:
                for entry in os.listdir(root):
                    dirs.append(os.path.join(root, entry, "Bin"))
                    dirs.append(os.path.join(root, entry, "bin"))
            except OSError:
                pass

    if tool == "dxc":
        dirs.extend(_windows_kit_dirs())
    if tool == "glslangValidator":
        for sdk in _android_sdk_roots():
            dirs.append(os.path.join(sdk, "emulator", "lib64", "vulkan"))
            dirs.append(os.path.join(sdk, "emulator", "lib64"))
            dirs.append(os.path.join(sdk, "build-tools"))
    if tool == "naga":
        dirs.append(os.path.join(os.path.expanduser("~"), ".cargo", "bin"))
        dirs.append(os.path.join(os.path.expanduser("~"), ".local", "bin"))
    return [d for d in dirs if d and os.path.isdir(d)]


def _which(tool: str) -> str | None:
    """Locate ``tool``: env override, then ``PATH``, then the known SDK roots."""
    override = _ENV_OVERRIDE.get(tool)
    if override:
        value = os.environ.get(override)
        if value:
            if os.path.isfile(value):
                return os.path.abspath(value)
            # An override that does not resolve is a configuration error the
            # caller must see, not something to paper over by searching PATH.
            return None

    found = shutil.which(tool)
    if found:
        return os.path.abspath(found)

    exe = tool + (".exe" if sys.platform == "win32" else "")
    for directory in _candidate_dirs(tool):
        cand = os.path.join(directory, exe)
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def _tool_version(tool: str, path: str) -> str | None:
    """First line of the tool's version banner.  Cached; never fatal."""
    if tool in _VERSION_CACHE:
        return _VERSION_CACHE[tool]
    args = [path, "--version"] if tool != "xcrun" else [path, "--version"]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=20)
        text = (proc.stdout or "") + (proc.stderr or "")
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), None)
    except Exception:  # noqa: BLE001 - a missing banner is not a validation failure
        first = None
    _VERSION_CACHE[tool] = first
    return first


def available(*, refresh: bool = False) -> dict[str, str | None]:
    """Map each supported tool to its resolved absolute path, or ``None``.

    Tools: ``glslangValidator``, ``dxc``, ``naga``, ``xcrun metal``, ``unity``.
    The search covers ``PATH``, the per-tool environment overrides, and the
    conventional SDK directories (see the module docstring).  Results are cached
    for the process; ``refresh=True`` re-probes.

    A ``None`` here is the reason a :func:`validate_source` call will come back
    with ``ok=None``.  It is **not** permission to assume the shader is fine.
    """
    if refresh:
        clear_cache()
    for tool in ("glslangValidator", "dxc", "naga", "unity"):
        if tool not in _TOOL_CACHE:
            _TOOL_CACHE[tool] = _which(tool)
    if "xcrun metal" not in _TOOL_CACHE:
        xcrun = _which("xcrun") if sys.platform == "darwin" else None
        _TOOL_CACHE["xcrun metal"] = xcrun
    return {
        "glslangValidator": _TOOL_CACHE["glslangValidator"],
        "dxc": _TOOL_CACHE["dxc"],
        "naga": _TOOL_CACHE["naga"],
        "xcrun metal": _TOOL_CACHE["xcrun metal"],
        "unity": _TOOL_CACHE["unity"],
    }


def clear_cache() -> None:
    """Forget cached tool paths and version banners (used by the tests)."""
    _TOOL_CACHE.clear()
    _VERSION_CACHE.clear()


# --------------------------------------------------------------------------- #
# Source inspection
# ---------------------------------------------------------------------------


def detect_target(source: str) -> str | None:
    """Guess the target language from the source text, or ``None`` if unsure.

    Deliberately conservative: a wrong guess would validate the file with the
    wrong compiler and produce a confident, meaningless verdict.  ``None`` means
    "say which target you mean".
    """
    head = source[:4000]
    if re.search(r"^\s*#version\s+\d+", head, re.M) or re.search(r"\bgl_FragColor\b|\bgl_FragCoord\b", head):
        return "glsl"
    if re.search(r"@fragment\b|@group\(\d+\)|var<uniform>|fn\s+\w+\s*\(", head):
        return "wgsl"
    if re.search(r"Shader\s+\"[^\"]+\"\s*\{", head):
        return "shaderlab"
    if re.search(r"#include\s*<metal_stdlib>|\[\[(?:stage_in|buffer|texture|sampler)\(", head):
        return "msl"
    if re.search(r"\bcbuffer\b|\bSV_Target\b|\bregister\(b\d+\)|float4\s+\w+\s*\([^)]*\)\s*:\s*SV_", head):
        return "hlsl"
    return None


def discover_entry(source: str, target: str) -> str:
    """Find the fragment entry-point name in ``source``.

    HLSL: the emitter writes ``float4 PSMain(PSInput) : SV_Target`` (see
    ``packages/ir/src/targets/README.md`` section 4.3), but this looks for *any*
    function returning ``float4``/``MaterialFloat4`` with an ``SV_Target``
    semantic, then for ``PSMain``/``PS_Main``/``main``, and only then falls back
    to :data:`DEFAULT_ENTRY`.  WGSL/MSL: the ``@fragment``/``[[stage_in]]``
    function.  GLSL: ``main``.
    """
    if target == "hlsl":
        match = re.search(
            r"\b(?:float4|MaterialFloat4)\s+([A-Za-z_]\w*)\s*\([^;{]*\)\s*:\s*SV_Target", source
        )
        if match:
            return match.group(1)
        for name in ("PSMain", "PS_Main", "ps_main", "main"):
            if re.search(rf"\b{re.escape(name)}\s*\(", source):
                return name
        return DEFAULT_ENTRY["hlsl"]
    if target == "wgsl":
        match = re.search(r"@fragment\s+fn\s+([A-Za-z_]\w*)", source)
        return match.group(1) if match else DEFAULT_ENTRY["wgsl"]
    if target == "msl":
        match = re.search(r"\bfragment\s+\w+\s+([A-Za-z_]\w*)\s*\(", source)
        return match.group(1) if match else DEFAULT_ENTRY["msl"]
    return DEFAULT_ENTRY.get(target, "main")


def entry_points(source: str, target: str) -> list[str]:
    """**Every** stage the emitter wrote, fragment first.

    HLSL emits a pair on both paths, and the two pairs are different shapes: a quad document's
    ``VSMain`` builds the fullscreen triangle from ``SV_VertexID``, while a **surface**
    document's ``VSMain`` consumes mesh attributes and interpolates world position / normal /
    uv. A checker that only ever looks at the fragment half reports a green tick for a document
    whose vertex stage was never fed to a compiler — and on this target the vertex stage is
    where the surface contract lives.

    Other targets return the single entry ``discover_entry`` finds; GLSL's surface output is a
    fragment shader whose vertex stage belongs to the *host* (see the emitted header), so there
    is nothing else in the file to compile.
    """
    if target == "hlsl":
        found = [n for n in ("PSMain", "VSMain") if re.search(rf"\b{n}\s*\(", source)]
        if found:
            return found
    return [discover_entry(source, target)]


# --------------------------------------------------------------------------- #
# Diagnostic parsing
# --------------------------------------------------------------------------- #

# Every pattern captures `line` and optionally `column`, plus the message.  They
# are tried most-specific first; anything unmatched is still preserved verbatim
# in `stdout`/`stderr`, so the raw text is never lost to a parsing gap.
_DIAG_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # dxc / clang style:  file:12:34: error: message
    (
        re.compile(r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<sev>error|warning|note|fatal error):\s*(?P<msg>.*)$"),
        "file:line:col: sev: msg",
    ),
    # glslangValidator:   ERROR: 0:12: 'x' : message   (file:line, no column)
    (
        re.compile(r"^(?P<sev>ERROR|WARNING):\s*(?P<file>\d+):(?P<line>\d+):\s*(?P<msg>.*)$"),
        "SEV: file:line: msg",
    ),
    # fxc style:          file(12,5): error X3000: message
    (
        re.compile(r"^(?P<file>\S+)\((?P<line>\d+)(?:,(?P<col>\d+))?\)\s*:\s*(?P<sev>error|warning)\s+\w*\d*\s*:\s*(?P<msg>.*)$"),
        "file(line,col): sev code: msg",
    ),
    # naga:               error: message   (+ a following "--> file:line:col")
    (
        re.compile(r"^(?P<sev>error|warning)(?:\[[^\]]*\])?:\s*(?P<msg>.*)$"),
        "sev: msg (+--> line)",
    ),
]

_LOCATION_PATTERN = re.compile(r"-->\s*(?P<file>\S+?):(?P<line>\d+):(?P<col>\d+)")


def parse_diagnostics(text: str) -> list[Diagnostic]:
    """Parse compiler output into structured diagnostics, tolerantly.

    Unrecognised lines are skipped here but remain available verbatim in
    :attr:`ValidationResult.stdout` / ``stderr``.  A parser that silently dropped
    the raw text would be worse than no parser at all.
    """
    out: list[Diagnostic] = []
    pending: Diagnostic | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        # naga prints the location on the line after the message.
        if pending is not None:
            loc = _LOCATION_PATTERN.search(line)
            if loc:
                pending.line = int(loc.group("line"))
                pending.column = int(loc.group("col"))
                pending = None
                continue
        matched = False
        for pattern, _kind in _DIAG_PATTERNS:
            m = pattern.match(line)
            if not m:
                continue
            groups = m.groupdict()
            severity = groups.get("sev", "error").lower().replace("fatal ", "")
            if severity not in ("error", "warning", "note", "info"):
                severity = "error" if severity.startswith("err") else "warning"
            diag = Diagnostic(
                line=int(groups["line"]) if groups.get("line") else None,
                column=int(groups["col"]) if groups.get("col") else None,
                message=(groups.get("msg") or "").strip(),
                severity=severity,
            )
            out.append(diag)
            # Only the naga-style bare "error:" form waits for a location.
            pending = diag if groups.get("line") is None else None
            matched = True
            break
        if not matched:
            pending = None
    return out


# --------------------------------------------------------------------------- #
# Writing the temp file
# --------------------------------------------------------------------------- #

#: ``dxc`` on a CJK code page cannot read non-ASCII without a BOM.  See the
#: module docstring for the full diagnosis.
_DXC_ENCODING_FAILURE = re.compile(r"error code 0x80070459", re.I)

_HLSL_HEADER_NOISE = re.compile(r"^(dxc failed|warning: unknown)", re.I)


def _write_source(source: str, target: str, directory: str) -> tuple[str, list[str]]:
    """Write the source to a temp file with the right extension.

    Returns ``(path, notes)``.  The only transformation ever applied is adding a
    UTF-8 BOM for HLSL when the text is not pure ASCII; that is recorded in
    ``notes`` and is a workaround for a dxc limitation, not a source edit.
    """
    path = os.path.join(directory, "shader" + EXTENSION_FOR_TARGET.get(target, ".txt"))
    notes: list[str] = []
    encoding = "utf-8"
    if target == "hlsl" and not source.isascii():
        encoding = "utf-8-sig"
        notes.append(
            "wrote the HLSL source with a UTF-8 BOM: dxc 1.6 cannot decode a non-ASCII source "
            "file without one on a CJK ANSI code page (it fails with 0x80070459 and no diagnostics)"
        )
    # newline="" preserves whatever the caller produced; we never rewrite line endings.
    with open(path, "w", encoding=encoding, newline="") as fh:
        fh.write(source)
    return path, notes


# --------------------------------------------------------------------------- #
# Per-tool invocation
# --------------------------------------------------------------------------- #


def _build_command(
    tool: str,
    tool_path: str,
    target: str,
    path: str,
    entry: str,
    out_path: str,
    extra_args: Sequence[str],
) -> list[str]:
    """The exact argv for one tool.  Kept separate so the CLI can print it."""
    if tool == "glslangValidator":
        # `-S frag` selects the fragment stage; glslang infers the language
        # version from the source's `#version` directive.
        return [tool_path, "-S", "frag", *extra_args, path]
    if tool == "dxc":
        return [
            tool_path,
            "-T", dxc_profile(entry),
            "-E", entry,
            *extra_args,
            path,
            "-Fo", out_path,
        ]
    if tool == "naga":
        # naga takes the file directly and infers the front end from it.
        return [tool_path, *extra_args, path]
    if tool == "xcrun metal":
        return [tool_path, "-sdk", "macosx", "metal", "-c", *extra_args, path, "-o", out_path]
    return [tool_path, *extra_args, path]


def _run(argv: Sequence[str], timeout: float) -> tuple[int | None, str, str, str | None]:
    """Run ``argv``.  Returns ``(returncode, stdout, stderr, failure_reason)``.

    A timeout or an OSError is reported as a reason rather than raised: those are
    "no verdict exists", which is a legitimate result this module must be able to
    carry, not an exception for the caller to translate.
    """
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "", "", f"timed out after {timeout:g}s"
    except FileNotFoundError:
        return None, "", "", "executable not found"
    except OSError as exc:
        return None, "", "", f"could not run the tool: {type(exc).__name__}: {exc}"
    return proc.returncode, proc.stdout or "", proc.stderr or "", None


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def validate_source(
    source: str,
    target: str,
    *,
    entry: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    extra_args: Sequence[str] = (),
    require_tool: str | None = None,
) -> ValidationResult:
    """Validate ``source`` for ``target`` with a real compiler.

    * ``source`` -- the shader text, exactly as emitted.
    * ``target`` -- ``"glsl"``, ``"wgsl"``, ``"hlsl"``, ``"msl"``, ``"shaderlab"``
      or ``"ue"``.
    * ``entry`` -- entry-point name.  Discovered from ``source`` by
      :func:`discover_entry` when omitted (the emitters use ``PSMain`` for HLSL).
    * ``require_tool`` -- assert that a *specific* tool performed the validation.
      If it is absent, ``ok`` is ``None`` and ``reason`` says so; the CLI turns
      that into a non-zero exit so a CI job can demand, say, ``--tool dxc``
      rather than accepting whichever compiler happened to exist.
    * returns: a :class:`ValidationResult`.  ``ok`` is ``True``/``False``/``None``;
      ``stdout`` and ``stderr`` are the compiler's output **verbatim**.

    ``ok=None`` whenever no real verdict exists (absent tool, unsupported target,
    timeout, crash with no diagnostics).  It is never ``True`` in that case.
    """
    target = (target or "").lower().strip()
    wanted = TOOL_FOR_TARGET.get(target, None)
    if target not in TOOL_FOR_TARGET:
        return ValidationResult(
            target=target,
            tool=None,
            tool_path=None,
            ok=None,
            reason=(
                f"unknown target {target!r}; expected one of "
                f"{', '.join(sorted(TOOL_FOR_TARGET))}"
            ),
        )

    # Targets with no command-line compiler at all: DESIGN.md section 5.4 says say
    # so plainly rather than reporting a green tick we cannot justify.
    if wanted is None or target in ("shaderlab", "ue"):
        reason = {
            "ue": (
                "Unreal Engine cannot headless-compile a material Custom node; "
                "the emitted HLSL is reviewed-by-construction, not verified "
                "(DESIGN.md section 5.4, packages/ir/src/targets/README.md section 4.6)"
            ),
            "shaderlab": (
                "no Unity installation was found; validating ShaderLab requires "
                "`Unity -batchmode -executeMethod`, which cannot run here"
            ),
        }.get(target, f"no command-line compiler exists for target {target!r}")
        return ValidationResult(
            target=target,
            tool=wanted,
            tool_path=None,
            ok=None,
            reason=f"{reason}. No verdict exists for this target -- this is NOT a pass.",
        )

    tools = available()
    tool_path = tools.get(wanted)

    if require_tool:
        key = require_tool if require_tool in tools else wanted
        if require_tool not in tools and require_tool != wanted:
            return ValidationResult(
                target=target,
                tool=require_tool,
                tool_path=None,
                ok=None,
                reason=(
                    f"--tool {require_tool!r} requested but {require_tool!r} is not a known tool; "
                    f"known tools are {', '.join(sorted(tools))}"
                ),
            )
        if tools.get(key) is None:
            return ValidationResult(
                target=target,
                tool=require_tool,
                tool_path=None,
                ok=None,
                reason=(
                    f"--tool {require_tool!r} was required but no executable was found for it "
                    f"(searched PATH, AI4SHADER_TOOL_PATHS, and the conventional SDK directories)"
                ),
            )
        tool_path = tools[key]
        wanted = require_tool

    if tool_path is None:
        searched = (
            "PATH, the per-tool environment overrides ("
            + ", ".join(sorted(_ENV_OVERRIDE.values()))
            + ") and the conventional SDK directories"
        )
        return ValidationResult(
            target=target,
            tool=wanted,
            tool_path=None,
            ok=None,
            reason=(
                f"{wanted!r} is not installed or not discoverable ({searched}). "
                "No verdict exists for this target -- this is NOT a pass."
            ),
        )

    resolved_entry = entry or discover_entry(source, target)
    with tempfile.TemporaryDirectory(prefix="ai4shader-validate-") as tmp:
        path, notes = _write_source(source, target, tmp)
        out_path = os.path.join(tmp, "out.bin")
        argv = _build_command(wanted, tool_path, target, path, resolved_entry, out_path, extra_args)
        code, stdout, stderr, failure = _run(argv, timeout)

    combined = stdout + ("\n" if stdout and stderr else "") + stderr
    diagnostics = parse_diagnostics(combined)
    errors = [d for d in diagnostics if d.severity == "error"]
    version = _tool_version(wanted, tool_path)
    base = dict(
        target=target,
        tool=wanted,
        tool_path=tool_path,
        stdout=stdout,
        stderr=stderr,
        returncode=code,
        diagnostics=diagnostics,
        command=argv,
        notes=notes,
        tool_version=version,
    )

    if failure is not None:
        return ValidationResult(
            ok=None,
            reason=f"{wanted} produced no verdict: {failure}. This is NOT a pass.",
            **base,
        )

    if code == 0 and not errors:
        return ValidationResult(
            ok=True,
            reason=(
                f"{wanted} accepted the source"
                + (f" ({len(diagnostics)} warning(s))" if diagnostics else " with no diagnostics")
            ),
            **base,
        )

    # Non-zero exit.  Distinguish "the compiler diagnosed a real problem" from
    # "the compiler could not read the file", because the second one looks like a
    # shader bug but is not.
    if not diagnostics and _DXC_ENCODING_FAILURE.search(combined):
        return ValidationResult(
            ok=False,
            reason=(
                "dxc could not decode the source file (error code 0x80070459) and emitted no "
                "diagnostics. This is dxc's encoding failure, not a shader error: on a host with a "
                "CJK ANSI code page it cannot read a source containing non-ASCII bytes without a "
                "UTF-8 BOM. Re-run with the emitted header comments stripped to ASCII, or add a BOM."
            ),
            **base,
        )
    if not diagnostics:
        return ValidationResult(
            ok=False,
            reason=(
                f"{wanted} exited with code {code} but produced no parseable diagnostic. "
                "The raw output above is all the compiler said."
            ),
            **base,
        )
    return ValidationResult(
        ok=False,
        reason=f"{wanted} rejected the source: {len(errors)} error(s), "
        f"{sum(1 for d in diagnostics if d.severity == 'warning')} warning(s)",
        **base,
    )


def validate_file(
    path: str | os.PathLike[str],
    target: str | None = None,
    *,
    entry: str | None = None,
    **kwargs: Any,
) -> ValidationResult:
    """Read ``path`` and validate it.  ``target`` is inferred from the extension.

    Extension map: ``.frag``/``.glsl`` -> glsl, ``.wgsl`` -> wgsl, ``.hlsl`` ->
    hlsl, ``.metal`` -> msl, ``.shader`` -> shaderlab.  An unknown extension with
    no ``target`` is an ``ok=None`` result, not a guess.
    """
    p = os.fspath(path)
    with open(p, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        source = fh.read()
    if target is None:
        ext = os.path.splitext(p)[1].lower()
        target = {
            ".frag": "glsl",
            ".glsl": "glsl",
            ".vert": "glsl",
            ".wgsl": "wgsl",
            ".hlsl": "hlsl",
            ".fx": "hlsl",
            ".metal": "msl",
            ".shader": "shaderlab",
        }.get(ext)
        if target is None:
            target = detect_target(source)
    if target is None:
        return ValidationResult(
            target="",
            tool=None,
            tool_path=None,
            ok=None,
            reason=(
                f"could not infer a target from {p!r}; pass target= explicitly. "
                "Guessing would validate the file with the wrong compiler."
            ),
        )
    return validate_source(source, target, entry=entry, **kwargs)
