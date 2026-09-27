"""Batch driver: emit every target for many documents, then validate each one.

This is what a CI regression job consumes.  Two properties matter more than
anything else here:

1. **Diffable output.**  The report is written with sorted keys, a stable key
   order inside every artefact record, and **content hashes instead of
   timestamps or absolute paths**.  Two runs over the same documents on different
   machines must produce byte-identical JSON, otherwise a regression job drowns
   in noise and nobody reads it.
2. **No false greens.**  A target whose compiler is absent is reported as
   ``"unverified"``, not ``"pass"``.  ``docs/IMPLEMENTATION.md`` section 4.2 is a
   list of things this repository could not verify; a report that silently
   upgraded those to passes would be worse than no report.

Verdict vocabulary (``status`` on every artefact):

``pass``
    A real compiler ran and accepted the source.
``fail``
    A real compiler ran and rejected the source.  ``diagnostics`` says where.
``unverified``
    No compiler exists for this target (or the required one is missing).  This is
    the honest state for WGSL/MSL/ShaderLab/UE on a machine without
    ``naga``/``xcrun``/Unity/UE -- and it is **not** a pass.
``emit-error``
    The Node CLI itself reported an error for this target.
``cli-missing``
    ``apps/cli/dist/cli.js`` or ``node`` was not found, so nothing could be
    emitted at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import validate as V

__all__ = [
    "BatchReport",
    "ArtefactReport",
    "run_matrix",
    "resolve_cli",
    "node_available",
    "TARGETS",
]

#: The six targets ``emit`` understands, in the IR's declared order.  Kept as an
#: explicit tuple: iterating a set or a dict here would make the report's key
#: order machine-dependent, which defeats the point of a diffable report.
TARGETS: tuple[str, ...] = ("glsl", "wgsl", "hlsl", "msl", "shaderlab", "ue")

#: Which emitted file carries each target's shader source.  These are *basenames*
#: only: the emitter nests them under ``<out>/<slug>/`` (and ``<slug>/ue/``), so
#: the directory is resolved from the emit manifest rather than guessed -- and
#: the basename is the fallback when a manifest entry is missing.
#: ``ue/CustomNode.hlsl`` is the Unreal Custom-node body, confirmed against
#: ``emit --target ue --json``.
SOURCE_FILE_FOR_TARGET: dict[str, str] = {
    "glsl": "shader.frag",
    "wgsl": "shader.wgsl",
    "hlsl": "shader.hlsl",
    "msl": "shader.metal",
    "shaderlab": "Shader.shader",
    "ue": "CustomNode.hlsl",
}

_DEFAULT_CLI = os.path.join("apps", "cli", "dist", "cli.js")


def _find_repo_root(start: str | None = None) -> str:
    """Walk up from ``start`` looking for the workspace root (``package.json`` + ``apps``)."""
    here = os.path.abspath(start or os.getcwd())
    for _ in range(8):
        if os.path.isfile(os.path.join(here, "package.json")) and os.path.isdir(os.path.join(here, "apps")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return os.path.abspath(start or os.getcwd())


def node_available() -> str | None:
    """Absolute path to ``node``, or ``None``.  Honours ``AI4SHADER_NODE``."""
    override = os.environ.get("AI4SHADER_NODE")
    if override and os.path.isfile(override):
        return os.path.abspath(override)
    return shutil.which("node")


def resolve_cli(root: str | None = None) -> str | None:
    """Absolute path to ``apps/cli/dist/cli.js``, or ``None``.

    ``AI4SHADER_CLI`` overrides the search.  Only the built ``dist`` entry point
    is used: the TypeScript sources need ``--experimental-strip-types`` and a
    resolver hook, which is not something a CI job should have to reproduce.
    """
    override = os.environ.get("AI4SHADER_CLI")
    if override:
        return os.path.abspath(override) if os.path.isfile(override) else None
    base = root or _find_repo_root()
    cand = os.path.join(base, _DEFAULT_CLI)
    return cand if os.path.isfile(cand) else None


def _hash_bytes(data: bytes) -> str:
    """Short content hash.  16 hex chars of SHA-256: enough to be unique in a
    repository, short enough to read in a diff."""
    return hashlib.sha256(data).hexdigest()[:16]


def _hash_text(text: str) -> str:
    return _hash_bytes(text.encode("utf-8", errors="replace"))


def _relpath(path: str, start: str) -> str:
    """``os.path.relpath`` that degrades to the basename on a cross-drive path.

    On Windows ``os.path.relpath(r"C:\\tmp\\x", r"D:\\repo")`` raises
    ``ValueError`` -- the two are on different mounts and no relative path exists.
    A batch run legitimately reads documents from a temp directory on another
    drive, so this must not explode; and an absolute path must never leak into
    the report, which is why it falls back to the basename rather than the
    original path.
    """
    try:
        return os.path.relpath(os.path.abspath(path), start).replace("\\", "/")
    except ValueError:
        return os.path.basename(path)


@dataclass
class ArtefactReport:
    """One emitted shader plus the verdict of a real compiler on it.

    Every field is either a scalar or a list of scalars, and there is **no
    timestamp and no absolute path** anywhere: the report must be identical
    across machines and runs.
    """

    doc: str
    target: str
    #: ``pass`` | ``fail`` | ``unverified`` | ``emit-error``
    status: str
    #: The emitter's own hash, verbatim from ``emit --json`` (or ``None``).
    emit_hash: str | None = None
    #: SHA-256/16 of the emitted source **as written to disk**.
    source_hash: str | None = None
    source_bytes: int | None = None
    #: SHA-256/16 of each sidecar, keyed by its path relative to the output dir.
    sidecars: dict[str, str] = field(default_factory=dict)
    #: Emitter diagnostics, in the emitter's order.
    emit_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    #: Tool that produced ``status``, or the tool that was missing.
    tool: str | None = None
    tool_path: str | None = None
    tool_version: str | None = None
    #: The exact argv, joined -- reproducible by hand.
    command: str | None = None
    reason: str = ""
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    notes: list[str] = field(default_factory=list)
    #: Relative paths of every file this target produced.
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Stable key order (declaration order of this dataclass)."""
        return {
            "doc": self.doc,
            "target": self.target,
            "status": self.status,
            "emit_hash": self.emit_hash,
            "source_hash": self.source_hash,
            "source_bytes": self.source_bytes,
            "sidecars": dict(sorted(self.sidecars.items())),
            "emit_diagnostics": self.emit_diagnostics,
            "tool": self.tool,
            "tool_path": self.tool_path,
            "tool_version": self.tool_version,
            "command": self.command,
            "reason": self.reason,
            "diagnostics": self.diagnostics,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "notes": self.notes,
            "files": sorted(self.files),
        }


@dataclass
class BatchReport:
    """The whole matrix run: documents x targets.

    ``summary`` counts are the CI gate.  ``environment`` records what was
    available, so a report that says ``unverified`` everywhere is
    self-explaining rather than looking like a broken build.
    """

    artefacts: list[ArtefactReport] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    docs: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[ArtefactReport]:
        return [a for a in self.artefacts if a.status == "fail"]

    @property
    def unverified(self) -> list[ArtefactReport]:
        return [a for a in self.artefacts if a.status == "unverified"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ai4shader.batch/1",
            "docs": list(self.docs),
            "targets": list(self.targets),
            "environment": self.environment,
            "summary": dict(sorted(self.summary.items())),
            "artefacts": [a.to_dict() for a in self.artefacts],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Diffable JSON: ``sort_keys=False`` on purpose.

        The key order is fixed by :meth:`to_dict`, which is *better* than
        alphabetical here -- a diff then shows a changed value next to its
        neighbours instead of scattered across the object.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, ensure_ascii=False) + "\n"

    def render_text(self) -> str:
        """A compact human table for a terminal."""
        lines = [
            f"{'doc':<34} {'target':<10} {'status':<11} {'tool':<18} detail",
            "-" * 118,
        ]
        for a in self.artefacts:
            detail = a.reason
            if a.status == "fail" and a.diagnostics:
                first = next((d for d in a.diagnostics if d.get("severity") == "error"), a.diagnostics[0])
                detail = f"{first.get('line')}:{first.get('column')} {first.get('message', '')}"
            lines.append(
                f"{a.doc:<34} {a.target:<10} {a.status:<11} {(a.tool or '-'):<18} {detail[:52]}"
            )
        lines.append("-" * 118)
        lines.append(
            "  ".join(f"{k}={v}" for k, v in sorted(self.summary.items()))
        )
        return "\n".join(lines)


def _run_cli(
    node: str, cli: str, doc: str, out_dir: str, target: str, timeout: float
) -> tuple[int | None, dict[str, Any] | None, str, str]:
    """Run ``cli.js emit <doc> --target <t> --out <dir> --json`` for one target.

    One call per target rather than one call with ``--target all``: the CLI does
    not accept a comma-separated list, and isolating the calls means a failure in
    one target cannot take the whole matrix down with it.

    Returns ``(returncode, parsed_json, stdout, stderr)``.  ``parsed_json`` is
    ``None`` when stdout was not JSON -- the raw text is always returned so the
    caller can report what actually happened instead of raising.
    """
    argv = [node, cli, "emit", doc, "--target", target, "--out", out_dir, "--json"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, None, "", f"emit timed out after {timeout:g}s"
    except OSError as exc:
        return None, None, "", f"could not run the CLI: {type(exc).__name__}: {exc}"
    parsed: dict[str, Any] | None = None
    text = (proc.stdout or "").strip()
    if text:
        try:
            candidate = json.loads(text)
            parsed = candidate if isinstance(candidate, dict) else None
        except json.JSONDecodeError:
            parsed = None
    return proc.returncode, parsed, proc.stdout or "", proc.stderr or ""


def _normalise_command(argv: Sequence[str], source_basename: str) -> str | None:
    """Rewrite a validator argv into a **machine-independent** form for the report.

    The executable and the validator's temporary directory are what make a raw
    command line differ between runs and machines, so they are replaced with
    placeholders.  The flags -- the part a human needs in order to reproduce a
    failure -- are kept exactly, and ``tool`` / ``tool_path`` already record
    *which* compiler ran.

    The temp directory is discovered from the argv itself (the argument whose
    basename matches the emitted file), because :func:`validate.validate_source`
    owns that directory and has already removed it by the time we get here.
    """
    if not argv:
        return None
    rest = list(argv[1:])  # drop the executable
    temp_dir = None
    for arg in rest:
        if os.path.sep in arg or "/" in arg:
            if os.path.basename(arg) == source_basename:
                temp_dir = os.path.dirname(os.path.abspath(arg))
                break
    parts: list[str] = []
    for arg in rest:
        if temp_dir is not None and (os.path.sep in arg or "/" in arg):
            candidate = os.path.abspath(arg)
            if os.path.dirname(candidate) == temp_dir:
                base = os.path.basename(candidate)
                parts.append("<source>" if base == source_basename else f"<{base}>")
                continue
        parts.append(arg)
    return " ".join(parts)


def _find_emitted_file(doc_out: str, payload: dict[str, Any], wanted_basename: str) -> str | None:
    """Locate the emitted source file: manifest first, then a bounded directory walk.

    The manifest's ``files[].path`` is relative to ``--out``, so using it avoids
    hard-coding the ``<slug>/`` nesting.  The walk is the safety net for an
    emitter that changes its layout.
    """
    for entry in payload.get("files") or []:
        if not isinstance(entry, Mapping):
            continue
        rel = entry.get("path")
        if not isinstance(rel, str):
            continue
        candidate = os.path.join(doc_out, rel.replace("/", os.sep))
        if os.path.basename(candidate) == wanted_basename and os.path.isfile(candidate):
            return candidate
    for root, _dirs, names in os.walk(doc_out):
        if wanted_basename in names:
            return os.path.join(root, wanted_basename)
    return None


def _find_sidecar(doc_out: str, source_path: str, name: str) -> str | None:
    """Sidecars sit next to the source; fall back to a walk for nested layouts."""
    beside = os.path.join(os.path.dirname(source_path), name)
    if os.path.isfile(beside):
        return beside
    for root, _dirs, names in os.walk(doc_out):
        if name in names:
            return os.path.join(root, name)
    return None


def run_matrix(
    docs: Sequence[str],
    targets: Sequence[str] = TARGETS,
    *,
    out_dir: str | None = None,
    root: str | None = None,
    timeout: float = 180.0,
    require_tool: str | None = None,
) -> BatchReport:
    """Emit ``docs`` for ``targets`` through the Node CLI, then validate each output.

    * ``docs`` -- paths to ``.shader.json`` documents.
    * ``targets`` -- any subset of :data:`TARGETS`, validated against that name.
    * ``out_dir`` -- where the emitter writes.  Defaults to a temporary directory
      so a batch run never litters the repository; only the *relative* paths
      appear in the report either way.
    * ``require_tool`` -- forwarded to :func:`ai4shader.validate.validate_source`;
      a CI job can demand a specific compiler instead of accepting whichever one
      happened to be installed.

    The returned :class:`BatchReport` has no timestamps and no absolute paths in
    its serialised form, so two runs of the same inputs produce identical JSON.
    """
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise ValueError(f"unknown target(s) {unknown}; known targets are {list(TARGETS)}")

    root_dir = root or _find_repo_root()
    node = node_available()
    cli = resolve_cli(root_dir)
    tools = V.available()

    env: dict[str, Any] = {
        "node": os.path.basename(node) if node else None,
        "cli": _relpath(cli, root_dir) if cli else None,
        "tools": {k: (os.path.basename(v) if v else None) for k, v in sorted(tools.items())},
    }

    report = BatchReport(
        docs=[os.path.basename(d) for d in docs],
        targets=list(targets),
        environment=env,
    )

    temp_holder = None
    if out_dir is None:
        import tempfile

        temp_holder = tempfile.TemporaryDirectory(prefix="ai4shader-batch-")
        out_dir = temp_holder.name
    os.makedirs(out_dir, exist_ok=True)

    try:
        for doc in docs:
            doc_rel = _relpath(doc, root_dir)
            if not os.path.isfile(doc):
                for target in targets:
                    report.artefacts.append(
                        ArtefactReport(
                            doc=doc_rel,
                            target=target,
                            status="cli-missing",
                            reason=f"document not found: {doc}",
                        )
                    )
                continue
            if node is None or cli is None:
                missing = "node" if node is None else f"CLI at {_DEFAULT_CLI}"
                for target in targets:
                    report.artefacts.append(
                        ArtefactReport(
                            doc=doc_rel,
                            target=target,
                            status="cli-missing",
                            reason=(
                                f"{missing} was not found, so nothing could be emitted. "
                                "Build the CLI (`node scripts/build.mjs`) or set AI4SHADER_NODE / AI4SHADER_CLI."
                            ),
                        )
                    )
                continue

            doc_out = os.path.join(out_dir, os.path.splitext(os.path.basename(doc))[0])

            for target in targets:
                code, payload, stdout, stderr = _run_cli(node, cli, doc, doc_out, target, timeout)

                entries = [
                    dict(e)
                    for e in (payload or {}).get("targets", []) or []
                    if isinstance(e, Mapping) and isinstance(e.get("target"), str)
                ]
                emitted = next((e for e in entries if e["target"] == target), None)

                artefact = ArtefactReport(doc=doc_rel, target=target, status="unverified")
                if emitted is None:
                    artefact.status = "emit-error"
                    artefact.reason = (
                        f"`emit --target {target}` produced no JSON entry "
                        f"(cli exit {code}). stderr: {stderr.strip()[:200] or '(empty)'}"
                    )
                    artefact.tool = V.TOOL_FOR_TARGET.get(target)
                    report.artefacts.append(artefact)
                    continue

                artefact.emit_hash = emitted.get("hash")
                artefact.emit_diagnostics = list(emitted.get("diagnostics") or [])

                # Locate the emitted source on disk and hash what is actually there.
                wanted = SOURCE_FILE_FOR_TARGET.get(target, "")
                abs_source = _find_emitted_file(doc_out, payload or {}, wanted)

                if abs_source is not None:
                    with open(abs_source, "rb") as fh:
                        raw = fh.read()
                    artefact.source_hash = _hash_bytes(raw)
                    artefact.source_bytes = len(raw)
                    artefact.files.append(_relpath(abs_source, doc_out))
                    for name in sorted(str(s) for s in (emitted.get("sidecars") or [])):
                        candidate = _find_sidecar(doc_out, abs_source, name)
                        if candidate is not None:
                            with open(candidate, "rb") as fh:
                                artefact.sidecars[name] = _hash_bytes(fh.read())
                            artefact.files.append(_relpath(candidate, doc_out))

                if not emitted.get("ok", True):
                    artefact.status = "emit-error"
                    artefact.reason = (
                        "the emitter reported an error for this target; see emit_diagnostics"
                    )
                    artefact.tool = V.TOOL_FOR_TARGET.get(target)
                    report.artefacts.append(artefact)
                    continue

                if abs_source is None:
                    artefact.status = "unverified"
                    artefact.reason = (
                        f"the emitter reported success but {wanted!r} was not found under the output "
                        f"directory, so no compiler could be run on it"
                    )
                    artefact.tool = V.TOOL_FOR_TARGET.get(target)
                    report.artefacts.append(artefact)
                    continue

                result = V.validate_file(abs_source, target, require_tool=require_tool)
                artefact.tool = result.tool
                artefact.tool_path = os.path.basename(result.tool_path) if result.tool_path else None
                artefact.tool_version = result.tool_version
                # Normalised: no executable, no temp paths -- see _normalise_command.
                artefact.command = _normalise_command(result.command, wanted)
                artefact.reason = result.reason
                artefact.diagnostics = [d.to_dict() for d in result.diagnostics]
                artefact.error_count = result.error_count
                artefact.warning_count = result.warning_count
                artefact.notes = list(result.notes)
                artefact.status = {True: "pass", False: "fail", None: "unverified"}[result.ok]
                report.artefacts.append(artefact)
    finally:
        if temp_holder is not None:
            temp_holder.cleanup()

    counts = {name: 0 for name in ("pass", "fail", "unverified", "emit-error", "cli-missing")}
    for a in report.artefacts:
        counts[a.status] = counts.get(a.status, 0) + 1
    report.summary = counts
    return report
