"""Tests for :mod:`ai4shader.batch`.

The property that matters for a CI regression job is **diffability**: two runs
over the same inputs must produce byte-identical JSON.  That means no timestamps,
no absolute paths, no set-iteration order.  These tests assert that directly
rather than trusting it.
"""

from __future__ import annotations

import json
import os

import pytest

from ai4shader import batch as B
from ai4shader import validate as V

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE = os.path.join(REPO_ROOT, "examples", "liquid-metal.shader.json")

has_cli = B.resolve_cli(REPO_ROOT) is not None and B.node_available() is not None
needs_cli = pytest.mark.skipif(not has_cli, reason="node or apps/cli/dist/cli.js not available")


def test_targets_are_a_stable_ordered_tuple():
    """Key order in the report depends on this; a set would make output unstable."""
    assert isinstance(B.TARGETS, tuple)
    assert B.TARGETS == ("glsl", "wgsl", "hlsl", "msl", "shaderlab", "ue")


def test_every_target_maps_to_a_source_file_and_a_tool():
    for target in B.TARGETS:
        assert target in B.SOURCE_FILE_FOR_TARGET
        assert target in V.TOOL_FOR_TARGET


def test_unknown_target_is_refused():
    with pytest.raises(ValueError, match="unknown target"):
        B.run_matrix([EXAMPLE], ["glsl", "cobol"])


def test_missing_document_is_reported_not_raised(tmp_path):
    report = B.run_matrix([str(tmp_path / "nope.shader.json")], ["glsl"])
    assert report.summary["cli-missing"] == 1
    assert "document not found" in report.artefacts[0].reason
    assert report.artefacts[0].status == "cli-missing"


@needs_cli
def test_run_matrix_over_the_shipped_example():
    report = B.run_matrix([EXAMPLE], list(B.TARGETS))
    assert len(report.artefacts) == len(B.TARGETS)
    statuses = {a.target: a.status for a in report.artefacts}
    # The emitter must succeed on every target for the shipped example.
    assert "emit-error" not in statuses.values(), statuses
    assert "cli-missing" not in statuses.values(), statuses
    # GLSL has a real compiler on this machine (or nothing does).
    if V.available()["glslangValidator"]:
        assert statuses["glsl"] == "pass", statuses
    else:
        assert statuses["glsl"] == "unverified"
    # WGSL/MSL/ShaderLab/UE have no headless compiler here: they must be
    # "unverified".  Never "pass" -- that is the whole point.
    for target in ("wgsl", "msl", "shaderlab", "ue"):
        if not V.available().get(V.TOOL_FOR_TARGET[target] or "unity"):
            assert statuses[target] == "unverified", f"{target}: {statuses[target]}"


@needs_cli
def test_report_is_byte_identical_across_runs():
    """The diffability requirement: same inputs, same bytes."""
    first = B.run_matrix([EXAMPLE], ["glsl", "wgsl"]).to_json()
    second = B.run_matrix([EXAMPLE], ["glsl", "wgsl"]).to_json()
    assert first == second


@needs_cli
def test_report_contains_no_absolute_paths_or_timestamps():
    """A diffable report must not carry anything that changes between machines or
    runs.  ``iTime`` in emitter diagnostics is fine -- a wall-clock timestamp is not."""
    text = B.run_matrix([EXAMPLE], ["glsl"]).to_json()
    assert REPO_ROOT.replace("\\", "/") not in text
    assert "C:\\\\" not in text and "C:/" not in text
    assert "/tmp/" not in text and "\\\\Temp\\\\" not in text
    lowered = text.lower().replace("itime", "")  # the shader's own uniform name
    for marker in ("timestamp", "datetime", "elapsed", "duration", "mtime", "generated_at"):
        assert marker not in lowered, f"report contains {marker!r}"


@needs_cli
def test_report_summary_counts_every_artefact():
    report = B.run_matrix([EXAMPLE], list(B.TARGETS))
    assert sum(report.summary.values()) == len(report.artefacts)
    assert set(report.summary) == {"pass", "fail", "unverified", "emit-error", "cli-missing"}


@needs_cli
def test_artefacts_carry_content_hashes_not_timestamps():
    report = B.run_matrix([EXAMPLE], ["glsl"])
    artefact = report.artefacts[0]
    assert artefact.emit_hash or artefact.source_hash
    assert artefact.source_hash and len(artefact.source_hash) == 16
    assert artefact.source_bytes and artefact.source_bytes > 100
    json.loads(report.to_json())


@needs_cli
def test_hashes_change_when_the_document_changes(tmp_path):
    """A hash that does not move when the input moves is useless as a regression key."""
    doc = json.load(open(EXAMPLE, encoding="utf-8"))

    def write(value, name):
        path = tmp_path / name
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(value, fh)
        return str(path)

    import copy

    a = write(copy.deepcopy(doc), "a.shader.json")
    modified = copy.deepcopy(doc)
    for node in modified["graph"]["nodes"]:
        if node["kind"] == "domain_warp":
            node["params"]["scale"]["value"] = float(node["params"]["scale"]["value"]) * 1.7
    b = write(modified, "b.shader.json")

    hash_a = B.run_matrix([a], ["glsl"]).artefacts[0].source_hash
    hash_b = B.run_matrix([b], ["glsl"]).artefacts[0].source_hash
    assert hash_a and hash_b and hash_a != hash_b


@needs_cli
def test_batch_requires_a_tool_when_asked():
    report = B.run_matrix([EXAMPLE], ["glsl"], require_tool="naga")
    artefact = report.artefacts[0]
    if V.available()["naga"]:
        assert artefact.status in ("pass", "fail")
    else:
        assert artefact.status == "unverified"
        assert "required" in artefact.reason


def test_render_text_is_readable():
    report = B.BatchReport(
        artefacts=[B.ArtefactReport(doc="d.shader.json", target="glsl", status="pass", reason="ok")],
        summary={"pass": 1},
    )
    text = report.render_text()
    assert "d.shader.json" in text and "glsl" in text and "pass" in text


def test_artefact_dict_key_order_is_stable():
    artefact = B.ArtefactReport(doc="d", target="glsl", status="pass")
    keys = list(artefact.to_dict())
    assert keys == ["doc", "target", "status", "emit_hash", "source_hash", "source_bytes",
                    "sidecars", "emit_diagnostics", "tool", "tool_path", "tool_version",
                    "command", "reason", "diagnostics", "error_count", "warning_count",
                    "notes", "files"]
    assert list(artefact.to_dict()) == keys  # same both times


def test_cli_resolution_honours_the_env_override(monkeypatch, tmp_path):
    fake = tmp_path / "cli.js"
    fake.write_text("// nothing\n")
    monkeypatch.setenv("AI4SHADER_CLI", str(fake))
    assert B.resolve_cli(REPO_ROOT) == str(fake)
    monkeypatch.setenv("AI4SHADER_CLI", str(tmp_path / "absent.js"))
    assert B.resolve_cli(REPO_ROOT) is None


def test_repo_root_discovery_finds_the_workspace():
    assert os.path.isfile(os.path.join(B._find_repo_root(REPO_ROOT), "package.json"))
    # from a nested directory too
    nested = os.path.join(REPO_ROOT, "packages", "ir", "src")
    if os.path.isdir(nested):
        assert B._find_repo_root(nested) == REPO_ROOT
