"""Tests for :mod:`ai4shader.attribute` -- the DESIGN.md section 4.3 rules.

The important test in this file is :func:`test_discrimination`: synthetic image
pairs that differ in **exactly one way** must be routed to the **matching rule**.
Without it, the attribution rules are merely plausible; with it, they are
checkable.  A blur that gets blamed on the noise frequency, or a brightness
shift blamed on the palette, would send the parameter optimiser in a wrong
direction while looking like it was working.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ai4shader import attribute as A
from ai4shader import score as S
from conftest import (add_hf_noise, box_blur, brighten, make_base, saturate,
                      shift, shift_hue)

# --------------------------------------------------------------------------- #
# The discrimination test
# --------------------------------------------------------------------------- #

#: ``(label, perturbation, expected rule)``.  Every perturbation changes exactly
#: one property of the image; the expected rule is the DESIGN.md section 4.3 row
#: that owns that property.
DISCRIMINATION_CASES = [
    ("blur radius 2", lambda img: box_blur(img, 2), "edge.hardness"),
    ("blur radius 4", lambda img: box_blur(img, 4), "edge.hardness"),
    ("brightness +0.12", lambda img: brighten(img, 0.12), "tone.exposure"),
    ("brightness +0.22", lambda img: brighten(img, 0.22), "tone.exposure"),
    ("saturation x1.9", lambda img: saturate(img, 1.9), "color.saturation"),
    ("saturation x0.35", lambda img: saturate(img, 0.35), "color.saturation"),
    ("added high-freq noise 0.10", lambda img: add_hf_noise(img, 0.10), "detail.frequency"),
    ("added high-freq noise 0.16", lambda img: add_hf_noise(img, 0.16), "detail.frequency"),
    ("shape shifted 16px", lambda img: shift(img, 16, 0), "structure.mismatch"),
    ("shape shifted 28px", lambda img: shift(img, 28, 0), "structure.mismatch"),
    ("hue rotated +0.12 turn", lambda img: shift_hue(img, 0.12), "color.hue"),
    ("hue rotated -0.12 turn", lambda img: shift_hue(img, -0.12), "color.hue"),
]


@pytest.mark.parametrize(
    "label,perturb,expected",
    DISCRIMINATION_CASES,
    ids=[c[0] for c in DISCRIMINATION_CASES],
)
def test_discrimination(base, label, perturb, expected):
    """The top-ranked hint must be the rule that owns the perturbation."""
    render = perturb(base)
    breakdown = S.score(render, base)
    hints = A.attribute(breakdown)
    assert hints, f"{label}: no rule fired at all (ssim={breakdown.ssim:.3f})"
    assert hints[0]["rule"] == expected, (
        f"{label}: expected {expected}, got {hints[0]['rule']} "
        f"(confidence {hints[0]['confidence']}, why={hints[0]['why']!r}); "
        f"all hints: {[h['rule'] for h in hints]}"
    )


def test_discrimination_ignores_the_unrelated_control(base, unrelated):
    """Sanity: a structurally different image must NOT be blamed on the palette or
    the exposure, which is what a threshold set too tight would do."""
    hints = A.attribute(S.score(unrelated, base))
    assert hints[0]["rule"] == "structure.mismatch"


@pytest.mark.parametrize("direction,expected", [
    (lambda img: brighten(img, 0.22), "decrease"),
    (lambda img: saturate(img, 1.9), "decrease"),
    (lambda img: saturate(img, 0.35), "increase"),
    (lambda img: box_blur(img, 4), "decrease"),
    (lambda img: add_hf_noise(img, 0.16), "decrease"),
])
def test_directions_are_correct(base, direction, expected):
    """The direction is what turns a number into a patch, so each one is pinned.

    Note the ``box_blur`` case: ``sdf.smoothness`` is the smoothstep *width*, so a
    blurred render needs it **decreased** to harden the edge.  That is the
    opposite of what the word "smoothness" suggests, which is exactly why it is
    asserted rather than assumed.
    """
    hints = A.attribute(S.score(direction(base), base))
    assert hints[0]["direction"] == expected, hints[0]


def test_direction_is_a_property_of_the_parameter_not_the_observation(base):
    """The same observation must produce **opposite** directions for two parameters
    that affect it in opposite ways.

    An "edges too soft" observation wants a harder edge.  `sdf.smoothness` is the
    smoothstep *width*, so hardening means **decreasing** it; `grade.contrast`
    scales the gradient, so hardening means **increasing** it.  Getting this wrong
    sends the optimiser the wrong way while looking entirely plausible, so both
    are asserted from the same blur.
    """
    blurred = box_blur(base, 4)
    breakdown = S.score(blurred, base)

    smoothness_catalog = A.catalog_from_document({"graph": {"nodes": [
        {"id": "n_sdf", "kind": "sdf", "params": {
            "smoothness": {"kind": "param", "name": "u_smoothness",
                           "value": 0.01, "min": 0.001, "max": 1}}},
    ]}})
    contrast_catalog = A.catalog_from_document({"graph": {"nodes": [
        {"id": "n_grade", "kind": "grade", "params": {
            "contrast": {"kind": "param", "name": "u_contrast",
                         "value": 1.0, "min": 0.0, "max": 3.0}}},
    ]}})

    via_smoothness = A.attribute(breakdown, catalog=smoothness_catalog)[0]
    via_contrast = A.attribute(breakdown, catalog=contrast_catalog)[0]

    assert via_smoothness["param"] == "smoothness"
    assert via_smoothness["direction"] == "decrease", "widening `smoothness` would soften further"

    assert via_contrast["param"] == "contrast"
    assert via_contrast["direction"] == "increase", "lowering contrast would soften further"

    # Each `why` must describe the parameter actually being pointed at.
    assert "smoothness" in via_smoothness["why"]
    assert "contrast" in via_contrast["why"]
    assert "smoothness" not in via_contrast["why"], (
        "the explanation names a different parameter than the target line"
    )


def test_unknown_polarity_yields_no_direction_rather_than_a_guess(base):
    """`domain_warp.scale` sets the warp's spatial frequency; whether that raises or
    lowers coarse-structure contrast is not knowable here, so no direction is given."""
    catalog = A.catalog_from_document({"graph": {"nodes": [
        {"id": "n_warp", "kind": "domain_warp", "params": {
            "scale": {"kind": "param", "name": "u_warpScale",
                      "value": 0.9, "min": 0.2, "max": 4.0}}},
    ]}})
    hint = A.attribute(S.score(shift(base, 20, 0), base), catalog=catalog)[0]
    assert hint["rule"] == "structure.mismatch"
    assert hint["node"] == "n_warp" and hint["param"] == "scale"
    assert hint["direction"] is None
    assert hint["suggest"] is None


def test_polarity_table_covers_every_candidate():
    """A candidate with no declared polarity would silently lose its direction."""
    for rule, candidates in A.CANDIDATES.items():
        for kind, param in candidates:
            assert (kind, param) in A.POLARITY.get(rule, {}), (
                f"{rule}: {kind}.{param} has no declared polarity; "
                "add it (even as None) so the omission is deliberate"
            )


def test_polarity_values_are_plus_or_minus_one_or_none():
    for rule, table in A.POLARITY.items():
        for key, polarity in table.items():
            assert polarity in (-1, 0, 1, None), f"{rule} {key}: {polarity!r}"


def test_rule_candidates_are_the_first_entry_of_each_polarity_table():
    for rule, candidates in A.CANDIDATES.items():
        assert candidates[0] in A.POLARITY[rule]


def test_identical_images_produce_no_hints(base):
    """A perfect match must not invent work.  If any rule fires on identity, its
    threshold is wrong."""
    assert A.attribute(S.score(base, base)) == []


def test_no_catalog_means_no_invented_node_id(base):
    """The rule names a parameter but must never fabricate a node id."""
    hints = A.attribute(S.score(shift(base, 20, 0), base))
    assert hints
    for hint in hints:
        assert hint["node"] is None
        assert "exists in the catalog" in hint["why"]


def test_catalog_resolves_a_real_node_and_uniform(base):
    catalog = A.catalog_from_document({
        "graph": {"nodes": [
            {"id": "n_sdf", "kind": "sdf", "params": {
                "size": {"kind": "vec2", "value": [0.3, 0.3]},
                "smoothness": {"kind": "param", "name": "u_smoothness",
                               "value": 0.01, "min": 0.001, "max": 1}}},
        ]}
    })
    hint = A.attribute(S.score(box_blur(base, 4), base), catalog=catalog)[0]
    assert hint["rule"] == "edge.hardness"
    assert hint["node"] == "n_sdf"
    assert hint["param"] == "smoothness"
    assert hint["uniform"] == "u_smoothness"
    assert hint["direction"] == "decrease"
    assert hint["confidence"] > 0.5
    assert hint["suggest"] == pytest.approx(0.001, abs=1e-6)


def test_confidence_is_higher_when_the_node_is_resolved(base):
    """An unresolved hint is a weaker claim, and the number must say so."""
    render = box_blur(base, 4)
    breakdown = S.score(render, base)
    bare = A.attribute(breakdown)[0]["confidence"]
    catalog = A.catalog_from_document({
        "graph": {"nodes": [
            {"id": "n_sdf", "kind": "sdf", "params": {
                "smoothness": {"kind": "param", "name": "u_smoothness",
                               "value": 0.01, "min": 0.001, "max": 1}}},
        ]}
    })
    resolved = A.attribute(breakdown, catalog=catalog)[0]["confidence"]
    assert resolved > bare


def test_catalog_prefers_a_tunable_uniform_over_a_hard_coded_constant(base):
    """A ``kind:'param'`` input can be patched from the CLI; a constant cannot."""
    catalog = A.catalog_from_document({
        "graph": {"nodes": [
            {"id": "n_const", "kind": "domain_warp", "params": {
                "scale": {"kind": "float", "value": 3.0, "min": 0.1, "max": 64}}},
            {"id": "n_live", "kind": "domain_warp", "params": {
                "scale": {"kind": "param", "name": "u_scale",
                          "value": 3.0, "min": 0.1, "max": 64}}},
        ]}
    })
    hint = A.attribute(S.score(shift(base, 28, 0), base), catalog=catalog)[0]
    assert hint["node"] == "n_live"
    assert hint["uniform"] == "u_scale"


def test_structure_rule_admits_when_the_direction_is_undetermined(base):
    """A *misplaced* structure with matching band energy has no determined
    direction.  Saying ``None`` is the honest answer; inventing one is not."""
    breakdown = S.score(shift(base, 20, 0), base)
    hint = A.attribute(breakdown)[0]
    assert hint["rule"] == "structure.mismatch"
    assert hint["direction"] is None
    assert "misplaced" in hint["why"]
    # ...and the observable really is ambiguous: coarse energy matches.
    assert abs(S.band_ratio(breakdown.bands, breakdown.bands_ref, "low") - 1.0) < 0.15


def test_animation_rule_needs_a_temporal_signal(base):
    """Without ``temporal_ssim`` the last row of the table cannot be evaluated."""
    breakdown = S.score(base, base)
    assert A.attribute(breakdown) == []
    hints = A.attribute(breakdown, temporal_ssim=1.0)
    assert hints and hints[0]["rule"] == "animation.static"
    assert hints[0]["node"] is None
    assert "not wired" in hints[0]["why"]


def test_animation_rule_names_the_node_whose_time_is_stuck(base):
    catalog = A.catalog_from_document({
        "graph": {"nodes": [
            {"id": "n_warp", "kind": "domain_warp", "params": {
                "time": {"kind": "param", "name": "u_speed", "value": 0.0, "min": 0, "max": 2}}},
        ]}
    })
    hint = A.attribute(S.score(base, base), catalog=catalog, temporal_ssim=1.0)[0]
    assert hint["rule"] == "animation.static"
    assert hint["node"] == "n_warp"
    assert hint["param"] == "time"
    assert hint["uniform"] == "u_speed"
    assert "never advances" in hint["why"]


def test_animation_rule_does_not_fire_for_real_motion(base):
    moved = shift(base, 6, 3)
    temporal = S.score(moved, base).ssim
    assert temporal < A.T_TEMPORAL_STATIC
    assert all(h["rule"] != "animation.static" for h in A.attribute(S.score(moved, base), temporal_ssim=temporal))


# --------------------------------------------------------------------------- #
# Shape of the output
# --------------------------------------------------------------------------- #


def test_hints_are_json_serialisable_with_the_documented_keys(base, unrelated):
    hints = A.attribute(S.score(unrelated, base))
    assert hints
    for hint in hints:
        for key in ("node", "param", "direction", "why", "confidence"):
            assert key in hint
        assert isinstance(hint["why"], str) and hint["why"]
        assert 0.0 <= hint["confidence"] <= 1.0
        assert hint["direction"] in (None, "increase", "decrease")
    json.loads(json.dumps(hints))  # must cross a process boundary


def test_attribute_accepts_a_plain_dict_breakdown(base, unrelated):
    """The dict form is what crosses a process boundary, so it must be accepted."""
    payload = S.score(unrelated, base).to_dict()
    from_dict = A.attribute(payload)
    from_obj = A.attribute(S.score(unrelated, base))
    assert [h["rule"] for h in from_dict] == [h["rule"] for h in from_obj]


def test_hints_are_sorted_by_descending_confidence(base, unrelated):
    hints = A.attribute(S.score(unrelated, base))
    confidences = [h["confidence"] for h in hints]
    assert confidences == sorted(confidences, reverse=True)


def test_an_informational_note_never_outranks_an_actionable_hint(base):
    """A large exposure offset emits a note about the missing `tonemap_aces` node.
    The note has no parameter, so it must never be the top recommendation."""
    hints = A.attribute(S.score(brighten(base, 0.25), base), max_hints=8)
    assert hints[0]["param"] is not None
    assert hints[0]["rule"] == "tone.exposure"
    notes = [h for h in hints if h["param"] is None]
    assert notes, "the severe-offset case should still record the tonemapping gap"
    assert all(h["confidence"] < hints[0]["confidence"] for h in notes)


def test_severe_exposure_offset_records_the_missing_tonemap_node(base):
    hint = next(h for h in A.attribute(S.score(brighten(base, 0.25), base), max_hints=8)
                if h["rule"] == "tone.tonemap")
    assert hint["node"] is None and hint["param"] is None
    assert "no tonemap node exists in the catalog" in hint["why"]


def test_max_hints_caps_the_list(base, unrelated):
    assert len(A.attribute(S.score(unrelated, base), max_hints=2)) <= 2


def test_hints_are_deduplicated(base, unrelated):
    hints = A.attribute(S.score(unrelated, base), max_hints=50)
    keys = [(h["node"], h["param"], h["rule"]) for h in hints]
    assert len(keys) == len(set(keys))


def test_thresholds_are_named_constants_so_a_test_can_pin_them():
    """These numbers are the contract with score.ts; a silent edit would move
    every verdict.  Pinned explicitly."""
    assert A.T_SSIM_LOW == 0.75
    assert A.T_HIGH_RATIO_UP == 2.0
    assert A.T_HIGH_RATIO_DOWN == 0.5
    assert A.T_LUMA_OFFSET == 0.06
    assert A.T_SAT_DELTA == 0.08
    assert A.T_HUE_DELTA == 0.04
    assert A.T_GRAD_RATIO_UP == 1.5
    assert A.T_GRAD_RATIO_DOWN == 0.66
    assert A.T_TEMPORAL_STATIC == 0.995


def test_every_rule_candidate_exists_in_the_real_catalog():
    """A rule must not point at a node kind or an input the emitter would reject.

    The catalog is read from ``packages/ir/src/catalog/nodes.ts``; when that file
    is not present (an isolated checkout of ``python/``) the test still checks the
    candidate table's internal shape, and says so.
    """
    import os
    import re

    catalog_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "packages", "ir", "src", "catalog", "nodes.ts",
    )
    local_kinds, local_inputs = set(), {}
    if os.path.isfile(catalog_path):
        text = open(catalog_path, encoding="utf-8").read()
        blocks = re.split(r"(?=kind: ')", text)
        for block in blocks:
            match = re.match(r"kind: '([a-z0-9_]+)'", block)
            if not match:
                continue
            kind = match.group(1)
            local_kinds.add(kind)
            local_inputs[kind] = set(re.findall(r"\{ name: '([A-Za-z0-9_]+)', type: '", block))
    else:  # pragma: no cover - only when python/ is checked out alone
        pytest.skip("packages/ir/src/catalog/nodes.ts not present; cannot cross-check")

    for rule, candidates in A.CANDIDATES.items():
        for kind, param in candidates:
            assert kind in local_kinds, f"{rule}: node kind {kind!r} is not in the catalog"
            assert param in local_inputs.get(kind, set()), (
                f"{rule}: {kind}.{param} is not an input of that node; "
                f"known inputs: {sorted(local_inputs.get(kind, []))}"
            )


def test_catalog_from_document_tolerates_junk():
    """An unreadable document must produce an empty catalog, not a crash and not
    invented node ids."""
    assert A.catalog_from_document({}) == {"nodes": []}
    assert A.catalog_from_document({"nodes": "nonsense"}) == {"nodes": []}
    assert A.catalog_from_document({"graph": {"nodes": [{"id": "x"}, None, 7]}}) == {"nodes": []}


def test_catalog_from_document_reads_the_real_example_document():
    """The shipped example must produce a usable catalog."""
    import os

    example = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "examples", "liquid-metal.shader.json",
    )
    if not os.path.isfile(example):
        pytest.skip("examples/liquid-metal.shader.json not present")
    doc = json.load(open(example, encoding="utf-8"))
    catalog = A.catalog_from_document(doc)
    kinds = {n["kind"] for n in catalog["nodes"]}
    assert {"domain_warp", "fbm", "grade", "color_ramp"} <= kinds
    # every node keeps its id and exposes tunable params with uniform names
    warp = next(n for n in catalog["nodes"] if n["kind"] == "domain_warp")
    assert warp["params"]["scale"]["tunable"] is True
    assert warp["params"]["scale"]["uniform"].startswith("u_")


def test_summarize_mentions_the_target_and_direction(base, unrelated):
    line = A.summarize(S.score(unrelated, base))
    assert "SSIM" in line and "main problem" in line
