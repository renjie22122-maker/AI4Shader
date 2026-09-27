"""Failure attribution -- DESIGN.md section 4.3 as literal, testable rules.

"SSIM 0.61, not good enough" is useless to a model and to a user.  This module
turns a :class:`~ai4shader.score.ScoreBreakdown` into a ranked list of concrete
``{node, param, direction}`` suggestions, which is what makes a score
*actionable*.

The rules are the seven rows of DESIGN.md section 4.3's table:

======================================  =========================  ======================
observable                              attribution                rule id
======================================  =========================  ======================
low-frequency structure mismatch        shape / layout wrong       ``structure.mismatch``
high-frequency band ratio far from 1    noise frequency wrong      ``detail.frequency``
brightness histogram offset             exposure / tonemapping     ``tone.exposure``
HSV saturation delta                    too pale / too vivid       ``color.saturation``
hue distribution difference             palette wrong              ``color.hue``
gradient magnitude distribution         edge too hard / too soft   ``edge.hardness``
inter-frame SSIM ~ 1                    animation not wired        ``animation.static``
======================================  =========================  ======================

Every hint names a **concrete node input** and a **direction**, because that is
what converts a number into a patch.  Where a rule genuinely cannot determine the
direction, ``direction`` is ``None`` and the ``why`` says so -- that is more
useful than a coin-flip.  Where no node can be identified, ``node`` is ``None``
and ``param`` is still the design-level parameter name, so a caller can see what
kind of fix is needed without being handed a node id that does not exist.

Direction is a property of the parameter, not of the observation
----------------------------------------------------------------
A rule observes an *effect* ("the edges need to be harder"); a parameter has a
* polarity* (which way of turning it produces that effect).  These are different
things and conflating them produces confidently wrong advice:

* ``sdf.smoothness`` is the smoothstep **width**, so a *harder* edge means a
  **smaller** value.
* ``grade.contrast`` scales the gradient, so a *harder* edge means a **larger**
  value.

Both are legitimate answers to the same observation, and they point in opposite
directions.  :data:`POLARITY` therefore records, per rule and per candidate
``(kind, param)``, whether increasing the input increases the desired effect
(``+1``), decreases it (``-1``), or is unknown (``None``).  A parameter whose
polarity is unknown gets ``direction=None`` rather than a guess --
``domain_warp.scale``, for instance, controls the warp's *spatial frequency*, and
"does a finer warp raise or lower coarse-structure contrast?" is not answerable
without knowing the noise underneath it.

Interpreting DESIGN.md section 4.3
----------------------------------
The table is a design sketch, and three of its rows are ambiguous.  The
interpretations chosen here, and why:

1. **``sdf.size`` and ``domain_warp.scale`` are named without a direction.**
   ``size`` is a ``vec2`` (there is no scalar "increase"), and ``scale`` means
   opposite things on the two nodes it appears on (a larger ``fbm.scale`` means
   *finer* noise; a larger ``domain_warp.scale`` means a *finer* warp).  This
   module reads the direction off the **low-band energy ratio**, which is a real
   measurement, and only when the energy matches the reference (so the structure
   is *misplaced* rather than *mis-energised*) does it admit that the direction is
   undetermined and report ``direction=None``.  ``score.ts`` always picks one.
2. **``sdf.smoothness`` is inverted relative to its name.**  In the catalog the
   mask is ``1.0 - smoothstep(0.0, smoothness, d)``: ``smoothness`` is the
   smoothstep **width**, so a *larger* value is a *softer* edge and a *lower*
   gradient.  An "edges too sharp" observation therefore calls for
   ``smoothness`` to **increase**, which is the opposite of what a naive reading
   of the word suggests.  ``tests/test_attribute.py`` pins this explicitly.
3. **``tonemap_aces`` does not exist.**  The exposure row says "insert
   ``tonemap_aces``", but the catalog has no tonemap node (the 35 declared kinds
   contain none).  A large exposure offset therefore produces a second hint with
   ``node=None`` recording that the suggested fix is not expressible today,
   rather than naming a node kind an emitter would reject.

Confidence
----------
``confidence = strength * resolution`` where

* ``strength`` is monotone in the size of the measured effect, normalised by the
  rule's own threshold (see the ``_S_*`` constants), clipped to ``0..1``;
* ``resolution`` is ``1.0`` when the catalog resolved a concrete node id, ``0.6``
  when it did not, and ``0.4`` when there was no catalog at all.

Confidence is about *how sure the diagnosis is*, not about how much a patch would
help.  It is deliberately never ``1.0`` for an unresolved hint.

Units
-----
All band ratios are render/reference on the ``0..1`` luma scale; ``1.0`` means
"same energy".  Mean-luma offsets are absolute differences in ``0..1``.  Hue
differences are in **turns** and are wrapped to ``-0.5 .. 0.5``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, TypedDict

from .score import ScoreBreakdown, band_ratio

__all__ = [
    "Hint",
    "attribute",
    "summarize",
    "catalog_from_document",
    "RULES",
    "CANDIDATES",
    "POLARITY",
]


class Hint(TypedDict, total=False):
    """One actionable attribution.

    Required keys (always present, possibly ``None``):

    ``node``
        Document node id to patch, or ``None`` when no node could be identified.
        ``None`` always means "I could not name it", never "any node will do".
    ``param``
        Node **input** name (e.g. ``"scale"``, ``"smoothness"``), not the uniform
        name.  ``None`` only when the rule cannot name a parameter at all.
    ``direction``
        ``"increase"``, ``"decrease"`` or ``None`` when the observable does not
        determine it, or when the resolved parameter's polarity is unknown.
    ``why``
        One sentence naming the **measured quantity**, not a vibe.  Always says
        which node kinds were wanted when ``node`` is ``None``.
    ``confidence``
        ``0.0 .. 1.0``.  How sure the diagnosis is.  See the module docstring.

    Optional extras (documented, safe to ignore):

    ``rule``
        Stable rule id (see the module docstring's table).  Useful for tests,
        logging and "did the new rule fire?" questions.
    ``uniform``
        The uniform name this input is exposed as, when the catalog knew it --
        this is the string ``ai4shader emit --set`` / ``params --set`` wants.
    ``suggest``
        A concrete next value, when the parameter has a numeric range and current
        value.  ``None`` for non-scalar parameters (``vec2``/``color``).
    ``candidates``
        ``(kind, param)`` pairs the rule considered, in preference order.  This is
        the audit trail: it shows what the rule looked for when it failed to
        resolve a node.
    """

    node: str | None
    param: str | None
    direction: str | None
    why: str
    confidence: float
    rule: str
    uniform: str | None
    suggest: float | None
    candidates: list[tuple[str, str]]


# --------------------------------------------------------------------------- #
# Thresholds -- every one of them is a named constant, so a test can pin it.
# --------------------------------------------------------------------------- #

#: Below this 1/4-scale SSIM the coarse structure is considered wrong.  Same value
#: as ``structureWeak`` in ``score.ts``.
T_SSIM_LOW = 0.75
#: Low-band energy ratio outside ``1 +/- this`` is a real coarse-energy change.
T_LOW_RATIO = 0.15  # -> 0.85 and 1.25
#: DESIGN.md's literal trigger for the high-frequency rule: "> 2x".
T_HIGH_RATIO_UP = 2.0
T_HIGH_RATIO_DOWN = 0.5
#: Absolute mean-luma offset, in ``0..1`` luma units, that counts as exposure
#: drift.  Matches ``score.ts``.
T_LUMA_OFFSET = 0.06
#: Beyond this the exposure fix alone will clip, so tonemapping is implicated.
T_LUMA_OFFSET_SEVERE = 0.20
#: Absolute mean-HSV-saturation delta.  Matches ``score.ts``.
T_SAT_DELTA = 0.08
#: Absolute hue difference in turns (about 14 degrees).  Matches ``score.ts``.
T_HUE_DELTA = 0.04
#: Gradient-magnitude ratio bounds.  Match ``score.ts``.
T_GRAD_RATIO_UP = 1.5
T_GRAD_RATIO_DOWN = 0.66
#: Temporal SSIM above which two frames count as identical.  Matches ``score.ts``.
T_TEMPORAL_STATIC = 0.995

#: A change confined to the high band while the mid band stays put is a
#: *low-pass* signature (edge width, blur).  A change that moves the mid and high
#: bands together is a *frequency-content* signature (octave count, added grain).
#: This is the discriminator between the ``edge.hardness`` and ``detail.frequency``
#: rules, which otherwise both fire on a blur AND on added noise and would have to
#: be ranked by a coin flip.  Measured (``dev/measure_thresholds.py``, 256x256):
#: box blur moves mid by 0.03 to 0.09; added white noise at sigma 0.08/0.16 moves
#: mid by 0.15/0.42.
T_MID_MOVED = 0.15

#: When a broadband frequency mismatch is present, a gradient-magnitude change is
#: *explained by* that mismatch rather than by edge width, and an HSV-saturation
#: delta is largely an artefact -- adding achromatic noise raises measured HSV
#: saturation because S = (max-min)/max grows with the per-channel spread.  Both
#: rules are attenuated by these factors so the causal rule outranks them.  The
#: factors are below the smallest observed causal strength (saturation 1.00 on a
#: pure saturation change; edge 1.00 on a pure blur).
ATTENUATE_EXPLAINED_EDGE = 0.35
ATTENUATE_EXPLAINED_SATURATION = 0.30

#: Normalisers for the strength curves, chosen from the measured table in
#: ``dev/measure_thresholds.py`` so a just-barely-triggering observation scores
#: low and a blatant one scores near 1.
_S_SSIM_LOW = 0.45  # ssimLow 0.75 -> 0.0, 0.30 -> 1.0
_S_LOW_RATIO = 0.40
_S_HIGH_RATIO = 3.0  # ratio 2x -> 0.33, 4x -> 1.0
_S_LUMA = 0.25  # offset 0.06 -> 0.24, 0.31 -> 1.0
_S_SAT = 0.35
_S_HUE = 0.25
_S_GRAD = 1.5  # ratio 1.5 -> 0.39 (log2), 4x -> 1.0

_RESOLVED = 1.0
_UNRESOLVED_CATALOG = 0.6
_NO_CATALOG = 0.4

#: ``(kind, param)`` pairs each rule looks for, in preference order.  These come
#: straight from DESIGN.md section 4.3 and from the real catalog in
#: ``packages/ir/src/catalog/nodes.ts`` -- a kind or input that is not in that
#: file must not appear here (see interpretation note 3, and the test that checks
#: every pair against the catalog source).
CANDIDATES: dict[str, tuple[tuple[str, str], ...]] = {
    "structure.mismatch": (("sdf", "size"), ("sdf", "radius"), ("domain_warp", "strength"), ("domain_warp", "scale")),
    "detail.frequency": (("fbm", "scale"), ("noise", "scale"), ("fbm", "octaves"), ("fbm", "lacunarity")),
    "tone.exposure": (("grade", "exposure"), ("hsv", "value")),
    "color.saturation": (("grade", "saturation"), ("hsv", "saturation")),
    "color.hue": (("hsv", "hue_shift"), ("color_ramp", "colorB"), ("color_ramp", "colorA"), ("color_ramp", "colorC")),
    "edge.hardness": (("sdf", "smoothness"), ("grade", "contrast")),
    "animation.static": (("domain_warp", "time"), ("raymarch", "time")),
}

RULES = tuple(CANDIDATES)

#: Per rule, per candidate ``(kind, param)``: does **increasing** the input
#: increase the effect the rule wants?  ``+1`` yes, ``-1`` no, ``None`` unknown.
#:
#: Read the effects as:
#:
#: * ``structure.mismatch`` -> coarse-structure contrast
#: * ``detail.frequency``   -> amount of high-frequency detail
#: * ``tone.exposure``      -> brightness
#: * ``color.saturation``   -> HSV saturation
#: * ``color.hue``          -> hue rotation in the +turns direction
#: * ``edge.hardness``      -> edge steepness (gradient magnitude)
#: * ``animation.static``   -> elapsed time
#:
#: A ``None`` entry is not a gap to be filled in later by guessing: it is the
#: honest answer for a parameter whose effect on the measured quantity is not
#: monotone or not knowable from the catalog alone.
POLARITY: dict[str, dict[tuple[str, str], int | None]] = {
    "structure.mismatch": {
        ("sdf", "size"): +1,            # a bigger subject has more coarse contrast
        ("sdf", "radius"): +1,          # same, scalar
        ("domain_warp", "strength"): +1,  # more displacement, more rearrangement
        ("domain_warp", "scale"): None,  # spatial frequency: direction not knowable here
    },
    "detail.frequency": {
        ("fbm", "scale"): +1,           # scale multiplies uv: bigger = finer noise
        ("noise", "scale"): +1,
        ("fbm", "octaves"): +1,
        ("fbm", "lacunarity"): +1,
    },
    "tone.exposure": {
        ("grade", "exposure"): +1,
        ("hsv", "value"): +1,
    },
    "color.saturation": {
        ("grade", "saturation"): +1,
        ("hsv", "saturation"): +1,
    },
    "color.hue": {
        ("hsv", "hue_shift"): +1,
        ("color_ramp", "colorB"): None,  # RGBA stop: no scalar direction
        ("color_ramp", "colorA"): None,
        ("color_ramp", "colorC"): None,
    },
    "edge.hardness": {
        ("sdf", "smoothness"): -1,      # smoothstep WIDTH: bigger = softer. See note 2.
        ("grade", "contrast"): +1,      # more contrast = steeper transitions
    },
    "animation.static": {
        ("domain_warp", "time"): +1,
        ("raymarch", "time"): +1,
    },
}

#: Human clause explaining *why* the resolved parameter has that polarity.  Shown
#: appended to ``why`` when the parameter was resolved, so the explanation always
#: matches the thing being pointed at.
PARAM_NOTES: dict[tuple[str, str], dict[int, str]] = {
    ("sdf", "smoothness"): {
        -1: "`sdf.smoothness` is the smoothstep width, so narrowing it hardens the edge",
        +1: "`sdf.smoothness` is the smoothstep width, so widening it softens the edge",
    },
    ("grade", "contrast"): {
        +1: "`grade.contrast` scales the gradient, so raising it steepens the edge",
        -1: "`grade.contrast` scales the gradient, so lowering it softens the edge",
    },
    ("sdf", "size"): {
        +1: "`sdf.size` is a vec2; increasing it means enlarging the subject in both axes",
        -1: "`sdf.size` is a vec2; decreasing it means shrinking the subject in both axes",
    },
    ("sdf", "radius"): {
        +1: "a larger `sdf.radius` fills more of the frame with the subject",
        -1: "a smaller `sdf.radius` leaves more background in the frame",
    },
    ("domain_warp", "strength"): {
        +1: "more `domain_warp.strength` displaces the domain further",
        -1: "less `domain_warp.strength` displaces the domain less",
    },
    ("fbm", "scale"): {
        +1: "`scale` multiplies the sample point, so raising it makes the noise finer",
        -1: "`scale` multiplies the sample point, so lowering it makes the noise coarser",
    },
    ("noise", "scale"): {
        +1: "`scale` multiplies the sample point, so raising it makes the noise finer",
        -1: "`scale` multiplies the sample point, so lowering it makes the noise coarser",
    },
    ("hsv", "hue_shift"): {
        +1: "`hsv.hue_shift` is a rotation in turns",
        -1: "`hsv.hue_shift` is a rotation in turns",
    },
}

_OPPOSITE = {"increase": "decrease", "decrease": "increase"}


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


def _coerce_param(value: Any) -> dict[str, Any]:
    """Normalise one IR param entry into ``{uniform, label, min, max, value, ...}``."""
    if not isinstance(value, Mapping):
        return {"value": value, "tunable": False}
    out: dict[str, Any] = {
        "uniform": value.get("name"),
        "label": value.get("label"),
        "min": value.get("min"),
        "max": value.get("max"),
        "value": value.get("value"),
        "type": value.get("type"),
        # A `kind:'param'` entry is a live uniform: it can be changed from the CLI
        # without touching the document.  A constant input cannot.
        "tunable": value.get("kind") == "param",
    }
    if isinstance(out["value"], Sequence) and not isinstance(out["value"], str):
        out["value"] = list(out["value"])
    return out


def catalog_from_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Build the ``catalog=`` argument for :func:`attribute` from a ``.shader.json``.

    Accepts a whole IR document (``{"graph": {"nodes": [...]}}``) or a bare
    ``{"nodes": [...]}``.  Each node's params are flattened from the IR's
    ``{kind, name, label, min, max, value, ...}`` form into plain dicts, keeping
    the uniform name so a hint can tell the caller what to pass to ``--set``.

    Unknown shapes are tolerated: a document this function cannot read produces an
    empty catalog, and the attribution then reports unresolved hints instead of
    crashing or inventing node ids.
    """
    nodes = doc.get("nodes") if "nodes" in doc else None
    if nodes is None and isinstance(doc.get("graph"), Mapping):
        nodes = doc["graph"].get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        return {"nodes": []}

    out_nodes = []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        node_id = node.get("id")
        kind = node.get("kind")
        if not isinstance(node_id, str) or not isinstance(kind, str):
            continue
        params = node.get("params")
        flat = {k: _coerce_param(v) for k, v in params.items()} if isinstance(params, Mapping) else {}
        out_nodes.append({"id": node_id, "kind": kind, "params": flat})
    return {"nodes": out_nodes}


def _resolve(
    catalog: Mapping[str, Any] | None,
    candidates: Sequence[tuple[str, str]],
) -> tuple[dict[str, Any] | None, str | None, str | None, str | None, bool]:
    """Find the first catalog node offering one of ``candidates``.

    Returns ``(param_info, node_id, kind, param_name, resolved)``.  Two passes: a
    tunable (live-uniform) match is worth more than a hard-coded constant, because
    only the former can be patched without editing the document.
    """
    if catalog is None:
        return None, None, None, None, False
    nodes = catalog.get("nodes") or []
    for want_tunable in (True, False):
        for kind, param in candidates:
            for node in nodes:
                if node.get("kind") != kind:
                    continue
                info = (node.get("params") or {}).get(param)
                if info is None:
                    continue
                if bool(info.get("tunable")) != want_tunable:
                    continue
                return dict(info), str(node.get("id")), kind, param, True
    return None, None, None, None, False


def _nudge(info: Mapping[str, Any], direction: str | None) -> float | None:
    """A concrete next value one twelfth of the range in ``direction``.

    Mirrors ``nudge()`` in ``score.ts`` (``span/12``, clamped to ``[min, max]``,
    rounded to 3 decimals) so a suggestion computed here and one computed in the
    browser agree.  Returns ``None`` for a non-scalar or range-less parameter
    rather than guessing.
    """
    if direction is None:
        return None
    value = info.get("value")
    lo, hi = info.get("min"), info.get("max")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    span = (hi - lo) if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) else abs(value) or 1.0
    step = span / 12.0
    nxt = value + step if direction == "increase" else value - step
    if isinstance(lo, (int, float)):
        nxt = max(lo, nxt)
    if isinstance(hi, (int, float)):
        nxt = min(hi, nxt)
    return round(float(nxt), 3)


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _fmt(x: float) -> str:
    """Two decimals, with ``-0.00`` and non-finite values made readable."""
    if math.isinf(x):
        return "inf"
    if math.isnan(x):
        return "n/a"
    return "0.00" if abs(x) < 0.005 else f"{x:.2f}"


def _wrap_turns(delta: float) -> float:
    """Wrap a hue difference into ``-0.5 .. 0.5`` turns (hue is a circle)."""
    while delta > 0.5:
        delta -= 1.0
    while delta < -0.5:
        delta += 1.0
    return delta


def _resolve_direction(
    rule: str, kind: str | None, param: str | None, want: str | None, resolved: bool
) -> str | None:
    """Turn "the effect should be more/less" into a direction for *this* parameter.

    ``want`` is ``"more"``, ``"less"`` or ``None`` (the observable does not
    determine it).  The parameter's polarity then decides which way to turn it, and
    an unknown polarity yields ``None`` rather than a guess.
    """
    if want is None:
        return None
    target = "increase" if want == "more" else "decrease"
    if not resolved:
        # No catalog: the design-level parameter is the primary candidate, so use
        # its polarity.  Only the *target* is unknown, and `node=None` says that.
        kind, param = CANDIDATES[rule][0]
    polarity = POLARITY.get(rule, {}).get((kind or "", param or ""))
    if polarity is None:
        return None
    return target if polarity > 0 else _OPPOSITE[target]


def _make(
    *,
    rule: str,
    effect_wanted: str | None,
    why: str,
    strength: float,
    catalog: Mapping[str, Any] | None,
    candidates: Sequence[tuple[str, str]],
    resolution: float | None = None,
    extra: str = "",
) -> Hint:
    """Assemble one hint: resolve the catalog, derive the direction, score confidence."""
    info, node_id, kind, param, resolved = _resolve(catalog, candidates)
    if not resolved:
        # Unresolved: keep the design-level parameter name (the rule *does* name a
        # parameter -- "fbm.scale" is one), but refuse to invent a node id.
        param = candidates[0][1]
        kind = candidates[0][0]

    direction = _resolve_direction(rule, kind, param, effect_wanted, resolved)

    # The explanatory clause must describe the parameter actually being pointed at,
    # never a different one from the candidate list -- otherwise the reader is told
    # to narrow `sdf.smoothness` while the target line says `grade.contrast`.
    if resolved:
        polarity = POLARITY.get(rule, {}).get((kind or "", param or ""), 0)
        clause = PARAM_NOTES.get((kind or "", param or ""), {}).get(1 if (polarity or 0) >= 0 else -1)
        if clause:
            extra = f"{extra}; {clause}" if extra else clause

    if resolution is None:
        resolution = _RESOLVED if resolved else (_UNRESOLVED_CATALOG if catalog else _NO_CATALOG)

    if not resolved:
        kinds = " or ".join(sorted({f"`{k}`" for k, _ in candidates}))
        extra = (
            f"{extra}; " if extra else ""
        ) + f"No {kinds} node with an input named `{candidates[0][1]}` exists in the catalog."

    return Hint(
        node=node_id,
        param=param,
        direction=direction,
        why=f"{why}; {extra}" if extra else why,
        confidence=round(_clamp01(strength) * resolution, 4),
        rule=rule,
        uniform=(info or {}).get("uniform"),
        suggest=_nudge(info or {}, direction),
        candidates=[tuple(c) for c in candidates],
    )


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #


def attribute(
    breakdown: ScoreBreakdown | Mapping[str, Any],
    *,
    catalog: Mapping[str, Any] | None = None,
    temporal_ssim: float | None = None,
    max_hints: int = 4,
) -> list[Hint]:
    """Turn a score breakdown into ranked, actionable patches.

    * ``breakdown`` -- a :class:`~ai4shader.score.ScoreBreakdown` or its
      ``to_dict()`` form (both are accepted; the dict form is what crosses a
      process boundary).
    * ``catalog`` -- optional.  Build it with :func:`catalog_from_document`.  With
      a catalog, hints name a real node id and the uniform to pass to ``--set``;
      without one, ``node`` is ``None`` and the ``why`` says which node kinds the
      rule looked for.  **No node id is ever invented.**
    * ``temporal_ssim`` -- SSIM between two frames of the *current* render at
      different times.  Required to evaluate the "animation is not wired" rule
      (DESIGN.md section 4.3, last row); with ``None`` that rule cannot fire.
      Compute it with ``score.score(frame_a, frame_b).ssim``.
    * ``max_hints`` -- cap on returned hints (default 4, matching ``attribute()``
      in ``score.ts``).  Hints are sorted by descending confidence, so the cap
      drops the weakest.

    Returns a list of :class:`Hint` dicts, JSON-serialisable, strongest first.  An
    empty list means "nothing in the table fired" -- it does **not** mean the
    render matches the reference; check ``breakdown.ssim`` for that.
    """
    if isinstance(breakdown, Mapping):
        breakdown = ScoreBreakdown.from_dict(dict(breakdown))

    hints: list[Hint] = []
    b = breakdown

    high_ratio = band_ratio(b.bands, b.bands_ref, "high")
    mid_ratio = band_ratio(b.bands, b.bands_ref, "mid")
    low_ratio = band_ratio(b.bands, b.bands_ref, "low")

    # -- Rule 1: coarse structure / layout ---------------------------------- #
    if b.ssim_low is not None and b.ssim_low < T_SSIM_LOW:
        strength = _clamp01((T_SSIM_LOW - b.ssim_low) / _S_SSIM_LOW)
        head = f"coarse structure disagrees (1/4-scale SSIM {_fmt(b.ssim_low)} < {T_SSIM_LOW})"
        if low_ratio < 1.0 - T_LOW_RATIO:
            hints.append(
                _make(
                    rule="structure.mismatch",
                    effect_wanted="more",
                    why=(
                        f"{head}; its coarse-band energy is only {_fmt(low_ratio)}x the reference, "
                        "so the subject is too small or too flat for the frame"
                    ),
                    strength=strength,
                    catalog=catalog,
                    candidates=CANDIDATES["structure.mismatch"],
                )
            )
        elif low_ratio > 1.0 + T_LOW_RATIO:
            hints.append(
                _make(
                    rule="structure.mismatch",
                    effect_wanted="less",
                    why=(
                        f"{head}; its coarse-band energy is {_fmt(low_ratio)}x the reference, "
                        "so the subject is too large or too broken up"
                    ),
                    strength=strength,
                    catalog=catalog,
                    candidates=CANDIDATES["structure.mismatch"],
                )
            )
        else:
            # Energy matches but the layout does not.  Band energy cannot say which
            # way to move, and inventing a direction here is how an optimiser gets
            # walked off a cliff.
            hints.append(
                _make(
                    rule="structure.mismatch",
                    effect_wanted=None,
                    why=(
                        f"{head}, but its coarse-band energy matches the reference "
                        f"({_fmt(low_ratio)}x): the structure is misplaced rather than mis-energised, "
                        "so no direction is determined -- explore this parameter"
                    ),
                    strength=strength,
                    catalog=catalog,
                    candidates=CANDIDATES["structure.mismatch"],
                )
            )

    # -- Rule 2: detail / noise frequency ----------------------------------- #
    if math.isinf(high_ratio) or high_ratio > T_HIGH_RATIO_UP or high_ratio < T_HIGH_RATIO_DOWN:
        if high_ratio > T_HIGH_RATIO_UP or math.isinf(high_ratio):
            want = "less"
            shown = "inf" if math.isinf(high_ratio) else _fmt(high_ratio)
            why = (
                f"fine-detail energy is {shown}x the reference (DESIGN.md's trigger is > {T_HIGH_RATIO_UP}x): "
                "the noise is finer than the reference"
            )
            extreme = math.isinf(high_ratio)
        else:
            want = "more"
            why = (
                f"fine-detail energy is only {_fmt(high_ratio)}x the reference "
                f"(trigger < {T_HIGH_RATIO_DOWN}x): the noise is coarser than the reference"
            )
            extreme = False
        magnitude = math.inf if math.isinf(high_ratio) else abs(math.log2(high_ratio))
        # Strength is linear in the energy ratio's distance from 1, not in its log:
        # a 4x surplus and a 4x deficit are equally blatant.
        strength = _clamp01((math.inf if math.isinf(high_ratio) else abs(high_ratio - 1.0)) / _S_HIGH_RATIO)
        # A pure low-pass (blur) leaves the mid band alone; say so, because the
        # edge-width rule is then the better explanation of the same observation.
        extra = ""
        if abs(mid_ratio - 1.0) < T_MID_MOVED:
            extra = (
                f"the mid band is unchanged ({_fmt(mid_ratio)}x), which is a low-pass signature -- "
                "if the render is simply too soft, the edge-width rule is the better explanation"
            )
        hints.append(
            _make(
                rule="detail.frequency",
                effect_wanted=want,
                why=why,
                strength=strength,
                catalog=catalog,
                candidates=CANDIDATES["detail.frequency"],
                extra=extra,
            )
        )
        # An extreme mismatch is usually the octave count, not the base scale:
        # changing `scale` alone cannot make up a 4x energy deficit.
        if extreme or magnitude >= 2.0:
            hints.append(
                _make(
                    rule="detail.frequency",
                    effect_wanted=want,
                    why=(
                        f"the fine-detail mismatch is {_fmt(magnitude)} octaves wide, which a base-frequency "
                        "change alone will not cover; the octave count is the likely cause"
                    ),
                    strength=strength * 0.8,
                    catalog=catalog,
                    candidates=(("fbm", "octaves"), ("fbm", "lacunarity")),
                )
            )

    # -- Rule 3: exposure / tonemapping ------------------------------------- #
    offset = b.luma_offset
    if offset is None and b.mean_luma:
        offset = b.mean_luma.get("cur", 0.0) - b.mean_luma.get("ref", 0.0)
    if offset is not None and abs(offset) > T_LUMA_OFFSET:
        want = "less" if offset > 0 else "more"
        word = "brighter" if offset > 0 else "darker"
        strength = _clamp01(abs(offset) / _S_LUMA)
        exposure_hint = _make(
            rule="tone.exposure",
            effect_wanted=want,
            why=f"mean luma is {_fmt(abs(offset))} {word} than the reference (0..1 luma scale)",
            strength=strength,
            catalog=catalog,
            candidates=CANDIDATES["tone.exposure"],
        )
        hints.append(exposure_hint)
        if abs(offset) > T_LUMA_OFFSET_SEVERE:
            # DESIGN.md asks for `tonemap_aces` here.  There is no such node kind in
            # the catalog (35 kinds, none of them a tonemap), so this hint says so
            # instead of naming a node the emitter would reject.
            #
            # Its confidence is pinned below the exposure hint it annotates: a note
            # with no target must never outrank the patch that does have one, or
            # "increase grade.exposure" would silently become "nothing you can do".
            hints.append(
                {
                    "node": None,
                    "param": None,
                    "direction": None,
                    "why": (
                        f"the mean-luma offset is {_fmt(abs(offset))}, large enough that an exposure change alone "
                        "will clip highlights; DESIGN.md section 4.3 suggests inserting `tonemap_aces`, but no "
                        "tonemap node exists in the catalog (packages/ir/src/catalog/nodes.ts), so this needs "
                        "either a new node or a manual curve"
                    ),
                    "confidence": round(float(exposure_hint["confidence"]) * 0.9, 4),
                    "rule": "tone.tonemap",
                    "uniform": None,
                    "suggest": None,
                    "candidates": [],
                }
            )

    # -- Rule 4: saturation -------------------------------------------------- #
    if b.saturation:
        delta = b.saturation.get("cur", 0.0) - b.saturation.get("ref", 0.0)
        if abs(delta) > T_SAT_DELTA:
            want = "less" if delta > 0 else "more"
            word = "more saturated" if delta > 0 else "less saturated"
            strength = abs(delta) / _S_SAT
            broadband = high_ratio > T_HIGH_RATIO_UP or high_ratio < T_HIGH_RATIO_DOWN
            extra = ""
            if broadband:
                # S = (max-min)/max grows with per-channel spread, so added achromatic
                # noise or a blur *moves* this observable without there being anything
                # wrong with the palette.  Do not let it outrank the frequency rule
                # that actually explains the difference.
                strength *= ATTENUATE_EXPLAINED_SATURATION
                extra = (
                    "this observable also moves when fine detail changes, and the high band "
                    f"is {_fmt(high_ratio)}x, so the palette may not be the cause"
                )
            hints.append(
                _make(
                    rule="color.saturation",
                    effect_wanted=want,
                    why=(
                        f"mean HSV saturation is {_fmt(abs(delta))} {word} than the reference "
                        f"({_fmt(b.saturation.get('cur', 0.0))} vs {_fmt(b.saturation.get('ref', 0.0))})"
                    ),
                    strength=strength,
                    catalog=catalog,
                    candidates=CANDIDATES["color.saturation"],
                    extra=extra,
                )
            )

    # -- Rule 5: hue / palette ----------------------------------------------- #
    if b.hue_mean:
        delta = _wrap_turns(b.hue_mean.get("cur", 0.0) - b.hue_mean.get("ref", 0.0))
        if abs(delta) > T_HUE_DELTA:
            want = "less" if delta > 0 else "more"
            hints.append(
                _make(
                    rule="color.hue",
                    effect_wanted=want,
                    why=(
                        f"the dominant hue is {_fmt(abs(delta) * 360.0)} degrees "
                        f"{'ahead of' if delta > 0 else 'behind'} the reference "
                        f"(circular mean, {_fmt(abs(delta))} turns)"
                    ),
                    strength=_clamp01(abs(delta) / _S_HUE),
                    catalog=catalog,
                    candidates=CANDIDATES["color.hue"],
                )
            )

    # -- Rule 6: edge hardness ----------------------------------------------- #
    if b.gradient:
        g_ratio = b.gradient.get("cur", 0.0) / max(b.gradient.get("ref", 0.0), 1e-12)
        if g_ratio > T_GRAD_RATIO_UP or g_ratio < T_GRAD_RATIO_DOWN:
            magnitude = abs(math.log2(max(g_ratio, 1e-12)))
            want = "less" if g_ratio > T_GRAD_RATIO_UP else "more"
            word = "harder" if want == "less" else "softer"
            why = f"edges are {_fmt(g_ratio)}x {word} than the reference (gradient RMS)"
            edge_strength = _clamp01(magnitude / _S_GRAD)

            # Which explanation does the pyramid support?  A gradient change that
            # comes with a mid-band change is broadband -- the edge statistic is a
            # *consequence* of a frequency-content change, and the detail rule owns
            # that.  A gradient change with the mid band intact is edge width.
            moved_mid = abs(mid_ratio - 1.0) >= T_MID_MOVED
            same_way = (g_ratio > 1.0) == (high_ratio > 1.0)
            extra = ""
            if moved_mid and same_way:
                edge_strength *= ATTENUATE_EXPLAINED_EDGE
                extra = (
                    f"the mid band also moved ({_fmt(mid_ratio)}x), so this is broadband fine detail "
                    "rather than edge width -- the frequency rule is the better explanation"
                    if g_ratio > 1.0
                    else f"the mid band also moved ({_fmt(mid_ratio)}x), so fine detail was lost "
                    "broadly rather than the edge being softened"
                )
            elif abs(mid_ratio - 1.0) < T_MID_MOVED:
                extra = f"the mid band matches ({_fmt(mid_ratio)}x), so this is edge width, not lost detail"

            hints.append(
                _make(
                    rule="edge.hardness",
                    effect_wanted=want,
                    why=why,
                    strength=edge_strength,
                    catalog=catalog,
                    candidates=CANDIDATES["edge.hardness"],
                    extra=extra,
                )
            )

    # -- Rule 7: is it animating at all? ------------------------------------- #
    if temporal_ssim is not None and temporal_ssim > T_TEMPORAL_STATIC:
        strength = _clamp01((temporal_ssim - T_TEMPORAL_STATIC) / max(1e-6, 1.0 - T_TEMPORAL_STATIC))
        strength = max(strength, 0.5)  # SSIM 1.000 exactly is not a marginal call
        info, node_id, _kind, param, resolved = _resolve(catalog, CANDIDATES["animation.static"])
        value = (info or {}).get("value")
        stuck = resolved and isinstance(value, (int, float)) and abs(float(value)) < 1e-9
        if stuck:
            extra = f"node `{node_id}` has `{param}` = {value}, i.e. time never advances"
        elif resolved:
            extra = "check that the `time` input is actually connected to iTime"
        else:
            extra = ""
        hints.append(
            _make(
                rule="animation.static",
                effect_wanted="more" if resolved else None,
                why=(
                    f"two frames at different times are identical (temporal SSIM {_fmt(temporal_ssim)}): "
                    "the animation is not wired"
                ),
                strength=strength,
                catalog=catalog,
                candidates=CANDIDATES["animation.static"],
                extra=extra,
            )
        )

    # -- Rank, dedupe, cap --------------------------------------------------- #
    # Sort key: actionable hints first, then by confidence, then by the design
    # table's row order.  "Actionable" means the hint names a parameter -- a note
    # with ``param=None`` (e.g. "this needs a tonemap node that does not exist") is
    # information, never the top recommendation.
    order = {rule: i for i, rule in enumerate(RULES)}

    def sort_key(h: Hint) -> tuple[int, float, int]:
        return (0 if h["param"] else 1, -h["confidence"], order.get(h["rule"], 99))

    hints.sort(key=sort_key)
    out: list[Hint] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()
    for hint in hints:
        key = (hint["node"], hint["param"], hint["rule"])
        if key in seen:
            continue
        seen.add(key)
        out.append(hint)
        if len(out) >= max_hints:
            break
    return out


def summarize(breakdown: ScoreBreakdown | Mapping[str, Any], hints: Sequence[Hint] | None = None) -> str:
    """One human-readable line for a UI, in the same shape as ``score.ts``'s verdict.

    ``"SSIM 0.612 . hist 0.088 -- main problem: coarse structure disagrees ... [node.param increase]"``
    """
    if isinstance(breakdown, Mapping):
        breakdown = ScoreBreakdown.from_dict(dict(breakdown))
    head = f"SSIM {breakdown.ssim:.3f} . hist {breakdown.hist:.3f}"
    if hints is None:
        hints = attribute(breakdown)
    if not hints:
        return f"{head} -- every dimension in the attribution table agrees with the reference"
    top = hints[0]
    target = f"{top['node']}.{top['param']}" if top["node"] else f"(no node).{top['param']}"
    direction = f" {top['direction']}" if top["direction"] else ""
    return f"{head} -- main problem: {top['why']} [{target}{direction}, confidence {top['confidence']:.2f}]"
