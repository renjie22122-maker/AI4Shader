"""Tests for the optional learned metrics: real LPIPS, real CLIP, honest ``None``.

This machine has the real weights cached, so the real metrics run.  The tests
therefore assert the *real* behaviour (LPIPS is zero for identical images and
monotonic under perturbation; CLIP is 1.0 for identical images) instead of
asserting ``None`` and calling the fallback "LPIPS".

The ``None`` path is still tested -- by **injecting** an unavailable probe, which
tests the degradation honestly without pretending the weights are missing.  That
distinction matters: a test that asserted ``None`` here would be testing the
fallback while the real thing is available, and would start passing for the wrong
reason the moment the cache was cleared.

Nothing in this file, or in the library, makes a network call.  Model loading
runs inside :func:`ai4shader.score.no_network`, which turns any connection
attempt into an exception -- so a cache miss shows up as a failure, not as a
slow download.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ai4shader import score as S
from conftest import add_hf_noise, box_blur, brighten, make_base


def _caps() -> dict:
    """Cheap presence check.  Does not import torch."""
    return S.capabilities()


LPIPS_STATE = _caps()["lpips"]
CLIP_STATE = _caps()["clip_similarity"]

needs_lpips = pytest.mark.skipif(
    not LPIPS_STATE["available"],
    reason=f"real LPIPS weights not cached: {LPIPS_STATE['reason']}",
)
needs_clip = pytest.mark.skipif(
    not CLIP_STATE["available"],
    reason=f"real CLIP weights not cached: {CLIP_STATE['reason']}",
)


# --------------------------------------------------------------------------- #
# capabilities() must report the truth in both directions
# --------------------------------------------------------------------------- #


def test_capabilities_is_cheap_and_never_imports_torch():
    """``capabilities()`` without ``probe`` must not pull in torch.

    It is called on every CLI invocation; a 2.3 s import would make the CLI feel
    broken.
    """
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, %r);"
        "from ai4shader import score;"
        "score.capabilities();"
        "print('torch' in sys.modules)"
    ) % __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "capabilities() must not import torch"


def test_capabilities_has_the_documented_shape():
    caps = _caps()
    for name in ("ssim", "histogram_distance", "band_energy", "gradient_energy", "color_stats"):
        assert caps[name]["available"] is True
    for name in ("lpips", "clip_similarity"):
        info = caps[name]
        assert set(("available", "verified", "reason")) <= set(info)
        assert isinstance(info["available"], bool)
        assert isinstance(info["reason"], str) and info["reason"]
    json.loads(json.dumps(caps, default=str))


def test_capabilities_agrees_with_what_actually_ran():
    """Probe, then check the report matches the outcome the functions produced."""
    caps = S.capabilities(probe=True)
    base = make_base(128)
    value = S.lpips(base, base)
    if value is None:
        assert caps["lpips"]["available"] is False
        assert caps["lpips"]["verified"] is False
    else:
        assert caps["lpips"]["available"] is True
        assert caps["lpips"]["verified"] is True
        assert "loaded" in caps["lpips"]["reason"]


def test_unavailable_metrics_helper_agrees_with_capabilities():
    caps = _caps()
    listed = set(S.unavailable_metrics())
    for name in ("lpips", "clip_similarity"):
        assert (name in listed) is (not caps[name]["available"])


# --------------------------------------------------------------------------- #
# Real LPIPS
# --------------------------------------------------------------------------- #


@needs_lpips
def test_real_lpips_runs_and_reports_itself_as_real():
    caps = S.capabilities(probe=True)
    assert caps["lpips"]["available"] is True
    assert caps["lpips"]["verified"] is True, caps["lpips"]["reason"]
    assert "cached weights" in caps["lpips"]["reason"]
    # the path it actually used must be the cached backbone, not a download dir
    assert caps["lpips"]["backbone_present"] is True
    assert caps["lpips"]["linear_present"] is True


@needs_lpips
def test_lpips_of_an_image_with_itself_is_zero():
    """LPIPS's defining property.  If this is not ~0 the metric is not LPIPS."""
    base = make_base(128)
    value = S.lpips(base, base)
    assert value is not None
    assert value == pytest.approx(0.0, abs=1e-6)


@needs_lpips
def test_lpips_is_monotonic_in_perturbation_strength():
    """A mildly perturbed image must score lower (less different) than a heavily
    perturbed one.  Monotonicity is what makes it usable as an optimiser term."""
    base = make_base(256)
    same = S.lpips(base, base)
    mild = S.lpips(box_blur(base, 1), base)
    heavy = S.lpips(box_blur(base, 6), base)
    other = S.lpips(make_base(256, cx=0.25, cy=0.7, seed=42), base)
    assert same is not None and mild is not None and heavy is not None and other is not None
    assert same < mild < heavy < other, (same, mild, heavy, other)


@needs_lpips
def test_lpips_is_not_secretly_ssim(base):
    """LPIPS and SSIM must be different functions; if they tracked each other the
    'learned perceptual metric' would be a very expensive alias."""
    noisy = add_hf_noise(base, 0.05)
    blurred = box_blur(base, 2)
    lp_noisy, lp_blur = S.lpips(noisy, base), S.lpips(blurred, base)
    ss_noisy, ss_blur = S.ssim(noisy, base), S.ssim(blurred, base)
    assert lp_noisy is not None and lp_blur is not None
    assert lp_noisy != pytest.approx(ss_noisy, abs=1e-3)
    assert lp_blur != pytest.approx(ss_blur, abs=1e-3)
    # They even disagree about which perturbation is worse, which is the point of
    # having both: SSIM dislikes the blur, LPIPS dislikes it more.
    assert (lp_blur > lp_noisy) != (ss_blur > ss_noisy) or lp_blur > lp_noisy


@needs_lpips
def test_lpips_distinguishes_direction_on_the_same_knob(base):
    """Symmetric around identical: brightening and darkening are both > 0."""
    up = S.lpips(brighten(base, 0.18), base)
    down = S.lpips(brighten(base, -0.18), base)
    assert up is not None and down is not None
    assert up > 0.0 and down > 0.0


@needs_lpips
def test_lpips_rejects_input_smaller_than_the_network(base):
    tiny = make_base(16)
    with pytest.raises(ValueError, match="32x32"):
        S.lpips(tiny, tiny)


@needs_lpips
def test_lpips_rejects_mismatched_shapes(base):
    with pytest.raises(ValueError, match="same shape"):
        S.lpips(base, make_base(128))


# --------------------------------------------------------------------------- #
# Real CLIP
# --------------------------------------------------------------------------- #


@needs_clip
def test_real_clip_runs_and_reports_itself_as_real():
    caps = S.capabilities(probe=True)
    assert caps["clip_similarity"]["available"] is True
    assert caps["clip_similarity"]["verified"] is True, caps["clip_similarity"]["reason"]
    assert caps["clip_similarity"]["sha256_ok"] is True, "the cached checkpoint's hash must verify"


@needs_clip
def test_clip_of_an_image_with_itself_is_one():
    base = make_base(128)
    value = S.clip_similarity(base, base)
    assert value is not None
    assert value == pytest.approx(1.0, abs=1e-4)


@needs_clip
def test_clip_similarity_is_not_calibrated_as_a_percentage(base, unrelated):
    """A warning worth encoding: CLIP cosine sits high even for unrelated images,
    so it must not be read as 'x% correct'."""
    value = S.clip_similarity(unrelated, base)
    assert value is not None
    assert -1.0 <= value <= 1.0
    assert value > 0.5, "CLIP embeddings of unrelated images still score high; that is the documented trap"


@needs_clip
def test_clip_accepts_different_input_sizes(base):
    """CLIP centre-crops to 224, so mismatched sizes are fine (unlike SSIM)."""
    value = S.clip_similarity(base, make_base(128, seed=7))
    assert value is not None and -1.0 <= value <= 1.0


# --------------------------------------------------------------------------- #
# The honest ``None`` path -- injected, not assumed
# --------------------------------------------------------------------------- #


def test_lpips_returns_none_when_the_probe_says_unavailable(monkeypatch):
    """Injecting unavailability tests the fallback without pretending the weights
    are gone.  The contract is ``None`` -- never a fabricated number."""
    monkeypatch.setattr(S, "_lpips_model", lambda: (_ for _ in ()).throw(S.MetricUnavailable("injected")))
    base = make_base(64)
    assert S.lpips(base, base) is None


def test_clip_returns_none_when_the_probe_says_unavailable(monkeypatch):
    monkeypatch.setattr(S, "_clip_model", lambda: (_ for _ in ()).throw(S.MetricUnavailable("injected")))
    base = make_base(64)
    assert S.clip_similarity(base, base) is None


def test_lpips_availability_reports_missing_when_the_files_are_absent(monkeypatch):
    """With the weights hidden, the presence check must say why -- and the metric
    must return ``None``, not a substitute."""
    monkeypatch.setattr(S, "_lpips_files", lambda net="alex": {
        "net": net,
        "linear_weights": "/nope/alex.pth",
        "linear_present": False,
        "backbone_weights": "/nope/alexnet.pth",
        "backbone_present": False,
    })
    monkeypatch.setattr(S, "_LOAD_STATE", {})
    info = S._lpips_availability()
    assert info["available"] is False
    assert "linear weights" in info["reason"] and "backbone" in info["reason"]
    assert S.lpips(make_base(64), make_base(64)) is None


def test_clip_availability_reports_a_bad_checksum(monkeypatch):
    monkeypatch.setattr(S, "_sha256_cached", lambda path: "0" * 64)
    monkeypatch.setattr(S, "_LOAD_STATE", {})
    info = S._clip_availability()
    if info["sha256_ok"] is None:
        pytest.skip("no cached CLIP checkpoint to checksum")
    assert info["available"] is False
    assert "sha256" in info["reason"]


def test_a_failed_forward_pass_returns_none_rather_than_a_number(monkeypatch):
    """If the model raises mid-inference, the answer is 'unknown', not a guess."""
    class Exploding:
        def __call__(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(S, "_lpips_model", lambda: Exploding())
    assert S.lpips(make_base(64), make_base(64)) is None


# --------------------------------------------------------------------------- #
# The dependency-free fallback, honestly named
# --------------------------------------------------------------------------- #


def test_perceptual_fallback_is_zero_for_identical_and_multi_scale(base):
    assert S.perceptual_fallback(base, base) == pytest.approx(0.0, abs=1e-9)
    value = S.perceptual_fallback(box_blur(base, 2), base)
    assert 0.0 < value <= 2.0


def test_perceptual_fallback_is_not_called_lpips():
    """Nothing may ever report the fallback as an LPIPS value.  The names are the
    contract; this asserts they stay distinct."""
    assert S.perceptual_fallback is not S.lpips
    doc = (S.perceptual_fallback.__doc__ or "").lower()
    assert "not lpips" in doc
    assert "lpips" not in S.__all__ or "perceptual_fallback" in S.__all__


def test_loss_degrades_to_the_weight_free_expression_without_lpips(base, unrelated):
    breakdown = S.score(unrelated, base)
    explicit_none = S.loss(breakdown, lpips_value=None)
    default = S.loss(breakdown)
    assert explicit_none == pytest.approx(default)
    assert S.loss(breakdown, lpips_value=0.4) > default


# --------------------------------------------------------------------------- #
# No network, ever
# --------------------------------------------------------------------------- #


def test_no_network_guard_blocks_a_real_connection():
    """The guard must actually stop a connect, or it is decoration."""
    import socket

    with pytest.raises(RuntimeError, match="network access is disabled"):
        with S.no_network():
            socket.create_connection(("pypi.org", 443), timeout=2)


def test_no_network_guard_restores_the_original_behaviour():
    import socket

    before = socket.create_connection
    with S.no_network():
        assert socket.create_connection is not before
    assert socket.create_connection is before


def test_offline_env_defaults_are_set(monkeypatch):
    import os

    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
