"""Tests for :mod:`ai4shader.score`.

Two things are being pinned here beyond "the metric runs":

1. **Agreement with ``apps/web/src/compare/score.ts``.**  The Python module is
   the reference implementation, but a number computed in the browser has to
   mean the same thing.  ``test_ssim_matches_the_typescript_formula`` transcribes
   the TypeScript expression literally and checks the two agree; if this test
   ever fails, the two implementations have drifted and every threshold in
   ``attribute.py`` is suspect.
2. **The ``high`` band bug.**  ``score.ts`` computes its high band against the
   wrong pyramid level, which makes it a proxy for global contrast instead of
   fine detail.  ``test_high_band_is_sensitive_to_fine_detail`` demonstrates the
   correct behaviour and ``test_ts_compat_reproduces_the_score_ts_bug`` pins the
   compatibility shim, so the divergence is explicit rather than accidental.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ai4shader import score as S
from conftest import add_hf_noise, box_blur, brighten, make_base, smooth_gradient, white_noise

# --------------------------------------------------------------------------- #
# SSIM
# --------------------------------------------------------------------------- #


def test_ssim_of_an_image_with_itself_is_exactly_one(base):
    assert abs(S.ssim(base, base) - 1.0) < 1e-6


def test_ssim_is_symmetric(base, unrelated):
    assert abs(S.ssim(base, unrelated) - S.ssim(unrelated, base)) < 1e-9
    noisy = add_hf_noise(base, 0.05)
    assert abs(S.ssim(base, noisy) - S.ssim(noisy, base)) < 1e-9


def test_ssim_of_unrelated_images_is_far_below_one(base, unrelated):
    assert S.ssim(base, unrelated) < 0.6


def test_ssim_is_monotonic_in_brightness_shift(headroom, unrelated):
    """A brightness shift must land strictly between identical and unrelated, and
    get worse as the shift grows.  A metric that is not monotone here cannot be
    used to drive a search."""
    shifts = [0.0, 0.01, 0.02, 0.04, 0.08, 0.12]
    scores = [S.ssim(brighten(headroom, d), headroom) for d in shifts]

    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    for earlier, later in zip(scores, scores[1:]):
        assert later < earlier, f"ssim not monotonic in the shift: {scores}"
    unrelated_score = S.ssim(unrelated, headroom)
    assert unrelated_score < scores[-1] < scores[0]


def test_ssim_matches_the_typescript_formula(base):
    """Literal transcription of ``ssimLuma`` from ``apps/web/src/compare/score.ts``.

    The TypeScript works on 0..255 luma with ``C1 = (0.01*255)**2``; this module
    works on 0..1 with ``C1 = 0.01**2``.  SSIM is invariant under that common
    rescaling, so the two must agree to within float noise -- not merely be
    "close".
    """
    a = box_blur(base, 2)
    b = base
    la = (S.to_luma(a) * 255.0).ravel()
    lb = (S.to_luma(b) * 255.0).ravel()
    w = 256
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    win = 8
    total, count = 0.0, 0
    for y0 in range(0, 256, win):
        y1 = min(256, y0 + win)
        for x0 in range(0, 256, win):
            x1 = min(256, x0 + win)
            n = (x1 - x0) * (y1 - y0)
            sa = sb = 0.0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    sa += la[y * w + x]
                    sb += lb[y * w + x]
            ma, mb = sa / n, sb / n
            va = vb = cov = 0.0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    da = la[y * w + x] - ma
                    db = lb[y * w + x] - mb
                    va += da * da
                    vb += db * db
                    cov += da * db
            va /= n
            vb /= n
            cov /= n
            total += ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
            count += 1
    assert S.ssim(a, b) == pytest.approx(total / count, abs=1e-9)


def test_ssim_handles_ragged_edge_windows_like_the_typescript(base):
    """A 250x250 buffer is not a multiple of 8; the partial edge block must be
    *covered*, not skipped and not padded."""
    odd = make_base(250)
    assert abs(S.ssim(odd, odd) - 1.0) < 1e-6
    assert 0.0 < S.ssim(box_blur(odd, 3), odd) < 1.0


@pytest.mark.parametrize("size", [1, 2, 7, 8, 9, 16, 17])
def test_ssim_is_defined_for_tiny_images(size):
    img = np.random.default_rng(size).random((size, size, 3)).astype(np.float32)
    assert abs(S.ssim(img, img) - 1.0) < 1e-6


def test_ssim_rejects_a_window_below_one(base):
    with pytest.raises(ValueError, match="window must be"):
        S.ssim(base, base, window=0)


# --------------------------------------------------------------------------- #
# Histogram
# --------------------------------------------------------------------------- #


def test_histogram_distance_is_zero_for_identical_and_one_for_disjoint():
    assert S.histogram_distance(np.zeros((4, 4, 3), dtype=np.float32),
                                np.zeros((4, 4, 3), dtype=np.float32)) == pytest.approx(0.0)
    assert S.histogram_distance(np.zeros((4, 4, 3), dtype=np.float32),
                                np.ones((4, 4, 3), dtype=np.float32)) == pytest.approx(1.0)


def test_histogram_distance_is_symmetric(base):
    other = add_hf_noise(base, 0.1)
    assert S.histogram_distance(base, other) == pytest.approx(S.histogram_distance(other, base))


def test_python_binning_matches_the_typescript_binning_for_all_256_levels():
    """``score.ts`` bins 8-bit data as ``floor(c / (256/bins))``; this module bins
    ``0..1`` floats as ``floor(v * bins)``.  For 64 bins the two coincide on every
    level -- asserted here rather than argued in a comment."""
    bins = S.HIST_BINS
    levels = np.arange(256)
    ts = np.minimum(bins - 1, (levels * (bins / 256)).astype(int))
    py = np.clip((levels / 255.0 * bins).astype(int), 0, bins - 1)
    assert np.array_equal(ts, py)


def test_histogram_offset_is_signed():
    dark = np.full((8, 8, 3), 0.2, dtype=np.float32)
    bright = np.full((8, 8, 3), 0.7, dtype=np.float32)
    assert S.histogram_offset(bright, dark) > 0
    assert S.histogram_offset(dark, bright) < 0
    assert S.histogram_offset(dark, dark) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Bands
# --------------------------------------------------------------------------- #


def test_band_energy_returns_three_non_negative_bands(base):
    bands = S.band_energy(base)
    assert set(bands) == {"low", "mid", "high"}
    assert all(isinstance(v, float) and v >= 0.0 for v in bands.values())


def test_white_noise_has_more_high_band_energy_than_a_smooth_gradient():
    """The required check: noise is high-frequency, a ramp is not."""
    noise = S.band_energy(white_noise(256))
    ramp = S.band_energy(smooth_gradient(256))
    assert noise["high"] > ramp["high"] * 10
    assert ramp["high"] < 0.02


def test_high_band_is_sensitive_to_fine_detail():
    """The property ``score.ts`` fails: blurring must collapse ``high``, adding
    grain must raise it.  Both were ~1.00x in the browser implementation."""
    base = make_base(256)
    reference = S.band_energy(base)
    blurred = S.band_energy(box_blur(base, 3))
    noisy = S.band_energy(add_hf_noise(base, 0.16))
    assert blurred["high"] < reference["high"] * 0.5
    assert noisy["high"] > reference["high"] * 2.0
    # ...and the mid band moves far less for a pure low-pass, which is the
    # discriminator the attribution rules rely on.
    assert abs(S.band_ratio(blurred, reference, "mid") - 1.0) < 0.15


def test_ts_compat_reproduces_the_score_ts_bug():
    """``ts_compat=True`` must actually reproduce the browser's numbers, or the
    cross-check it exists for is worthless."""
    base = make_base(256)
    reference = S.band_energy(base, ts_compat=True)
    blurred = S.band_energy(box_blur(base, 3), ts_compat=True)
    noisy = S.band_energy(add_hf_noise(base, 0.16), ts_compat=True)
    # The buggy statistic is nearly blind to both perturbations.
    assert abs(S.band_ratio(blurred, reference, "high") - 1.0) < 0.05
    assert abs(S.band_ratio(noisy, reference, "high") - 1.0) < 0.05
    # While the correct one separates them by more than 3x either way.
    correct_ref = S.band_energy(base)
    assert S.band_ratio(S.band_energy(box_blur(base, 3)), correct_ref, "high") < 0.5
    assert S.band_ratio(S.band_energy(add_hf_noise(base, 0.16)), correct_ref, "high") > 2.0


def test_band_energy_of_a_constant_image_is_all_zero():
    flat = np.full((64, 64, 3), 0.4, dtype=np.float32)
    bands = S.band_energy(flat)
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in bands.values())


def test_band_energy_is_defined_for_a_one_pixel_image():
    assert S.band_energy(np.full((1, 1, 3), 0.5, dtype=np.float32)) == {"low": 0.0, "mid": 0.0, "high": 0.0}


def test_band_ratio_handles_a_zero_reference():
    """Never divide by zero and never return a silently wrong finite number."""
    assert S.band_ratio({"high": 0.0}, {"high": 0.0}, "high") == 1.0
    assert math.isinf(S.band_ratio({"high": 0.5}, {"high": 0.0}, "high"))
    assert S.band_ratio({"high": 0.3}, {"high": 0.6}, "high") == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Colour / gradient statistics
# --------------------------------------------------------------------------- #


def test_color_stats_reports_saturation_and_circular_hue():
    red = np.zeros((8, 8, 3), dtype=np.float32)
    red[:, :, 0] = 1.0
    stats = S.color_stats(red)
    assert stats["saturation"] == pytest.approx(1.0)
    assert stats["hue"] == pytest.approx(0.0, abs=1e-6)

    grey = np.full((8, 8, 3), 0.5, dtype=np.float32)
    assert S.color_stats(grey)["saturation"] == pytest.approx(0.0)


def test_color_stats_saturation_tracks_the_saturation_perturbation(base):
    from conftest import saturate

    assert S.color_stats(saturate(base, 1.9))["saturation"] > S.color_stats(base)["saturation"]
    assert S.color_stats(saturate(base, 0.35))["saturation"] < S.color_stats(base)["saturation"]


def test_color_stats_hue_is_invariant_to_saturation(base):
    from conftest import saturate

    plain = S.color_stats(base)["hue"]
    boosted = S.color_stats(saturate(base, 1.9))["hue"]
    delta = abs(((boosted - plain + 0.5) % 1.0) - 0.5)
    assert delta < 0.01, "scaling saturation must not rotate the hue"


def test_gradient_energy_is_zero_for_a_flat_image_and_positive_for_an_edge():
    flat = np.full((16, 16, 3), 0.5, dtype=np.float32)
    assert S.gradient_energy(flat) == pytest.approx(0.0)
    edge = np.zeros((16, 16, 3), dtype=np.float32)
    edge[:, 8:] = 1.0
    assert S.gradient_energy(edge) > 0.1
    assert S.gradient_energy(np.full((2, 2, 3), 0.5, dtype=np.float32)) == 0.0


# --------------------------------------------------------------------------- #
# Breakdown, JSON schema, loss
# --------------------------------------------------------------------------- #


def test_score_refuses_mismatched_shapes(base):
    """Silently resizing here is how a comparison becomes meaningless."""
    with pytest.raises(ValueError, match="must have the same shape"):
        S.score(base, make_base(128))


def test_breakdown_emits_exactly_the_typescript_key_names(base, unrelated):
    """The JSON must be consumable by ``apps/web`` without a translation table."""
    payload = S.score(add_hf_noise(base, 0.1), base).to_dict()
    assert set(payload) <= {
        "ssim", "hist", "bands", "bandsRef", "ssimLow",
        "meanLuma", "saturation", "hueMean", "gradient", "lumaOffset",
    }
    for required in ("ssim", "hist", "bands", "bandsRef"):
        assert required in payload
    assert set(payload["bands"]) == {"low", "mid", "high"}
    assert set(payload["bandsRef"]) == {"low", "mid", "high"}
    assert set(payload["meanLuma"]) == {"ref", "cur"}
    # must survive a process boundary
    assert json.loads(json.dumps(payload)) == payload


def test_breakdown_round_trips_through_dict(base, unrelated):
    original = S.score(base, unrelated)
    restored = S.ScoreBreakdown.from_dict(original.to_dict())
    assert restored.ssim == pytest.approx(original.ssim)
    assert restored.bands == pytest.approx(original.bands)
    # snake_case is accepted too, so a hand-written dict works
    assert S.ScoreBreakdown.from_dict({"ssim": 0.5, "hist": 0.1}).ssim == 0.5


def test_breakdown_has_no_nan_or_infinity(base, unrelated):
    payload = S.score(base, unrelated).to_dict()
    for key, value in payload.items():
        if isinstance(value, float):
            assert math.isfinite(value), f"{key} is {value}"
        elif isinstance(value, dict):
            assert all(math.isfinite(v) for v in value.values()), key


def test_loss_is_zero_for_identical_and_larger_for_unrelated(base, unrelated):
    assert S.loss(S.score(base, base)) == pytest.approx(0.0, abs=1e-9)
    assert S.loss(S.score(unrelated, base)) > S.loss(S.score(box_blur(base, 1), base)) > 0.0


def test_loss_includes_the_lpips_term_only_when_a_value_is_supplied(base, unrelated):
    breakdown = S.score(unrelated, base)
    without = S.loss(breakdown)
    with_lpips = S.loss(breakdown, lpips_value=0.5)
    assert with_lpips > without
    assert with_lpips - without == pytest.approx(S.LAMBDA_LPIPS_DEFAULT * 0.5)


def test_uint8_input_is_accepted_and_equivalent(base):
    as_u8 = (base * 255.0 + 0.5).astype(np.uint8)
    from_u8 = S.score(as_u8, as_u8)
    assert from_u8.ssim == pytest.approx(1.0, abs=1e-6)


def test_summarize_is_available_for_a_one_line_ui_string(base, unrelated):
    from ai4shader import attribute as A

    line = A.summarize(S.score(unrelated, base))
    assert line.startswith("SSIM ")
    assert "main problem" in line
