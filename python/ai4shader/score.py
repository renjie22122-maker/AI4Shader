"""Image-similarity metrics -- the reference implementation.

This module is the *reference* implementation.  ``apps/web/src/compare/score.ts``
is the in-browser approximation; it uses the same constants and the same
windowing so that a number computed here can be interpreted there and vice
versa.  Any change to a constant in this file must be mirrored in that one.

Interoperability contract
-------------------------
:meth:`ScoreBreakdown.to_dict` emits the **exact key names** of the TypeScript
``ScoreBreakdown`` type (``ssim``, ``hist``, ``bands``, ``bandsRef``,
``ssimLow``, ``meanLuma``, ``saturation``, ``hueMean``, ``gradient``) so that a
JSON blob produced here can be handed to the browser or to the Node CLI without
a translation table.  Python attribute names are snake_case; the emitted keys
are camelCase.  :meth:`ScoreBreakdown.from_dict` accepts either spelling.

Scale equivalence (why the two implementations agree exactly)
-------------------------------------------------------------
``score.ts`` computes SSIM on luma in ``0..255`` with ``C1 = (0.01*255)^2`` and
``C2 = (0.03*255)^2``.  This module computes it on luma in ``0..1`` with
``C1 = 0.01^2`` and ``C2 = 0.03^2``.  SSIM is invariant under a common scaling
of both images *together with* the constants: scaling ``a``, ``b`` and both
constants by ``s`` multiplies numerator and denominator by ``s^2``.  The two
are therefore the same number, not merely close.  ``tests/test_score.py``
asserts this against a literal transcription of the TypeScript formula.

Metrics at a glance
-------------------
===================== ============================== ============ ==============
function              meaning                        range        better
===================== ============================== ============ ==============
``ssim``              structure agreement            ``-1 .. 1``  higher
``histogram_distance`` tonal/colour distribution       ``0 .. 1``   lower
``band_energy``       per-band detail energy          ``>= 0``     n/a (ratio)
``lpips``             learned perceptual distance     ``>= 0``     lower
``clip_similarity``   CLIP embedding cosine           ``-1 .. 1``  higher
===================== ============================== ============ ==============
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import math
import os
import socket
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Literal, Sequence

import numpy as np

# Defensive offline defaults, applied before anything can lazily import a model
# loader.  ``setdefault`` is used deliberately: if the host has already set these
# (or deliberately wants the opposite), we do not override it.  These stop
# huggingface_hub / transformers from even *attempting* a HEAD request when a
# checkpoint is missing; :func:`no_network` is the hard backstop underneath.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

__all__ = [
    "ScoreBreakdown",
    "ssim",
    "histogram_distance",
    "band_energy",
    "color_stats",
    "gradient_energy",
    "score",
    "loss",
    "lpips",
    "clip_similarity",
    "unavailable_metrics",
    "perceptual_fallback",
    "capabilities",
    "LUMA_WEIGHTS",
    "SSIM_WINDOW",
    "SSIM_C1",
    "SSIM_C2",
    "HIST_BINS",
    "LOSS_WEIGHTS",
]

# --------------------------------------------------------------------------- #
# Constants -- shared with apps/web/src/compare/score.ts.  Change both or neither.
# --------------------------------------------------------------------------- #

#: Rec.709 luma weights, matching ``LUMA_R/G/B`` in ``score.ts``.
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

#: Non-overlapping SSIM window side, matching ``win = 8`` in ``score.ts``.
SSIM_WINDOW = 8

#: Wang et al. (2004) stabilisers for a dynamic range of 1.0: ``(K1*L)^2`` with
#: ``K1 = 0.01``.  ``score.ts`` uses ``L = 255`` and gets ``6.5025``; see the
#: module docstring for why these are the same number.
SSIM_C1 = (0.01 * 1.0) ** 2

#: ``(K2*L)^2`` with ``K2 = 0.03``.  ``score.ts`` gets ``58.5225`` at ``L = 255``.
SSIM_C2 = (0.03 * 1.0) ** 2

#: Histogram bins per channel, matching ``HIST_BINS = 64`` in ``score.ts``.
HIST_BINS = 64

#: Loss weights.  ``hist``/``high_ratio``/``low_ratio`` follow ``lossFrom()`` in
#: ``score.ts``, which in turn follows DESIGN.md section 6.6.  DESIGN.md names
#: lambda_1 (the LPIPS term) without fixing its value: it is ``0.0`` when no
#: LPIPS value was supplied, and :data:`LAMBDA_LPIPS_DEFAULT` when one was.
LOSS_WEIGHTS = {
    "ssim": 1.0,
    "hist": 0.5,
    "high_ratio": 0.25,
    "low_ratio": 0.15,
}
LAMBDA_LPIPS_DEFAULT = 0.5


# --------------------------------------------------------------------------- #
# Array plumbing
# --------------------------------------------------------------------------- #


def _as_image(img: Any, name: str = "image") -> np.ndarray:
    """Coerce to ``(H, W, 3) float64`` in ``0..1``.

    ``float64`` is used for all statistics: ``float32`` accumulation over a
    512x512 window sum loses enough precision to move SSIM in the 6th decimal,
    which is the tolerance the tests assert.
    """
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"{name} must be (H, W, 3); got shape {a.shape}")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"{name} is empty")
    if a.dtype == np.uint8:
        return a.astype(np.float64) / 255.0
    return a.astype(np.float64)


def _check_pair(a: np.ndarray, b: np.ndarray, an: str = "render", bn: str = "reference") -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{an} and {bn} must have the same shape, got {a.shape} and {b.shape}. "
            "Resize both to a common size first -- build/load with "
            "`ai4shader.imageio.load(path, size=(w, h))`, which letterboxes rather "
            "than stretches. Comparing different sizes is meaningless."
        )


def to_luma(img: Any) -> np.ndarray:
    """Rec.709 luma of a ``(H, W, 3)`` image, returned ``(H, W) float64`` in ``0..1``."""
    a = _as_image(img)
    r, g, b = LUMA_WEIGHTS
    return a @ np.array([r, g, b], dtype=np.float64)


# --------------------------------------------------------------------------- #
# SSIM
# --------------------------------------------------------------------------- #


def _block_sums(x: np.ndarray, window: int) -> np.ndarray:
    """Sum ``x`` over non-overlapping ``window x window`` blocks, ragged at the edge.

    ``np.add.reduceat`` sums ``x[starts[i]:starts[i+1]]`` and, for the final
    index, ``x[starts[-1]:]`` -- which is exactly "the last block is a partial
    block covering the remainder", the behaviour ``score.ts`` implements with
    ``y1 = min(h, y0 + win)``.  A clamped/replicated pad would *not* be
    equivalent: it would change the mean and variance of the partial block.
    """
    rows = np.arange(0, x.shape[0], window)
    cols = np.arange(0, x.shape[1], window)
    return np.add.reduceat(np.add.reduceat(x, rows, axis=0), cols, axis=1)


def ssim(a: Any, b: Any, *, window: int = SSIM_WINDOW) -> float:
    """Global SSIM over Rec.709 luma, mean over non-overlapping ``window`` blocks.

    Standard formula (Wang et al. 2004):

        SSIM = ((2*ma*mb + C1) * (2*cov + C2)) / ((ma^2 + mb^2 + C1) * (va + vb + C2))

    with ``C1 = (0.01)^2 = 1e-4`` and ``C2 = (0.03)^2 = 9e-4`` for a dynamic
    range of 1.0 (see the module docstring for the equivalence with the
    ``L = 255`` form used in TypeScript).

    * ``a``, ``b``: ``(H, W, 3)`` in ``0..1`` (``uint8`` in ``0..255`` also accepted).
    * ``window``: block side.  Blocks are **non-overlapping** and the final
      row/column block is partial rather than skipped, so a 250x250 buffer is
      fully covered.  ``window`` must be ``>= 1``.
    * returns: mean SSIM in ``-1 .. 1``.  **1.0 = identical, 0.0 = unrelated,
      negative = anti-correlated.**  Higher is better.

    Unlike ``skimage.metrics.structural_similarity`` this is the global-mean
    formulation with fixed non-overlapping windows, which is what the browser
    computes; the two will not agree to the last decimal and are not expected to.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    x = _as_image(a, "a")
    y = _as_image(b, "b")
    _check_pair(x, y, "a", "b")
    return ssim_luma(to_luma(x), to_luma(y), window=window)


def ssim_luma(la: np.ndarray, lb: np.ndarray, *, window: int = SSIM_WINDOW) -> float:
    """SSIM on two pre-computed ``(H, W)`` luma planes.  See :func:`ssim`."""
    la = np.asarray(la, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    if la.shape != lb.shape or la.ndim != 2:
        raise ValueError(f"luma planes must be matching (H, W); got {la.shape} and {lb.shape}")

    h, w = la.shape
    rows = np.arange(0, h, window)
    cols = np.arange(0, w, window)
    counts = np.outer(np.diff(np.append(rows, h)), np.diff(np.append(cols, w))).astype(np.float64)

    ma = _block_sums(la, window) / counts
    mb = _block_sums(lb, window) / counts
    # E[x^2] - E[x]^2.  The two-pass form in score.ts is numerically kinder, but
    # in float64 the difference is ~1e-15 and C1/C2 dominate it entirely.
    va = _block_sums(la * la, window) / counts - ma * ma
    vb = _block_sums(lb * lb, window) / counts - mb * mb
    cov = _block_sums(la * lb, window) / counts - ma * mb
    # Clamp the round-off; a truly constant block can land on -1e-17.
    np.maximum(va, 0.0, out=va)
    np.maximum(vb, 0.0, out=vb)

    num = (2.0 * ma * mb + SSIM_C1) * (2.0 * cov + SSIM_C2)
    den = (ma * ma + mb * mb + SSIM_C1) * (va + vb + SSIM_C2)
    # den >= C1 * C2 > 0 always, so the `den === 0 ? 1 : ...` guard in score.ts
    # is unreachable.  Kept out of the hot path rather than copied.
    return float(np.mean(num / den))


# --------------------------------------------------------------------------- #
# Histogram
# --------------------------------------------------------------------------- #


def _bin_indices(channel: np.ndarray, bins: int) -> np.ndarray:
    """Bin a ``0..1`` channel.  ``floor(v * bins)``, clamped to ``bins - 1``.

    ``score.ts`` bins 8-bit data as ``floor(c / (256/bins))``.  For ``bins = 64``
    the two agree on every one of the 256 levels: the exact difference between
    the two rules is ``c/1020``, which can only push a value across a boundary at
    ``c = 255`` -- and that bin is clamped anyway.  ``tests/test_score.py``
    checks all 256 levels rather than trusting that argument.
    """
    return np.clip((channel * bins).astype(np.int64), 0, bins - 1)


def _histogram(img: np.ndarray, bins: int) -> np.ndarray:
    """``(3, bins)`` per-channel normalised histogram of a ``(H, W, 3)`` image."""
    out = np.empty((3, bins), dtype=np.float64)
    for c in range(3):
        idx = _bin_indices(img[:, :, c], bins)
        out[c] = np.bincount(idx.ravel(), minlength=bins)[:bins]
    total = img.shape[0] * img.shape[1]
    return out / float(total)


def histogram_distance(a: Any, b: Any, *, bins: int = HIST_BINS) -> float:
    """Per-channel histogram **intersection distance**, averaged over RGB.

        distance = 1 - mean_RGB( sum_i min(p_i, q_i) )

    * ``a``, ``b``: ``(H, W, 3)`` in ``0..1`` (``uint8`` accepted).
    * ``bins``: bin count per channel.  64 matches ``HIST_BINS`` in ``score.ts``.
    * returns: ``0.0 .. 1.0``.  **0.0 = identical distributions, 1.0 = disjoint.**
      Lower is better.

    Intersection rather than chi-squared or EMD because it is bounded, cheap, and
    reads directly as "the fraction of tonal mass landing in the same bin" --
    which is a number that can be explained next to a slider.  It is *not* a
    perceptual difference: two very different images with similar tone histograms
    score near 0 here, which is why it is never used alone.
    """
    x = _as_image(a, "a")
    y = _as_image(b, "b")
    _check_pair(x, y, "a", "b")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    ha = _histogram(x, bins)
    hb = _histogram(y, bins)
    intersection = np.minimum(ha, hb).sum(axis=1)  # per channel
    return float(1.0 - intersection.mean())


def histogram_offset(a: Any, b: Any) -> float:
    """Signed mean-luma difference ``mean(a) - mean(b)``, in ``-1 .. 1``.

    Positive means ``a`` is brighter than ``b``.  This is the observable behind
    the exposure rule in :mod:`ai4shader.attribute`; the histogram *distance*
    above is unsigned and cannot say which image is brighter.
    """
    x = _as_image(a, "a")
    y = _as_image(b, "b")
    return float(to_luma(x).mean() - to_luma(y).mean())


# --------------------------------------------------------------------------- #
# Frequency bands
# --------------------------------------------------------------------------- #


def _downsample2(x: np.ndarray) -> np.ndarray:
    """2x2 box downsample of the two **spatial** axes.  Odd dimensions drop the final row/column.

    Spatial axes are axes 0 and 1, matching the ``(H, W)`` / ``(H, W, C)``
    convention used throughout this package -- so a 2-D luma plane and a colour
    image go through the same code path and a colour image is never accidentally
    downsampled along its channel axis.

    Matches ``downsample()`` in ``score.ts`` exactly, including the degenerate
    ``1 -> 1`` case where its ``x0 + 1 < w`` guards silently replicate the only
    sample (here: clipping the index does the same thing).
    """
    h, w = x.shape[0], x.shape[1]
    rest = x.shape[2:]
    nh, nw = max(1, h // 2), max(1, w // 2)
    ys = np.minimum(np.arange(nh * 2), h - 1)
    xs = np.minimum(np.arange(nw * 2), w - 1)
    # np.take on explicit axes: fancy indexing like `x[ys][:, xs]` would be wrong
    # for a 3-D input, because the trailing axis is channels, not width.
    sub = np.take(np.take(x, ys, axis=0), xs, axis=1)
    return sub.reshape(nh, 2, nw, 2, *rest).mean(axis=(1, 3))


def _upsample2(x: np.ndarray, w: int, h: int) -> np.ndarray:
    """Nearest-neighbour 2x upsample to ``(h, w)``.

    Nearest neighbour, not bilinear, because this only ever feeds a residual band
    (``P0 - up(P1)``); the interpolation kernel's smoothness is irrelevant to an
    RMS of the difference, and matching the TypeScript bit-for-bit is not.
    """
    sh, sw = x.shape
    ys = np.minimum(np.arange(h) // 2, sh - 1)
    xs = np.minimum(np.arange(w) // 2, sw - 1)
    return x[np.ix_(ys, xs)]


def _rms(x: np.ndarray) -> float:
    """RMS about the mean (population standard deviation), DC removed."""
    return float(np.std(x))


def _rms_nodc(x: np.ndarray) -> float:
    """RMS without removing the DC term -- for a difference image, which has no DC."""
    return float(np.sqrt(np.mean(x * x)))


def band_energy(img: Any, *, ts_compat: bool = False) -> dict[str, float]:
    """3-level pyramid band energies of a ``(H, W, 3)`` image.

    ``P0 = luma``, ``P1 = down2(P0)``, ``P2 = down2(P1)``, ``P3 = down2(P2)``::

        low  = RMS about the mean of P3     -- coarse structure / composition
        mid  = RMS of P2 - up2(P3)          -- medium features (blobs, streaks)
        high = RMS of P0 - up2(P1)          -- fine detail (grain, thin filaments)

    * returns ``{"low": float, "mid": float, "high": float}``, each ``>= 0`` on
      the ``0..1`` luma scale.  Higher = more energy in that band.

    These are **not** calibrated in cycles/pixel: a box filter is a poor low-pass
    (it aliases a little) and this is not an FFT.  They are meant to be used as
    **ratios between two images**, where the filter's exact response largely
    cancels.  "The render's high band is 3.2x the reference's" survives any
    reasonable filter choice, and that is all the attribution needs.

    ``ts_compat`` -- **reproduces a bug in ``score.ts``; do not use it.**
    ``score.ts`` computes ``high`` against ``up2(P2)`` (the 1/4-scale plane
    upsampled with a *single* doubling) instead of ``up2(P1)``.  Because its
    upsample maps ``y -> min(sh-1, y>>1)``, that clamps three quarters of the
    result to one constant, so the browser's ``high`` is really "RMS of the image
    about one pixel of P2" -- it tracks overall contrast and is nearly blind to
    fine detail (measured: 0.99x under a 4-pixel box blur and 1.01x with
    sigma=0.16 white noise added, where the correct statistic gives 0.19x and
    3.36x).  ``ts_compat=True`` reproduces those wrong numbers exactly, so the two
    implementations can be cross-checked while the TypeScript is being fixed; the
    default is the correct statistic.  See ``python/dev/verify_band_bug.py``.

    A degenerate image (1x1) returns zeros rather than raising.
    """
    p0 = to_luma(img)
    if p0.size == 0:
        return {"low": 0.0, "mid": 0.0, "high": 0.0}
    d1 = _downsample2(p0)
    d2 = _downsample2(d1)
    d3 = _downsample2(d2)
    up2 = _upsample2(d3, d2.shape[1], d2.shape[0])
    # `high` is the residual above the 2-pixel scale, so it must compare P0 with
    # the 1/2-scale plane upsampled once -- i.e. P1, not P2.  See `ts_compat`.
    source = d2 if ts_compat else d1
    up1 = _upsample2(source, p0.shape[1], p0.shape[0])
    return {
        "low": _rms(d3),
        "mid": _rms_nodc(d2 - up2),
        "high": _rms_nodc(p0 - up1),
    }


def band_ratio(render: dict[str, float], reference: dict[str, float], band: str) -> float:
    """``render[band] / reference[band]``, with the degenerate cases pinned.

    Returns ``1.0`` when both are ~0 (no evidence of a mismatch), and ``inf``
    when the reference is ~0 but the render is not -- never a division error and
    never a silently wrong finite number.
    """
    r = float(reference.get(band, 0.0))
    c = float(render.get(band, 0.0))
    if r <= 1e-9:
        return 1.0 if c <= 1e-9 else math.inf
    return c / r


# --------------------------------------------------------------------------- #
# Colour / edge statistics
# --------------------------------------------------------------------------- #


def color_stats(img: Any) -> dict[str, float]:
    """Mean HSV saturation and saturation-weighted circular mean hue.

    * returns ``{"saturation": float, "hue": float}``.
    * ``saturation`` is the mean of ``(max - min) / max`` over pixels, ``0..1``.
      Higher = more colourful.
    * ``hue`` is the circular mean of the hue histogram **weighted by
      saturation** (grey pixels have no meaningful hue), expressed in **turns**
      in ``0..1`` where ``0.0 = red``, ``1/3 = green``, ``2/3 = blue``.  It is a
      direction on a circle, so compare it with a wrapped difference, never with
      ``abs(a - b)``.
    """
    a = _as_image(img)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn

    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(mx > 0.0, d / np.where(mx > 0.0, mx, 1.0), 0.0)
    saturation = float(sat.mean())

    is_r = mx == r
    is_g = (~is_r) & (mx == g)
    safe_d = np.where(d > 0.0, d, 1.0)
    hue6 = np.where(
        is_r,
        np.mod((g - b) / safe_d, 6.0),
        np.where(is_g, (b - r) / safe_d + 2.0, (r - g) / safe_d + 4.0),
    )
    hue6 = np.mod(hue6 + 6.0, 6.0)
    angle = hue6 / 6.0 * (2.0 * math.pi)

    weight = np.where(d > 1e-4, sat, 0.0)
    hx = float((np.cos(angle) * weight).sum())
    hy = float((np.sin(angle) * weight).sum())
    if hx == 0.0 and hy == 0.0:
        return {"saturation": saturation, "hue": 0.0}
    hue = (math.atan2(hy, hx) / (2.0 * math.pi) + 1.0) % 1.0
    return {"saturation": saturation, "hue": float(hue)}


def gradient_energy(img: Any) -> float:
    """RMS of the central-difference gradient magnitude of luma.

    * returns ``>= 0`` on the ``0..1`` luma scale.  Higher = harder edges / more
      high-frequency contrast.

    Interior pixels only (``y, x`` from 1 to ``H-2``/``W-2``), so the border does
    not contribute a phantom one-sided difference.  This is a proxy for edge
    *hardness*, not a perceptual sharpness metric: it rises for blur-free noise
    just as much as for genuinely crisp structure, which is why the attribution
    looks at the mid band as well before blaming ``sdf.smoothness``.
    """
    l = to_luma(img)
    h, w = l.shape
    if h < 3 or w < 3:
        return 0.0
    gx = (l[1:-1, 2:] - l[1:-1, :-2]) * 0.5
    gy = (l[2:, 1:-1] - l[:-2, 1:-1]) * 0.5
    return float(np.sqrt(np.mean(gx * gx + gy * gy)))


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #

BreakdownDict = dict[str, Any]


@dataclass
class ScoreBreakdown:
    """Every observable the attribution rules read, for one render/reference pair.

    Field names are snake_case in Python; :meth:`to_dict` emits the camelCase
    keys of the TypeScript ``ScoreBreakdown`` so the JSON is directly consumable
    by ``apps/web`` and the Node CLI.

    ``ssim``/``hist``/``bands``/``bands_ref`` are the contract.  The rest are
    optional signals that make the attribution specific; a caller holding only
    the first four still gets (coarser) hints.
    """

    #: Global SSIM over luma, 8x8 blocks.  ``-1..1``, 1 = identical, higher better.
    ssim: float
    #: 1 - mean per-channel histogram intersection.  ``0..1``, 0 = identical, lower better.
    hist: float
    #: Band energies of the **render**.  Each ``>= 0``; meaningful as a ratio.
    bands: dict[str, float] = field(default_factory=dict)
    #: Band energies of the **reference**.
    bands_ref: dict[str, float] = field(default_factory=dict)
    #: SSIM on 1/4-scale luma -- isolates "wrong layout" from "wrong grain". ``-1..1``.
    ssim_low: float | None = None
    #: Mean luma of each image, ``0..1``: ``{"ref": float, "cur": float}``.
    mean_luma: dict[str, float] | None = None
    #: Mean HSV saturation, ``0..1``: ``{"ref": float, "cur": float}``.
    saturation: dict[str, float] | None = None
    #: Circular mean hue in turns, ``0..1``: ``{"ref": float, "cur": float}``.
    hue_mean: dict[str, float] | None = None
    #: RMS gradient magnitude, ``>= 0``: ``{"ref": float, "cur": float}``.
    gradient: dict[str, float] | None = None
    #: Signed mean-luma offset ``render - reference``, ``-1..1``.  Positive = too bright.
    luma_offset: float | None = None

    # -- serialisation ----------------------------------------------------- #

    _CAMEL = {
        "ssim": "ssim",
        "hist": "hist",
        "bands": "bands",
        "bands_ref": "bandsRef",
        "ssim_low": "ssimLow",
        "mean_luma": "meanLuma",
        "saturation": "saturation",
        "hue_mean": "hueMean",
        "gradient": "gradient",
        "luma_offset": "lumaOffset",
    }

    def to_dict(self) -> BreakdownDict:
        """JSON-serialisable dict using the TypeScript ``ScoreBreakdown`` key names.

        ``None`` values are **omitted** rather than emitted as ``null``, matching
        the optional (``?``) fields of the TypeScript type: a consumer doing
        ``if (b.ssimLow !== undefined)`` sees exactly the same thing either way.
        Every remaining value is a plain ``float``/``dict``/``str`` -- no numpy
        scalars, no NaN, no Infinity (an infinite band ratio is not stored here;
        it is derived, and :func:`ai4shader.attribute.attribute` handles it).
        """
        out: BreakdownDict = {}
        for py_name, ts_name in self._CAMEL.items():
            value = getattr(self, py_name)
            if value is None:
                continue
            if isinstance(value, dict):
                out[ts_name] = {k: float(v) for k, v in value.items()}
            else:
                out[ts_name] = float(value)
        return out

    @classmethod
    def from_dict(cls, data: BreakdownDict) -> "ScoreBreakdown":
        """Inverse of :meth:`to_dict`; accepts snake_case or camelCase keys."""
        reverse = {v: k for k, v in cls._CAMEL.items()}
        kwargs: BreakdownDict = {}
        for key, value in data.items():
            py_name = reverse.get(key, key)
            if py_name in cls._CAMEL:
                kwargs[py_name] = value
        return cls(**kwargs)


def score(render: Any, reference: Any, *, ts_compat: bool = False) -> ScoreBreakdown:
    """Compare a rendered frame against a reference image.

    Both must already be the **same shape**: this function refuses to resize,
    because silently stretching one to match the other corrupts every metric
    downstream.  Use :func:`ai4shader.imageio.load` with ``size=`` on both, which
    letterboxes instead.

    * ``render``: ``(H, W, 3)`` in ``0..1`` -- the shader output.
    * ``reference``: ``(H, W, 3)`` in ``0..1`` -- the user's target image.
    * ``ts_compat``: reproduce ``score.ts``'s buggy ``high`` band.  See
      :func:`band_energy`.  **Default ``False`` (correct); do not ship ``True``.**
    * returns: a :class:`ScoreBreakdown`; ``bands`` describe ``render`` and
      ``bandsRef`` describe ``reference``, so a band ratio is ``bands/bandsRef``;
      use :func:`band_ratio` to avoid the division-by-zero trap.

    Argument order is ``(render, reference)``, the reverse of ``compareFrames``
    in ``score.ts``, which takes ``(ref, cur)``.  That asymmetry is deliberate:
    the Python entry point is named for what it does, and the breakdown's own
    keys (``ref``/``cur``) say which is which.
    """
    cur = _as_image(render, "render")
    ref = _as_image(reference, "reference")
    _check_pair(cur, ref)

    cur_luma = to_luma(cur)
    ref_luma = to_luma(ref)
    cur_q = _downsample2(_downsample2(cur_luma))
    ref_q = _downsample2(_downsample2(ref_luma))
    cur_stats = color_stats(cur)
    ref_stats = color_stats(ref)

    return ScoreBreakdown(
        ssim=ssim_luma(ref_luma, cur_luma),
        hist=histogram_distance(cur, ref),
        bands=band_energy(cur, ts_compat=ts_compat),
        bands_ref=band_energy(ref, ts_compat=ts_compat),
        ssim_low=ssim_luma(ref_q, cur_q),
        mean_luma={"ref": float(ref_luma.mean()), "cur": float(cur_luma.mean())},
        saturation={"ref": ref_stats["saturation"], "cur": cur_stats["saturation"]},
        hue_mean={"ref": ref_stats["hue"], "cur": cur_stats["hue"]},
        gradient={"ref": gradient_energy(ref), "cur": gradient_energy(cur)},
        luma_offset=float(cur_luma.mean() - ref_luma.mean()),
    )


def loss(
    breakdown: ScoreBreakdown,
    *,
    lpips_value: float | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Scalar objective for the parameter optimiser.  **Lower is better.**

        L = w_ssim*(1 - SSIM) + w_hist*|hist| + w_high*|high_ratio - 1|
            + w_low*|low_ratio - 1| + lambda_1 * LPIPS

    Follows DESIGN.md section 6.6 and ``lossFrom()`` in ``score.ts``.  DESIGN.md
    writes ``lambda_1`` without a value; it is ``0`` unless ``lpips_value`` is
    supplied, in which case :data:`LAMBDA_LPIPS_DEFAULT` is used.  Pass an
    explicit ``weights={"lpips": ...}`` to override.

    ``lpips_value`` of ``None`` (weights unavailable) degrades to the
    weight-free expression rather than to a fabricated term.
    """
    w = dict(LOSS_WEIGHTS)
    w["lpips"] = LAMBDA_LPIPS_DEFAULT if lpips_value is not None else 0.0
    if weights:
        w.update(weights)

    high = band_ratio(breakdown.bands, breakdown.bands_ref, "high")
    low = band_ratio(breakdown.bands, breakdown.bands_ref, "low")
    total = (
        w["ssim"] * (1.0 - breakdown.ssim)
        + w["hist"] * abs(breakdown.hist)
        + w["high_ratio"] * (0.0 if math.isinf(high) else abs(high - 1.0))
        + w["low_ratio"] * (0.0 if math.isinf(low) else abs(low - 1.0))
    )
    if lpips_value is not None:
        total += w["lpips"] * float(lpips_value)
    return float(total)


# --------------------------------------------------------------------------- #
# Offline enforcement
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def no_network() -> Iterator[None]:
    """Make every outbound socket connection raise, for the duration of the block.

    This project never makes a network call.  Rather than *promising* that, the
    weight-loading paths run inside this guard: if a model loader decides it
    needs to download something, it fails loudly here instead of silently
    reaching the internet.  It is also the mechanism that proves the cached
    weights were genuinely used -- see ``tests/test_metrics_optional.py``.
    """

    def _blocked(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "AI4Shader: network access is disabled. A model loader tried to open a "
            "connection, which means the required weights are not cached locally. "
            "Nothing in this project downloads anything."
        )

    patched = []
    for target in ("connect", "connect_ex"):
        if hasattr(socket.socket, target):
            patched.append((socket.socket, target, getattr(socket.socket, target)))
            setattr(socket.socket, target, _blocked)
    patched.append((socket, "create_connection", socket.create_connection))
    socket.create_connection = _blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        for owner, name, original in patched:
            setattr(owner, name, original)


# --------------------------------------------------------------------------- #
# Optional learned metrics -- availability probing
# --------------------------------------------------------------------------- #

#: ``sha256`` of CLIP ``ViT-B-32.pt`` as published in the OpenAI CLIP ``_MODELS``
#: table.  Verified before use, so a truncated or substituted cache file is
#: reported as unavailable instead of producing a meaningless embedding.
CLIP_VITB32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

#: LPIPS backbone checkpoints, by net name.  ``alex`` is the LPIPS default and
#: the smallest.  The filename includes the ``torchvision`` URL hash prefix,
#: which is exactly how ``torch.hub.load_state_dict_from_url`` caches it.
LPIPS_BACKBONES = {
    "alex": "alexnet-owt-7be5be79.pth",
    "vgg": "vgg16-397923af.pth",
    "squeeze": "squeezenet1_1-b8a52dc0.pth",
}

#: Populated by an actual load attempt so :func:`capabilities` can distinguish
#: "the files are on disk" from "loading them really worked".
_LOAD_STATE: dict[str, dict[str, Any]] = {}


def _find_spec(name: str) -> bool:
    """Is ``name`` importable?  Uses ``find_spec``, which does **not** import it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _torch_hub_checkpoints() -> str:
    """The torch hub ``checkpoints`` directory, without importing torch."""
    root = os.environ.get("TORCH_HOME")
    if not root:
        xdg = os.environ.get("XDG_CACHE_HOME")
        root = os.path.join(xdg, "torch") if xdg else os.path.join(os.path.expanduser("~"), ".cache", "torch")
    return os.path.join(root, "hub", "checkpoints")


def _lpips_bundle_dir() -> str | None:
    """Directory of the installed ``lpips`` package, or ``None``."""
    try:
        spec = importlib.util.find_spec("lpips")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(spec.origin)


def _lpips_files(net: str = "alex") -> dict[str, Any]:
    """Where LPIPS's two weight files would live, and whether each is present."""
    bundle = _lpips_bundle_dir()
    linear = os.path.join(bundle, "weights", "v0.1", f"{net}.pth") if bundle else None
    trunk = os.path.join(_torch_hub_checkpoints(), LPIPS_BACKBONES.get(net, ""))
    return {
        "net": net,
        "linear_weights": linear,
        "linear_present": bool(linear and os.path.isfile(linear)),
        "backbone_weights": trunk,
        "backbone_present": os.path.isfile(trunk),
    }


def _lpips_availability() -> dict[str, Any]:
    """Cheap presence check for the real LPIPS.  Imports nothing heavy."""
    info = _lpips_files("alex")
    missing = []
    if not _find_spec("torch"):
        missing.append("torch")
    if not _find_spec("torchvision"):
        missing.append("torchvision")
    if not _find_spec("lpips"):
        missing.append("lpips")
    if not info["linear_present"]:
        missing.append("lpips linear weights (weights/v0.1/alex.pth)")
    if not info["backbone_present"]:
        missing.append(f"alexnet backbone ({LPIPS_BACKBONES['alex']})")

    real = _LOAD_STATE.get("lpips")
    if missing:
        return {
            "available": False,
            "verified": False,
            "reason": "missing: " + ", ".join(missing),
            **info,
        }
    return {
        "available": True,
        "verified": bool(real and real.get("ok")),
        "reason": "cached weights present" if not real else str(real.get("reason")),
        **info,
    }


def _clip_cache_dir() -> str:
    return os.environ.get("AI4SHADER_CLIP_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "clip"))


def _clip_availability() -> dict[str, Any]:
    """Cheap presence + checksum check for OpenAI CLIP ``ViT-B/32``."""
    path = os.path.join(_clip_cache_dir(), "ViT-B-32.pt")
    present = os.path.isfile(path)
    digest_ok: bool | None = None
    if present:
        digest_ok = _sha256_cached(path) == CLIP_VITB32_SHA256

    missing = []
    if not _find_spec("torch"):
        missing.append("torch")
    if not _find_spec("clip"):
        missing.append("clip")
    if not present:
        missing.append(f"ViT-B-32.pt under {_clip_cache_dir()}")
    elif not digest_ok:
        missing.append("ViT-B-32.pt has the wrong sha256")

    real = _LOAD_STATE.get("clip")
    if missing:
        return {
            "available": False,
            "verified": False,
            "reason": "missing: " + ", ".join(missing),
            "weights": path,
            "sha256_ok": digest_ok,
            "model": "ViT-B/32",
        }
    return {
        "available": True,
        "verified": bool(real and real.get("ok")),
        "reason": "cached weights present, sha256 verified" if not real else str(real.get("reason")),
        "weights": path,
        "sha256_ok": digest_ok,
        "model": "ViT-B/32",
    }


_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def _sha256_cached(path: str) -> str:
    """SHA-256 of a file, memoised on ``(path, size, mtime_ns)``.

    Hashing 354 MB on every call would be absurd; hashing it never would mean
    trusting a file we did not check.  Keying on size and mtime gives both.
    """
    try:
        st = os.stat(path)
    except OSError:
        return ""
    key = (path, st.st_size, st.st_mtime_ns)
    if key in _SHA_CACHE:
        return _SHA_CACHE[key]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    _SHA_CACHE[key] = digest
    return digest


def capabilities(*, probe: bool = False) -> dict[str, Any]:
    """Report which metrics can actually run, so callers can branch instead of guess.

    * ``probe=False`` (default): a **presence** check.  It never imports ``torch``
      and never loads a model, so it is cheap enough for every CLI invocation.
      ``verified`` is ``False`` for anything not yet loaded -- "the files are
      there" is not the same claim as "it works".  The one non-trivial cost is
      the CLIP checkpoint's sha256 (354 MB, ~0.25 s on NVMe); it is memoised on
      ``(path, size, mtime_ns)``, so only the first call in a process pays it.
    * ``probe=True``: actually loads LPIPS and CLIP (inside :func:`no_network`)
      and reports the real outcome.  This costs seconds and ~600 MB of RSS.

    Returns a JSON-serialisable dict::

        {
          "ssim": {"available": true, "detail": "numpy, no extra deps"},
          ...,
          "lpips": {"available": bool, "verified": bool, "reason": str,
                    "net": "alex", "linear_weights": path, "backbone_weights": path,
                    "linear_present": bool, "backbone_present": bool},
          "clip_similarity": {"available": bool, "verified": bool, "reason": str,
                              "model": "ViT-B/32", "weights": path, "sha256_ok": bool},
          "environment": {"numpy": str, "pillow": bool, "torch": bool}
        }

    A ``False`` here is a licence to omit the metric, never to substitute a
    fabricated number: an invented similarity silently steers the optimiser in a
    direction the data does not support.
    """
    report: dict[str, Any] = {
        "ssim": {"available": True, "detail": "numpy only; window=8, C1/C2 for L=1"},
        "histogram_distance": {"available": True, "detail": "numpy only; 64 bins, intersection"},
        "band_energy": {"available": True, "detail": "numpy only; 3-level box pyramid"},
        "gradient_energy": {"available": True, "detail": "numpy only; central difference"},
        "color_stats": {"available": True, "detail": "numpy only; HSV S + circular hue"},
        "lpips": _lpips_availability(),
        "clip_similarity": _clip_availability(),
        "environment": {
            "numpy": np.__version__,
            "pillow": _find_spec("PIL"),
            "torch": _find_spec("torch"),
        },
    }
    if not probe:
        return report
    for name, fn in (("lpips", _lpips_model), ("clip", _clip_model)):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - the reason is the deliverable
            _LOAD_STATE.setdefault(name, {"ok": False, "reason": f"{type(exc).__name__}: {exc}"})
    report["lpips"] = _lpips_availability()
    report["clip_similarity"] = _clip_availability()
    return report


class MetricUnavailable(RuntimeError):
    """Raised internally when a learned metric cannot run.  Callers see ``None``."""


# --------------------------------------------------------------------------- #
# LPIPS
# --------------------------------------------------------------------------- #


def _lpips_model() -> Any:
    """Load (and cache) the LPIPS model.  Raises :class:`MetricUnavailable`."""
    if "model" in _LOAD_STATE.get("lpips", {}):
        return _LOAD_STATE["lpips"]["model"]
    state = _lpips_availability()
    if not state["available"]:
        _LOAD_STATE["lpips"] = {"ok": False, "reason": state["reason"]}
        raise MetricUnavailable(f"LPIPS unavailable: {state['reason']}")

    try:
        with no_network():
            import warnings

            import lpips as lpips_pkg  # noqa: PLC0415 - deliberately lazy
            import torch  # noqa: PLC0415

            # lpips calls `tv.alexnet(pretrained=True)`, which torchvision 0.13+
            # deprecates loudly on stderr.  The warnings are about lpips's code,
            # not ours, and they would otherwise be the first thing a CLI user
            # sees.  Scoped strictly to model construction.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = lpips_pkg.LPIPS(net="alex", verbose=False)
            model.eval()
            torch.set_grad_enabled(False)
    except Exception as exc:  # noqa: BLE001
        _LOAD_STATE["lpips"] = {"ok": False, "reason": f"load failed: {type(exc).__name__}: {exc}"}
        raise MetricUnavailable(str(exc)) from exc
    _LOAD_STATE["lpips"] = {"ok": True, "reason": "loaded from cached weights", "model": model}
    return model


def _lpips_tensor(img: np.ndarray) -> Any:
    import torch  # noqa: PLC0415

    # LPIPS expects NCHW in [-1, 1] (it re-maps internally from [0,1]).
    x = np.ascontiguousarray(np.transpose(img, (2, 0, 1))[None], dtype=np.float32)
    return torch.from_numpy(x)


def lpips(a: Any, b: Any) -> float | None:
    """Learned Perceptual Image Patch Similarity (Zhang et al. 2018), AlexNet trunk.

    * ``a``, ``b``: ``(H, W, 3)`` in ``0..1``; must be the same shape.
    * returns: distance ``>= 0``.  **0.0 = identical, higher = more different**
      (lower is better).  Typical values for unrelated shader frames are
      ``0.3 .. 0.8``; ``< 0.05`` is usually visually very close.
    * returns ``None`` -- **never a fabricated number** -- when the weights are
      absent, ``torch``/``torchvision``/``lpips`` is missing, or loading fails.
      Call :func:`capabilities` first if you need to branch before rendering.

    The model is loaded lazily on first use (importing ``torch`` costs seconds
    and hundreds of MB) and cached for the process.  Loading happens inside
    :func:`no_network`, so a loader that wants to download raises instead of
    reaching the internet.

    This is *not* the metric to optimise alone: LPIPS is trained to mimic human
    judgements of natural-image patches, and procedural shader output is far
    outside its training distribution.  It is a useful tie-breaker next to SSIM,
    not a replacement.
    """
    x = _as_image(a, "a")
    y = _as_image(b, "b")
    _check_pair(x, y, "a", "b")
    if min(x.shape[:2]) < 32:
        raise ValueError(
            f"LPIPS needs at least 32x32 input (AlexNet has 5 pooling stages); got {x.shape[:2]}"
        )
    try:
        model = _lpips_model()
    except MetricUnavailable:
        return None
    try:
        import torch  # noqa: PLC0415

        with torch.no_grad():
            value = model(_lpips_tensor(x), _lpips_tensor(y))
        return float(value.item())
    except Exception:  # noqa: BLE001 - a failed forward pass is "unavailable", not a score
        return None


# --------------------------------------------------------------------------- #
# CLIP
# --------------------------------------------------------------------------- #

#: CLIP's published preprocessing constants (``clip.clip._transform``).
_CLIP_SIZE = 224
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float64)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float64)


def _clip_model() -> Any:
    """Load (and cache) OpenAI CLIP ViT-B/32.  Raises :class:`MetricUnavailable`."""
    if "model" in _LOAD_STATE.get("clip", {}):
        return _LOAD_STATE["clip"]["model"]
    state = _clip_availability()
    if not state["available"]:
        _LOAD_STATE["clip"] = {"ok": False, "reason": state["reason"]}
        raise MetricUnavailable(f"CLIP unavailable: {state['reason']}")

    try:
        with no_network():
            import clip as clip_pkg  # noqa: PLC0415
            import torch  # noqa: PLC0415

            model, _preprocess = clip_pkg.load("ViT-B/32", device="cpu", jit=False, download_root=_clip_cache_dir())
            model.eval()
            torch.set_grad_enabled(False)
    except Exception as exc:  # noqa: BLE001
        _LOAD_STATE["clip"] = {"ok": False, "reason": f"load failed: {type(exc).__name__}: {exc}"}
        raise MetricUnavailable(str(exc)) from exc
    _LOAD_STATE["clip"] = {"ok": True, "reason": "loaded from cached weights, sha256 verified", "model": model}
    return model


def _clip_preprocess(img: np.ndarray) -> Any:
    """CLIP's preprocessing, reimplemented on numpy.

    ``clip.clip._transform`` is ``Resize(224) -> CenterCrop(224) -> ToTensor ->
    Normalize``.  We reproduce it with our own resampler instead of going through
    ``PIL.Image`` so the result does not depend on whether Pillow is installed.
    The resampling kernel differs slightly (area/bilinear vs bicubic); for a
    learned embedding that difference is far below the noise floor of the metric,
    and reproducibility is worth more than matching torchvision's kernel.
    """
    import torch  # noqa: PLC0415

    from .imageio import fit as _fit  # noqa: PLC0415

    small = _fit(img.astype(np.float32), (_CLIP_SIZE, _CLIP_SIZE), mode="crop")
    norm = (small.astype(np.float64) - _CLIP_MEAN) / _CLIP_STD
    chw = np.ascontiguousarray(np.transpose(norm, (2, 0, 1))[None], dtype=np.float32)
    return torch.from_numpy(chw)


def clip_similarity(a: Any, b: Any) -> float | None:
    """Cosine similarity of OpenAI CLIP ``ViT-B/32`` image embeddings.

    * ``a``, ``b``: ``(H, W, 3)`` in ``0..1``.  They need not be the same size --
      both are centre-cropped and resized to 224x224 first.
    * returns: cosine in ``-1 .. 1``.  **Higher is more similar.**
    * returns ``None`` -- **never a fabricated number** -- when the weights are
      absent or loading fails.

    Calibration warning: this is *not* a 0..1 similarity, and it is not
    threshold-able in the abstract.  Two arbitrary CLIP image embeddings
    typically sit at ``0.5 .. 0.9``; unrelated procedural noise often lands near
    ``0.6``.  Read it as a *relative* signal between candidate renders of the
    same document, never as "0.8 means 80% correct".

    The sha256 of the cached checkpoint is verified before loading, so a
    truncated download is reported unavailable rather than silently producing a
    meaningless embedding.
    """
    x = _as_image(a, "a")
    y = _as_image(b, "b")
    try:
        model = _clip_model()
    except MetricUnavailable:
        return None
    try:
        import torch  # noqa: PLC0415

        with torch.no_grad():
            fx = model.encode_image(_clip_preprocess(x))
            fy = model.encode_image(_clip_preprocess(y))
            fx = fx / fx.norm(dim=-1, keepdim=True)
            fy = fy / fy.norm(dim=-1, keepdim=True)
            return float((fx * fy).sum().item())
    except Exception:  # noqa: BLE001
        return None


def unavailable_metrics() -> list[str]:
    """Names of the learned metrics that cannot run right now, from a cached probe.

    Convenience wrapper over :func:`capabilities` for the common branch::

        if ai4shader.score.unavailable_metrics():
            ...  # degrade honestly, do not invent a number
    """
    caps = capabilities()
    return [name for name in ("lpips", "clip_similarity") if not caps[name]["available"]]


def perceptual_fallback(a: Any, b: Any, *, levels: int = 4) -> float:
    """Dependency-free multi-scale structural distance.  **This is not LPIPS.**

    A mean of ``1 - SSIM`` over ``levels`` successive 2x downsamplings, i.e. a
    coarse MS-SSIM-style objective.  It exists so a caller with no learned
    weights still has *some* multi-scale perceptual signal for the optimiser, and
    it is deliberately given its own name so that nothing can ever report it as
    an LPIPS value.

    * returns ``0.0 .. 2.0``; **0.0 = identical, higher = more different**
      (lower is better) -- the same direction as LPIPS, so it is a drop-in for
      the *sign* of the gradient and nothing else.
    * It is sensitive to exactly what SSIM is sensitive to (structure, local
      contrast) and blind to exactly what SSIM is blind to (texture statistics,
      style).  Do not interpret its magnitude against LPIPS thresholds.

    Prefer :func:`lpips` whenever ``capabilities()["lpips"]["available"]`` is
    true.  This function is the honest degradation path, not a substitute.
    """
    x = _as_image(a, "a")
    y = _as_image(b, "b")
    _check_pair(x, y, "a", "b")
    if levels < 1:
        raise ValueError("levels must be >= 1")
    total = 0.0
    used = 0
    for level in range(levels):
        lx, ly = to_luma(x), to_luma(y)
        if min(lx.shape) < SSIM_WINDOW:
            break
        total += 1.0 - ssim_luma(lx, ly)
        used += 1
        if min(lx.shape) < 2 * SSIM_WINDOW:
            break
        x = _downsample2(_downsample2(x))
        y = _downsample2(_downsample2(y))
    return float(total / used) if used else 0.0
