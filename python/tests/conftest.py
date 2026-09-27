"""Shared fixtures and synthetic-image builders for the test suite.

The builders live here rather than in ``dev/`` so the suite is self-contained:
``python -m pytest tests/`` must work on a checkout that has nothing else.

Everything is numpy-only and seeded, so every run produces identical images.
"""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# A shader-like base image: soft SDF disc + fbm detail + a two-colour ramp.
# --------------------------------------------------------------------------- #


def value_noise(shape: tuple[int, int], freq: float, seed: int) -> np.ndarray:
    """Smoothstep-interpolated value noise at ``freq`` cycles per unit."""
    rng = np.random.default_rng(seed)
    h, w = shape
    gh, gw = max(2, int(h * freq) + 2), max(2, int(w * freq) + 2)
    grid = rng.random((gh, gw))
    ys = np.linspace(0, gh - 1, h)
    xs = np.linspace(0, gw - 1, w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    y1 = np.minimum(y0 + 1, gh - 1)
    x1 = np.minimum(x0 + 1, gw - 1)
    a = grid[np.ix_(y0, x0)]
    b = grid[np.ix_(y0, x1)]
    c = grid[np.ix_(y1, x0)]
    d = grid[np.ix_(y1, x1)]
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    top = a + (b - a) * fx
    bot = c + (d - c) * fx
    return top + (bot - top) * fy


def fbm(shape: tuple[int, int], base: float = 3.0, octaves: int = 5, gain: float = 0.5,
        lacunarity: float = 2.0, seed: int = 7) -> np.ndarray:
    out = np.zeros(shape)
    amp, freq = 0.5, base
    for i in range(octaves):
        out += amp * value_noise(shape, freq, seed + i * 13)
        amp *= gain
        freq *= lacunarity
    return out / max(out.max(), 1e-9)


def make_base(n: int = 256, cx: float = 0.5, cy: float = 0.5, radius: float = 0.30,
              sharp: float = 0.06, seed: int = 7, level: float = 1.0) -> np.ndarray:
    """A shader-like RGB image in ``0..1``.  ``level`` scales it to leave headroom."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64) / (n - 1)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    shape = 1.0 - np.clip((d - radius) / sharp + 0.5, 0.0, 1.0)
    detail = fbm((n, n), seed=seed)
    v = np.clip(shape * 0.72 + detail * 0.42 - 0.06, 0.0, 1.0)
    lo = np.array([0.05, 0.07, 0.16])
    hi = np.array([0.95, 0.62, 0.24])
    img = lo + (hi - lo) * v[:, :, None]
    img *= (0.85 + 0.30 * shape)[:, :, None]
    return np.clip(img * level, 0.0, 1.0).astype(np.float32)


def smooth_gradient(n: int = 256) -> np.ndarray:
    """A perfectly smooth linear ramp -- almost no high-frequency energy."""
    ramp = np.linspace(0.05, 0.95, n)
    img = np.stack([ramp, ramp * 0.8 + 0.1, ramp * 0.5 + 0.2], axis=1)
    return np.repeat(img[None, :, :], n, axis=0).astype(np.float32)


def white_noise(n: int = 256, seed: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).random((n, n, 3)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Perturbations.  Each changes exactly ONE thing.
# --------------------------------------------------------------------------- #


def box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Edge-replicated box blur.  Changes edge *width* and nothing else."""
    if radius <= 0:
        return img.copy()
    k = 2 * radius + 1
    pad = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(img)
    for dy in range(k):
        for dx in range(k):
            out += pad[dy : dy + img.shape[0], dx : dx + img.shape[1]]
    return (out / (k * k)).astype(np.float32)


def brighten(img: np.ndarray, delta: float) -> np.ndarray:
    """Add ``delta`` to every channel.  Clips at 1.0 -- keep ``delta`` small."""
    return np.clip(img + delta, 0.0, 1.0).astype(np.float32)


def saturate(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale each pixel's distance from its own luma.  Changes S, not hue."""
    luma = (img @ np.array([0.2126, 0.7152, 0.0722])).astype(np.float32)
    return np.clip(luma[:, :, None] + (img - luma[:, :, None]) * factor, 0.0, 1.0).astype(np.float32)


def add_hf_noise(img: np.ndarray, sigma: float, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0.0, sigma, img.shape).astype(np.float32), 0.0, 1.0).astype(np.float32)


def shift(img: np.ndarray, dx: int, dy: int = 0) -> np.ndarray:
    """Translate by wrapping.  Changes *where* structure is, not how much."""
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1).astype(np.float32)


def shift_hue(img: np.ndarray, turns: float) -> np.ndarray:
    """Rotate hue in HSV, preserving S and V.

    HSV->RGB is ``(C, X, 0) + m`` with chroma ``C = V*S = max - min``.  Using the
    normalised ``S`` instead of the chroma silently shrinks saturation, which
    would make a hue change masquerade as a saturation change -- worth getting
    right, since this helper is what the discrimination test relies on.
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    safe = np.where(chroma > 1e-9, chroma, 1.0)
    hue = np.where(mx == r, np.mod((g - b) / safe, 6.0),
                   np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    hue = np.mod(hue + turns * 6.0, 6.0)
    x = chroma * (1.0 - np.abs(np.mod(hue, 2.0) - 1.0))
    m = mx - chroma
    z = np.zeros_like(chroma)
    conds = [hue < 1, hue < 2, hue < 3, hue < 4, hue < 5, hue < 6]
    out = np.stack([
        np.select(conds, [chroma, x, z, z, x, chroma]),
        np.select(conds, [x, chroma, chroma, x, z, z]),
        np.select(conds, [z, z, x, chroma, chroma, x]),
    ], axis=2) + m[:, :, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def base() -> np.ndarray:
    """One 256x256 shader-like image, shared by every test in the session."""
    return make_base(256)


@pytest.fixture(scope="session")
def headroom() -> np.ndarray:
    """A dimmer copy, so brightness shifts can be applied without clipping."""
    return make_base(256, level=0.62)


@pytest.fixture(scope="session")
def unrelated(base: np.ndarray) -> np.ndarray:
    """A structurally different image: different seed, position and size."""
    return make_base(256, cx=0.28, cy=0.72, radius=0.18, seed=99)
