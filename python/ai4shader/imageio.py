"""Image loading and saving for AI4Shader's offline layer.

Contract
--------
Every image crossing this module's boundary is a ``numpy.ndarray`` of shape
``(H, W, 3)``, dtype ``float32``, values in ``0.0 .. 1.0`` (linear-ish sRGB
encoded, no gamma conversion is performed anywhere).  Height first, then width:
``img[y, x, c]``.  Alpha is **dropped**, not composited -- scoring compares RGB,
and compositing would silently invent colour that neither image contains.

PIL is used when importable, but it is *not* a hard dependency: a pure-stdlib
PNG codec (``zlib`` + ``struct``) and a pure-stdlib PPM codec cover the formats
this project actually produces.  ``load``/``save`` take ``prefer=`` so the
fallback path is reachable on a machine that *does* have PIL -- without that,
the fallback would be dead code that nobody ever exercises.

Resizing is done with our own numpy resampler, **never** with PIL, even when PIL
is present.  Two reasons:

1. Reproducibility.  A score computed on a machine with PIL must equal the score
   computed on a machine without it.  Two different bicubic/filter
   implementations would break that.
2. Quality.  Downscaling uses an area (box) average, not bilinear point
   sampling, so a 4K reference reduced to 512x512 does not alias into fake
   high-frequency detail -- which matters because the whole point of the
   frequency-band metric is to measure high-frequency detail honestly.

Letterboxing instead of stretching
----------------------------------
A reference image and a render are almost never the same size.  Stretching them
to match silently corrupts every metric: SSIM drops because structure moved,
band energy shifts because the stretch changed the effective sampling rate, and
the attribution then blames the shader for a resize artefact.

``load(..., size=(w, h))`` therefore letterboxes by default, and the padding is
filled with **each image's own mean colour** rather than black.  The reasoning:
black padding is identical in both images, so it contributes SSIM ~= 1.0 over the
bar area and *inflates* the score.  Mean-colour padding carries each image's own
tone, so the bars differ whenever the images differ and the padding cannot be
mistaken for agreement.  ``pad="black"`` is still available for callers that
want conventional letterboxing.

Units and ranges
----------------
* ``load`` -> ``float32``, shape ``(H, W, 3)``, range ``0.0 .. 1.0``.
* ``save`` accepts ``float32``/``float64`` in ``0..1`` or ``uint8`` in ``0..255``.
* No function here performs any colour-space conversion.
"""

from __future__ import annotations

import os
import struct
import zlib
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "load",
    "save",
    "resize",
    "fit",
    "to_uint8",
    "png_decode",
    "png_encode",
    "ppm_decode",
    "ppm_encode",
    "pil_available",
    "SUPPORTED_EXTENSIONS",
]

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Extensions this module can write without PIL.
STDLIB_EXTENSIONS = frozenset({".png", ".ppm", ".pgm", ".pnm"})
#: Extensions that additionally become available when PIL is importable.
PIL_EXTENSIONS = frozenset({".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"})
SUPPORTED_EXTENSIONS = STDLIB_EXTENSIONS | PIL_EXTENSIONS


# --------------------------------------------------------------------------- #
# Optional PIL
# --------------------------------------------------------------------------- #

_PIL_CACHE: bool | None = None


def pil_available() -> bool:
    """Return True when ``PIL.Image`` can be imported.  Result is cached.

    Only the probe is cached, never a decoded image: the caller must not be able
    to observe stale pixels.
    """
    global _PIL_CACHE
    if _PIL_CACHE is None:
        try:
            import PIL.Image  # noqa: F401

            _PIL_CACHE = True
        except Exception:
            _PIL_CACHE = False
    return _PIL_CACHE


def _want_pil(prefer: str) -> bool:
    if prefer not in ("auto", "pil", "stdlib"):
        raise ValueError(f"prefer must be 'auto', 'pil' or 'stdlib', got {prefer!r}")
    if prefer == "stdlib":
        return False
    if prefer == "pil":
        if not pil_available():
            raise RuntimeError("prefer='pil' requested but PIL is not importable")
        return True
    return pil_available()


# --------------------------------------------------------------------------- #
# Pixel-format helpers
# --------------------------------------------------------------------------- #


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Quantise to ``uint8`` with round-half-away-from-zero, saturating at 0/255.

    Accepts ``0..1`` floats or ``0..255`` integers.  Rounding is ``floor(x*255 +
    0.5)``, which is the inverse of the ``c/255`` used by :func:`load`, so an
    8-bit -> float -> 8-bit round trip is the identity for every level.
    """
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.copy()
    if np.issubdtype(a.dtype, np.floating):
        return np.clip(np.floor(a * 255.0 + 0.5), 0.0, 255.0).astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)


def _to_float01(a: np.ndarray) -> np.ndarray:
    """Normalise any decoded buffer to ``(H, W, 3) float32 0..1``."""
    a = np.asarray(a)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    else:
        a = a.astype(np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D image array, got shape {a.shape}")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.shape[2] == 2:  # grey + alpha -> drop alpha
        a = np.repeat(a[:, :, :1], 3, axis=2)
    elif a.shape[2] >= 4:  # drop alpha, keep RGB
        a = a[:, :, :3]
    elif a.shape[2] != 3:
        raise ValueError(f"unsupported channel count {a.shape[2]}")
    return np.ascontiguousarray(a, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Pure-stdlib PNG
# --------------------------------------------------------------------------- #


def _paeth(a: int, b: int, c: int) -> int:
    """PNG Paeth predictor (spec 9.4).  Scalar form, used by the byte loop."""
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _png_chunks(data: bytes):
    if data[:8] != _PNG_SIGNATURE:
        raise ValueError("not a PNG file (bad signature)")
    pos = 8
    end = len(data)
    while pos + 8 <= end:
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        pos += 8
        if pos + length + 4 > end:
            raise ValueError(f"truncated PNG chunk {ctype!r}")
        payload = data[pos : pos + length]
        pos += length
        (crc,) = struct.unpack(">I", data[pos : pos + 4])
        pos += 4
        expected = zlib.crc32(ctype + payload) & 0xFFFFFFFF
        if crc != expected:
            raise ValueError(f"PNG chunk {ctype!r} CRC mismatch")
        yield ctype, payload


def _unfilter(raw: bytes, height: int, stride: int, bpp: int) -> np.ndarray:
    """Reverse PNG per-scanline filtering.  Returns ``(height, stride) uint8``."""
    out = np.empty((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    pos = 0
    for y in range(height):
        if pos + 1 + stride > len(raw):
            raise ValueError("truncated PNG image data")
        ft = raw[pos]
        pos += 1
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=pos).copy()
        pos += stride

        if ft == 0:  # None
            cur = line
        elif ft == 1:  # Sub -- per-channel prefix sum, vectorised
            plane = line.reshape(-1, bpp)
            cur = np.cumsum(plane, axis=0, dtype=np.uint32).astype(np.uint8).reshape(stride)
        elif ft == 2:  # Up -- fully vectorised
            cur = (line + prev).astype(np.uint8)
        elif ft == 3:  # Average
            cur = line.copy()
            pv = prev
            for i in range(stride):
                left = int(cur[i - bpp]) if i >= bpp else 0
                cur[i] = (int(cur[i]) + ((left + int(pv[i])) >> 1)) & 0xFF
        elif ft == 4:  # Paeth
            cur = line.copy()
            pv = prev
            for i in range(stride):
                if i >= bpp:
                    left = int(cur[i - bpp])
                    upleft = int(pv[i - bpp])
                else:
                    left = 0
                    upleft = 0
                cur[i] = (int(cur[i]) + _paeth(left, int(pv[i]), upleft)) & 0xFF
        else:
            raise ValueError(f"unknown PNG filter type {ft} on row {y}")
        out[y] = cur
        prev = cur
    return out


def _unpack_bits(row_bytes: np.ndarray, width: int, channels: int, depth: int) -> np.ndarray:
    """Expand 1/2/4-bit samples to ``uint8`` 0..255 (PNG samples are MSB-first)."""
    per_byte = 8 // depth
    mask = (1 << depth) - 1
    flat = row_bytes.reshape(row_bytes.shape[0], -1)
    idx = np.arange(width * channels)
    byte_i = idx // per_byte
    shift = 8 - depth * (idx % per_byte + 1)
    vals = (flat[:, byte_i] >> shift) & mask
    if depth < 8:  # scale to full range, matching the 16-bit -> 8-bit convention
        vals = (vals * (255 // mask)).astype(np.uint8)
    return vals.astype(np.uint8)


def png_decode(data: bytes) -> np.ndarray:
    """Decode a PNG byte string to ``(H, W, 3) float32 0..1`` with pure stdlib.

    Supports bit depths 1/2/4/8/16, colour types 0 (grey), 2 (RGB), 3 (palette),
    4 (grey+alpha) and 6 (RGBA).  Adam7 interlacing is **not** supported and
    raises ``NotImplementedError`` rather than returning a wrong image.

    Alpha is dropped (see the module docstring).  16-bit samples are scaled by
    ``1/65535``; sub-8-bit samples are scaled by ``255/max`` so that "white stays
    white" regardless of the source depth.
    """
    ihdr = None
    idat = bytearray()
    palette: bytes | None = None
    transparency: bytes | None = None

    for ctype, payload in _png_chunks(data):
        if ctype == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif ctype == b"IDAT":
            idat += payload
        elif ctype == b"PLTE":
            palette = bytes(payload)
        elif ctype == b"tRNS":
            transparency = bytes(payload)
        elif ctype == b"IEND":
            break

    if ihdr is None:
        raise ValueError("PNG has no IHDR chunk")

    width, height, depth, colour, compression, filt, interlace = ihdr
    if compression != 0:
        raise ValueError(f"unsupported PNG compression method {compression}")
    if filt != 0:
        raise ValueError(f"unsupported PNG filter method {filt}")
    if interlace != 0:
        raise NotImplementedError(
            "Adam7-interlaced PNG is not supported by the stdlib decoder; "
            "install Pillow or re-save the image without interlacing"
        )

    channels_of = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if colour not in channels_of:
        raise ValueError(f"unsupported PNG colour type {colour}")
    channels = channels_of[colour]
    if depth not in (1, 2, 4, 8, 16):
        raise ValueError(f"unsupported PNG bit depth {depth}")
    if colour not in (0, 3) and depth < 8:
        raise ValueError(f"PNG colour type {colour} cannot use bit depth {depth}")

    raw = zlib.decompress(bytes(idat))
    stride = (width * channels * depth + 7) // 8
    bpp = max(1, channels * depth // 8)
    if len(raw) < height * (stride + 1):
        raise ValueError("PNG image data is shorter than its declared dimensions")

    rows = _unfilter(raw, height, stride, bpp)

    if depth == 16:
        samples = rows.reshape(height, width * channels, 2)
        arr = (samples[:, :, 0].astype(np.uint16) << 8) | samples[:, :, 1]
        arr = arr.reshape(height, width, channels)
        if colour == 3:
            raise ValueError("palette PNG cannot use bit depth 16")
        return _to_float01(arr.astype(np.float32) / 65535.0)

    if depth < 8:
        arr = _unpack_bits(rows, width, channels, depth).reshape(height, width, channels)
    else:
        arr = rows.reshape(height, width, channels)

    if colour == 3:
        if palette is None:
            raise ValueError("palette PNG has no PLTE chunk")
        entries = np.frombuffer(palette, dtype=np.uint8).reshape(-1, 3)
        idx = np.clip(arr[:, :, 0].astype(np.int32), 0, len(entries) - 1)
        rgb = entries[idx]
        if transparency is not None:
            alpha = np.frombuffer(transparency, dtype=np.uint8)
            keep = np.ones(len(entries), dtype=bool)
            keep[: len(alpha)] = alpha > 0
            rgb = np.where(keep[idx][:, :, None], rgb, 0)
        return _to_float01(rgb.astype(np.uint8))

    return _to_float01(arr.astype(np.uint8))


def png_encode(img: np.ndarray) -> bytes:
    """Encode to an 8-bit RGB PNG byte string with pure stdlib.

    Filter type 0 (None) is used on every scanline: it costs a few percent of
    file size and makes the output trivially auditable, which is the right
    trade-off for test fixtures and diffable artefacts.
    """
    u8 = to_uint8(img)
    if u8.ndim == 2:
        u8 = np.repeat(u8[:, :, None], 3, axis=2)
    if u8.ndim != 3 or u8.shape[2] != 3:
        raise ValueError(f"cannot encode shape {u8.shape} as RGB PNG")
    height, width = u8.shape[:2]

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw += u8[y].tobytes()
    idat = zlib.compress(bytes(raw), 9)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return _PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


# --------------------------------------------------------------------------- #
# Pure-stdlib PPM / PGM
# --------------------------------------------------------------------------- #


def _ppm_header(data: bytes) -> tuple[bytes, int, int, int, int]:
    """Parse a PNM header.  Returns (magic, width, height, maxval, data_offset)."""
    pos = 0
    fields: list[bytes] = []
    while len(fields) < 4:
        while pos < len(data) and data[pos : pos + 1].isspace():
            pos += 1
        if pos < len(data) and data[pos : pos + 1] == b"#":
            while pos < len(data) and data[pos] != 0x0A:
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos : pos + 1].isspace():
            pos += 1
        if start == pos:
            raise ValueError("truncated PNM header")
        fields.append(data[start:pos])
    pos += 1  # exactly one whitespace byte separates the header from the raster
    magic = fields[0]
    try:
        width, height, maxval = int(fields[1]), int(fields[2]), int(fields[3])
    except ValueError as exc:
        raise ValueError(f"malformed PNM header {fields!r}") from exc
    if maxval <= 0 or maxval > 65535:
        raise ValueError(f"PNM maxval out of range: {maxval}")
    return magic, width, height, maxval, pos


def ppm_decode(data: bytes) -> np.ndarray:
    """Decode binary PPM (P6) or PGM (P5) to ``(H, W, 3) float32 0..1``.

    ASCII P3/P2 variants are rejected with a clear message rather than decoded
    slowly and partially.  ``maxval`` may be up to 65535 (2-byte big-endian
    samples, one byte per sample when ``maxval < 256``).
    """
    magic, width, height, maxval, offset = _ppm_header(data)
    if magic == b"P6":
        channels = 3
    elif magic == b"P5":
        channels = 1
    elif magic in (b"P3", b"P2"):
        raise NotImplementedError(
            f"ASCII PNM ({magic.decode()}) is not supported; convert to binary {magic[:1].decode()}6/{magic[:1].decode()}5"
        )
    else:
        raise ValueError(f"unsupported PNM magic {magic!r}")

    if maxval < 256:
        count = width * height * channels
        raster = np.frombuffer(data, dtype=np.uint8, count=count, offset=offset)
        if raster.size < count:
            raise ValueError("truncated PNM raster")
        arr = raster.reshape(height, width, channels).astype(np.float32) / float(maxval)
    else:
        count = width * height * channels
        raster = np.frombuffer(data, dtype=">u2", count=count, offset=offset)
        if raster.size < count:
            raise ValueError("truncated PNM raster")
        arr = raster.reshape(height, width, channels).astype(np.float32) / float(maxval)
    return _to_float01(arr)


def ppm_encode(img: np.ndarray) -> bytes:
    """Encode to binary PPM (P6) with 8-bit samples."""
    u8 = to_uint8(img)
    if u8.ndim == 2:
        u8 = np.repeat(u8[:, :, None], 3, axis=2)
    if u8.ndim != 3 or u8.shape[2] != 3:
        raise ValueError(f"cannot encode shape {u8.shape} as RGB PPM")
    height, width = u8.shape[:2]
    return b"P6\n%d %d\n255\n" % (width, height) + u8.tobytes()


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


def _blend_axis(img: np.ndarray, axis: int, i0: np.ndarray, i1: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """Linear blend ``img[..., i0, ...] * (1-f) + img[..., i1, ...] * f`` along ``axis``."""
    a = np.take(img, i0, axis=axis)
    b = np.take(img, i1, axis=axis)
    shape = [1] * img.ndim
    shape[axis] = len(frac)
    f = frac.reshape(shape).astype(img.dtype)
    return a * (1.0 - f) + b * f


def _resample_axis(img: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Resample one axis to ``n_out`` samples.

    Downscaling uses an exact **area average** (each output sample is the mean of
    the input interval it covers), which is a box low-pass: it cannot invent
    detail that was not there.  Upscaling uses bilinear interpolation.  PIL is
    deliberately not used -- see the module docstring.
    """
    n_in = img.shape[axis]
    if n_in == n_out:
        return img

    if n_out < n_in:
        edges = np.linspace(0.0, float(n_in), n_out + 1)
        starts = np.floor(edges[:-1]).astype(np.int64)
        stops = np.ceil(edges[1:]).astype(np.int64)
        stops = np.maximum(stops, starts + 1)
        stops = np.minimum(stops, n_in)
        # Dense weight matrix (n_out x n_in).  Bounded by the preview sizes this
        # project uses (<= 2048); it is a few MB at worst and pays for exactness.
        weights = np.zeros((n_out, n_in), dtype=np.float64)
        for i in range(n_out):
            lo, hi = float(edges[i]), float(edges[i + 1])
            idx = np.arange(starts[i], stops[i])
            weights[i, idx] = np.minimum(idx + 1.0, hi) - np.maximum(idx.astype(np.float64), lo)
        weights /= weights.sum(axis=1, keepdims=True)
        moved = np.moveaxis(img.astype(np.float64), axis, 0)
        out = np.tensordot(weights, moved, axes=([1], [0]))
        return np.moveaxis(out, 0, axis)

    src = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5
    i0 = np.floor(src).astype(np.int64)
    frac = (src - i0).astype(np.float64)
    i0c = np.clip(i0, 0, n_in - 1)
    # Below the first / above the last sample the linear ramp is clamped, which
    # is what edge extension means; frac is pinned to 0 so no extrapolation occurs.
    frac = np.where(i0 < 0, 0.0, frac)
    frac = np.where(i0 > n_in - 2, 0.0, frac)
    i0c = np.clip(i0, 0, max(0, n_in - 2)) if n_in >= 2 else np.zeros_like(i0)
    i1c = np.clip(i0c + 1, 0, n_in - 1)
    return _blend_axis(img.astype(np.float64), axis, i0c, i1c, frac)


def resize(img: np.ndarray, size: int | Sequence[int]) -> np.ndarray:
    """Resample to ``size`` with **no** aspect-ratio preservation (see :func:`fit`).

    ``size`` is ``(width, height)``, or a single int for a square.  Returns
    ``float32`` in ``0..1``.  This is the "stretch" primitive; calling it
    directly on a reference/render pair is almost always a mistake.
    """
    w, h = _parse_size(size)
    out = np.asarray(img, dtype=np.float64)
    out = _resample_axis(out, w, axis=1)
    out = _resample_axis(out, h, axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


def _parse_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, (int, np.integer)):
        side = int(size)
        if side <= 0:
            raise ValueError("size must be positive")
        return side, side
    seq = tuple(int(v) for v in size)
    if len(seq) != 2:
        raise ValueError(f"size must be an int or (width, height), got {size!r}")
    if seq[0] <= 0 or seq[1] <= 0:
        raise ValueError("size components must be positive")
    return seq[0], seq[1]


def fit(
    img: np.ndarray,
    size: int | Sequence[int],
    *,
    mode: str = "letterbox",
    pad: str | float = "mean",
) -> np.ndarray:
    """Scale ``img`` to ``size`` preserving aspect ratio.

    ``mode``:

    * ``"letterbox"`` (default) -- scale to fit inside, pad the remainder.
    * ``"crop"`` -- scale to cover, centre-crop the overflow.
    * ``"stretch"`` -- plain :func:`resize`; **corrupts metrics**, provided only
      so callers can reproduce the mistake deliberately.

    ``pad`` is used by ``"letterbox"`` only:

    * ``"mean"`` (default) -- each image's own mean colour.  Padding cannot be
      mistaken for agreement, so SSIM is not inflated by the bars.
    * ``"black"`` -- conventional black bars.  Identical in both images, which
      *does* inflate SSIM over the bar area.
    * ``"edge"`` -- replicate the border pixels.
    * a float in ``0..1`` -- that grey level.

    Returns ``float32`` ``(H, W, 3)`` in ``0..1``; the result is exactly
    ``(size[1], size[0], 3)``.
    """
    w, h = _parse_size(size)
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"fit expects (H, W, 3), got {arr.shape}")
    src_h, src_w = arr.shape[:2]
    if src_h == 0 or src_w == 0:
        raise ValueError("cannot fit an empty image")

    if mode == "stretch":
        return resize(arr, (w, h))
    if mode not in ("letterbox", "crop"):
        raise ValueError(f"mode must be 'letterbox', 'crop' or 'stretch', got {mode!r}")

    scale = min(w / src_w, h / src_h) if mode == "letterbox" else max(w / src_w, h / src_h)
    new_w = max(1, min(w, int(round(src_w * scale))))
    new_h = max(1, min(h, int(round(src_h * scale)))) if mode == "letterbox" else max(1, int(round(src_h * scale)))
    if mode == "crop":
        new_w = max(w, int(round(src_w * scale)))
        new_h = max(h, int(round(src_h * scale)))
    scaled = resize(arr, (new_w, new_h))

    if mode == "crop":
        x0 = (new_w - w) // 2
        y0 = (new_h - h) // 2
        return np.ascontiguousarray(scaled[y0 : y0 + h, x0 : x0 + w])

    canvas = np.empty((h, w, 3), dtype=np.float32)
    if pad == "edge":
        canvas[:] = scaled[0, 0]
        canvas[: scaled.shape[0], : scaled.shape[1]] = scaled
        if scaled.shape[0] < h:
            canvas[scaled.shape[0] :, : scaled.shape[1]] = scaled[-1:]
        if scaled.shape[1] < w:
            canvas[:, scaled.shape[1] :] = canvas[:, scaled.shape[1] - 1 : scaled.shape[1]]
        return canvas

    if pad == "mean":
        fill = scaled.reshape(-1, 3).mean(axis=0)
    elif pad == "black":
        fill = np.zeros(3, dtype=np.float32)
    elif isinstance(pad, (int, float, np.floating)):
        fill = np.full(3, float(pad), dtype=np.float32)
    else:
        raise ValueError(f"pad must be 'mean', 'black', 'edge' or a float, got {pad!r}")
    canvas[:] = fill
    y0 = (h - scaled.shape[0]) // 2
    x0 = (w - scaled.shape[1]) // 2
    canvas[y0 : y0 + scaled.shape[0], x0 : x0 + scaled.shape[1]] = scaled
    return canvas


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def _decode(path: str | os.PathLike[str], prefer: str):
    ext = os.path.splitext(os.fspath(path))[1].lower()
    use_pil = _want_pil(prefer)

    if ext in PIL_EXTENSIONS and not use_pil:
        raise RuntimeError(
            f"{ext} requires Pillow, which is not importable. "
            f"Without PIL only {sorted(STDLIB_EXTENSIONS)} can be read."
        )

    if use_pil and (ext in PIL_EXTENSIONS or ext not in STDLIB_EXTENSIONS):
        from PIL import Image

        with Image.open(os.fspath(path)) as im:
            return _to_float01(np.asarray(im.convert("RGB")))

    with open(os.fspath(path), "rb") as fh:
        data = fh.read()
    if ext == ".png" or data[:8] == _PNG_SIGNATURE:
        return png_decode(data)
    if ext in (".ppm", ".pgm", ".pnm") or data[:2] in (b"P6", b"P5", b"P3", b"P2"):
        return ppm_decode(data)
    if use_pil:
        from PIL import Image

        with Image.open(os.fspath(path)) as im:
            return _to_float01(np.asarray(im.convert("RGB")))
    raise ValueError(f"unsupported image format for {os.fspath(path)!r} (extension {ext!r})")


def load(
    path: str | os.PathLike[str],
    *,
    size: int | Sequence[int] | None = None,
    mode: str = "letterbox",
    pad: str | float = "mean",
    prefer: str = "auto",
) -> np.ndarray:
    """Load an image as ``(H, W, 3) float32`` in ``0.0 .. 1.0`` (higher = brighter).

    Reads PNG and PPM/PGM with no third-party dependency at all; JPEG/BMP/TIFF/
    WebP/GIF additionally when PIL is importable.  Alpha is dropped.

    ``size`` -- if given, the image is scaled to that ``(width, height)`` (or a
    square of that side).  **Do not pass the render's size and the reference's
    size as the same number without thinking**: see the module docstring on why
    stretching is wrong and what the default letterbox padding does.
    ``mode``/``pad`` are forwarded to :func:`fit`.

    ``prefer`` -- ``"auto"`` (default) uses PIL when available, ``"pil"``
    demands it, ``"stdlib"`` forces the built-in codecs (this is how the tests
    reach the fallback path on a machine that has Pillow).
    """
    img = _decode(path, prefer)
    if size is not None:
        img = fit(img, size, mode=mode, pad=pad)
    return np.ascontiguousarray(img, dtype=np.float32)


def save(img: np.ndarray, path: str | os.PathLike[str], *, prefer: str = "auto") -> None:
    """Write an image, choosing the codec from the file extension.

    Accepts ``float32``/``float64`` in ``0..1`` or ``uint8`` in ``0..255``.
    Quantisation is round-half-away-from-zero and saturating.

    PNG and PPM/PGM are written by the pure-stdlib encoders (PNG is always
    8-bit RGB, filter type 0).  ``.jpg``/``.bmp``/``.tif``/``.webp`` require PIL
    and raise ``RuntimeError`` without it -- JPEG is lossy, so it is never used
    for anything a metric reads back.
    """
    ext = os.path.splitext(os.fspath(path))[1].lower()
    if not ext:
        raise ValueError(f"cannot infer an image format from {os.fspath(path)!r}")
    use_pil = _want_pil(prefer) and ext in PIL_EXTENSIONS
    if ext in PIL_EXTENSIONS and not use_pil:
        raise RuntimeError(
            f"writing {ext} requires Pillow, which is not importable. "
            f"Without PIL only {sorted(STDLIB_EXTENSIONS)} can be written."
        )
    if use_pil:
        from PIL import Image

        Image.fromarray(to_uint8(img), mode="RGB").save(os.fspath(path))
        return
    if ext == ".png":
        payload: bytes = png_encode(img)
    elif ext in (".ppm", ".pgm", ".pnm"):
        payload = ppm_encode(img)
    else:
        raise ValueError(f"cannot write {ext!r} without Pillow")
    with open(os.fspath(path), "wb") as fh:
        fh.write(payload)
