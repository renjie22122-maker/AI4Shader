"""Tests for :mod:`ai4shader.imageio`.

The stdlib PNG/PPM codecs are the *fallback* path, and a fallback that is never
executed is a fallback that does not work.  Every codec test therefore runs twice:
once with ``prefer="stdlib"`` (the pure-stdlib path, which is what a machine
without Pillow would use) and, when Pillow is installed, once with
``prefer="pil"``.  Parametrising by ``prefer`` is what keeps the fallback honest.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ai4shader import imageio as iio

PREFERS = ["stdlib"] + (["pil"] if iio.pil_available() else [])


# --------------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("prefer", PREFERS)
@pytest.mark.parametrize("ext", [".png", ".ppm"])
def test_round_trip_is_lossless_within_8bit(base, tmp_path, prefer, ext):
    """save -> load must reproduce the 8-bit quantisation of the input exactly."""
    path = tmp_path / f"rt{ext}"
    iio.save(base, path, prefer=prefer)
    back = iio.load(path, prefer=prefer)

    assert back.shape == base.shape
    assert back.dtype == np.float32
    assert 0.0 <= back.min() and back.max() <= 1.0

    expected = iio.to_uint8(base)
    got = iio.to_uint8(back)
    assert np.array_equal(got, expected), (
        f"{ext} round trip changed pixels: max delta {int(np.abs(got.astype(int) - expected.astype(int)).max())}"
    )


@pytest.mark.parametrize("ext", [".png", ".ppm"])
def test_round_trip_written_by_stdlib_read_by_pil(base, tmp_path, ext):
    """A file written by the fallback encoder must be readable by Pillow.

    This is the cross-check that matters in practice: CI (no Pillow) writes the
    fixture, a developer's machine (with Pillow) reads it.
    """
    if not iio.pil_available():
        pytest.skip("Pillow not installed; cross-decoding cannot be checked")
    path = tmp_path / f"cross{ext}"
    iio.save(base, path, prefer="stdlib")
    assert np.array_equal(iio.to_uint8(iio.load(path, prefer="pil")), iio.to_uint8(base))


@pytest.mark.parametrize("ext", [".png", ".ppm"])
def test_round_trip_written_by_pil_read_by_stdlib(base, tmp_path, ext):
    """...and the other direction."""
    if not iio.pil_available():
        pytest.skip("Pillow not installed")
    path = tmp_path / f"cross2{ext}"
    iio.save(base, path, prefer="pil")
    assert np.array_equal(iio.to_uint8(iio.load(path, prefer="stdlib")), iio.to_uint8(base))


def test_quantisation_is_the_identity_on_all_256_levels():
    """``to_uint8`` must invert ``c/255`` for every 8-bit level, or round trips drift."""
    levels = np.arange(256, dtype=np.uint8)
    img = np.repeat(levels[None, :, None], 3, axis=2).astype(np.float32) / 255.0
    img = np.repeat(img, 2, axis=0)
    assert np.array_equal(iio.to_uint8(img), np.repeat(levels[None, :, None], 3, axis=2).repeat(2, 0))


# --------------------------------------------------------------------------- #
# Stdlib codec coverage: the PNG variants a real reference image might use
# --------------------------------------------------------------------------- #


def test_png_encoder_emits_a_valid_ihdr_and_crc():
    """The hand-rolled encoder must produce a structurally valid PNG."""
    import struct
    import zlib

    data = iio.png_encode(np.full((4, 3, 3), 0.5, dtype=np.float32))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, colour, comp, filt, interlace = struct.unpack(">IIBBBBB", data[16:29])
    assert (width, height, depth, colour, comp, filt, interlace) == (3, 4, 8, 2, 0, 0, 0)
    # every chunk's CRC must verify (png_decode checks this and raises otherwise)
    assert iio.png_decode(data).shape == (4, 3, 3)
    assert b"IEND" in data and zlib.decompress(data[data.index(b"IDAT") + 4 : -12]) is not None


@pytest.mark.parametrize(
    "colour,channels",
    [(0, 1), (2, 3), (4, 2), (6, 4)],
)
def test_stdlib_png_decodes_every_8bit_colour_type(colour, channels):
    """Grey, RGB, grey+alpha and RGBA must all decode; alpha is dropped, never composited."""
    from PIL import Image

    mode = {0: "L", 2: "RGB", 4: "LA", 6: "RGBA"}[colour]
    rng = np.random.default_rng(colour)
    arr = rng.integers(0, 256, (8, 9, channels), dtype=np.uint8)
    img = Image.fromarray(arr[:, :, 0] if channels == 1 else arr, mode=mode)
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    decoded = iio.png_decode(buf.getvalue())
    assert decoded.shape == (8, 9, 3)
    # RGB channels must survive; alpha must be dropped, and grey replicated to RGB.
    if channels >= 3:
        expected = arr[:, :, :3]
    else:
        expected = np.repeat(arr[:, :, :1], 3, axis=2)
    assert np.array_equal(iio.to_uint8(decoded), expected.astype(np.uint8))


def test_stdlib_png_decodes_palette_and_16bit():
    from PIL import Image

    import io

    # palette
    pal = Image.fromarray(
        np.random.default_rng(1).integers(0, 4, (6, 7), dtype=np.uint8), mode="P"
    )
    pal.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * (256 * 3 - 12))
    buf = io.BytesIO()
    pal.save(buf, format="PNG")
    out = iio.png_decode(buf.getvalue())
    assert out.shape == (6, 7, 3)

    # 16-bit greyscale
    arr16 = (np.random.default_rng(2).random((5, 5)) * 65535).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(arr16, mode="I;16").save(buf, format="PNG")
    out16 = iio.png_decode(buf.getvalue())
    assert out16.shape == (5, 5, 3)
    assert out16.max() <= 1.0 and out16.min() >= 0.0


def test_stdlib_png_rejects_interlaced_with_a_clear_error():
    """Adam7 must raise, not return a wrong image."""
    from PIL import Image

    import io

    arr = np.random.default_rng(3).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG", interlace=True)
    if b"Adam7" not in buf.getvalue() and _is_not_interlaced(buf.getvalue()):
        pytest.skip("Pillow did not produce an interlaced PNG")
    with pytest.raises(NotImplementedError, match="Adam7"):
        iio.png_decode(buf.getvalue())


def _is_not_interlaced(data: bytes) -> bool:
    return data[28] == 0


def test_ppm_handles_grey_and_16bit_maxval():
    grey = b"P5\n2 2\n255\n" + bytes([0, 85, 170, 255])
    out = iio.ppm_decode(grey)
    assert out.shape == (2, 2, 3)
    assert np.allclose(out[0, 0], 0.0) and np.allclose(out[1, 1], 1.0)

    wide = b"P6\n1 1\n65535\n" + (65535).to_bytes(2, "big") * 3
    assert np.allclose(iio.ppm_decode(wide), 1.0)


def test_ppm_rejects_ascii_variants_explicitly():
    with pytest.raises(NotImplementedError, match="ASCII PNM"):
        iio.ppm_decode(b"P3\n1 1\n255\n255 0 0\n")


def test_png_decode_rejects_a_corrupt_chunk():
    data = bytearray(iio.png_encode(np.zeros((4, 4, 3), dtype=np.float32)))
    data[30] ^= 0xFF  # damage an IDAT byte so the CRC no longer matches
    with pytest.raises(ValueError, match="CRC"):
        iio.png_decode(bytes(data))


# --------------------------------------------------------------------------- #
# Fitting -- the part that silently corrupts metrics when it is wrong
# --------------------------------------------------------------------------- #


def test_letterbox_preserves_aspect_ratio(base):
    """256x256 squeezed into 128x64 must letterbox to *square* 64x64 content with
    32px bars on the left and right, not stretch the disc into an ellipse."""
    out = iio.fit(base, (128, 64), mode="letterbox", pad=0.0)
    assert out.shape == (64, 128, 3)
    # scale = min(128/256, 64/256) = 0.25 -> 64x64 content, vertically centred in
    # a 128x64 canvas: bars on the left/right, none on the top/bottom.
    assert np.allclose(out[:, :32], 0.0), "left bar should be pure padding"
    assert np.allclose(out[:, 96:], 0.0), "right bar should be pure padding"
    content = out[:, 32:96]
    assert content.max() > 0.3, "the content column band should hold the image"
    # Square content: the disc stays a disc.  Its vertical and horizontal extents
    # of non-background pixels must match, which stretching would break.
    mask = content.max(axis=2) > 0.02
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert abs((rows.max() - rows.min()) - (cols.max() - cols.min())) <= 2, (
        "content is not square: the aspect ratio was not preserved"
    )


def test_stretch_and_letterbox_differ_and_stretch_is_documented_as_wrong(base):
    stretched = iio.fit(base, (128, 64), mode="stretch")
    letterboxed = iio.fit(base, (128, 64), mode="letterbox")
    assert stretched.shape == letterboxed.shape == (64, 128, 3)
    # Stretching an isotropic image to 2:1 must visibly differ from letterboxing it.
    assert float(np.abs(stretched - letterboxed).mean()) > 0.05


def test_letterbox_pad_mean_avoids_black_bars(base):
    """Default padding is each image's own mean, so bars cannot fake agreement."""
    out = iio.fit(base, (128, 64), mode="letterbox", pad="mean")
    mean = float(base.reshape(-1, 3).mean())
    assert abs(float(out[0, 0].mean()) - mean) < 0.05
    black = iio.fit(base, (128, 64), mode="letterbox", pad="black")
    assert np.allclose(black[0, 0], 0.0)


def test_crop_mode_covers_the_target(base):
    out = iio.fit(base, (100, 200), mode="crop")
    assert out.shape == (200, 100, 3)
    assert out.max() > 0.3  # real content, not mostly padding


def test_fit_rejects_an_unknown_mode(base):
    with pytest.raises(ValueError, match="mode must be"):
        iio.fit(base, (8, 8), mode="squash")


@pytest.mark.parametrize("size", [(64, 64), (37, 91), (1, 1), 33])
def test_fit_always_returns_exactly_the_requested_shape(base, size):
    w, h = (size, size) if isinstance(size, int) else size
    assert iio.fit(base, size).shape == (h, w, 3)


def test_resize_downscale_is_an_area_average_not_point_sampling():
    """A 2x2 checkerboard reduced to 1x1 must average to the midpoint, not pick a corner."""
    checker = np.zeros((8, 8, 3), dtype=np.float32)
    checker[::2, ::2] = 1.0
    checker[1::2, 1::2] = 1.0
    out = iio.resize(checker, (1, 1))
    assert abs(float(out[0, 0, 0]) - 0.5) < 1e-6


# --------------------------------------------------------------------------- #
# Dispatching and error messages
# --------------------------------------------------------------------------- #


def test_load_with_size_fits_but_load_without_size_does_not(base, tmp_path):
    path = tmp_path / "a.png"
    iio.save(base, path)
    assert iio.load(path).shape == base.shape
    assert iio.load(path, size=(64, 32)).shape == (32, 64, 3)


def test_jpeg_requires_pillow_or_works(tmp_path, base):
    path = tmp_path / "a.jpg"
    if iio.pil_available():
        iio.save(base, path)
        assert iio.load(path).shape == base.shape
    else:
        with pytest.raises(RuntimeError, match="requires Pillow"):
            iio.save(base, path)


def test_unknown_extension_is_refused(tmp_path, base):
    with pytest.raises((ValueError, RuntimeError)):
        iio.save(base, tmp_path / "a.xyz")


def test_prefer_pil_without_pil_raises_clearly(tmp_path, base, monkeypatch):
    monkeypatch.setattr(iio, "_PIL_CACHE", False)
    path = tmp_path / "a.png"
    iio.save(base, path, prefer="stdlib")
    with pytest.raises(RuntimeError, match="PIL is not importable"):
        iio.load(path, prefer="pil")


def test_bad_prefer_value_is_rejected(base, tmp_path):
    iio.save(base, tmp_path / "a.png")
    with pytest.raises(ValueError, match="prefer must be"):
        iio.load(tmp_path / "a.png", prefer="magic")
