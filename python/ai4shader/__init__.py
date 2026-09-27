"""AI4Shader offline Python layer.

Two jobs that belong in Python rather than in the browser or in Node:

1. **Real compiler validation** -- :mod:`ai4shader.validate` wraps
   ``glslangValidator`` / ``dxc`` / ``naga`` / ``xcrun metal`` and returns the
   compiler's own verdict.  An absent compiler yields ``ok=None``, never a false
   green.
2. **Image-similarity scoring and failure attribution** -- :mod:`ai4shader.score`
   computes the metrics and :mod:`ai4shader.attribute` turns them into a concrete
   ``{node, param, direction}`` patch suggestion (DESIGN.md section 4.3).

Nothing in this package makes a network call.  See ``score.no_network``.

Typical use::

    from ai4shader import imageio, score, attribute

    render = imageio.load("dist/demo/shader/frame.png", size=(512, 512))
    ref = imageio.load("examples/reference.jpg", size=(512, 512))
    bd = score.score(render, ref)
    for hint in attribute.attribute(bd):
        print(hint)
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import attribute, imageio, score  # noqa: E402

__all__ = ["attribute", "imageio", "score", "__version__"]


def __getattr__(name: str):
    """Import the subprocess-touching modules on first access only.

    ``validate`` and ``batch`` shell out and import ``subprocess``/``json``; a
    caller that only wants the metrics should not pay for that, and more
    importantly ``import ai4shader`` should never be able to fail because of a
    missing external tool.
    """
    if name in ("validate", "batch", "cli"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
