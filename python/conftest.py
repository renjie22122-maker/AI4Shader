"""Pytest bootstrap: make ``import ai4shader`` work when run from ``python/``.

``python -m pytest tests/`` inserts the *test file's* directory into ``sys.path``,
not the project directory, so the package would not be importable without this.
It also works from a checkout that was never installed (there is no install step;
see ``pyproject.toml``).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
