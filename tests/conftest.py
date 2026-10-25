"""fake fcntl module for windows so the tests can import play_storage.py
(it uses fcntl file locks, which windows doesnt have)."""

from __future__ import annotations

import sys
import types

if sys.platform == "win32":
    try:
        import fcntl  # noqa: F401
    except ImportError:
        _stub = types.ModuleType("fcntl")
        _stub.LOCK_SH = 1
        _stub.LOCK_EX = 2
        _stub.LOCK_NB = 4
        _stub.flock = lambda *_args, **_kwargs: None
        sys.modules["fcntl"] = _stub
