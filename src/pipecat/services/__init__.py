#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Dict


def _warn_deprecated_access(globals: Dict[str, Any], attr, old: str, new: str):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(
            f"Module `pipecat.services.{old}` is deprecated, use `pipecat.services.{new}` instead",
            DeprecationWarning,
        )

    return globals[attr]


class DeprecatedModuleProxy:
    def __init__(self, globals: Dict[str, Any], old: str, new: str):
        self._globals = globals
        self._old = old
        self._new = new

    def __getattr__(self, attr):
        if attr in self._globals:
            return _warn_deprecated_access(self._globals, attr, self._old, self._new)
        raise AttributeError(f"module 'pipecat.{self._old}' has no attribute '{attr}'")
