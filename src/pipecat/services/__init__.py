#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Dict

# Track which modules we've already warned about
_warned_modules = set()


def _warn_deprecated_access(globals: Dict[str, Any], attr, old: str, new: str):
    import warnings

    # Only warn once per old->new module pair
    module_key = (old, new)
    if module_key not in _warned_modules:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                f"Module `pipecat.services.{old}` is deprecated, use `pipecat.services.{new}` instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        _warned_modules.add(module_key)

    return globals[attr]


class DeprecatedModuleProxy:
    """Proxy class to handle access to deprecated modules.

    This class is used to redirect access from an old module to a new one,
    while issuing a deprecation warning. It should be used in the `__init__.py` of a module
    that has been renamed or moved, to ensure that users are informed about
    the change when they try to access the old module.
    """

    def __init__(self, globals: Dict[str, Any], old: str, new: str):
        """Initialize the DeprecatedModuleProxy.

        Args:
            globals (Dict[str, Any]): The globals dictionary of the module.
            old (str): The name of the old module.
            new (str): The name of the new module to redirect to.
        """
        self._globals = globals
        self._old = old
        self._new = new

    def __getattr__(self, attr):
        if attr in self._globals:
            return _warn_deprecated_access(self._globals, attr, self._old, self._new)
        raise AttributeError(f"module 'pipecat.services.{self._old}' has no attribute '{attr}'")
