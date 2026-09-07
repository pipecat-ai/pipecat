#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Types shared across Pipecat, independent of any provider SDK.

The ``NOT_GIVEN`` sentinel lives here so that settings, contexts and anything
else needing "this value was not provided" share one meaning. Provider SDKs
ship their own equivalents; those belong to the SDK and are translated at the
adapter boundary rather than used inside Pipecat.
"""

from typing import Literal, TypeGuard, TypeVar


class NotGiven:
    """Sentinel type meaning "this value was not provided".

    Distinct from ``None``, which is a meaningful value in most of the places
    this is used (typically "this service doesn't support this field").
    ``NOT_GIVEN`` is the singleton instance; test for it with :func:`is_given`
    rather than comparing directly.

    Falsy, so ``value or default`` treats a missing value as absent.
    """

    _instance: "NotGiven | None" = None

    def __new__(cls) -> "NotGiven":
        """Return the singleton instance, creating it on first use."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """Return the sentinel's name."""
        return "NOT_GIVEN"

    def __bool__(self) -> Literal[False]:
        """Return ``False``, so a missing value reads as absent.

        Typed as ``Literal[False]`` rather than ``bool`` so type checkers can
        narrow ``value or fallback`` to the fallback's type.
        """
        return False


NOT_GIVEN: NotGiven = NotGiven()
"""Singleton sentinel meaning "this value was not provided"."""


_T = TypeVar("_T")


def is_given(value: _T | NotGiven) -> TypeGuard[_T]:
    """Check whether a value was explicitly provided.

    Also acts as a type guard: inside a true branch, the value is narrowed to
    exclude :class:`NotGiven` (e.g. ``str | None | NotGiven`` becomes
    ``str | None``)::

        if is_given(delta.voice):
            # caller wants to change the voice
            ...

    Args:
        value: The value to check.

    Returns:
        ``True`` if *value* is anything other than ``NOT_GIVEN``.
    """
    return not isinstance(value, NotGiven)


def assert_given(value: _T | NotGiven) -> _T:
    """Extract a value that must have been provided.

    Intended for reads where ``NOT_GIVEN`` should never appear, such as a
    store-mode settings object (see :mod:`pipecat.services.settings`). Narrows
    away :class:`NotGiven` at the type level and raises at runtime if that
    invariant is violated::

        resolved_model = assert_given(self._settings.model)  # narrowed str | None

    Args:
        value: The value to extract.

    Returns:
        The value, narrowed to exclude :class:`NotGiven`.

    Raises:
        RuntimeError: If *value* is ``NOT_GIVEN``.
    """
    if not is_given(value):
        raise RuntimeError("Expected a value, got NOT_GIVEN")
    return value
