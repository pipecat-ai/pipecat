#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local models, loaded once a process and shared by every pipeline that asks for them."""

import asyncio
import threading
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

_M = TypeVar("_M")

_models: dict[Hashable, Any] = {}
_loading = threading.Lock()


async def load_shared_model(key: Hashable, load: Callable[[], _M]) -> _M:
    """Load a model off the event loop, or share the one already loaded for ``key``.

    Every caller asking for the same ``key`` gets the same model, loaded once a
    process. A model that can't be loaded isn't kept, so the next call tries
    again.

    Args:
        key: What identifies the model and its settings.
        load: Loads the model, downloading its files the first time.

    Returns:
        The model.

    Raises:
        Exception: Whatever ``load`` raised.
    """
    return await asyncio.to_thread(_shared_model, key, load)


def _shared_model(key: Hashable, load: Callable[[], _M]) -> _M:
    """The model loaded for ``key``, loading it first if no one has."""
    with _loading:
        if key not in _models:
            _models[key] = load()
        return _models[key]
