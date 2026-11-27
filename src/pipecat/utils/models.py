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
# One lock per key, so loading one model never holds up another.
_key_locks: dict[Hashable, threading.Lock] = {}
_key_locks_lock = threading.Lock()


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
    return await asyncio.to_thread(get_shared_model, key, load)


def get_shared_model(key: Hashable, load: Callable[[], _M]) -> _M:
    """Load a model in the calling thread, or share the one already loaded for ``key``.

    The blocking form of :func:`load_shared_model`, for models that must be
    ready as soon as their owner is constructed. The first call for a key
    blocks while ``load`` runs, so keep it to models that load quickly from
    local files; later calls return the loaded model without locking.

    Args:
        key: What identifies the model and its settings.
        load: Loads the model.

    Returns:
        The model.

    Raises:
        Exception: Whatever ``load`` raised.
    """
    model = _models.get(key)
    if model is not None:
        return model
    with _key_locks_lock:
        key_lock = _key_locks.setdefault(key, threading.Lock())
    with key_lock:
        if key not in _models:
            _models[key] = load()
        return _models[key]
