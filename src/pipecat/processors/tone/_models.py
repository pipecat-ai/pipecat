#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Tone's models, loaded once a process and shared by every call."""

import asyncio
import threading
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

from pipecat.processors.frame_processor import FrameProcessor

M = TypeVar("M")

_models: dict[Hashable, Any] = {}
_loading = threading.Lock()


def _shared(key: Hashable, load: Callable[[], M]) -> M:
    with _loading:
        if key not in _models:
            _models[key] = load()
        return _models[key]


async def load_model(processor: FrameProcessor, key: Hashable, load: Callable[[], M]) -> M | None:
    """Load a model off the event loop, or share the one already loaded for ``key``.

    A model that can't be loaded is reported as an error that leaves the
    processor unusable, and the next call tries again.

    Args:
        processor: The processor the model is for.
        key: What identifies the model and its settings.
        load: Loads the model, downloading its files the first time.

    Returns:
        The model, or None when it can't be loaded.
    """
    try:
        return await asyncio.to_thread(_shared, key, load)
    except Exception as e:
        await processor.push_error(
            f"{processor}: couldn't load its model: {e}", e, force_treat_as_permanent=True
        )
        return None
