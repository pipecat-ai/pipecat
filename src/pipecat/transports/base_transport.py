#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import inspect
import types

from abc import ABC, abstractmethod
from functools import partial

from pydantic import ConfigDict
from pydantic.main import BaseModel

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.vad.vad_analyzer import VADAnalyzer

from loguru import logger


class TransportParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_out_enabled: bool = False
    camera_out_is_live: bool = False
    camera_out_width: int = 1024
    camera_out_height: int = 768
    camera_out_bitrate: int = 800000
    camera_out_framerate: int = 30
    camera_out_color_format: str = "RGB"
    audio_out_enabled: bool = False
    audio_out_sample_rate: int = 16000
    audio_out_channels: int = 1
    audio_in_enabled: bool = False
    audio_in_sample_rate: int = 16000
    audio_in_channels: int = 1
    vad_enabled: bool = False
    vad_audio_passthrough: bool = False
    vad_analyzer: VADAnalyzer | None = None


class BaseTransport(ABC):

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._event_handlers: dict = {}

    @abstractmethod
    def input(self) -> FrameProcessor:
        raise NotImplementedError

    @abstractmethod
    def output(self) -> FrameProcessor:
        raise NotImplementedError

    def event_handler(self, event_name: str):
        def decorator(handler):
            self._add_event_handler(event_name, handler)
            return handler
        return decorator

    def _register_event_handler(self, event_name: str):
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        if event_name not in [method[0] for method in methods]:
            raise Exception(f"Event handler {event_name} not found")

        self._event_handlers[event_name] = [getattr(self, event_name)]

        patch_method = types.MethodType(partial(self._patch_method, event_name), self)
        setattr(self, event_name, patch_method)

    def _add_event_handler(self, event_name: str, handler):
        if event_name not in self._event_handlers:
            raise Exception(f"Event handler {event_name} not registered")
        self._event_handlers[event_name].append(types.MethodType(handler, self))

    def _patch_method(self, event_name, *args, **kwargs):
        try:
            for handler in self._event_handlers[event_name]:
                if inspect.iscoroutinefunction(handler):
                    # Beware, if handler() calls another event handler it
                    # will deadlock. You shouldn't do that anyways.
                    future = asyncio.run_coroutine_threadsafe(
                        handler(*args[1:], **kwargs), self._loop)

                    # wait for the coroutine to finish. This will also
                    # raise any exceptions raised by the coroutine.
                    future.result()
                else:
                    handler(*args[1:], **kwargs)
        except Exception as e:
            logger.error(f"Exception in event handler {event_name}: {e}")
            raise e
