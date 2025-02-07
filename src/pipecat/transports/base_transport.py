#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.utils.utils import obj_count, obj_id


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
    audio_out_is_live: bool = False
    audio_out_sample_rate: Optional[int] = None
    audio_out_channels: int = 1
    audio_out_bitrate: int = 96000
    audio_out_mixer: Optional[BaseAudioMixer] = None
    audio_in_enabled: bool = False
    audio_in_sample_rate: Optional[int] = None
    audio_in_channels: int = 1
    audio_in_filter: Optional[BaseAudioFilter] = None
    vad_enabled: bool = False
    vad_audio_passthrough: bool = False
    vad_analyzer: Optional[VADAnalyzer] = None


class BaseTransport(ABC):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        self._id: int = obj_id()
        self._name = name or f"{self.__class__.__name__}#{obj_count(self)}"
        self._input_name = input_name
        self._output_name = output_name
        self._event_handlers: dict = {}

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def input(self) -> FrameProcessor:
        raise NotImplementedError

    @abstractmethod
    def output(self) -> FrameProcessor:
        raise NotImplementedError

    def event_handler(self, event_name: str):
        def decorator(handler):
            self.add_event_handler(event_name, handler)
            return handler

        return decorator

    def add_event_handler(self, event_name: str, handler):
        if event_name not in self._event_handlers:
            raise Exception(f"Event handler {event_name} not registered")
        self._event_handlers[event_name].append(handler)

    def _register_event_handler(self, event_name: str):
        if event_name in self._event_handlers:
            raise Exception(f"Event handler {event_name} already registered")
        self._event_handlers[event_name] = []

    async def _call_event_handler(self, event_name: str, *args, **kwargs):
        try:
            for handler in self._event_handlers[event_name]:
                if inspect.iscoroutinefunction(handler):
                    await handler(self, *args, **kwargs)
                else:
                    handler(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in event handler {event_name}: {e}")

    def __str__(self):
        return self.name
