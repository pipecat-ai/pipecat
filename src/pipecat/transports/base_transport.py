#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import abstractmethod
from typing import Optional

from pydantic import BaseModel, ConfigDict

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.utils.base_object import BaseObject


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
    audio_out_sample_rate: Optional[int] = None
    audio_out_channels: int = 1
    audio_out_bitrate: int = 96000
    audio_out_mixer: Optional[BaseAudioMixer] = None
    audio_in_enabled: bool = False
    audio_in_sample_rate: Optional[int] = None
    audio_in_channels: int = 1
    audio_in_filter: Optional[BaseAudioFilter] = None
    audio_in_stream_on_start: bool = True
    vad_enabled: bool = False
    vad_audio_passthrough: bool = False
    vad_analyzer: Optional[VADAnalyzer] = None


class BaseTransport(BaseObject):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._input_name = input_name
        self._output_name = output_name

    @abstractmethod
    def input(self) -> FrameProcessor:
        pass

    @abstractmethod
    def output(self) -> FrameProcessor:
        pass
