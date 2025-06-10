#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import abstractmethod
from typing import List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.utils.base_object import BaseObject


class TransportParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_in_enabled: bool = False
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
    audio_out_10ms_chunks: int = 4
    audio_out_mixer: Optional[BaseAudioMixer | Mapping[Optional[str], BaseAudioMixer]] = None
    audio_out_destinations: List[str] = Field(default_factory=list)
    audio_in_enabled: bool = False
    audio_in_sample_rate: Optional[int] = None
    audio_in_channels: int = 1
    audio_in_filter: Optional[BaseAudioFilter] = None
    audio_in_stream_on_start: bool = True
    audio_in_passthrough: bool = True
    video_in_enabled: bool = False
    video_out_enabled: bool = False
    video_out_is_live: bool = False
    video_out_width: int = 1024
    video_out_height: int = 768
    video_out_bitrate: int = 800000
    video_out_framerate: int = 30
    video_out_color_format: str = "RGB"
    video_out_destinations: List[str] = Field(default_factory=list)
    vad_enabled: bool = False
    vad_audio_passthrough: bool = False
    vad_analyzer: Optional[VADAnalyzer] = None
    turn_analyzer: Optional[BaseTurnAnalyzer] = None


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
