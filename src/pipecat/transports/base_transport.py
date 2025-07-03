#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base transport classes for Pipecat.

This module provides the foundation for transport implementations including
parameter configuration and abstract base classes for input/output transport
functionality.
"""

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
    """Configuration parameters for transport implementations.

    Parameters:
        camera_in_enabled: Enable camera input (deprecated, use video_in_enabled).

            .. deprecated:: 0.0.66
               The `camera_in_enabled` parameter is deprecated, use
               `video_in_enabled` instead.

        camera_out_enabled: Enable camera output (deprecated, use video_out_enabled).

            .. deprecated:: 0.0.66
               The `camera_out_enabled` parameter is deprecated, use
               `video_out_enabled` instead.

        camera_out_is_live: Enable real-time camera output (deprecated).

            .. deprecated:: 0.0.66
               The `camera_out_is_live` parameter is deprecated, use
               `video_out_is_live` instead.

        camera_out_width: Camera output width in pixels (deprecated).

            .. deprecated:: 0.0.66
               The `camera_out_width` parameter is deprecated, use
               `video_out_width` instead.

        camera_out_height: Camera output height in pixels (deprecated).

            .. deprecated:: 0.0.66
                The `camera_out_height` parameter is deprecated, use
                `video_out_height` instead.

        camera_out_bitrate: Camera output bitrate in bits per second (deprecated).

            .. deprecated:: 0.0.66
                The `camera_out_bitrate` parameter is deprecated, use
                `video_out_bitrate` instead.

        camera_out_framerate: Camera output frame rate in FPS (deprecated).

            .. deprecated:: 0.0.66
                The `camera_out_framerate` parameter is deprecated, use
                `video_out_framerate` instead.

        camera_out_color_format: Camera output color format string (deprecated).

            .. deprecated:: 0.0.66
                The `camera_out_color_format` parameter is deprecated, use
                `video_out_color_format` instead.

        audio_out_enabled: Enable audio output streaming.
        audio_out_sample_rate: Output audio sample rate in Hz.
        audio_out_channels: Number of output audio channels.
        audio_out_bitrate: Output audio bitrate in bits per second.
        audio_out_10ms_chunks: Number of 10ms chunks to buffer for output.
        audio_out_mixer: Audio mixer instance or destination mapping.
        audio_out_destinations: List of audio output destination identifiers.
        audio_in_enabled: Enable audio input streaming.
        audio_in_sample_rate: Input audio sample rate in Hz.
        audio_in_channels: Number of input audio channels.
        audio_in_filter: Audio filter to apply to input audio.
        audio_in_stream_on_start: Start audio streaming immediately on transport start.
        audio_in_passthrough: Pass through input audio frames downstream.
        video_in_enabled: Enable video input streaming.
        video_out_enabled: Enable video output streaming.
        video_out_is_live: Enable real-time video output streaming.
        video_out_width: Video output width in pixels.
        video_out_height: Video output height in pixels.
        video_out_bitrate: Video output bitrate in bits per second.
        video_out_framerate: Video output frame rate in FPS.
        video_out_color_format: Video output color format string.
        video_out_destinations: List of video output destination identifiers.
        vad_enabled: Enable Voice Activity Detection (deprecated).

            .. deprecated:: 0.0.66
               The `vad_enabled` parameter is deprecated, use `audio_in_enabled`
               and `TransportParams.vad_analyzer` instead.

        vad_audio_passthrough: Enable VAD audio passthrough (deprecated).

            .. deprecated:: 0.0.66
                The `vad_audio_passthrough` parameter is deprecated, use `audio_in_passthrough`
                instead.

        vad_analyzer: Voice Activity Detection analyzer instance.
        turn_analyzer: Turn-taking analyzer instance for conversation management.
    """

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
    """Base class for transport implementations.

    Provides the foundation for transport classes that handle media streaming,
    including input and output frame processors for audio and video data.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the base transport.

        Args:
            name: Optional name for the transport instance.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(name=name)
        self._input_name = input_name
        self._output_name = output_name

    @abstractmethod
    def input(self) -> FrameProcessor:
        """Get the input frame processor for this transport.

        Returns:
            The frame processor that handles incoming frames.
        """
        pass

    @abstractmethod
    def output(self) -> FrameProcessor:
        """Get the output frame processor for this transport.

        Returns:
            The frame processor that handles outgoing frames.
        """
        pass
