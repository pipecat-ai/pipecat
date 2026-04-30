#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base transport classes for Pipecat.

This module provides the foundation for transport implementations including
parameter configuration and abstract base classes for input/output transport
functionality.
"""

from abc import abstractmethod
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.utils.base_object import BaseObject


class TransportParams(BaseModel):
    """Configuration parameters for transport implementations.

    Parameters:
        audio_out_enabled: Enable audio output streaming.
        audio_out_sample_rate: Output audio sample rate in Hz.
        audio_out_channels: Number of output audio channels.
        audio_out_bitrate: Output audio bitrate in bits per second.
        audio_out_10ms_chunks: Number of 10ms chunks to buffer for output.
        audio_out_mixer: Audio mixer instance or destination mapping.
        audio_out_destinations: List of audio output destination identifiers.
        audio_out_end_silence_secs: How much silence to send after an EndFrame (0 for no silence).
        audio_out_auto_silence: Insert silence frames when the audio output queue is empty.
            When False, the transport will wait for audio data instead of inserting silence.
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
        video_out_bitrate: [DEPRECATED] Video output bitrate in bits per second.
        video_out_framerate: Video output frame rate in FPS.
        video_out_color_format: Video output color format string.
        video_out_codec: Preferred video codec for output (e.g., 'VP8', 'H264', 'H265').
        video_out_destinations: List of video output destination identifiers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_out_enabled: bool = False
    audio_out_sample_rate: int | None = None
    audio_out_channels: int = 1
    audio_out_bitrate: int = 96000
    audio_out_10ms_chunks: int = 4
    audio_out_mixer: BaseAudioMixer | Mapping[str | None, BaseAudioMixer] | None = None
    audio_out_destinations: list[str] = Field(default_factory=list)
    audio_out_end_silence_secs: int = 2
    audio_out_auto_silence: bool = True
    audio_in_enabled: bool = False
    audio_in_sample_rate: int | None = None
    audio_in_channels: int = 1
    audio_in_filter: BaseAudioFilter | None = None
    audio_in_stream_on_start: bool = True
    audio_in_passthrough: bool = True
    video_in_enabled: bool = False
    video_out_enabled: bool = False
    video_out_is_live: bool = False
    video_out_width: int = 1024
    video_out_height: int = 768
    video_out_bitrate: int | None = None
    video_out_framerate: int = 30
    video_out_color_format: str = "RGB"
    video_out_codec: str | None = None
    video_out_destinations: list[str] = Field(default_factory=list)


class BaseTransport(BaseObject):
    """Base class for transport implementations.

    Provides the foundation for transport classes that handle media streaming,
    including input and output frame processors for audio and video data.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
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
