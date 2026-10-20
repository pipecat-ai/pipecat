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
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.utils.base_object import BaseObject


class VideoInSourceParams(BaseModel):
    """How a transport captures one video source from the user.

    Parameters:
        framerate: Frames per second to pass on from this source.
        on_request_only: Pass on frames only to answer a ``UserImageRequestFrame``,
            rather than continuously at ``framerate``.
    """

    framerate: int = Field(default=30, ge=1)
    on_request_only: bool = False

    @field_validator("framerate", mode="before")
    @classmethod
    def _check_framerate(cls, framerate: Any) -> Any:
        if framerate == 0:
            raise ValueError(
                "a framerate of 0 isn't a rate; use on_request_only=True to pass on "
                "frames only to answer image requests"
            )
        return framerate

    @model_validator(mode="after")
    def _check_on_request_only(self) -> "VideoInSourceParams":
        if self.on_request_only and "framerate" in self.model_fields_set:
            raise ValueError("framerate doesn't apply when on_request_only=True")
        return self


class TransportParams(BaseModel):
    """Configuration parameters for transport implementations.

    Parameters:
        audio_out_enabled: Enable audio output streaming.
        audio_out_sample_rate: Output audio sample rate in Hz.
        audio_out_channels: Number of output audio channels.
        audio_out_bitrate: Output audio bitrate in bits per second.
        audio_out_10ms_chunks: Number of 10ms chunks to buffer for output.
        audio_out_filter: Audio filter to apply to output audio of the default destination,
            before the mixer. Audio the transport generates itself (mixer audio, end silence,
            DTMF tones) is not filtered.
        audio_out_volume: Volume of the output audio, as a multiplier on its samples, applied
            after the filter and before the mixer. A ``VolumeFrame`` changes it while the
            transport runs, and a ``VolumeGainFrame`` adjusts it for some audio.
        audio_out_mixer: Audio mixer instance or destination mapping.
        audio_out_destinations: List of audio output destination identifiers.
        audio_out_end_silence_secs: How much silence to send after an EndFrame (0 for no silence).
        audio_out_auto_silence: Insert silence frames when the audio output queue is empty.
            When False, the transport will wait for audio data instead of inserting silence.
        audio_out_write_timeout_secs: How long a single write to the transport may take
            before the peer is considered gone. A client that stops reading leaves the
            write waiting with nothing to fail, so it would otherwise never return.
        audio_in_enabled: Enable audio input streaming.
        audio_in_sample_rate: Input audio sample rate in Hz.
        audio_in_channels: Number of input audio channels.
        audio_in_filter: Audio filter to apply to input audio.
        audio_in_stream_on_start: Start audio streaming immediately on transport start.
        audio_in_passthrough: Pass through input audio frames downstream.
        video_in_enabled: Enable video input streaming.
        video_in_sources: Video sources to capture from each user as they connect,
            keyed by source (``"camera"``, ``"screenVideo"``, or a transport-specific
            custom source). Sources not listed here are captured only when the
            application asks for them, e.g. with the transport's
            ``capture_participant_video()``. Requires ``video_in_enabled``.
            Supported by ``DailyTransport``, ``LiveKitTransport`` and
            ``SmallWebRTCTransport``; other transports log a warning and ignore
            it, so check the logs if video doesn't arrive as configured.
        video_out_enabled: Enable video output streaming.
        video_out_is_live: Enable real-time video output streaming.
        video_out_width: Video output width in pixels.
        video_out_height: Video output height in pixels.
        video_out_bitrate: Video output bitrate in bits per second.

            .. deprecated:: 1.1.0
                Use provider-specific settings instead (e.g.,
                ``DailyParams.camera_out_send_settings``).
                Will be removed in 2.0.0.

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
    audio_out_filter: BaseAudioFilter | None = None
    audio_out_volume: float = 1.0
    audio_out_mixer: BaseAudioMixer | Mapping[str | None, BaseAudioMixer] | None = None
    audio_out_destinations: list[str] = Field(default_factory=list)
    audio_out_end_silence_secs: int = 2
    audio_out_auto_silence: bool = True
    audio_out_write_timeout_secs: float = 10.0
    audio_in_enabled: bool = False
    audio_in_sample_rate: int | None = None
    audio_in_channels: int = 1
    audio_in_filter: BaseAudioFilter | None = None
    audio_in_stream_on_start: bool = True
    audio_in_passthrough: bool = True
    video_in_enabled: bool = False
    video_in_sources: dict[str, VideoInSourceParams] = Field(default_factory=dict)
    video_out_enabled: bool = False
    video_out_is_live: bool = False
    video_out_width: int = 1024
    video_out_height: int = 768
    video_out_bitrate: int | None = None
    video_out_framerate: int = 30
    video_out_color_format: str = "RGB"
    video_out_codec: str | None = None
    video_out_destinations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_video_in_sources(self) -> "TransportParams":
        if self.video_in_sources and not self.video_in_enabled:
            raise ValueError("video_in_sources requires video_in_enabled=True")
        return self


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

    async def capture_participant_video(
        self,
        participant_id: str,
        framerate: int | None = 30,
        video_source: str = "camera",
        *,
        on_request_only: bool = False,
    ):
        """Capture a participant's video source at a framerate.

        This captures a source on demand, where ``TransportParams.video_in_sources``
        captures the listed sources as each user connects, and takes precedence
        over it for the source. Transports that receive user video implement this;
        the others log a warning.

        Args:
            participant_id: The participant to capture, as from :meth:`get_client_id`.
            framerate: Frames per second to pass on, or ``None`` for every frame. It
                doesn't apply with ``on_request_only``.

                .. deprecated:: 1.13.0
                    Use ``on_request_only=True`` instead of a ``framerate`` of ``0``.
                    Will be removed in 2.0.0.

            video_source: The video source, e.g. ``"camera"`` or ``"screenVideo"``.
            on_request_only: Pass on only the frames that answer image requests.
        """
        logger.warning(f"{self}: capturing participant video isn't supported.")

    def get_client_id(self, client: Any) -> str:
        """The id of a client, as passed to ``on_client_connected``.

        This is the ``participant_id`` that :meth:`capture_participant_video` and
        image requests take. Transports with client events implement this; the
        others log a warning and return an empty string.

        Args:
            client: The client, as passed to the transport's client events.

        Returns:
            The client's id.
        """
        logger.warning(f"{self}: getting a client id isn't supported.")
        return ""
