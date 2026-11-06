#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame serialization interfaces for Pipecat."""

from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, model_validator

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.base_object import BaseObject
from pipecat.utils.deprecation import warn_deprecated


class FrameSerializer(BaseObject):
    """Abstract base class for frame serialization implementations.

    Defines the interface for converting frames to/from serialized formats
    for transmission or storage. Subclasses must implement the core
    serialize/deserialize methods.

    A serializer whose wire format has a fixed sample rate states it in
    ``audio_in_sample_rate`` and ``audio_out_sample_rate``, and converts audio
    with :meth:`_resample_audio_in` and :meth:`_resample_audio_out`: incoming
    audio to the pipeline's input rate, outgoing audio to the wire's rate.
    Outgoing audio has already been paced by the output transport, and the
    stream resampler keeps that pace.
    """

    class InputParams(BaseModel):
        """Base configuration parameters for FrameSerializer.

        Parameters:
            ignore_rtvi_messages: Whether to ignore RTVI protocol messages during serialization.
                Defaults to True to prevent RTVI messages from being sent to external transports.
            audio_in_sample_rate: Sample rate of the audio received on the wire, which is
                converted to the pipeline's input rate. None when the wire format carries
                the rate with the audio.
            audio_out_sample_rate: Sample rate of the audio sent on the wire, which audio is
                converted to before it is encoded. None when the wire format carries the
                rate with the audio.
            resampler_clear_after_secs: Has no effect: the serializer's resamplers clear
                their history after their default inactivity.

                .. deprecated:: 1.16.0
                    No replacement. Will be removed in 2.0.0.
        """

        ignore_rtvi_messages: bool = True
        audio_in_sample_rate: int | None = None
        audio_out_sample_rate: int | None = None
        resampler_clear_after_secs: float | None = 0.2

        @model_validator(mode="before")
        @classmethod
        def _warn_resampler_clear_after_secs(cls, data: Any) -> Any:
            if isinstance(data, dict) and "resampler_clear_after_secs" in data:
                warn_deprecated(
                    "`FrameSerializer.InputParams.resampler_clear_after_secs` is deprecated "
                    "since 1.16.0 and will be removed in 2.0.0. No replacement.",
                    stacklevel=4,
                )
            return data

    def __init__(self, params: InputParams | None = None, **kwargs):
        """Initialize the FrameSerializer.

        Args:
            params: Configuration parameters.
            **kwargs: Additional arguments passed to BaseObject (e.g., name).
        """
        super().__init__(**kwargs)
        self._params = params or FrameSerializer.InputParams()
        # The pipeline's input rate, which incoming audio is converted to. Set
        # in setup().
        self._sample_rate = 0
        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._output_resampler_rate = 0

    @property
    def audio_in_sample_rate(self) -> int | None:
        """Sample rate of the audio received on the wire, or None if the wire format carries it."""
        return self._params.audio_in_sample_rate

    @property
    def audio_out_sample_rate(self) -> int | None:
        """Sample rate of the audio sent on the wire, or None if the wire format carries it."""
        return self._params.audio_out_sample_rate

    def should_ignore_frame(self, frame: Frame) -> bool:
        """Check if a frame should be ignored during serialization.

        This method filters out RTVI protocol messages when ignore_rtvi_messages is enabled.
        Subclasses can override this to add additional filtering logic.

        Args:
            frame: The frame to check.

        Returns:
            True if the frame should be ignored, False otherwise.
        """
        if (
            self._params.ignore_rtvi_messages
            and isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame))
            and isinstance(frame.message, dict)
            and frame.message.get("label") == RTVI.MESSAGE_LABEL
        ):
            return True
        return False

    async def setup(self, setup: FrameProcessorSetup):
        """Initialize the serializer with pipeline configuration.

        Args:
            setup: Configuration object containing setup parameters.
        """
        self._sample_rate = setup.audio_in_sample_rate

    @abstractmethod
    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Convert a frame to its serialized representation.

        Args:
            frame: The frame to serialize.

        Returns:
            Serialized frame data as string or bytes, or None when there is
            nothing to send for this frame: it isn't handled, or the serializer
            is buffering it and will emit it on a later call.
        """
        pass

    @abstractmethod
    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Convert serialized data back to a frame object.

        Args:
            data: Serialized frame data as string or bytes.

        Returns:
            Reconstructed Frame object, or None if deserialization fails.
        """
        pass

    async def _resample_audio_in(self, audio: bytes) -> bytes:
        """Convert audio received at `audio_in_sample_rate` to the pipeline's input rate."""
        rate = self.audio_in_sample_rate
        if rate is None:
            return audio
        return await self._input_resampler.resample(audio, rate, self._sample_rate)

    async def _resample_audio_out(self, frame: AudioRawFrame) -> bytes:
        """Convert a frame's audio to `audio_out_sample_rate`."""
        rate = self.audio_out_sample_rate
        if rate is None:
            return frame.audio
        # A stream resampler converts from one rate, so audio arriving at
        # another one starts a new resampler.
        if self._output_resampler_rate and frame.sample_rate != self._output_resampler_rate:
            self._output_resampler = create_stream_resampler()
        self._output_resampler_rate = frame.sample_rate
        return await self._output_resampler.resample(frame.audio, frame.sample_rate, rate)

    async def _reset_audio_out(self):
        """Discard the outgoing audio still in the resampler, such as after an interruption."""
        await self._output_resampler.reset()
