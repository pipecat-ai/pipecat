#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LiveKit frame serializer for Pipecat."""

import ctypes
import pickle

from loguru import logger

from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType

try:
    from livekit.rtc import AudioFrame
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use LiveKit, you need to `pip install pipecat-ai[livekit]`.")
    raise Exception(f"Missing module: {e}")


class LivekitFrameSerializer(FrameSerializer):
    """Serializer for converting between Pipecat frames and LiveKit audio frames.

    This serializer handles the conversion of Pipecat's OutputAudioRawFrame objects
    to LiveKit AudioFrame objects for transmission, and the reverse conversion
    for received audio data.
    """

    @property
    def type(self) -> FrameSerializerType:
        """Get the serializer type.

        Returns:
            The serializer type indicating binary serialization.
        """
        return FrameSerializerType.BINARY

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize a Pipecat frame to LiveKit AudioFrame format.

        Args:
            frame: The Pipecat frame to serialize. Only OutputAudioRawFrame
                  instances are supported.

        Returns:
            Pickled LiveKit AudioFrame bytes if frame is OutputAudioRawFrame,
            None otherwise.
        """
        if not isinstance(frame, OutputAudioRawFrame):
            return None
        audio_frame = AudioFrame(
            data=frame.audio,
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=len(frame.audio) // ctypes.sizeof(ctypes.c_int16),
        )
        return pickle.dumps(audio_frame)

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize LiveKit AudioFrame data to a Pipecat frame.

        Args:
            data: Pickled data containing a LiveKit AudioFrame.

        Returns:
            InputAudioRawFrame containing the deserialized audio data,
            or None if deserialization fails.
        """
        audio_frame: AudioFrame = pickle.loads(data)["frame"]
        return InputAudioRawFrame(
            audio=bytes(audio_frame.data),
            sample_rate=audio_frame.sample_rate,
            num_channels=audio_frame.num_channels,
        )
