#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import ctypes
import pickle

from pipecat.frames.frames import AudioRawFrame, Frame
from pipecat.serializers.base_serializer import FrameSerializer

from loguru import logger

try:
    from livekit.rtc import AudioFrame
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use LiveKit, you need to `pip install pipecat-ai[livekit]`.")
    raise Exception(f"Missing module: {e}")


class LivekitFrameSerializer(FrameSerializer):
    SERIALIZABLE_TYPES = {
        AudioRawFrame: "audio",
    }

    def serialize(self, frame: Frame) -> str | bytes | None:
        if not isinstance(frame, AudioRawFrame):
            return None
        audio_frame = AudioFrame(
            data=frame.audio,
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=len(frame.audio) // ctypes.sizeof(ctypes.c_int16),
        )
        return pickle.dumps(audio_frame)

    def deserialize(self, data: str | bytes) -> Frame | None:
        audio_frame: AudioFrame = pickle.loads(data)['frame']
        return AudioRawFrame(
            audio=bytes(audio_frame.data),
            sample_rate=audio_frame.sample_rate,
            num_channels=audio_frame.num_channels,
        )
