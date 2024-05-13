#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import numpy as np

from pipecat.frames.frames import AudioRawFrame, Frame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.vad.vad_analyzer import VADAnalyzer, VADState

from loguru import logger

try:
    import torch
    # We don't use torchaudio here, but we need to try importing it because
    # Silero uses it.
    import torchaudio

    torch.set_num_threads(1)

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Silero VAD, you need to `pip install pipecat-ai[silero]`.")
    raise Exception(f"Missing module(s): {e}")


# Provided by Alexander Veysov
def int2float(sound):
    try:
        abs_max = np.abs(sound).max()
        sound = sound.astype("float32")
        if abs_max > 0:
            sound *= 1 / 32768
        sound = sound.squeeze()  # depends on the use case
        return sound
    except ValueError:
        return sound


class SileroVAD(FrameProcessor, VADAnalyzer):

    def __init__(self, sample_rate=16000, audio_passthrough=False):
        FrameProcessor.__init__(self)
        VADAnalyzer.__init__(self, sample_rate=sample_rate, num_channels=1)

        logger.debug("Loading Silero VAD model...")

        (self._model, self._utils) = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )

        self._processor_vad_state: VADState = VADState.QUIET
        self._audio_passthrough = audio_passthrough

        logger.debug("Loaded Silero VAD")

    #
    # VADAnalyzer
    #

    def num_frames_required(self) -> int:
        return int(self.sample_rate / 100) * 4  # 40ms

    def voice_confidence(self, buffer) -> float:
        try:
            audio_int16 = np.frombuffer(buffer, np.int16)
            audio_float32 = int2float(audio_int16)
            new_confidence = self._model(torch.from_numpy(audio_float32), self.sample_rate).item()
            return new_confidence
        except BaseException as e:
            # This comes from an empty audio array
            logger.error(f"Error analyzing audio with Silero VAD: {e}")
            return 0

    #
    # FrameProcessor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, AudioRawFrame):
            await self._analyze_audio(frame)
            if self._audio_passthrough:
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _analyze_audio(self, frame: AudioRawFrame):
        # Check VAD and push event if necessary. We just care about changes
        # from QUIET to SPEAKING and vice versa.
        new_vad_state = self.analyze_audio(frame.audio)
        if new_vad_state != self._processor_vad_state and new_vad_state != VADState.STARTING and new_vad_state != VADState.STOPPING:
            new_frame = None

            if new_vad_state == VADState.SPEAKING:
                new_frame = UserStartedSpeakingFrame()
            elif new_vad_state == VADState.QUIET:
                new_frame = UserStoppedSpeakingFrame()

            if new_frame:
                await self.push_frame(new_frame)
                self._processor_vad_state = new_vad_state
