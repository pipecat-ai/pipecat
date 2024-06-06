#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time

import numpy as np

from pipecat.frames.frames import AudioRawFrame, Frame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.vad.vad_analyzer import VADAnalyzer, VADParams, VADState

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

# How often should we reset internal model state
_MODEL_RESET_STATES_TIME = 5.0


class SileroVADAnalyzer(VADAnalyzer):

    def __init__(self, sample_rate=16000, params: VADParams = VADParams()):
        super().__init__(sample_rate=sample_rate, num_channels=1, params=params)

        logger.debug("Loading Silero VAD model...")

        (self._model, utils) = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )

        self._last_reset_time = 0

        logger.debug("Loaded Silero VAD")

    #
    # VADAnalyzer
    #

    def num_frames_required(self) -> int:
        return int(self.sample_rate / 100) * 4  # 40ms

    def voice_confidence(self, buffer) -> float:
        try:
            audio_int16 = np.frombuffer(buffer, np.int16)
            # Divide by 32768 because we have signed 16-bit data.
            audio_float32 = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0
            new_confidence = self._model(torch.from_numpy(audio_float32), self.sample_rate).item()

            # We need to reset the model from time to time because it doesn't
            # really need all the data and memory will keep growing otherwise.
            curr_time = time.time()
            diff_time = curr_time - self._last_reset_time
            if diff_time >= _MODEL_RESET_STATES_TIME:
                self._model.reset_states()
                self._last_reset_time = curr_time

            return new_confidence
        except BaseException as e:
            # This comes from an empty audio array
            logger.error(f"Error analyzing audio with Silero VAD: {e}")
            return 0


class SileroVAD(FrameProcessor):

    def __init__(
            self,
            sample_rate: int = 16000,
            vad_params: VADParams = VADParams(),
            audio_passthrough: bool = False):
        super().__init__()

        self._vad_analyzer = SileroVADAnalyzer(sample_rate=sample_rate, params=vad_params)
        self._audio_passthrough = audio_passthrough

        self._processor_vad_state: VADState = VADState.QUIET

    #
    # FrameProcessor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            await self._analyze_audio(frame)
            if self._audio_passthrough:
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _analyze_audio(self, frame: AudioRawFrame):
        # Check VAD and push event if necessary. We just care about changes
        # from QUIET to SPEAKING and vice versa.
        new_vad_state = self._vad_analyzer.analyze_audio(frame.audio)
        if new_vad_state != self._processor_vad_state and new_vad_state != VADState.STARTING and new_vad_state != VADState.STOPPING:
            new_frame = None

            if new_vad_state == VADState.SPEAKING:
                new_frame = UserStartedSpeakingFrame()
            elif new_vad_state == VADState.QUIET:
                new_frame = UserStoppedSpeakingFrame()

            if new_frame:
                await self.push_frame(new_frame)
                self._processor_vad_state = new_vad_state
