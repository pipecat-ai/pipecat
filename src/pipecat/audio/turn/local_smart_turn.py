#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import numpy as np
from loguru import logger

from pipecat.audio.turn.base_turn_analyzer import BaseEndOfTurnAnalyzer, EndOfTurnState


class LocalSmartTurnAnalyzer(BaseEndOfTurnAnalyzer):
    def __init__(self):
        super().__init__()
        self._audio_buffer = bytearray()

        logger.debug("Loading Local Smart Turn model...")

        # TODO: implement it

        logger.debug("Loaded Local Smart Turn")

    def analyze_audio(self, buffer: bytes) -> EndOfTurnState:
        self._audio_buffer += buffer

        # TODO: we probably don't need this
        # Checking if we have at least 6 seconds of audio
        # if len(self._audio_buffer) < 16000 * 2 * 6:
        #    return EndOfTurnState.INCOMPLETE

        audio_int16 = np.frombuffer(self._audio_buffer, dtype=np.int16)

        # Divide by 32768 because we have signed 16-bit data.
        audio_float32 = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0

        # TODO: implement to use the smart turn
        # for now it is always returning as complete only for testing it
        prediction = 1

        state = EndOfTurnState.COMPLETE if prediction == 1 else EndOfTurnState.INCOMPLETE

        if state == EndOfTurnState.COMPLETE:
            # clears the buffer completely
            self._audio_buffer = bytearray()
        else:
            # TODO: implement it
            pass

        return state
