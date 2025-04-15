#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os

import numpy as np
from loguru import logger

from pipecat.audio.turn.base_turn_analyzer import BaseEndOfTurnAnalyzer, EndOfTurnState

try:
    import coremltools as ct
    from transformers import AutoFeatureExtractor
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the LocalSmartTurnAnalyzer, you need to `pip install pipecat-ai[local-smart-turn]`."
    )
    raise Exception(f"Missing module: {e}")


class LocalSmartTurnAnalyzer(BaseEndOfTurnAnalyzer):
    def __init__(self):
        super().__init__()
        self._audio_buffer = bytearray()

        logger.debug("Loading Local Smart Turn model...")

        # To use this locally, set the environment variable LOCAL_SMART_TURN_MODEL_PATH
        # to the path where the smart-turn repo is cloned.
        #
        # Example setup:
        #
        #   # Git LFS (Large File Storage)
        #   brew install git-lfs
        #   # Hugging Face uses LFS to store large model files, including .mlpackage
        #   git lfs install
        #   # Clone the repo with the smart_turn_classifier.mlpackage
        #   git clone https://huggingface.co/pipecat-ai/smart-turn
        #
        # Then set the env variable:
        #   export LOCAL_SMART_TURN_MODEL_PATH=./smart-turn
        # or add it to your .env file
        smart_turn_model_path = os.getenv("LOCAL_SMART_TURN_MODEL_PATH")

        if not smart_turn_model_path:
            logger.error("LOCAL_SMART_TURN_MODEL_PATH is not set.")
            raise Exception("LOCAL_SMART_TURN_MODEL_PATH environment variable must be provided.")

        core_ml_model_path = f"{smart_turn_model_path}/coreml/smart_turn_classifier.mlpackage"

        # Only load the processor, not the torch model
        processor = AutoFeatureExtractor.from_pretrained(smart_turn_model_path)
        model = ct.models.MLModel(core_ml_model_path)

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
