#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os
import time

import numpy as np
import torch
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


# TODO: we should convert all this to params
STOP_MS = 1000
PRE_SPEECH_MS = 200
MAX_DURATION_SECONDS = 8  # Maximum duration for the smart turn model


class LocalSmartTurnAnalyzer(BaseEndOfTurnAnalyzer):
    def __init__(self):
        super().__init__()
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

        logger.debug("Loading Local Smart Turn model...")
        # Only load the processor, not the torch model
        self._turn_processor = AutoFeatureExtractor.from_pretrained(smart_turn_model_path)
        self._turn_model = ct.models.MLModel(core_ml_model_path)
        logger.debug("Loaded Local Smart Turn")

        self._audio_buffer = []
        self._speech_triggered = False
        self._silence_frames = 0
        self._speech_start_time = None

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        audio_int16 = np.frombuffer(buffer, dtype=np.int16)
        # Divide by 32768 because we have signed 16-bit data.
        audio_float32 = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0

        state = EndOfTurnState.INCOMPLETE
        if is_speech:
            self._silence_frames = 0
            self._speech_triggered = True
            if self._speech_start_time is None:
                self._speech_start_time = time.time()
            self._audio_buffer.append((time.time(), audio_float32))
        else:
            if self._speech_triggered:
                self._audio_buffer.append((time.time(), audio_float32))
                self._silence_frames += 1
                if self._silence_frames * self._chunk_size_ms >= STOP_MS:
                    logger.debug("End of Turn complete due to STOP_MS.")
                    state = EndOfTurnState.COMPLETE
                    self._clear()
            else:
                # Keep buffering some silence before potential speech starts
                self._audio_buffer.append((time.time(), audio_float32))
                # Keep the buffer size reasonable, assuming CHUNK is small
                max_buffer_time = (
                    PRE_SPEECH_MS + STOP_MS
                ) / 1000 + MAX_DURATION_SECONDS  # Some extra buffer
                while (
                    self._audio_buffer and self._audio_buffer[0][0] < time.time() - max_buffer_time
                ):
                    self._audio_buffer.pop(0)

        return state

    def analyze_end_of_turn(self) -> EndOfTurnState:
        logger.debug("Analyzing End of Turn...")
        state = self._process_speech_segment(self._audio_buffer)
        if state == EndOfTurnState.COMPLETE:
            self._clear()

        logger.debug(f"End of Turn result: {state}")
        return state

    def _clear(self):
        self._speech_triggered = False
        self._audio_buffer = []
        self._speech_start_time = None
        self._silence_frames = 0

    def _process_speech_segment(self, audio_buffer) -> EndOfTurnState:
        state = EndOfTurnState.INCOMPLETE

        if not audio_buffer:
            return state

        # Find start and end indices for the segment
        start_time = self._speech_start_time - (PRE_SPEECH_MS / 1000)
        start_index = 0
        for i, (t, _) in enumerate(audio_buffer):
            if t >= start_time:
                start_index = i
                break

        end_index = len(audio_buffer) - 1

        # Extract the audio segment
        segment_audio_chunks = [chunk for _, chunk in audio_buffer[start_index : end_index + 1]]
        segment_audio = np.concatenate(segment_audio_chunks)

        # Remove (STOP_MS - 200)ms from the end of the segment
        samples_to_remove = int((STOP_MS - 200) / 1000 * self.sample_rate)
        segment_audio = segment_audio[:-samples_to_remove]

        # Limit maximum duration
        if len(segment_audio) / self.sample_rate > MAX_DURATION_SECONDS:
            segment_audio = segment_audio[: int(MAX_DURATION_SECONDS * self.sample_rate)]

        # No resampling needed as both recording and prediction use 16000 Hz
        segment_audio_resampled = segment_audio

        if len(segment_audio_resampled) > 0:
            # Call the new predict_endpoint function with the audio data
            start_time = time.perf_counter()

            result = self._predict_endpoint(segment_audio_resampled)

            state = (
                EndOfTurnState.COMPLETE if result["prediction"] == 1 else EndOfTurnState.INCOMPLETE
            )

            end_time = time.perf_counter()

            logger.debug("--------")
            logger.debug(f"Prediction: {'Complete' if result['prediction'] == 1 else 'Incomplete'}")
            logger.debug(f"Probability of complete: {result['probability']:.4f}")
            logger.debug(f"Prediction took {(end_time - start_time) * 1000:.2f}ms seconds")
        else:
            logger.debug("Captured empty audio segment, skipping prediction.")

        return state

    def _predict_endpoint(self, audio_array):
        """
        Predict whether an audio segment is complete (turn ended) or incomplete.

        Args:
            audio_array: Numpy array containing audio samples at 16kHz

        Returns:
            Dictionary containing prediction results:
            - prediction: 1 for complete, 0 for incomplete
            - probability: Probability of completion class
        """

        inputs = self._turn_processor(
            audio_array,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            max_length=800,  # Maximum length as specified in training
            return_attention_mask=True,
            return_tensors="pt",
        )

        output = self._turn_model.predict(dict(inputs))
        logits = output["logits"]  # Core ML returns numpy array
        logits_tensor = torch.tensor(logits)
        probabilities = torch.nn.functional.softmax(logits_tensor, dim=1)
        completion_prob = probabilities[0, 1].item()  # Probability of class 1 (Complete)
        prediction = 1 if completion_prob > 0.5 else 0

        return {
            "prediction": prediction,
            "probability": completion_prob,
        }
