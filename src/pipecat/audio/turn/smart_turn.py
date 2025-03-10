#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import numpy as np
import torch
from loguru import logger
from transformers import AutoFeatureExtractor, Wav2Vec2BertForSequenceClassification

from pipecat.audio.turn.base_turn_analyzer import BaseEndOfTurnAnalyzer, EndOfTurnState

# MODEL_PATH = "model-v1"
MODEL_PATH = "pipecat-ai/smart-turn"


class SmartTurnAnalyzer(BaseEndOfTurnAnalyzer):
    def __init__(self):
        super().__init__()
        self._audio_buffer = bytearray()

        logger.debug("Loading Smart Turn model...")

        # Load model and processor
        model = Wav2Vec2BertForSequenceClassification.from_pretrained(MODEL_PATH)
        self._processor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)

        # Set model to evaluation mode and move to GPU if available
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._model.eval()

        logger.debug("Loaded Smart Turn")

    def analyze_audio(self, buffer: bytes) -> EndOfTurnState:
        self._audio_buffer += buffer
        if len(self._audio_buffer) < 16000 * 2 * 6:
            return EndOfTurnState.INCOMPLETE

        audio_int16 = np.frombuffer(self._audio_buffer, dtype=np.int16)

        # Divide by 32768 because we have signed 16-bit data.
        audio_float32 = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0
        print(audio_float32)

        # Process audio
        inputs = self._processor(
            audio_float32,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            max_length=800,  # Maximum length as specified in training
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

            # Get probabilities using softmax
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            completion_prob = probabilities[0, 1].item()  # Probability of class 1 (Complete)

            # Make prediction (1 for Complete, 0 for Incomplete)
            prediction = 1 if completion_prob > 0.5 else 0

        state = EndOfTurnState.COMPLETE if prediction == 1 else EndOfTurnState.INCOMPLETE

        if state == EndOfTurnState.COMPLETE:
            self._audio_buffer = bytearray()
        else:
            self._audio_buffer = self._audio_buffer[len(buffer) :]

        print("AAAAAAAAAAAA", state)

        return state
