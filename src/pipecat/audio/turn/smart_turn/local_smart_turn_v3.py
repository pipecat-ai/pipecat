#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local PyTorch turn analyzer for on-device ML inference using the smart-turn-v3 model.

This module provides a smart turn analyzer that uses an ONNX model for
local end-of-turn detection without requiring network connectivity.
"""

from typing import Any, Dict

import numpy as np
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn

try:
    from transformers import WhisperFeatureExtractor
    import onnxruntime as ort
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use LocalSmartTurnAnalyzerV3, you need to `pip install pipecat-ai[local-smart-turn-v3]`."
    )
    raise Exception(f"Missing module: {e}")


class LocalSmartTurnAnalyzerV3(BaseSmartTurn):
    """Local turn analyzer using the smart-turn-v2 PyTorch model.

    Provides end-of-turn detection using locally-stored PyTorch models,
    enabling offline operation without network dependencies. Uses
    Wav2Vec2 architecture for audio sequence classification.
    """

    def __init__(self, *, smart_turn_model_path: str, **kwargs):
        """Initialize the local PyTorch smart-turn-v3 analyzer.

        Args:
            smart_turn_model_path: Path to the ONNX model file.
            **kwargs: Additional arguments passed to BaseSmartTurn.
        """
        super().__init__(**kwargs)

        if not smart_turn_model_path:
            raise ValueError("smart_turn_model_path must be provided")

        logger.debug("Loading Local Smart Turn v3 model...")

        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        self._session = ort.InferenceSession(smart_turn_model_path)

        logger.debug("Loaded Local Smart Turn v3")

    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Predict end-of-turn using local ONNX model."""

        def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
            """Truncate audio to last n seconds or pad with zeros to meet n seconds."""
            max_samples = n_seconds * sample_rate
            if len(audio_array) > max_samples:
                return audio_array[-max_samples:]
            elif len(audio_array) < max_samples:
                # Pad with zeros at the beginning
                padding = max_samples - len(audio_array)
                return np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
            return audio_array

        # Truncate to 8 seconds (keeping the end) or pad to 8 seconds
        audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

        # Process audio using Whisper's feature extractor
        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )

        # Convert to numpy and ensure correct shape for ONNX
        input_features = inputs.input_features.squeeze(0).numpy().astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension

        # Run ONNX inference
        outputs = self._session.run(None, {"input_features": input_features})

        # Extract probability (ONNX model returns sigmoid probabilities)
        probability = outputs[0][0].item()

        # Make prediction (1 for Complete, 0 for Incomplete)
        prediction = 1 if probability > 0.5 else 0

        return {
            "prediction": prediction,
            "probability": probability,
        }
