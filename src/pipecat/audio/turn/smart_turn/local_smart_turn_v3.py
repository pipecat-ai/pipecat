#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local turn analyzer for on-device ML inference using the smart-turn-v3 model.

This module provides a smart turn analyzer that uses an ONNX model for
local end-of-turn detection without requiring network connectivity.
"""

from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn

try:
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use LocalSmartTurnAnalyzerV3, you need to `pip install pipecat-ai[local-smart-turn-v3]`."
    )
    raise Exception(f"Missing module: {e}")


class LocalSmartTurnAnalyzerV3(BaseSmartTurn):
    """Local turn analyzer using the smart-turn-v3 ONNX model.

    Provides end-of-turn detection using locally-stored ONNX model,
    enabling offline operation without network dependencies.
    """

    def __init__(self, *, smart_turn_model_path: Optional[str] = None, **kwargs):
        """Initialize the local ONNX smart-turn-v3 analyzer.

        Args:
            smart_turn_model_path: Path to the ONNX model file. If this is not
                set, the bundled smart-turn-v3.0 model will be used.
            **kwargs: Additional arguments passed to BaseSmartTurn.
        """
        super().__init__(**kwargs)

        logger.debug("Loading Local Smart Turn v3 model...")

        if not smart_turn_model_path:
            # Load bundled model
            model_name = "smart-turn-v3.0.onnx"
            package_path = "pipecat.audio.turn.smart_turn.data"

            try:
                import importlib_resources as impresources

                smart_turn_model_path = str(impresources.files(package_path).joinpath(model_name))
            except BaseException:
                from importlib import resources as impresources

                try:
                    with impresources.path(package_path, model_name) as f:
                        smart_turn_model_path = f
                except BaseException:
                    smart_turn_model_path = str(
                        impresources.files(package_path).joinpath(model_name)
                    )

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        self._session = ort.InferenceSession(smart_turn_model_path, sess_options=so)

        logger.debug("Loaded Local Smart Turn v3")

    def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Predict end-of-turn using local ONNX model."""

        def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
            """Truncate audio to last n seconds or pad with zeros to meet n seconds."""
            max_samples = n_seconds * sample_rate
            if len(audio_array) > max_samples:
                return audio_array[-max_samples:]
            elif len(audio_array) < max_samples:
                # Pad with zeros at the beginning
                padding = max_samples - len(audio_array)
                return np.pad(audio_array, (padding, 0), mode="constant", constant_values=0)
            return audio_array

        # Truncate to 8 seconds (keeping the end) or pad to 8 seconds
        audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

        # Process audio using Whisper's feature extractor
        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )

        # Extract features and ensure correct shape for ONNX
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
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
