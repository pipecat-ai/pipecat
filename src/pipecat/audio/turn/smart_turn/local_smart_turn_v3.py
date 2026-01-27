#
# Copyright (c) 2024-2026, Daily
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
from pipecat.utils.env import env_truthy

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

    def __init__(
        self, *, smart_turn_model_path: Optional[str] = None, cpu_count: int = 1, **kwargs
    ):
        """Initialize the local ONNX smart-turn-v3 analyzer.

        Args:
            smart_turn_model_path: Path to the ONNX model file. If this is not
                set, the bundled smart-turn-v3.2-cpu model will be used.
            cpu_count: The number of CPUs to use for inference. Defaults to 1.
            **kwargs: Additional arguments passed to BaseSmartTurn.
        """
        super().__init__(**kwargs)

        self._log_data = env_truthy("PIPECAT_SMART_TURN_LOG_DATA", default=False)

        if not smart_turn_model_path:
            # Load bundled model
            model_name = "smart-turn-v3.2-cpu.onnx"
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

        logger.debug(f"Loading Local Smart Turn v3.x model from {smart_turn_model_path}...")

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = cpu_count
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        self._session = ort.InferenceSession(smart_turn_model_path, sess_options=so)

        logger.debug("Loaded Local Smart Turn v3.x")

    def _write_audio_to_wav(
        self, audio_array: np.ndarray, sample_rate: int = 16000, suffix: str = ""
    ) -> None:
        """Write audio data to a WAV file in a background thread.

        Args:
            audio_array: The audio data as a numpy array (float32, normalized to [-1, 1]).
            sample_rate: The sample rate of the audio data.
            suffix: Optional suffix to append to the filename (e.g., "_raw", "_padded").
        """
        import os
        import threading
        import wave
        from datetime import datetime

        # Generate filename with current timestamp (millisecond precision)
        timestamp = datetime.now().strftime("%Y-%m-%d__%H:%M:%S.%f")[:-3]
        log_dir = "./smart_turn_audio_log"
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir, f"{timestamp}{suffix}.wav")

        # Make a copy of the audio data to avoid issues with the array being modified
        audio_copy = audio_array.copy()

        def write_wav():
            try:
                # Convert float32 audio to int16 for WAV file
                audio_int16 = (audio_copy * 32767).astype(np.int16)

                with wave.open(filename, "wb") as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 2 bytes for int16
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())

                logger.debug(f"Wrote audio to {filename}")
            except Exception as e:
                logger.error(f"Failed to write audio to {filename}: {e}")

        # Start background thread to write the WAV file
        thread = threading.Thread(target=write_wav, daemon=True)
        thread.start()

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

        audio_for_logging = audio_array

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

        if self._log_data:
            suffix = "_complete" if prediction == 1 else "_incomplete"
            self._write_audio_to_wav(audio_for_logging, sample_rate=16000, suffix=suffix)

        return {
            "prediction": prediction,
            "probability": probability,
        }
