#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local PyTorch smart turn analyzer for on-device ML inference.

This module provides a smart turn analyzer that uses PyTorch models for
local end-of-turn detection without requiring network connectivity.
"""

from typing import Any, Dict

import numpy as np
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn

try:
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2BertForSequenceClassification
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the LocalSmartTurnAnalyzer, you need to `pip install pipecat-ai[local-smart-turn]`."
    )
    raise Exception(f"Missing module: {e}")


class LocalSmartTurnAnalyzer(BaseSmartTurn):
    """Local smart turn analyzer using PyTorch models.

    Provides end-of-turn detection using locally-stored PyTorch models,
    enabling offline operation without network dependencies. Uses
    Wav2Vec2-BERT architecture for audio sequence classification.
    """

    def __init__(self, *, smart_turn_model_path: str, **kwargs):
        """Initialize the local PyTorch smart turn analyzer.

        Args:
            smart_turn_model_path: Path to directory containing the PyTorch model
                and feature extractor files. If empty, uses default HuggingFace model.
            **kwargs: Additional arguments passed to BaseSmartTurn.
        """
        super().__init__(**kwargs)

        if not smart_turn_model_path:
            # Define the path to the pretrained model on Hugging Face
            smart_turn_model_path = "pipecat-ai/smart-turn"

        logger.debug("Loading Local Smart Turn model...")
        # Load the pretrained model for sequence classification
        self._turn_model = Wav2Vec2BertForSequenceClassification.from_pretrained(
            smart_turn_model_path
        )
        # Load the corresponding feature extractor for preprocessing audio
        self._turn_processor = AutoFeatureExtractor.from_pretrained(smart_turn_model_path)
        # Set device to GPU if available, else CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move model to selected device and set it to evaluation mode
        self._turn_model = self._turn_model.to(self._device)
        self._turn_model.eval()
        logger.debug("Loaded Local Smart Turn")

    def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Predict end-of-turn using local PyTorch model."""
        inputs = self._turn_processor(
            audio_array,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            max_length=800,  # Maximum length as specified in training
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Move input tensors to the same device as the model
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = self._turn_model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            completion_prob = probabilities[0, 1].item()  # Probability of class 1 (Complete)
            prediction = 1 if completion_prob > 0.5 else 0

        return {
            "prediction": prediction,
            "probability": completion_prob,
        }
