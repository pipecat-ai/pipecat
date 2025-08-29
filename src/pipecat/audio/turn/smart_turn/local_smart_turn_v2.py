#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local PyTorch turn analyzer for on-device ML inference using the smart-turn-v2 model.

This module provides a smart turn analyzer that uses PyTorch models for
local end-of-turn detection without requiring network connectivity.
"""

from typing import Any, Dict

import numpy as np
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
        Wav2Vec2Processor,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use LocalSmartTurnAnalyzerV2, you need to `pip install pipecat-ai[local-smart-turn]`."
    )
    raise Exception(f"Missing module: {e}")


class LocalSmartTurnAnalyzerV2(BaseSmartTurn):
    """Local turn analyzer using the smart-turn-v2 PyTorch model.

    Provides end-of-turn detection using locally-stored PyTorch models,
    enabling offline operation without network dependencies. Uses
    Wav2Vec2 architecture for audio sequence classification.
    """

    def __init__(self, *, smart_turn_model_path: str, **kwargs):
        """Initialize the local PyTorch smart-turn-v2 analyzer.

        Args:
            smart_turn_model_path: Path to directory containing the PyTorch model
                and feature extractor files. If empty, uses default HuggingFace model.
            **kwargs: Additional arguments passed to BaseSmartTurn.
        """
        super().__init__(**kwargs)

        if not smart_turn_model_path:
            # Define the path to the pretrained model on Hugging Face
            smart_turn_model_path = "pipecat-ai/smart-turn-v2"

        logger.debug("Loading Local Smart Turn v2 model...")
        # Load the pretrained model for sequence classification
        self._turn_model = _Wav2Vec2ForEndpointing.from_pretrained(smart_turn_model_path)
        # Load the corresponding feature extractor for preprocessing audio
        self._turn_processor = Wav2Vec2Processor.from_pretrained(smart_turn_model_path)
        # Use platform-optimized backend if available (MPS for Apple silicon, CUDA for NVIDIA)
        self._device = "cpu"
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        # Move model to selected device and set it to evaluation mode
        self._turn_model = self._turn_model.to(self._device)
        self._turn_model.eval()
        logger.debug("Loaded Local Smart Turn v2")

    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Predict end-of-turn using local PyTorch model."""
        inputs = self._turn_processor(
            audio_array,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            max_length=16000 * 16,  # 16 seconds at 16kHz
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._turn_model(**inputs)

            # The model returns sigmoid probabilities directly in the logits field
            probability = outputs["logits"][0].item()

            # Make prediction (1 for Complete, 0 for Incomplete)
            prediction = 1 if probability > 0.5 else 0

        return {
            "prediction": prediction,
            "probability": probability,
        }


class _Wav2Vec2ForEndpointing(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)

        self.pool_attention = nn.Sequential(
            nn.Linear(config.hidden_size, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

        for module in self.pool_attention:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

    def attention_pool(self, hidden_states, attention_mask):
        # Calculate attention weights
        attention_weights = self.pool_attention(hidden_states)

        if attention_mask is None:
            raise ValueError("attention_mask must be provided for attention pooling")

        attention_weights = attention_weights + (
            (1.0 - attention_mask.unsqueeze(-1).to(attention_weights.dtype)) * -1e9
        )

        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention to hidden states
        weighted_sum = torch.sum(hidden_states * attention_weights, dim=1)

        return weighted_sum

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]

        # Create transformer padding mask
        if attention_mask is not None:
            input_length = attention_mask.size(1)
            hidden_length = hidden_states.size(1)
            ratio = input_length / hidden_length
            indices = (torch.arange(hidden_length, device=attention_mask.device) * ratio).long()
            attention_mask = attention_mask[:, indices]
            attention_mask = attention_mask.bool()
        else:
            attention_mask = None

        pooled = self.attention_pool(hidden_states, attention_mask)

        logits = self.classifier(pooled)

        if torch.isnan(logits).any():
            raise ValueError("NaN values detected in logits")

        if labels is not None:
            # Calculate positive sample weight based on batch statistics
            pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(min=0.1, max=10.0)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1), labels.view(-1))

            # Add L2 regularization for classifier layers
            l2_lambda = 0.01
            l2_reg = torch.tensor(0.0, device=logits.device)
            for param in self.classifier.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            probs = torch.sigmoid(logits.detach())
            return {"loss": loss, "logits": probs}

        probs = torch.sigmoid(logits)
        return {"logits": probs}
