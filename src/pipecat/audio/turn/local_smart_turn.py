#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os
from typing import Dict

import numpy as np
import torch
from loguru import logger

from pipecat.audio.turn.base_smart_turn import BaseSmartTurn

try:
    import coremltools as ct
    from transformers import AutoFeatureExtractor
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the LocalSmartTurnAnalyzer, you need to `pip install pipecat-ai[local-smart-turn]`."
    )
    raise Exception(f"Missing module: {e}")


class LocalCoreMLSmartTurnAnalyzer(BaseSmartTurn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

    def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, any]:
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
