#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Moondream vision service implementation.

This module provides integration with the Moondream vision-language model
for image analysis and description generation.
"""

import asyncio
from typing import AsyncGenerator

from loguru import logger
from PIL import Image

from pipecat.frames.frames import ErrorFrame, Frame, TextFrame, VisionImageRawFrame
from pipecat.services.vision_service import VisionService

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Moondream, you need to `pip install pipecat-ai[moondream]`.")
    raise Exception(f"Missing module(s): {e}")


def detect_device():
    """Detect the appropriate device to run on.

    Detects available hardware acceleration and selects the best device
    and data type for optimal performance.

    Returns:
        tuple: A tuple containing (device, dtype) where device is a torch.device
               and dtype is the recommended torch data type for that device.
    """
    try:
        import intel_extension_for_pytorch

        if torch.xpu.is_available():
            return torch.device("xpu"), torch.float32
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


class MoondreamService(VisionService):
    """Moondream vision-language model service.

    Provides image analysis and description generation using the Moondream
    vision-language model. Supports various hardware acceleration options
    including CUDA, MPS, and Intel XPU.
    """

    def __init__(
        self, *, model="vikhyatk/moondream2", revision="2025-01-09", use_cpu=False, **kwargs
    ):
        """Initialize the Moondream service.

        Args:
            model: Hugging Face model identifier for the Moondream model.
            revision: Specific model revision to use.
            use_cpu: Whether to force CPU usage instead of hardware acceleration.
            **kwargs: Additional arguments passed to the parent VisionService.
        """
        super().__init__(**kwargs)

        self.set_model_name(model)

        if not use_cpu:
            device, dtype = detect_device()
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        logger.debug("Loading Moondream model...")

        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            revision=revision,
            device_map={"": device},
            torch_dtype=dtype,
        ).eval()

        logger.debug("Loaded Moondream model")

    async def run_vision(self, frame: VisionImageRawFrame) -> AsyncGenerator[Frame, None]:
        """Analyze an image and generate a description.

        Args:
            frame: Vision frame containing the image data and optional question text.

        Yields:
            Frame: TextFrame containing the generated image description, or ErrorFrame
                  if analysis fails.
        """
        if not self._model:
            logger.error(f"{self} error: Moondream model not available ({self.model_name})")
            yield ErrorFrame("Moondream model not available")
            return

        logger.debug(f"Analyzing image: {frame}")

        def get_image_description(frame: VisionImageRawFrame):
            """Generate description for the given image frame.

            Args:
                frame: Vision frame containing image data and question.

            Returns:
                str: Generated description of the image.
            """
            image = Image.frombytes(frame.format, frame.size, frame.image)
            image_embeds = self._model.encode_image(image)
            description = self._model.query(image_embeds, frame.text)["answer"]
            return description

        description = await asyncio.to_thread(get_image_description, frame)

        yield TextFrame(text=description)
