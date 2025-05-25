#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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
    """Detects the appropriate device to run on, and return the device and dtype."""
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
    def __init__(
        self, *, model="vikhyatk/moondream2", revision="2024-08-26", use_cpu=False, **kwargs
    ):
        super().__init__(**kwargs)

        self.set_model_name(model)

        if not use_cpu:
            device, dtype = detect_device()
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)

        logger.debug("Loading Moondream model...")

        self._model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, revision=revision
        ).to(device=device, dtype=dtype)
        self._model.eval()

        logger.debug("Loaded Moondream model")

    async def run_vision(self, frame: VisionImageRawFrame) -> AsyncGenerator[Frame, None]:
        if not self._model:
            logger.error(f"{self} error: Moondream model not available ({self.model_name})")
            yield ErrorFrame("Moondream model not available")
            return

        logger.debug(f"Analyzing image: {frame}")

        def get_image_description(frame: VisionImageRawFrame):
            image = Image.frombytes(frame.format, frame.size, frame.image)
            image_embeds = self._model.encode_image(image)
            description = self._model.answer_question(
                image_embeds=image_embeds, question=frame.text, tokenizer=self._tokenizer
            )
            return description

        description = await asyncio.to_thread(get_image_description, frame)

        yield TextFrame(text=description)
