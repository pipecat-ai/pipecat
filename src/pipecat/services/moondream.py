#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from pipecat.frames.frames import TextFrame, VisionImageRawFrame
from pipecat.services.ai_services import VisionService

from PIL import Image

from loguru import logger

try:
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Moondream, you need to `pip install pipecat-ai[moondream]`.")
    raise Exception(f"Missing module(s): {e}")


def detect_device():
    """
    Detects the appropriate device to run on, and return the device and dtype.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


class MoondreamService(VisionService):
    def __init__(
        self,
        model_id="vikhyatk/moondream2",
        revision="2024-04-02",
        use_cpu=False
    ):
        super().__init__()

        if not use_cpu:
            device, dtype = detect_device()
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        logger.debug("Loading Moondream model...")

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        ).to(device=device, dtype=dtype)
        self._model.eval()

        logger.debug("Loaded Moondream model")

    async def run_vision(self, frame: VisionImageRawFrame):
        if not self._model:
            logger.error("Moondream model not available")
            return

        logger.debug(f"Analyzing image: {frame}")

        def get_image_description(frame: VisionImageRawFrame):
            image = Image.frombytes(frame.format, (frame.size[0], frame.size[1]), frame.image)
            image_embeds = self._model.encode_image(image)
            description = self._model.answer_question(
                image_embeds=image_embeds,
                question=frame.text,
                tokenizer=self._tokenizer)
            return description

        description = await asyncio.to_thread(get_image_description, frame)

        await self.push_frame(TextFrame(text=description))
