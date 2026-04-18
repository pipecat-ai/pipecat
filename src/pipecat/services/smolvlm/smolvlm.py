"""SmolVLM vision service implementation.

This module provides integration with the SmolVLM vision-language model
for image analysis and description generation.
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from loguru import logger
from PIL import Image

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    UserImageRawFrame,
    VisionFullResponseEndFrame,
    VisionFullResponseStartFrame,
    VisionTextFrame,
)
from pipecat.services.settings import VisionSettings
from pipecat.services.vision_service import VisionService

try:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use SmolVLM, you need to `pip install pipecat-ai[smolvlm]`."
    )
    raise Exception(f"Missing module(s): {e}")


def detect_device() -> tuple["torch.device", "torch.dtype"]:
    """Detect the appropriate device and dtype to run on.

    Detects available hardware acceleration and selects the best device
    and data type for optimal performance.

    Returns:
        tuple: A tuple containing (device, dtype) where device is a torch.device
               and dtype is the recommended torch data type for that device.
    """
    try:
        import intel_extension_for_pytorch  # noqa: F401

        if torch.xpu.is_available():
            return torch.device("xpu"), torch.float32
    except ImportError:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    else:
        return torch.device("cpu"), torch.float32


@dataclass
class SmolVlmSettings(VisionSettings):
    """Settings for the SmolVLM vision service.

    Attributes:
        model: SmolVLM model identifier from HuggingFace.
        max_new_tokens: Maximum number of tokens to generate.
    """

    model: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    max_new_tokens: int = 500


class SmolVlmService(VisionService):
    """SmolVLM vision-language model service.

    Provides image analysis and description generation using the SmolVLM
    vision-language model. Supports various hardware acceleration options
    including CUDA, MPS (Apple Silicon), Intel XPU, and CPU fallback.

    Example usage::

        service = SmolVlmService()

        # With custom settings
        service = SmolVlmService(
            settings=SmolVlmService.Settings(
                model="HuggingFaceTB/SmolVLM-500M-Instruct",
                max_new_tokens=256,
            )
        )
    """

    Settings = SmolVlmSettings
    _settings : Settings


    def __init__(
        self,
        *,
        model: str | None = None,
        use_cpu: bool = False,
        settings: SmolVlmSettings | None = None,
        **kwargs,
    ):
        """Initialize the SmolVLM service.

        Args:
            model: HuggingFace model identifier for the SmolVLM model.

                .. deprecated:: 0.0.105
                    Use ``settings=SmolVlmService.Settings(model=...)`` instead.

            use_cpu: Whether to force CPU usage instead of hardware acceleration.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to the parent VisionService.
        """
        default_settings = SmolVlmSettings()

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)

        if not use_cpu:
            self._device, dtype = detect_device()
        else:
            self._device = torch.device("cpu")
            dtype = torch.float32

        logger.debug(f"SmolVLM using device: {self._device} with dtype: {dtype}")

        logger.debug("Loading SmolVLM processor...")
        self._processor = AutoProcessor.from_pretrained(self._settings.model)
        logger.debug("SmolVLM processor loaded")

        logger.debug(f"Loading SmolVLM model: {self._settings.model}...")
        self._model = AutoModelForVision2Seq.from_pretrained(
            self._settings.model,
            torch_dtype=dtype,

            _attn_implementation="flash_attention_2"
            if self._device.type == "cuda"
            else "eager",
        ).to(self._device)
        logger.debug("SmolVLM model loaded")

    def _build_prompt(self, text: str | None) -> str:
        """Build the prompt from frame text or return the default.

        Args:
            text: Optional prompt text from the image frame.

        Returns:
            The prompt string to use for inference.
        """
        return text.strip() if text and text.strip() else "Describe the given image."

    async def run_vision(self, frame: UserImageRawFrame) -> AsyncGenerator[Frame, None]:
        """Analyze an image and generate a description.

        Args:
            frame: The image frame to process, optionally containing
                   a prompt in frame.text.

        Yields:
            VisionFullResponseStartFrame, TextFrame with description,
            VisionFullResponseEndFrame, or ErrorFrame on failure.
        """
        if not self._processor:
            yield ErrorFrame("SmolVLM processor not available")
            return

        if not self._model:
            yield ErrorFrame("SmolVLM model not available")
            return

        logger.debug(f"Analyzing image (bytes length: {len(frame.image)})")

        def get_image_description(image_bytes: bytes, text: str | None) -> str:
            image = Image.frombytes(frame.format, frame.size, image_bytes)

            prompt_text = self._build_prompt(text)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]

            prompt = self._processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self._processor(text=prompt, images=[image], return_tensors="pt")
            inputs = inputs.to(self._device)

            generated_ids = self._model.generate(
                **inputs, max_new_tokens=self._settings.max_new_tokens
            )
            generated_texts = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

            raw_output = generated_texts[0]

            if "Assistant:" in raw_output:
                description = raw_output.split("Assistant:")[1].strip()
            else:
                description = raw_output.strip()

            return description

        description = await asyncio.to_thread(
            get_image_description, frame.image, frame.text
        )

        yield VisionFullResponseStartFrame()
        yield VisionTextFrame(text=description)
        yield VisionFullResponseEndFrame()