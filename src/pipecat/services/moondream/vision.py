#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Moondream vision service implementation.

This module provides integration with the Moondream vision-language model
for image analysis and description generation.
"""

import asyncio
import importlib
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
from pipecat.utils.types import assert_given

try:
    import torch
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Moondream, you need to `uv add "pipecat-ai[moondream]"`.')
    raise ImportError(f"Missing module(s): {e}") from e


def detect_device():
    """Detect the appropriate device to run on.

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
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


def load_model(model_path: str, revision: str | None, device: torch.device, dtype: torch.dtype):
    """Load a Moondream model from the Hugging Face Hub.

    Transformers builds models on the meta device and relies on two hooks that
    Moondream's remote model class does not provide: calling ``post_init()``
    from the constructor, and recomputing non-persistent buffers in
    ``_init_weights()``. The remote class is
    subclassed here to provide both; without them, loading fails on an
    accelerator and the attention mask and rotary embedding buffers are left
    uninitialized.

    Args:
        model_path: Hugging Face model identifier or local path.
        revision: Specific model revision to use.
        device: Device to load the model on.
        dtype: Data type to load the model weights as.

    Returns:
        The loaded model, in evaluation mode.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, revision=revision)
    model_class = get_class_from_dynamic_module(
        config.auto_map["AutoModelForCausalLM"], model_path, revision=revision
    )
    rope = importlib.import_module(f"{model_class.__module__.rpartition('.')[0]}.rope")

    class Transformers5Moondream(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.post_init()

        def _init_weights(self, module):
            super()._init_weights(module)
            model = self.model
            if module is model:
                max_context = model.config.text.max_context
                patch_w = model.config.vision.crop_size // model.config.vision.enc_patch_size
                prefix_attn_len = 1 + patch_w**2
                attn_mask = torch.tril(torch.ones(max_context, max_context, dtype=torch.bool))
                attn_mask[:prefix_attn_len, :prefix_attn_len] = True
                module.attn_mask.copy_(attn_mask)
            elif module is model.text:
                text_config = model.config.text
                freqs_cis = rope.precompute_freqs_cis(
                    text_config.dim // (2 * text_config.n_heads), text_config.max_context
                )
                module.freqs_cis.copy_(freqs_cis)

    return Transformers5Moondream.from_pretrained(
        model_path,
        config=config,
        revision=revision,
        device_map={"": device},
        dtype=dtype,
    ).eval()


@dataclass
class MoondreamSettings(VisionSettings):
    """Settings for the Moondream vision service.

    Parameters:
        model: Moondream model identifier.
    """


class MoondreamService(VisionService):
    """Moondream vision-language model service.

    Provides image analysis and description generation using the Moondream
    vision-language model. Supports various hardware acceleration options
    including CUDA, MPS, and Intel XPU.
    """

    Settings = MoondreamSettings
    _settings: Settings

    def __init__(
        self,
        *,
        model: str | None = None,
        revision="2025-06-21",
        use_cpu=False,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Moondream service.

        Args:
            model: Hugging Face model identifier for the Moondream model.

                .. deprecated:: 0.0.105
                    Use ``settings=MoondreamService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            revision: Specific model revision to use.
            use_cpu: Whether to force CPU usage instead of hardware acceleration.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to the parent VisionService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="vikhyatk/moondream2")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)

        if not use_cpu:
            device, dtype = detect_device()
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        logger.debug("Loading Moondream model...")

        model_path = assert_given(self._settings.model)
        if model_path is None:
            raise ValueError("Moondream model must be specified")
        self._model = load_model(model_path, revision, device, dtype)

        logger.debug("Loaded Moondream model")

    async def run_vision(self, frame: UserImageRawFrame) -> AsyncGenerator[Frame, None]:
        """Analyze an image and generate a description.

        Args:
            frame: The image frame to process.
        """
        if not self._model:
            yield ErrorFrame("Moondream model not available")
            return

        logger.debug(f"Analyzing image (bytes length: {len(frame.image)})")

        def get_image_description(image_bytes: bytes, text: str | None) -> str:
            if frame.format is None:
                raise ValueError("Cannot decode image bytes without a format")
            image = Image.frombytes(frame.format, frame.size, image_bytes)
            # `encode_image` and `query` are custom methods provided by the
            # moondream2 model code (via `trust_remote_code=True`) that pyright
            # can't see on `AutoModelForCausalLM`'s base type.
            image_embeds = self._model.encode_image(image)  # pyright: ignore[reportCallIssue]
            description = self._model.query(image_embeds, text)["answer"]  # pyright: ignore[reportCallIssue]
            return description

        description = await asyncio.to_thread(get_image_description, frame.image, frame.text)

        yield VisionFullResponseStartFrame()
        yield VisionTextFrame(text=description)
        yield VisionFullResponseEndFrame()
