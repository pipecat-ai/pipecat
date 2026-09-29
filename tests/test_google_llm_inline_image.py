#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for inline image output in GoogleLLMService."""

import io
from unittest.mock import patch

import pytest
from google.genai.types import Blob, Candidate, Content, GenerateContentResponse, Part
from PIL import Image

from pipecat.frames.frames import AssistantImageRawFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService


def _png_bytes(mode: str, size: tuple[int, int] = (8, 4)) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, "red").save(buffer, "PNG")
    return buffer.getvalue()


async def _image_frames(mode: str) -> list[AssistantImageRawFrame]:
    service = GoogleLLMService(api_key="test-key")
    frames = []

    async def capture_frame(frame, direction=None):
        frames.append(frame)

    async def fake_stream(context):
        async def generator():
            part = Part(inline_data=Blob(data=_png_bytes(mode), mime_type="image/png"))
            yield GenerateContentResponse(
                candidates=[Candidate(content=Content(role="model", parts=[part]))]
            )

        return generator()

    with (
        patch.object(service, "push_frame", capture_frame),
        patch.object(service, "_stream_content", fake_stream),
    ):
        await service._process_context(LLMContext())

    return [f for f in frames if isinstance(f, AssistantImageRawFrame)]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P", "L"])
async def test_inline_image_is_rgb(mode):
    """The frame's pixel data matches its declared RGB format whatever the source mode."""
    (frame,) = await _image_frames(mode)

    assert frame.format == "RGB"
    assert frame.size == (8, 4)
    assert len(frame.image) == 8 * 4 * 3
