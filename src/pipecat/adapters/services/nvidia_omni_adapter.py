#
# Copyright (c) 2026, Daily
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA Nemotron Omni LLM adapter for Pipecat.

Nemotron Omni is served through an OpenAI-compatible Chat Completions endpoint
that also accepts audio and video. It reads media as data URLs under
``audio_url`` and ``video_url`` content parts, so this adapter renames the
universal context's audio and media file parts to those shapes at the provider
boundary. Text and image parts are passed through unchanged.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContextMessage


class NvidiaOmniLLMAdapter(OpenAILLMAdapter):
    """Adapter for NVIDIA Nemotron Omni models.

    Extends ``OpenAILLMAdapter`` with the media content parts Omni accepts:

    - ``input_audio`` parts, as produced by ``LLMContext.create_audio_message()``
      and ``add_audio_frames_message()``, become ``audio_url`` data URLs.
    - ``audio/*`` and ``video/*`` files, as produced by
      ``LLMContext.create_file_message()``, become ``audio_url`` and
      ``video_url`` data URLs.

    Converted messages are new objects, so the context keeps its universal shape.
    """

    def to_provider_content_parts(self, parts: Sequence[Mapping[str, Any]]) -> list[Any]:
        """Convert universal content parts that are sent outside the context.

        Args:
            parts: Universal content parts, such as ``input_audio`` and ``text``.

        Returns:
            The parts in the shape Omni reads.
        """
        return [_audio_url_part(part) for part in parts]

    def _from_universal_context_messages(
        self,
        messages: list[LLMContextMessage],
        *,
        convert_developer_to_user: bool,
    ) -> list[ChatCompletionMessageParam]:
        converted = super()._from_universal_context_messages(
            messages, convert_developer_to_user=convert_developer_to_user
        )
        return [_rename_input_audio(message) for message in converted]

    def _inline_file_item(self, file_data_url: str, filename: str, mime_type: str) -> dict:
        """Build the content item for an inline (base64) file, including audio and video."""
        if mime_type.startswith("audio/"):
            return {"type": "audio_url", "audio_url": {"url": file_data_url}}
        if mime_type.startswith("video/"):
            return {"type": "video_url", "video_url": {"url": file_data_url}}
        return super()._inline_file_item(file_data_url, filename, mime_type)


def _rename_input_audio(message: ChatCompletionMessageParam) -> ChatCompletionMessageParam:
    """Rewrite a message's ``input_audio`` parts as ``audio_url`` data URLs."""
    content = message.get("content")
    if not isinstance(content, list):
        return message
    parts = [_audio_url_part(part) if isinstance(part, Mapping) else part for part in content]
    return cast(ChatCompletionMessageParam, {**message, "content": parts})


def _audio_url_part(part: Mapping[str, Any]) -> Mapping[str, Any]:
    if part.get("type") != "input_audio":
        return part
    payload = part.get("input_audio") or {}
    audio_format = payload.get("format") or "wav"
    data = payload.get("data") or ""
    return {"type": "audio_url", "audio_url": {"url": f"data:audio/{audio_format};base64,{data}"}}
