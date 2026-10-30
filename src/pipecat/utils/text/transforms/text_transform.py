#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The text transform callable shared by TTS services and the LLM text processor."""

from collections.abc import Awaitable, Callable

from pipecat.utils.text.base_text_aggregator import TextType

TextTransform = Callable[[str, str | TextType], Awaitable[str]]
"""An async callable that rewrites text of a given type: ``(text, text_type) -> text``.

Where a transform is registered decides who sees the rewrite. On a TTS service it
reaches the synthesizer only, and the context keeps the text as the LLM wrote it. On
:class:`~pipecat.processors.aggregators.llm_text_processor.LLMTextProcessor` it reaches
everything downstream: the voice, the context and the client.
"""
