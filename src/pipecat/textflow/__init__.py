#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Trace a theoretical LLM response through the text path of a pipeline.

Pipecat splits one LLM response into three texts: what the conversation
context records, what downstream consumers such as an RTVI client render, and
what the TTS provider is asked to speak. This package runs a string through
the real processors that do the splitting, with the LLM and the TTS provider
replaced by stand-ins, and records what each stage did to every segment.

A :class:`TextFlowSetup` describes the pipeline downstream of the LLM: whether
an ``LLMTextProcessor`` is present and how it aggregates and transforms, how
the TTS service aggregates, skips and transforms, and which ``RTVIObserver``
transforms apply. :func:`run_textflow` runs the setup over a response and
returns a :class:`TextFlowTrace`.

Example::

    from pipecat.textflow import TextFlowSetup, run_textflow

    setup = TextFlowSetup.preset("code-helper")
    trace = await run_textflow(setup, "Run <code>npm install</code> to start.")
    for segment in trace.segments:
        print(segment.text_type, segment.llm_text, segment.tts_text)

A setup can also name a bot file, in which case the run captures the bot's own
processors and traces them as they are (see :func:`capture_bot`).

The ``pipecat textflow`` command renders a trace in the terminal or serves the
interactive page.
"""

from pipecat.textflow.capture import BotSummary, TextFlowCaptureError, capture_bot
from pipecat.textflow.setup import (
    AggregatorSpec,
    ChunkingSpec,
    LLMTextProcessorSpec,
    PatternSpec,
    RTVISpec,
    TextFlowSetup,
    TokenizationSpec,
    TransformSpec,
    TTSSpec,
)
from pipecat.textflow.trace import (
    StageEvent,
    TextFlowRunError,
    TextFlowSegment,
    TextFlowTrace,
    TimelineEntry,
    WordEvent,
    run_textflow,
)
from pipecat.textflow.transforms import register_transform

__all__ = [
    "AggregatorSpec",
    "BotSummary",
    "ChunkingSpec",
    "LLMTextProcessorSpec",
    "PatternSpec",
    "RTVISpec",
    "StageEvent",
    "TTSSpec",
    "TextFlowCaptureError",
    "TextFlowRunError",
    "TextFlowSegment",
    "TextFlowSetup",
    "TextFlowTrace",
    "TimelineEntry",
    "TokenizationSpec",
    "TransformSpec",
    "WordEvent",
    "capture_bot",
    "register_transform",
    "run_textflow",
]
