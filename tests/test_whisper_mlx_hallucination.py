#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how the MLX Whisper STT service filters hallucinated segments.

``mlx_whisper`` is Apple-Silicon only, so the tests stand a fake module in
``sys.modules``: the service imports it lazily inside ``run_stt``, and they only
need the JSON-shaped result it returns.
"""

import sys
import types
from collections.abc import AsyncIterator

import pytest

# The service raises ImportError (not ModuleNotFoundError) when its extra is absent.
pytest.importorskip("pipecat.services.whisper.stt", exc_type=ImportError)

from pipecat.frames.frames import TranscriptionFrame  # noqa: E402
from pipecat.services.whisper.stt import WhisperSTTServiceMLX  # noqa: E402


def _segments(*segments: dict) -> dict:
    """Build the result ``mlx_whisper.transcribe`` returns."""
    return {"text": "", "segments": list(segments)}


async def _transcribe(segments: dict) -> list[str]:
    """Transcribe with a fake ``mlx_whisper`` returning *segments*."""
    fake = types.ModuleType("mlx_whisper")
    fake.transcribe = lambda audio, **kwargs: segments
    previous = sys.modules.get("mlx_whisper")
    sys.modules["mlx_whisper"] = fake
    try:
        service = WhisperSTTServiceMLX(settings=WhisperSTTServiceMLX.Settings(model="tiny.en"))
        frames: AsyncIterator = service.run_stt(b"\x00\x00" * 1600)
        # Segments are joined with a space, so the transcript is padded.
        return [
            frame.text.strip() async for frame in frames if isinstance(frame, TranscriptionFrame)
        ]
    finally:
        if previous is None:
            sys.modules.pop("mlx_whisper", None)
        else:
            sys.modules["mlx_whisper"] = previous


@pytest.mark.asyncio
async def test_repetitive_segment_is_dropped():
    """A compression ratio past Whisper's threshold marks a hallucination.

    A segment that repeats itself compresses far better than speech, so Whisper
    reports a high ``compression_ratio`` for it. Several segments are decoded
    together from one window, so the other one has to survive.
    """
    texts = await _transcribe(
        _segments(
            {"text": " Thank you for watching!", "compression_ratio": 3.0},
            {"text": " Hello.", "compression_ratio": 1.2},
        )
    )
    assert texts == ["Hello."]


@pytest.mark.asyncio
async def test_segment_without_compression_ratio_is_kept():
    """A backend result that omits the field carries no hallucination signal."""
    assert await _transcribe(_segments({"text": " Hello."})) == ["Hello."]
