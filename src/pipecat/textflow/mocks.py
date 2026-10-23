#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Stand-ins for the LLM and the TTS provider.

The LLM is a list of frames built from the response text, streamed the way a
service would stream it. The TTS provider is a :class:`TextFlowTTSService`,
a real ``TTSService`` whose synthesis records the text it was asked to speak
and reports word timestamps built from that text, so the service's tracking
runs exactly as it does against a provider.
"""

import asyncio
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame

from .setup import ChunkingSpec, TokenizationSpec

SAMPLE_RATE = 16000


def chunk_text(text: str, chunking: ChunkingSpec) -> list[str]:
    """Cut the response into the chunks the LLM streams."""
    if chunking.style == "whole":
        return [text]
    if chunking.style == "custom":
        return list(chunking.chunks)
    if chunking.style == "characters":
        size = max(1, chunking.size)
        return [text[i : i + size] for i in range(0, len(text), size)]
    # Each word keeps the whitespace that follows it, as LLM tokens do.
    return re.findall(r"\S+\s*|\s+", text)


def llm_response_frames(text: str, chunking: ChunkingSpec) -> tuple[list[str], list[Frame]]:
    """The frames an LLM service pushes for a response.

    Returns:
        The chunks and the frames: a response start, one ``LLMTextFrame`` per chunk
        with the chunking's interval between them, and a response end.
    """
    chunks = chunk_text(text, chunking)
    frames: list[Frame] = [LLMFullResponseStartFrame()]
    for chunk in chunks:
        frames.append(LLMTextFrame(chunk))
        if chunking.interval_s > 0:
            frames.append(SleepFrame(sleep=chunking.interval_s))
    frames.append(LLMFullResponseEndFrame())
    return chunks, frames


def tokenize(text: str, spec: TokenizationSpec) -> list[tuple[str, float]]:
    """The word timestamps the stand-in provider reports for ``text``."""
    if spec.style == "characters":
        tokens = [c for c in text if not c.isspace()]
    elif spec.style == "split_punctuation":
        tokens = re.findall(r"[^\s.,!?;:]+|[.,!?;:]+", text)
    else:
        tokens = text.split()
    tokens = [t for t in tokens if t not in spec.drop_words]
    return [(token, i * spec.gap_s) for i, token in enumerate(tokens)]


@dataclass
class ProviderRequest:
    """One synthesis request the stand-in provider received.

    Parameters:
        context_id: The TTS context the request belongs to.
        text: The text the service asked the provider to speak.
        tokens: The words the provider reported back, with their timestamps.
    """

    context_id: str
    text: str
    tokens: list[tuple[str, float]] = field(default_factory=list)


class TextFlowTTSService(TTSService):
    """A ``TTSService`` whose provider is a recorder.

    Synthesis records the request, reports word timestamps built from the
    request text when the setup says the provider has them, and yields a
    short silent audio frame. ``delivery`` picks the shape of a real service:
    ``http`` yields the audio from the request, ``websocket`` delivers it later
    through the audio context, as a streaming provider does.
    """

    def __init__(
        self,
        *,
        word_timestamps: bool = True,
        delivery: str = "http",
        tokenization: TokenizationSpec | None = None,
        **kwargs,
    ):
        """Initialize the stand-in service.

        Args:
            word_timestamps: Whether the provider reports word timestamps.
            delivery: ``http`` or ``websocket``.
            tokenization: How the provider reports the words it spoke.
            **kwargs: Passed to ``TTSService``.
        """
        super().__init__(
            push_start_frame=True,
            push_stop_frames=delivery == "http",
            push_text_frames=not word_timestamps,
            pause_frame_processing=False,
            sample_rate=SAMPLE_RATE,
            **kwargs,
        )
        self._word_timestamps = word_timestamps
        self._delivery = delivery
        self._tokenization = tokenization or TokenizationSpec()
        # Seconds of audio already produced per context, so a context's second
        # request reports its words after the first request's, as a provider's
        # timestamps run on from the audio before them.
        self._elapsed: dict[str, float] = {}
        self.requests: list[ProviderRequest] = []

    def can_generate_metrics(self) -> bool:
        """The stand-in reports no metrics."""
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Record the request and produce the provider's response.

        Args:
            text: The text to speak.
            context_id: The TTS context the request belongs to.

        Yields:
            The audio for an ``http`` delivery; nothing for ``websocket``, whose audio
            arrives through the audio context.
        """
        request = ProviderRequest(context_id=context_id, text=text)
        self.requests.append(request)
        if self._word_timestamps:
            offset = self._elapsed.get(context_id, 0.0)
            request.tokens = [(w, offset + t) for w, t in tokenize(text, self._tokenization)]
        self._elapsed[context_id] = self._elapsed.get(context_id, 0.0) + self._duration_s(request)

        if self._delivery == "websocket":
            self.create_task(self._deliver(request), name=f"textflow_deliver_{context_id}")
            return

        if self._word_timestamps:
            await self._report_words(request)
        yield self._audio(request)

    async def _deliver(self, request: ProviderRequest):
        await asyncio.sleep(0.01)
        if self._word_timestamps:
            await self._report_words(request)
        await self.append_to_audio_context(request.context_id, self._audio(request))
        await self.append_to_audio_context(
            request.context_id, TTSStoppedFrame(context_id=request.context_id)
        )
        await self.remove_audio_context(request.context_id)

    async def _report_words(self, request: ProviderRequest):
        await self.add_word_timestamps(
            request.tokens,
            context_id=request.context_id,
            includes_inter_frame_spaces=self._tokenization.includes_inter_frame_spaces,
            pre_merge_tokens=self._tokenization.pre_merge_tokens,
        )

    def _duration_s(self, request: ProviderRequest) -> float:
        # Silence long enough to cover every word, so the output transport
        # releases the words in order as the audio plays, as a provider's audio does.
        gap = self._tokenization.gap_s
        words = len(request.tokens) if request.tokens else len(request.text.split())
        return gap * (words + 1)

    def _audio(self, request: ProviderRequest) -> TTSAudioRawFrame:
        samples = max(int(SAMPLE_RATE * self._duration_s(request)), SAMPLE_RATE // 100)
        return TTSAudioRawFrame(
            audio=b"\x00\x00" * samples,
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            context_id=request.context_id,
        )
