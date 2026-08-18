#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speechify Text-to-Speech Service Implementation.

Contains SpeechifyHttpTTSService, which streams audio and word-level speech marks
from Speechify's ``/v1/audio/stream/with-timestamps`` endpoint over Server-Sent Events.
"""

import base64
import json
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from loguru import logger

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

# Identifies the integration to Speechify's platform attribution.
CALLER_HEADERS = {
    "Speechify-Caller": "pipecat",
    "X-Pipecat-Version": pipecat_version(),
}

# PCM rates Speechify can synthesize, as the `pcm_<rate>` output formats.
SPEECHIFY_PCM_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)

SPEECHIFY_DEFAULT_SAMPLE_RATE = 24000


def language_to_speechify_language(language: Language) -> str | None:
    """Convert a Language enum to a Speechify language tag.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Speechify language tag (e.g. ``"en-US"``). Languages outside
        the officially supported set fall back to their BCP-47 value with a warning.
    """
    LANGUAGE_MAP = {
        Language.DE: "de-DE",
        Language.EN: "en-US",
        Language.ES: "es-ES",
        Language.FR: "fr-FR",
        Language.IT: "it-IT",
        Language.PT: "pt-BR",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


def _output_format_from_sample_rate(sample_rate: int) -> tuple[str, int]:
    """Pick the Speechify PCM output format for a sample rate.

    Args:
        sample_rate: The desired audio sample rate in Hz.

    Returns:
        Tuple of (output_format, sample_rate), where the returned sample rate is the one
        Speechify will actually synthesize at. It differs from the requested rate when
        Speechify has no matching PCM format (e.g. 32000 Hz), so callers must stamp the
        returned rate onto their audio frames and let the output transport resample.
    """
    if sample_rate in SPEECHIFY_PCM_SAMPLE_RATES:
        return f"pcm_{sample_rate}", sample_rate
    logger.warning(
        f"Speechify has no PCM output format for {sample_rate} Hz, "
        f"synthesizing at {SPEECHIFY_DEFAULT_SAMPLE_RATE} Hz instead"
    )
    return f"pcm_{SPEECHIFY_DEFAULT_SAMPLE_RATE}", SPEECHIFY_DEFAULT_SAMPLE_RATE


def _mark_span(mark: Mapping[str, Any], text: str) -> tuple[int, int] | None:
    """Return a mark's character span within the synthesized text, if it has one.

    ``start`` and ``end`` are offsets into the request's ``input``. They are the
    authority on a mark's text: a mark's ``value`` is the word as Speechify
    pronounced it, which sometimes normalizes characters (typographic apostrophes
    become ASCII, for instance), and downstream word tracking matches words against
    the text that was sent, where a normalized word derails the match for the rest
    of the sentence.
    """
    start, end = mark.get("start"), mark.get("end")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
        return start, end
    return None


class _SpeechMarkAccumulator:
    """Assembles a request's speech marks into whole words.

    Speechify marks intra-word punctuation separately, and the resulting run of marks
    can straddle the boundary between two ``speech.chunk`` events. Marks that abut in
    the source text — the next starting exactly where the previous ended — are joined
    into one word. A word is released as soon as the text shows it cannot continue
    (the next character is whitespace, or the text ends), which is every word not
    split this way.

    Example marks for "text-to-speech is well-known.", split across two events::

        event 1: "text"  "-"  "to"  "-"
        event 2: "speech"  "is"  "well"  "-"  "known."

    Which are assembled into::

        ["text-to-speech", "is", "well-known."]
    """

    def __init__(self, text: str, time_offset: float = 0.0):
        """Initialize the accumulator.

        Args:
            text: The text sent for synthesis, which the marks index into.
            time_offset: Seconds to add to each word, carrying over the duration of
                the utterances already synthesized in this turn.
        """
        self._text = text
        self._time_offset = time_offset
        self._pending: tuple[int, int, float] | None = None
        # The last mark's end, in seconds and without the offset applied.
        self.end_time = 0.0

    def add(self, speech_marks: list[Mapping[str, Any]] | None) -> list[tuple[str, float]]:
        """Add one event's speech marks.

        Args:
            speech_marks: The ``speech_marks`` payload from a ``speech.chunk`` event,
                absent on an audio-only event.

        Returns:
            The (word, seconds) pairs completed by these marks.
        """
        word_times: list[tuple[str, float]] = []

        for mark in speech_marks or []:
            # The stream carries word marks; skip any other type Speechify may add.
            if mark.get("type", "word") != "word":
                continue

            self.end_time = max(self.end_time, mark.get("end_time", 0) / 1000)
            start_time = self._time_offset + mark.get("start_time", 0) / 1000

            span = _mark_span(mark, self._text)
            if span is None:
                word_times.extend(self.flush())
                value = mark.get("value", "").strip()
                if value:
                    word_times.append((value, start_time))
                continue

            start, end = span
            if self._pending and self._pending[1] == start:
                self._pending = (self._pending[0], end, self._pending[2])
            else:
                word_times.extend(self.flush())
                self._pending = (start, end, start_time)

            if end >= len(self._text) or self._text[end].isspace():
                word_times.extend(self.flush())

        return word_times

    def flush(self) -> list[tuple[str, float]]:
        """Release the word still being assembled, if any."""
        if self._pending is None:
            return []
        start, end, start_time = self._pending
        self._pending = None
        word = self._text[start:end].strip()
        return [(word, start_time)] if word else []


def _parse_sse_event(block: str) -> tuple[str, dict[str, Any]] | None:
    """Parse one Server-Sent Events block into its event name and decoded data.

    Args:
        block: The block's lines, newline-joined and without the terminating blank line.

    Returns:
        Tuple of (event_name, data), or None if the block carries no decodable data.
    """
    event_name = ""
    data_lines: list[str] = []

    for line in block.splitlines():
        field_name, _, value = line.partition(":")
        if not field_name:  # Comment line.
            continue
        value = value.removeprefix(" ")
        if field_name == "event":
            event_name = value
        elif field_name == "data":
            data_lines.append(value)

    if not data_lines:
        return None

    try:
        return event_name, json.loads("\n".join(data_lines))
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Speechify SSE event: {e}")
        return None


async def _iter_sse_events(
    response: aiohttp.ClientResponse,
) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
    """Yield (event_name, data) for each Server-Sent Events block in a response."""
    block: list[str] = []

    async for raw_line in response.content:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if line:
            block.append(line)
            continue
        if block:
            event = _parse_sse_event("\n".join(block))
            block.clear()
            if event:
                yield event

    if block:
        event = _parse_sse_event("\n".join(block))
        if event:
            yield event


@dataclass
class SpeechifyTTSSettings(TTSSettings):
    """Settings for SpeechifyHttpTTSService.

    Parameters:
        loudness_normalization: Whether to normalize audio loudness to a standard level.
            Adds latency.
        text_normalization: Whether to spell out numbers, dates and similar tokens
            before synthesis. Adds latency.
    """

    loudness_normalization: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_normalization: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SpeechifyHttpTTSService(TTSService):
    """Speechify HTTP-based TTS service with word timestamps.

    Streams PCM audio and word-level speech marks over Server-Sent Events from
    Speechify's ``/v1/audio/stream/with-timestamps`` endpoint. Speech marks are only
    produced by the streaming-native models, ``simba-3.2`` (English) and ``simba-3.0``
    (multilingual); the legacy ``simba-english`` and ``simba-multilingual`` models are
    rejected by this endpoint.
    """

    Settings = SpeechifyTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.speechify.ai",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        **kwargs,
    ):
        """Initialize the Speechify HTTP TTS service.

        Args:
            api_key: Speechify API key for authentication.
            aiohttp_session: aiohttp ClientSession for HTTP requests.
            base_url: Base URL for the Speechify API.
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline's rate.
            settings: Runtime-updatable settings.
            text_aggregation_mode: How to aggregate incoming text before synthesis.
            **kwargs: Additional arguments passed to the parent service.
        """
        default_settings = self.Settings(
            model="simba-3.2",
            voice="geffen_32",
            language=None,
            loudness_normalization=None,
            text_normalization=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            push_text_frames=False,
            push_stop_frames=True,
            push_start_frame=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._session = aiohttp_session
        self._url = f"{base_url}/v1/audio/stream/with-timestamps"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **CALLER_HEADERS,
        }

        self._output_format = ""  # Initialized in start().
        self._audio_sample_rate = 0  # Initialized in start().

        # Speech-mark times restart at zero for every request, so successive utterances
        # in a turn are offset by the duration of everything synthesized before them.
        self._cumulative_time = 0.0

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Speechify TTS service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a Speechify language tag.

        Args:
            language: The language to convert.

        Returns:
            The Speechify language tag, or None if not supported.
        """
        return language_to_speechify_language(language)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._output_format, self._audio_sample_rate = _output_format_from_sample_rate(
            self.sample_rate
        )
        self._cumulative_time = 0.0

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, TTSStoppedFrame)):
            self._cumulative_time = 0.0

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio synthesis.

        Note:
            HTTP-based service doesn't require explicit flushing.
        """
        pass

    def _build_payload(self, text: str) -> dict[str, Any]:
        """Build the request body for synthesizing a piece of text.

        Args:
            text: Text to convert to speech.

        Returns:
            The JSON request body, carrying the settings in effect for this request.
        """
        payload: dict[str, Any] = {
            "input": text,
            "voice_id": assert_given(self._settings.voice),
            "model": assert_given(self._settings.model),
            "output_format": self._output_format,
        }

        language = assert_given(self._settings.language)
        if language:
            payload["language"] = language

        options: dict[str, Any] = {}
        loudness_normalization = assert_given(self._settings.loudness_normalization)
        if loudness_normalization is not None:
            options["loudness_normalization"] = loudness_normalization
        text_normalization = assert_given(self._settings.text_normalization)
        if text_normalization is not None:
            options["text_normalization"] = text_normalization
        if options:
            payload["options"] = options

        return payload

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using the Speechify streaming API with timestamps.

        Args:
            text: Text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        try:
            async with self._session.post(
                self._url, json=self._build_payload(text), headers=self._headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=f"Speechify API error: {error_text}")
                    return

                await self.start_tts_usage_metrics(text)

                async for frame in self._process_stream(response, text, context_id):
                    yield frame

        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

        finally:
            await self.stop_all_metrics()

    async def _process_stream(
        self, response: aiohttp.ClientResponse, text: str, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        """Process the Server-Sent Events stream from the Speechify API.

        Args:
            response: The streaming response from the Speechify API.
            text: The text sent for synthesis, which the speech marks index into.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        # An event may carry audio, speech marks, or both, and marks lag their audio, so
        # the last event of a stream is often marks-only.
        words = _SpeechMarkAccumulator(text, self._cumulative_time)
        utterance_duration = 0.0

        async for event_name, data in _iter_sse_events(response):
            if event_name == "speech.chunk":
                audio = data.get("audio")
                if audio:
                    await self.stop_ttfb_metrics()
                    yield TTSAudioRawFrame(
                        audio=base64.b64decode(audio),
                        sample_rate=self._audio_sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )

                await self.add_word_timestamps(words.add(data.get("speech_marks")), context_id)
            elif event_name == "speech.done":
                # Preferred over the last mark's end time, since it includes any
                # trailing silence.
                audio_duration_ms = data.get("audio_duration_ms")
                if audio_duration_ms:
                    utterance_duration = max(utterance_duration, audio_duration_ms / 1000)
            elif event_name == "speech.error":
                # The status code is already committed once the stream has started, so
                # mid-stream failures arrive as an event rather than an HTTP error.
                error = data.get("error", {})
                await self.push_error(
                    error_msg=(
                        f"Speechify API error ({error.get('code', 'unknown')}): "
                        f"{error.get('message', data)}"
                    )
                )

        # A word split across marks is still pending if the text ran out mid-run.
        await self.add_word_timestamps(words.flush(), context_id)
        self._cumulative_time += max(utterance_duration, words.end_time)
