#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime speech-to-text service implementation.

Rime transcribes one utterance per stream and does not detect when the speaker
stops, so this service streams each VAD speech segment as its own request.
"""

import asyncio
import re
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from loguru import logger

from pipecat.audio.speech_segment_buffer import SpeechSegmentBuffer
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import RIME_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given

try:
    from rimelabs_sdk import (
        PCMFormat,
        Rime,
        RimeAudioFormatError,
        RimeAuthenticationError,
        RimeCancelledError,
        RimeInputError,
        RimePermissionError,
        RimeTimeoutError,
        RimeUnavailableError,
        TranscriptionMode,
        TranscriptionOutputContract,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use RimeSTTService, you need to `uv add "pipecat-ai[rime]"`.')
    raise ImportError(f"Missing module: {e}") from e

_SUPPORTED_SAMPLE_RATES = (8000, 16000, 24000, 48000)
_FALLBACK_SAMPLE_RATE = 16000
_MAX_CONTEXT_TERMS_BYTES = 1024
_CONTROL_MARKER = re.compile(r"<\|.*?\|>", re.DOTALL)


@dataclass
class RimeSTTSettings(STTSettings):
    """Settings for RimeSTTService.

    Parameters:
        language: Spoken BCP-47 tag or Language value, passed to Rime unchanged.
            Rime documents ``en`` and ``es``.
        mode: Written formatting or verbatim recognition.
        context_terms: Recognition hints, kept in order. Each term must be
            nonempty, and the terms joined with ``", "`` must fit in 1,024
            UTF-8 bytes.
        model: Must be None; Rime selects the recognizer.
        output_contract: Replacement hypotheses or append-only fragments.
            Both produce cumulative interim transcripts and one final.
    """

    mode: Literal["written", "verbatim"] | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    context_terms: Sequence[str] | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    output_contract: Literal["revised_hypotheses", "append_fragments"] | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


def _validate_settings(settings: RimeSTTSettings) -> None:
    if settings.model is not None:
        raise ValueError("Rime STT does not support model selection")
    if not isinstance(settings.language, str) or not settings.language.strip():
        raise ValueError("Rime STT language must be a nonempty BCP-47 string")
    if settings.mode not in ("written", "verbatim"):
        raise ValueError("Rime STT mode must be 'written' or 'verbatim'")
    terms = settings.context_terms
    if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
        raise ValueError("Rime STT context_terms must be a sequence of strings")
    if any(not isinstance(term, str) or not term.strip() for term in terms):
        raise ValueError("Rime STT context_terms must be nonempty strings")
    if any(_CONTROL_MARKER.search(term) for term in terms):
        raise ValueError("Rime STT context_terms cannot contain model control markers")
    if len(", ".join(terms).encode("utf-8")) > _MAX_CONTEXT_TERMS_BYTES:
        raise ValueError("Rime STT context_terms must fit in 1,024 UTF-8 bytes when joined")
    if settings.output_contract not in ("revised_hypotheses", "append_fragments"):
        raise ValueError("Rime STT output_contract must be revised_hypotheses or append_fragments")
    if settings.extra:
        raise ValueError("Rime STT does not support additional recognition settings")


def _error_category(error: Exception) -> ErrorCategory:
    if isinstance(error, RimeAuthenticationError):
        return ErrorCategory.AUTHENTICATION
    if isinstance(error, RimePermissionError):
        return ErrorCategory.AUTHORIZATION
    if isinstance(error, (RimeTimeoutError, RimeUnavailableError)):
        return ErrorCategory.CONNECTIVITY
    if isinstance(error, RimeInputError):
        return ErrorCategory.INVALID_REQUEST
    if isinstance(error, RimeAudioFormatError):
        return ErrorCategory.APPLICATION
    return ErrorCategory.UNKNOWN


def _frame_language(language: str | None) -> Language | None:
    try:
        return Language(language)
    except ValueError:
        return None


@dataclass
class _Segment:
    """One VAD speech segment and the stream that transcribes it.

    Parameters:
        number: The segment's position in the user's speech, from 1.
        settings: The settings the segment's stream uses.
        previous: The segment before this one, until its final is published.
        audio: Audio for the stream, ended by None.
        published: Set once this segment's final is published or abandoned.
    """

    number: int
    settings: RimeSTTSettings
    previous: "_Segment | None"
    audio: asyncio.Queue[bytes | None] = field(default_factory=asyncio.Queue)
    published: asyncio.Event = field(default_factory=asyncio.Event)

    async def audio_stream(self) -> AsyncIterator[bytes]:
        while (audio := await self.audio.get()) is not None:
            yield audio


class RimeSTTService(STTService):
    """Speech-to-text service using Rime's streaming recognition.

    Rime transcribes one utterance per stream, so the service needs VAD frames
    (from the user aggregator or a ``VADProcessor``): each VAD speech segment
    is streamed as its own request, starting with the audio from just before
    the VAD reported speech. Ending the segment's audio asks Rime for the final
    transcript. Finals are pushed in speech order, and one is finalized only
    while its segment is the user's latest speech.

    Rime's model runs out of context after about 90 to 110 seconds of speech,
    so a segment with no pause that long fails with no transcript.
    """

    Settings = RimeSTTSettings
    _settings: RimeSTTSettings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        sample_rate: int | None = None,
        settings: RimeSTTSettings | None = None,
        ttfs_p99_latency: float | None = RIME_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Rime STT service.

        Args:
            api_key: Rime API key. Defaults to the ``RIME_API_KEY`` environment variable.
            sample_rate: Audio sample rate. If None, uses the pipeline's input rate.
            settings: Runtime-updatable settings.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to the parent STTService.
        """
        default_settings = self.Settings(
            model=None,
            language="en",
            mode="written",
            context_terms=(),
            output_contract="revised_hypotheses",
        )
        if settings is not None:
            default_settings.apply_update(settings)
        _validate_settings(default_settings)

        super().__init__(
            sample_rate=sample_rate,
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

        self._api_key = api_key
        self._client: Rime | None = None
        # Never started, so it keeps only the audio from just before speech.
        self._pre_speech = SpeechSegmentBuffer()
        self._resampler = create_stream_resampler()
        self._segment: _Segment | None = None
        self._last_segment: _Segment | None = None
        self._segment_count = 0
        self._segment_tasks: set[asyncio.Task] = set()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Rime STT supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Pass the language tag to Rime unchanged.

        Args:
            language: The language to convert.

        Returns:
            The BCP-47 tag of ``language``.
        """
        return language.value

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta, used by segments that start afterwards.

        ``model`` and ``extra`` are ignored with a warning. A delta with an
        invalid value is rejected whole and reported as an error.

        Args:
            delta: A Rime STT settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        delta = cast(RimeSTTSettings, delta)
        unsupported = set(delta.extra)
        if is_given(delta.model) and delta.model is not None:
            unsupported.add("model")
        self._warn_unhandled_updated_settings(unsupported)
        delta.model = NOT_GIVEN
        delta.extra = {}
        candidate = self._settings.copy()
        candidate.apply_update(delta)
        try:
            _validate_settings(candidate)
        except ValueError as error:
            await self.push_error(
                f"Rime STT settings update rejected: {error}",
                exception=error,
                category=ErrorCategory.APPLICATION,
            )
            return {}
        return await super()._update_settings(delta)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._pre_speech.set_sample_rate(self.sample_rate)

    async def start(self, frame: StartFrame):
        """Start the service and create the Rime client.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        await self._ensure_client()

    async def stop(self, frame: EndFrame):
        """Finish transcribing the segments in progress, then stop.

        Args:
            frame: The end frame.
        """
        self._end_segment()
        if self._segment_tasks:
            await asyncio.gather(*self._segment_tasks, return_exceptions=True)
        await self._close_client()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Stop at once, dropping the segments in progress.

        Args:
            frame: The cancel frame.
        """
        await self._cancel_segments()
        await self._close_client()
        await super().cancel(frame)

    async def cleanup(self):
        """Release the Rime client at teardown."""
        await super().cleanup()
        await self._cancel_segments()
        await self._close_client()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, streaming each VAD speech segment as its own request.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._start_segment()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._segment is not None:
                self.request_finalize()
            self._end_segment()

    async def process_audio_frame(self, frame: InputAudioRawFrame, direction: FrameDirection):
        """Stream audio within a speech segment, and keep the audio just before one.

        Args:
            frame: The audio frame to process.
            direction: The direction of frame processing.
        """
        if self._segment is not None:
            await super().process_audio_frame(frame, direction)
        elif not self.is_muted:
            self._pre_speech.append(frame.audio)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to the current segment's stream.

        Args:
            audio: Raw 16-bit mono PCM audio at the service's sample rate.

        Yields:
            None, as transcripts arrive from the segment's stream.
        """
        if self._segment is not None:
            await self._send_audio(self._segment, audio)
        yield None

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    @property
    def _send_sample_rate(self) -> int:
        """The rate audio is sent at, resampled when Rime doesn't take the input rate."""
        if self.sample_rate in _SUPPORTED_SAMPLE_RATES:
            return self.sample_rate
        return _FALLBACK_SAMPLE_RATE

    async def _send_audio(self, segment: _Segment, audio: bytes):
        audio = await self._resampler.resample(audio, self.sample_rate, self._send_sample_rate)
        segment.audio.put_nowait(audio)

    async def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        try:
            self._client = Rime(api_key=self._api_key)
        except Exception as e:
            await self._push_rime_error(e)
            return False
        return True

    async def _close_client(self):
        if self._client is not None:
            client, self._client = self._client, None
            await client.close()

    async def _start_segment(self):
        self._end_segment()
        if self.is_muted or not self.is_usable or not await self._ensure_client():
            return
        self._segment_count += 1
        segment = _Segment(
            number=self._segment_count,
            settings=self._settings.copy(),
            previous=self._last_segment,
        )
        self._segment = self._last_segment = segment
        pre_speech = self._pre_speech.stop()
        if pre_speech:
            self._record_stt_audio_usage(pre_speech)
            await self._send_audio(segment, pre_speech)
        task = self.create_task(self._transcribe(segment), name=f"rime_stt_{segment.number}")
        self._segment_tasks.add(task)
        task.add_done_callback(self._segment_tasks.discard)

    def _end_segment(self):
        if self._segment is not None:
            self._segment.audio.put_nowait(None)
            self._segment = None

    async def _cancel_segments(self):
        self._segment = None
        for task in list(self._segment_tasks):
            await self.cancel_task(task)

    async def _transcribe(self, segment: _Segment):
        """Stream a segment to Rime and push its transcripts in speech order."""
        assert self._client is not None
        settings = segment.settings
        text = ""
        final = None
        try:
            stream = self._client.stt.stream(
                segment.audio_stream(),
                language=cast(str, settings.language),
                mode=TranscriptionMode(settings.mode),
                context_terms=tuple(cast(Sequence[str], settings.context_terms)),
                output_contract=TranscriptionOutputContract(settings.output_contract),
                input_format=PCMFormat(
                    sample_rate=cast(Literal[8000, 16000, 24000, 48000], self._send_sample_rate),
                    channels=1,
                ),
            )
            async with stream:
                async for update in stream:
                    if update.kind == "final":
                        final = update
                        continue
                    text = text + update.text if update.kind == "delta" else update.text
                    # Drop a later segment's interims until the earlier final is
                    # out; the next interim replaces them.
                    previous = segment.previous
                    if text and (previous is None or previous.published.is_set()):
                        await self.push_frame(
                            InterimTranscriptionFrame(
                                text,
                                self._user_id,
                                time_now_iso8601(),
                                _frame_language(cast(str, settings.language)),
                                result=update,
                            )
                        )
            if segment.previous is not None:
                await segment.previous.published.wait()
            if final is not None:
                await self._push_final(segment, final.text, final.language, final)
        except RimeCancelledError:
            pass
        except Exception as e:
            await self._push_rime_error(e)
        finally:
            segment.published.set()
            segment.previous = None
            if self._segment is segment:
                self._segment = None

    async def _push_final(self, segment: _Segment, text: str, language: str, result: Any):
        # Once the user speaks again, this final covers only the words before
        # they resumed, so only the latest segment's final is finalized.
        latest = segment.number == self._segment_count
        # An empty final has no words to add, but the latest one still tells
        # the user turn that no more transcripts are coming.
        if not text and not latest:
            return
        if latest:
            self.confirm_finalize()
        frame_language = _frame_language(language)
        # Report usage before the transcription frame so tracing can attach it
        # to the STT span the frame closes.
        await self.emit_stt_usage_metrics()
        await self.push_frame(
            TranscriptionFrame(
                text, self._user_id, time_now_iso8601(), frame_language, result=result
            )
        )
        await self._handle_transcription(text, True, frame_language)

    async def _push_rime_error(self, error: Exception):
        # Rime's messages can echo request data such as context terms, so the
        # message names only the error type and request.
        request_id = getattr(error, "request_id", None)
        request = f" (request {request_id})" if request_id else ""
        await self.push_error(
            f"Rime STT request failed: {type(error).__name__}{request}",
            exception=error,
            category=_error_category(error),
        )
