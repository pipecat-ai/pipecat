#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""VAD-driven Rime speech recognition using the public Rime SDK.

The service requires VAD start/stop frames, from either a VADProcessor or the
user aggregator. Each VAD segment is one streaming request. Input is PCM16
little-endian at 8, 16, 24 or 48 kHz, mono or stereo. Without VAD, only a
one-second prefix is retained.
"""

import asyncio
import math
import re
import time
from collections import deque
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    MetricsFrame,
    StartFrame,
    STTMuteFrame,
    TranscriptionFrame,
    UserAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    MetricsData,
    STTUsage,
    STTUsageMetricsData,
    TTFBMetricsData,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import NOT_GIVEN, NotGiven

try:
    from rimelabs_sdk import (
        PCMFormat,
        Rime,
        RimeAudioFormatError,
        RimeAuthenticationError,
        RimeCancelledError,
        RimeInputError,
        RimePermissionError,
        RimeResourceLimitError,
        RimeStreamError,
        RimeTimeoutError,
        RimeUnavailableError,
        TranscriptionMode,
        TranscriptionOutputContract,
    )
except ImportError as e:
    raise ImportError(
        "Rime STT requires the optional dependency: pip install 'pipecat-ai[rime-stt]'"
    ) from e

_MAX_REQUESTS = 4
_MAX_AUDIO_SECONDS = 120
_MAX_TEXT_BYTES = 1024 * 1024


@dataclass
class RimeSTTSettings(STTSettings):
    """Recognition options applied to the next speech segment.

    Parameters:
        language: Spoken BCP-47 tag or Language value; region tags are preserved.
        mode: Written formatting or verbatim recognition.
        context_terms: Recognition hints, copied in their original order.
        model: Must be None; Rime selects the recognizer.
        output_contract: Replacement hypotheses or append-only fragments.
            Both produce cumulative Pipecat interim transcripts and one final.
    """

    mode: Literal["written", "verbatim"] | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    context_terms: Sequence[str] | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    output_contract: Literal["revised_hypotheses", "append_fragments"] | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


def _validate_settings(settings: RimeSTTSettings) -> None:
    if settings.model is not None:
        raise ValueError("Rime STT does not support model selection")
    if isinstance(settings.language, Language):
        settings.language = settings.language.value
    if not isinstance(settings.language, str) or not settings.language.strip():
        raise ValueError("Rime STT language must be a nonempty BCP-47 string")
    if settings.mode not in ("written", "verbatim"):
        raise ValueError("Rime STT mode must be 'written' or 'verbatim'")
    terms = settings.context_terms
    if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
        raise ValueError("Rime STT context_terms must be a sequence of strings")
    if any(not isinstance(term, str) for term in terms):
        raise ValueError("Rime STT context_terms must contain strings")
    settings.context_terms = tuple(terms)
    if settings.output_contract not in (
        "revised_hypotheses",
        "append_fragments",
    ):
        raise ValueError("Rime STT output_contract must be revised_hypotheses or append_fragments")
    if settings.extra:
        raise ValueError("Rime STT does not support additional recognition settings")


def _settings_store(delta: RimeSTTSettings | None) -> RimeSTTSettings:
    settings = RimeSTTSettings(
        model=None,
        language="en",
        mode="written",
        context_terms=(),
        output_contract="revised_hypotheses",
    )
    if delta is not None:
        settings.apply_update(delta)
    _validate_settings(settings)
    return settings


class _AudioSource:
    """A byte-bounded source whose close signal needs no queue capacity."""

    def __init__(self, byte_limit: int):
        self.chunks: deque[bytes] = deque()
        self.size = 0
        self.byte_limit = byte_limit
        self.closed = False
        self.ready = asyncio.Event()

    def append(self, audio: bytes):
        if self.size + len(audio) > self.byte_limit:
            raise RimeResourceLimitError("Unsent audio exceeds five seconds")
        self.chunks.append(audio)
        self.size += len(audio)
        self.ready.set()

    def close(self):
        self.closed = True
        self.ready.set()

    def discard(self):
        self.chunks.clear()
        self.size = 0
        self.close()

    async def __aiter__(self):
        while True:
            if self.chunks:
                audio = self.chunks.popleft()
                self.size -= len(audio)
                yield audio
            elif self.closed:
                return
            else:
                self.ready.clear()
                await self.ready.wait()


@dataclass
class _Utterance:
    sequence: int
    settings: RimeSTTSettings
    user_id: str = ""
    audio_format: tuple[int, int] | None = None
    source: _AudioSource | None = None
    task: asyncio.Task | None = None
    speech_end: float | None = None
    received_at: float | None = None
    accepted_bytes: int = 0
    uploaded_seconds: float = 0
    text: str = ""
    request_id: str | None = None
    interim: InterimTranscriptionFrame | None = None
    final: TranscriptionFrame | None = None
    done: bool = False
    discarded: bool = False


class _Recognition:
    """SDK lifetime, ordered publication and per-request accounting."""

    def __init__(
        self,
        service: STTService,
        api_key: str | None,
        client: Rime | None,
        request_timeout: float,
        shutdown_timeout: float,
    ):
        if client is not None and api_key is not None:
            raise ValueError("Pass either client or api_key, not both")
        for name, value in (
            ("request_timeout", request_timeout),
            ("shutdown_timeout", shutdown_timeout),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite seconds")
        self.service = service
        self.api_key = api_key
        self.timeout = request_timeout
        self.shutdown_timeout = shutdown_timeout
        self.client = client
        self._owns_client = client is None
        self.pending: deque[_Utterance] = deque()
        self.active: _Utterance | None = None
        self.sequence = 0
        self.speaking = False
        self.suppressed = False
        self.closed = False
        self.credentials_rejected = False
        self.publish_lock = asyncio.Lock()

    async def start(self):
        try:
            if self.client is None:
                self.client = Rime(api_key=self.api_key)
            self.closed = False
        except Exception as error:
            self.closed = True
            await self.error(error)

    async def close(self):
        self.closed = True
        if self._owns_client and self.client is not None:
            client, self.client = self.client, None
            await client.close()

    async def begin(self) -> _Utterance | None:
        self.sequence += 1
        self.suppressed = False
        if (
            self.closed
            or self.service.is_muted
            or not self.service.is_usable
            or self.credentials_rejected
        ):
            self.suppressed = True
            return None
        if len(self.pending) >= _MAX_REQUESTS:
            self.suppressed = True
            await self.error(RimeResourceLimitError("Too many pending speech segments"))
            return None
        record = _Utterance(self.sequence, cast(RimeSTTSettings, self.service._settings).copy())
        self.pending.append(record)
        self.active = record
        return record

    async def update(self, delta: STTSettings) -> dict[str, Any]:
        candidate = cast(RimeSTTSettings, self.service._settings).copy()
        candidate.apply_update(cast(RimeSTTSettings, delta))
        _validate_settings(candidate)
        # Authentication is independent of recognition settings.
        changed = await AIService._update_settings(self.service, candidate)
        if self.credentials_rejected:
            await self.service.set_usable(False)
        return changed

    def frame(
        self, record: _Utterance, text: str, *, language: str | None = None, final: bool = False
    ):
        try:
            language_value = Language(language or cast(str, record.settings.language))
        except ValueError:
            language_value = None
        cls = TranscriptionFrame if final else InterimTranscriptionFrame
        return cls(
            text=text,
            user_id=record.user_id,
            timestamp=time_now_iso8601(),
            language=language_value,
            result={
                "request_id": record.request_id,
                "language": language or record.settings.language,
            },
        )

    async def publish(self):
        async with self.publish_lock:
            while self.pending and not self.closed:
                record = self.pending[0]
                if not record.discarded and record.interim is not None:
                    interim, record.interim = record.interim, None
                    await self.service.push_frame(interim)
                if not record.done:
                    break
                self.pending.popleft()
                if not record.discarded and record.final is not None:
                    record.final.finalized = record.sequence == self.sequence and not self.speaking
                    # Bypass STTService's single shared latency timer and use
                    # Rime's per-utterance finalization state.
                    await AIService.push_frame(self.service, record.final)

    async def metrics(self, record: _Utterance):
        seconds, record.uploaded_seconds = record.uploaded_seconds, 0
        data: list[MetricsData] = []
        if seconds and self.service.usage_metrics_enabled:
            data.append(
                STTUsageMetricsData(
                    processor=self.service.name, value=STTUsage(audio_seconds=seconds)
                )
            )
        if (
            self.service.metrics_enabled
            and not record.discarded
            and record.received_at is not None
            and record.speech_end is not None
        ):
            data.append(
                TTFBMetricsData(
                    processor=self.service.name,
                    value=max(0, record.received_at - record.speech_end),
                )
            )
        if data:
            frame = MetricsFrame(data=data)
            frame.metadata.update(user_id=record.user_id, request_id=record.request_id)
            await self.service.push_frame(frame)

    async def error(
        self, error: Exception, record: _Utterance | None = None, *, detail: str | None = None
    ):
        if record is not None and record.discarded:
            return
        if record is not None:
            record.discarded = record.done = True
            record.interim = record.final = None
        if not isinstance(error, (RimeCancelledError, asyncio.CancelledError)):
            category = ErrorCategory.UNKNOWN
            if isinstance(error, RimeAuthenticationError):
                category = ErrorCategory.AUTHENTICATION
                self.credentials_rejected = True
            elif isinstance(error, RimePermissionError):
                category = ErrorCategory.AUTHORIZATION
                self.credentials_rejected = True
            elif isinstance(error, (RimeTimeoutError, RimeUnavailableError, TimeoutError)):
                category = ErrorCategory.CONNECTIVITY
            elif isinstance(error, RimeInputError):
                category = ErrorCategory.INVALID_REQUEST
            elif isinstance(error, (RimeAudioFormatError, ValueError)):
                category = ErrorCategory.APPLICATION
            request_id = getattr(error, "request_id", None) or (
                record.request_id if record else None
            )
            identity = (
                f" (request {request_id})"
                if isinstance(request_id, str) and re.fullmatch(r"[\w.-]{1,128}", request_id)
                else ""
            )
            # SDK messages can contain echoed request data. Report only the
            # public error type and validated request identity.
            await self.service.push_error(
                f"Rime STT request failed: {type(error).__name__}{identity}"
                + (f": {detail}" if detail else ""),
                category=category,
            )
        await self.publish()


class RimeSTTService(STTService):
    """Stream each VAD speech segment to Rime with ordered transcript delivery.

    A one-second idle prefix covers VAD detection delay. At most four utterances
    may await publication, each with at most five seconds of unsent audio.
    Overflow discards the affected segment and reports an error. Muting discards
    pending speech; unmuting during speech starts a fresh segment on new audio.
    Interruption frames preserve recognition of the user's barge-in.
    An older segment's completed transcript has ``finalized=False`` while
    newer speech is pending, so it cannot finalize that newer segment.
    """

    Settings = RimeSTTSettings
    _settings: RimeSTTSettings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Rime | None = None,
        settings: RimeSTTSettings | None = None,
        request_timeout: float = 120,
        shutdown_timeout: float = 5.0,
        sample_rate: int | None = None,
        **kwargs,
    ):
        """Initialize streaming recognition without opening a connection.

        Args:
            api_key: Rime API credential. Defaults to RIME_API_KEY when no client is supplied.
            client: Caller-owned SDK client on the pipeline's event loop. Mutually
                exclusive with api_key; the service never closes this client.
            settings: Recognition options for subsequent utterances.
            request_timeout: Overall request deadline, including speaking and
                upload time. Server deadlines still apply.
            shutdown_timeout: Positive finite time to drain speech on EndFrame,
                in seconds, before cancelling unfinished requests.
            sample_rate: Expected input rate, if specified. Audio is never relabeled.
            **kwargs: Additional STTService options, including ttfs_p99_latency.
        """
        super().__init__(settings=_settings_store(settings), sample_rate=sample_rate, **kwargs)
        self._recognition = _Recognition(self, api_key, client, request_timeout, shutdown_timeout)
        self._prefix = bytearray()
        self._prefix_format: tuple[int, int] | None = None
        self._prefix_user = ""
        self._tasks: set[asyncio.Task] = set()

    def can_generate_metrics(self) -> bool:
        """Whether this service reports audio usage and speech-end latency."""
        return True

    async def start(self, frame: StartFrame):
        """Create the SDK client on the pipeline's event loop."""
        await super().start(frame)
        await self._recognition.start()

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        return await self._recognition.update(delta)

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        state = self._recognition
        if state.speaking:
            return
        state.speaking = True
        record = await state.begin()
        if record is not None and self._prefix_format is not None and self._prefix:
            self._open_source(record, self._prefix_format, self._prefix_user)
            await self._append_audio(record, bytes(self._prefix))
        self._prefix.clear()

    def _open_source(self, record: _Utterance, audio_format: tuple[int, int], user_id: str):
        record.audio_format = audio_format
        record.user_id = user_id
        record.source = _AudioSource(audio_format[0] * audio_format[1] * 2 * 5)
        record.task = self.create_task(self._read(record), name=f"rime_stt_{record.sequence}")
        self._tasks.add(record.task)
        record.task.add_done_callback(self._tasks.discard)

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        state = self._recognition
        if not state.speaking:
            return
        state.speaking = False
        state.suppressed = False
        record, state.active = state.active, None
        self._prefix.clear()
        if record is not None and not record.discarded:
            record.speech_end = frame.timestamp - frame.stop_secs
            if record.source is None:
                record.done = True
                await state.publish()
            elif record.accepted_bytes % (cast(tuple[int, int], record.audio_format)[1] * 2):
                await self._fail_audio(record, RimeAudioFormatError("Incomplete PCM sample frame"))
            else:
                record.source.close()

    async def process_audio_frame(self, frame: InputAudioRawFrame, direction: FrameDirection):
        """Queue audio without waiting on SDK connection or upload."""
        state = self._recognition
        if (
            self.is_muted
            or state.closed
            or not self.is_usable
            or state.suppressed
            or not frame.audio
        ):
            return
        audio_format = (frame.sample_rate, frame.num_channels)
        user_id = frame.user_id if isinstance(frame, UserAudioRawFrame) else ""
        if (
            frame.sample_rate not in (8000, 16000, 24000, 48000)
            or frame.num_channels not in (1, 2)
            or (self._init_sample_rate is not None and frame.sample_rate != self._init_sample_rate)
        ):
            await self._fail_audio(state.active, RimeAudioFormatError("Unsupported input format"))
            return
        if state.speaking and state.active is None:
            await state.begin()
        record = state.active
        if record is not None:
            if record.source is None:
                self._open_source(record, audio_format, user_id)
            if record.audio_format != audio_format or record.user_id != user_id:
                await self._fail_audio(
                    record, RimeAudioFormatError("Audio format or speaker changed during speech")
                )
                return
            await self._append_audio(record, frame.audio)
        else:
            if audio_format != self._prefix_format or user_id != self._prefix_user:
                self._prefix.clear()
            self._prefix_format, self._prefix_user = audio_format, user_id
            self._prefix.extend(frame.audio)
            block_size = frame.num_channels * 2
            budget = frame.sample_rate * block_size
            if len(self._prefix) > budget:
                discard = ((len(self._prefix) - budget + block_size - 1) // block_size) * block_size
                del self._prefix[:discard]

    async def _append_audio(self, record: _Utterance, audio: bytes):
        if record.discarded:
            return
        assert record.source is not None and record.audio_format is not None
        rate, channels = record.audio_format
        try:
            if record.accepted_bytes + len(audio) > rate * channels * 2 * _MAX_AUDIO_SECONDS:
                raise RimeResourceLimitError("Speech segment exceeds 120 seconds")
            record.source.append(audio)
            record.accepted_bytes += len(audio)
        except RimeResourceLimitError as e:
            await self._fail_audio(record, e)

    async def _fail_audio(self, record: _Utterance | None, error: Exception):
        self._prefix.clear()
        self._recognition.suppressed = True
        if record is not None and record.source is not None:
            record.source.discard()
        await self._recognition.error(error, record, detail=str(error))
        if record is not None and record.task is not None and not record.task.done():
            task = self.create_task(self.cancel_task(record.task), name="rime_stt_discard")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _audio(self, record: _Utterance):
        assert record.source is not None and record.audio_format is not None
        rate, channels = record.audio_format
        async for audio in record.source:
            record.uploaded_seconds += len(audio) / (rate * channels * 2)
            yield audio

    async def _read(self, record: _Utterance):
        state = self._recognition
        assert state.client is not None and record.audio_format is not None
        settings = record.settings
        stream = None
        try:
            async with asyncio.timeout(state.timeout):
                stream = state.client.stt.stream(
                    self._audio(record),
                    language=cast(str, settings.language),
                    mode=TranscriptionMode(settings.mode),
                    context_terms=cast(Sequence[str], settings.context_terms),
                    output_contract=TranscriptionOutputContract(settings.output_contract),
                    input_format=PCMFormat(
                        sample_rate=cast(
                            Literal[8000, 16000, 24000, 48000], record.audio_format[0]
                        ),
                        channels=cast(Literal[1, 2], record.audio_format[1]),
                    ),
                    timeout=state.timeout,
                )
                async with stream:
                    async for update in stream:
                        if record.discarded or state.closed:
                            return
                        record.request_id = stream.request_id
                        if record.final is not None:
                            raise RimeStreamError("Update after final transcript")
                        record.text = (
                            record.text + update.text if update.kind == "delta" else update.text
                        )
                        if len(record.text.encode("utf-8")) > _MAX_TEXT_BYTES:
                            raise RimeResourceLimitError("Transcript exceeds one MiB")
                        if update.kind == "final":
                            if record.source is None or not record.source.closed:
                                raise RimeStreamError("Final transcript before source exhaustion")
                            record.received_at = time.time()
                            record.final = cast(
                                TranscriptionFrame,
                                state.frame(
                                    record, update.text, language=update.language, final=True
                                ),
                            )
                        else:
                            record.interim = cast(
                                InterimTranscriptionFrame, state.frame(record, record.text)
                            )
                            await state.publish()
                if record.final is None:
                    raise RimeStreamError("Stream ended without a final transcript")
                record.done = True
                await state.publish()
        except asyncio.CancelledError:
            record.discarded = record.done = True
            raise
        except Exception as e:
            if stream is not None:
                record.request_id = stream.request_id
            await state.error(e, record)
        finally:
            if record.source is not None:
                record.source.discard()
            if record.discarded and state.active is record:
                state.suppressed = True
            await state.metrics(record)
            await state.publish()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Accept raw PCM for an already opened utterance.

        Normal pipeline input uses process_audio_frame to retain audio metadata.
        """
        record = self._recognition.active
        if record is not None and record.source is not None:
            await self._append_audio(record, audio)
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handle STT muting while preserving physical VAD state."""
        if isinstance(frame, STTMuteFrame) and frame.mute:
            await self._discard()
        elif isinstance(frame, STTMuteFrame) and self.is_muted:
            self._recognition.suppressed = False
        await super().process_frame(frame, direction)

    async def _discard(self):
        state = self._recognition
        self._prefix.clear()
        for record in state.pending:
            record.discarded = record.done = True
            if record.source is not None:
                record.source.discard()
        state.pending.clear()
        state.active = None
        state.suppressed = False
        for task in list(self._tasks):
            await self.cancel_task(task)

    async def stop(self, frame: EndFrame):
        """Drain accepted speech within the shutdown deadline and release owned resources."""
        if self._recognition.speaking:
            await self._handle_vad_user_stopped_speaking(VADUserStoppedSpeakingFrame())
        try:
            async with asyncio.timeout(self._recognition.shutdown_timeout):
                if self._tasks:
                    await asyncio.gather(*list(self._tasks), return_exceptions=True)
        except TimeoutError:
            await self._recognition.error(RimeTimeoutError("Shutdown deadline expired"))
        finally:
            await self._discard()
            await self._recognition.close()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Discard in-progress recognition and close an owned SDK client."""
        self._recognition.closed = True
        await self._discard()
        await self._recognition.close()
        await super().cancel(frame)

    async def cleanup(self):
        """Release SDK resources, including after partial startup."""
        self._recognition.closed = True
        await self._discard()
        await self._recognition.close()
        await super().cleanup()
