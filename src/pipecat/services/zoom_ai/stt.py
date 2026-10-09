#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Zoom Scribe speech-to-text services.

Two services, matching the two Scribe processing modes:

- :class:`ZoomScribeLiveSTTService` (Scribe Live): streaming recognition over a
  WebSocket. Scribe's own voice activity detection ends each turn and returns
  its final transcript.
- :class:`ZoomScribeFastSTTService` (Scribe Fast): one HTTP request per
  utterance, segmented by the pipeline's VAD.

See https://developers.zoom.us/docs/ai-services/scribe/ for the API.

Requests identify the integration to Zoom with a ``User-Agent`` header,
``ZoomAI/1.0 (integration=Pipecat; plugin=pipecat-ai/<version>)``. It
contains only the integration name and version, never user data.
"""

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from loguru import logger
from websockets.protocol import State

from pipecat import version as pipecat_version
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import ZOOM_SCRIBE_FAST_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService, WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.errors import ErrorCategory, classify_http_status_code
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven

LIVE_URL = "wss://api.zoom.us/v2/aiservices/scribe/live"
FAST_URL = "https://api.zoom.us/v2/aiservices/scribe/transcribe"

# Scribe Live accepts 16 kHz mono PCM16 only.
SCRIBE_SAMPLE_RATE = 16000
# 50 ms of audio per WebSocket message. The docs suggest about 100 ms; smaller
# messages keep Scribe's end-of-speech detection, and so each turn, a little faster.
SEND_CHUNK_BYTES = SCRIBE_SAMPLE_RATE * 2 // 20
# Silence that ends a Scribe Live turn, tuned for voice agents.
DEFAULT_SILENCE_DURATION_MS = 350
# How long stop() waits for Scribe to send the last transcript and session.closed.
SESSION_CLOSE_TIMEOUT_SECS = 2.0

# Scribe adds speaker or channel tags to transcripts when diarization is on.
_SPEAKER_TAG = re.compile(r"\[(?:speaker[ _]?\d+|spk_\d+|channel\d+)\]\s*", re.IGNORECASE)


def _clean(text: str | None) -> str:
    return _SPEAKER_TAG.sub("", text or "").strip()


def _headers(api_key: str) -> dict[str, str]:
    # Identifies this integration to Zoom (name and version only, no user data).
    user_agent = f"ZoomAI/1.0 (integration=Pipecat; plugin=pipecat-ai/{pipecat_version()})"
    return {"Authorization": f"Bearer {api_key}", "User-Agent": user_agent}


def _frame_language(value: object) -> Language | None:
    """Convert a stored Scribe locale back to a Language for transcription frames."""
    if isinstance(value, Language):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return Language(value)
    except ValueError:
        return None


SUPPORTED_LANGUAGES_URL = "https://developers.zoom.us/docs/ai-services/scribe/#supported-languages"

# Scribe's supported locales, as in its docs (SUPPORTED_LANGUAGES_URL).
_LANGUAGE_MAP = {
    Language.AR: "ar-SA",
    Language.AR_AE: "ar-AE",
    Language.AR_SA: "ar-SA",
    Language.DE: "de-DE",
    Language.DE_DE: "de-DE",
    Language.EN: "en-US",
    Language.EN_US: "en-US",
    Language.ES: "es-ES",
    Language.ES_ES: "es-ES",
    Language.FR: "fr-FR",
    Language.FR_FR: "fr-FR",
    Language.IT: "it-IT",
    Language.IT_IT: "it-IT",
    Language.JA: "ja-JP",
    Language.JA_JP: "ja-JP",
    Language.PT: "pt-BR",
    Language.PT_BR: "pt-BR",
    Language.PT_PT: "pt-PT",
    Language.ZH: "zh-CN",
    Language.ZH_CN: "zh-CN",
}


def language_to_zoom_scribe_language(language: Language) -> str:
    """Convert a Language to a Scribe locale, e.g. ``Language.EN`` to ``"en-US"``.

    Args:
        language: The language to convert.

    Returns:
        The Scribe locale.

    Raises:
        ValueError: If this version of the service doesn't support the language;
            see https://developers.zoom.us/docs/ai-services/scribe/#supported-languages
    """
    locale = _LANGUAGE_MAP.get(language)
    if locale is None:
        raise ValueError(
            f"Language {language.value!r} isn't supported by this version of the Zoom Scribe "
            f"service. See Scribe's supported languages: {SUPPORTED_LANGUAGES_URL}"
        )
    return locale


@dataclass
class ZoomScribeLiveSTTSettings(STTSettings):
    """Settings for :class:`ZoomScribeLiveSTTService`.

    Changing a setting reconnects with a new Scribe session.

    Parameters:
        silence_duration_ms: Milliseconds of silence that end a turn in Scribe's
            voice activity detection. Defaults to 350, tuned for voice agents;
            lower is faster, higher keeps sentences with longer pauses together.
            For transcription without an agent, a longer value such as 600 can
            improve accuracy. ``None`` uses Scribe's default (see the Scribe docs).
        diarization: Whether Scribe labels each completed transcript with a
            speaker. Off by default. The labels are in the Scribe event attached
            as ``TranscriptionFrame.result``: ``speaker``, plus ``speaker_segments``
            with each one's span when several speakers spoke. Scribe's speaker
            tags are removed from the transcript text.
    """

    silence_duration_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    diarization: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class ZoomScribeLiveSTTService(WebsocketSTTService):
    """Zoom Scribe Live streaming speech-to-text service.

    Streams 16 kHz PCM16 audio to Scribe Live over a WebSocket. Scribe's own
    voice activity detection decides when the user starts and stops speaking:
    its ``speech_started`` event broadcasts a
    :class:`ProposedUserStartedSpeakingFrame`, and each completed transcript is
    pushed as a :class:`TranscriptionFrame` followed by a
    :class:`ProposedUserStoppedSpeakingFrame`. ``service_metadata_frame()``
    recommends :class:`~pipecat.turns.user_turn_strategies.ExternalUserTurnStrategies`
    so the user aggregator resolves those proposals instead of running local
    VAD. Scribe Live returns final transcripts only. A transcript that arrives
    without a preceding ``speech_started`` (for example right after a
    reconnect) opens its own turn, so it is never dropped; a turn whose
    transcript comes back empty ends with no transcription.

    Example::

        stt = ZoomScribeLiveSTTService(
            api_key=os.getenv("ZOOM_SCRIBE_API_KEY"),
            settings=ZoomScribeLiveSTTService.Settings(language=Language.EN_US),
        )
    """

    Settings = ZoomScribeLiveSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = LIVE_URL,
        sample_rate: int | None = None,
        should_interrupt: bool = True,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Zoom Scribe Live STT service.

        Args:
            api_key: Zoom AI Services API key or JWT. It is sent on every request and
                reconnect, so it must stay valid for as long as the service runs.
            url: Scribe Live WebSocket URL.
            sample_rate: Input audio sample rate. Audio is resampled to 16 kHz.
            should_interrupt: Whether a user turn started by Scribe interrupts the bot.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent WebsocketSTTService.
        """
        default_settings = self.Settings(
            model=None,
            language=Language.EN_US,
            silence_duration_ms=DEFAULT_SILENCE_DURATION_MS,
            diarization=False,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            # Scribe ends a session after 30 seconds without audio.
            keepalive_timeout=10,
            keepalive_interval=5,
            settings=default_settings,
            **kwargs,
        )
        self._api_key = api_key
        self._url = url
        self._should_interrupt = should_interrupt
        self._resampler = create_stream_resampler()
        self._send_buffer = bytearray()
        self._user_turn_open = False
        self._receive_task = None
        self._session_closed = asyncio.Event()
        self._awaiting_final = False

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports metrics.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language to a Scribe locale.

        Args:
            language: The language to convert.

        Returns:
            The Scribe locale. Regional variants Scribe doesn't list resolve to
            the locale of their base language.
        """
        return language_to_zoom_scribe_language(language)

    @property
    def supports_ttfs(self) -> bool:
        """Scribe defines the turn boundary, so there is no TTFS to report."""
        return False

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend external turn strategies, since Scribe detects the turns.

        Returns:
            The STT metadata frame.
        """
        frame = super().service_metadata_frame()
        frame.user_turn_strategies = ExternalUserTurnStrategies(
            enable_interruptions=self._should_interrupt
        )
        return frame

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta and reconnect so the new session config takes effect."""
        changed = await super()._update_settings(delta)
        if changed:
            await self._request_reconnect()
        return changed

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect to Scribe Live.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Close the Scribe session, then stop the service.

        Args:
            frame: The end frame.
        """
        # Scribe answers session.close by closing the socket; mark the shutdown first so
        # the receive loop doesn't treat that as a dropped connection and reconnect.
        self._disconnecting = True
        await self._flush_audio()
        if self._websocket and self._websocket.state is State.OPEN:
            await self._send_json({"type": "session.close"})
            # As the Live docs describe: keep reading until Scribe sends the remaining
            # final transcript and session.closed, so the last utterance isn't lost.
            try:
                await asyncio.wait_for(self._session_closed.wait(), SESSION_CLOSE_TIMEOUT_SECS)
            except TimeoutError:
                logger.debug(f"{self} Scribe did not confirm session.close; closing")
        await super().stop(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to Scribe Live.

        Transcriptions are pushed from the receive task, not yielded here.

        Args:
            audio: Raw PCM16 mono audio at the pipeline sample rate.

        Yields:
            None.
        """
        if self.sample_rate != SCRIBE_SAMPLE_RATE:
            audio = await self._resampler.resample(audio, self.sample_rate, SCRIBE_SAMPLE_RATE)
        self._send_buffer.extend(audio)
        if len(self._send_buffer) >= SEND_CHUNK_BYTES:
            await self._flush_audio()
        yield None

    async def _flush_audio(self):
        if not self._send_buffer:
            return
        chunk, self._send_buffer = bytes(self._send_buffer), bytearray()
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(chunk)
            except Exception as e:
                logger.warning(f"{self} failed to send audio: {e}")

    async def _send_json(self, message: dict):
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(json.dumps(message))
            except Exception as e:
                logger.warning(f"{self} failed to send {message.get('type')}: {e}")

    def _session_update(self) -> dict[str, Any]:
        update: dict[str, Any] = {
            "type": "session.update",
            "language": self._settings.language or "en-US",
            "audio": {"format": "pcm16"},
        }
        config: dict[str, Any] = {}
        if self._settings.silence_duration_ms is not None:
            config["turn_detection"] = {
                "type": "acoustic_vad",
                "silence_duration_ms": self._settings.silence_duration_ms,
            }
        if self._settings.diarization:
            config["diarization"] = True
        if config:
            update["config"] = config
        return update

    async def _connect(self):
        await self._connect_websocket()
        await super()._connect()
        # Started even when the connection failed: with no socket the receive
        # loop goes straight to its reconnect path, which retries the connect.
        if not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error), name="receive"
            )

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Zoom Scribe Live")
            websocket = await self._websocket_connect(
                self._url,
                subprotocols=["live-asr"],
                additional_headers=_headers(self._api_key),
            )
            self._websocket = websocket
            self._session_closed.clear()
            await websocket.send(json.dumps(self._session_update()))
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(
                error_msg=f"Unable to connect to Zoom Scribe Live: {e}", exception=e
            )

    async def _disconnect_websocket(self):
        if not self._websocket:
            return
        try:
            logger.debug("Disconnecting from Zoom Scribe Live")
            await self._websocket.close()
        except Exception as e:
            logger.warning(f"{self} error closing websocket: {e}")
        finally:
            self._websocket = None
            self._send_buffer.clear()
            self._awaiting_final = False
            self._can_reconnect = True
            await self._end_open_turn()
            await self._call_event_handler("on_disconnected")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _receive_messages(self):
        if not self._websocket:
            return
        async for message in self._websocket:
            if not isinstance(message, str):
                continue
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} ignoring a non-JSON message")
                continue
            if isinstance(event, dict):
                await self._handle_event(event)

    async def _handle_event(self, event: dict[str, Any]):
        kind = event.get("type")
        if kind == "input_audio_buffer.speech_started":
            # Hold settings-change reconnects until the turn ends, even without a local VAD.
            self._can_reconnect = False
            if not self._user_turn_open:
                self._user_turn_open = True
                await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        elif kind == "input_audio_buffer.speech_stopped":
            # Processing time: from Scribe's end of speech to its final transcript.
            self._awaiting_final = True
            await self.start_processing_metrics()
        elif kind == "transcription.completed":
            if self._awaiting_final:
                self._awaiting_final = False
                await self.stop_processing_metrics()
            await self._push_final_transcript(event)
        elif kind == "session.updated":
            # Scribe never rejects diarization; it reports whether labels will come.
            if self._settings.diarization and event.get("diarization_enabled") is False:
                logger.warning(f"{self} Scribe can't label speakers for this session")
        elif kind == "error":
            error = event.get("error")
            await self._handle_error_event(error if isinstance(error, dict) else {})
        elif kind == "session.closed":
            logger.debug(f"{self} session closed: {event.get('reason')}")
            self._session_closed.set()

    async def _handle_error_event(self, error: dict[str, Any]):
        message = f"Zoom Scribe Live error {error.get('code')}: {error.get('message')}"
        if not error.get("fatal"):
            logger.warning(f"{self} {message}")
            return
        # A config error (e.g. UNSUPPORTED_LANGUAGE) would fail again, so it isn't retried.
        code = str(error.get("code") or "").upper()
        is_config_error = "INVALID" in code or "UNSUPPORTED" in code
        category = ErrorCategory.INVALID_REQUEST if is_config_error else ErrorCategory.UNKNOWN
        await self.push_error(error_msg=message, category=category)

    async def _push_final_transcript(self, event: dict[str, Any]):
        text = _clean(event.get("transcript"))
        if text:
            if not self._user_turn_open:
                # No speech_started for this transcript (e.g. it was lost in a
                # reconnect): open the turn so the aggregator doesn't drop it.
                self._user_turn_open = True
                await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
            language = _frame_language(self._settings.language)
            await self.emit_stt_usage_metrics()
            await self.push_frame(
                TranscriptionFrame(
                    text=text,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                    language=language,
                    result=event,
                    finalized=True,
                )
            )
            await self._handle_transcription(text, True, language)
        await self._end_open_turn()
        # Scribe's turn is over: apply a settings change held since speech_started. In its
        # own task, because the reconnect cancels the receive task this runs in.
        self._can_reconnect = True
        if self._need_reconnect:
            self.create_task(self._reconnect(), name="settings_reconnect")

    async def _end_open_turn(self):
        if self._user_turn_open:
            self._user_turn_open = False
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)


@dataclass
class ZoomScribeFastSTTSettings(STTSettings):
    """Settings for :class:`ZoomScribeFastSTTService`.

    Parameters:
        diarization: Whether Scribe labels each transcript segment with a
            speaker. Off by default. The labels are in the segments of the
            Scribe response attached as ``TranscriptionFrame.result``. Scribe's
            speaker tags are removed from the transcript text.
        word_time_offsets: Whether Scribe returns word-level timestamps. Off by
            default. The words are in the segments of ``TranscriptionFrame.result``.
    """

    diarization: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    word_time_offsets: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class ZoomScribeFastSTTService(SegmentedSTTService):
    """Zoom Scribe Fast speech-to-text service.

    Transcribes one utterance per HTTP request. The pipeline's VAD cuts the
    audio into utterances, and each one is uploaded to Scribe Fast as a WAV
    file, so a VAD is required.

    Example::

        stt = ZoomScribeFastSTTService(api_key=os.getenv("ZOOM_SCRIBE_API_KEY"))
    """

    Settings = ZoomScribeFastSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = FAST_URL,
        aiohttp_session: aiohttp.ClientSession | None = None,
        timeout_secs: float = 10.0,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = ZOOM_SCRIBE_FAST_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Zoom Scribe Fast STT service.

        Args:
            api_key: Zoom AI Services API key or JWT. It is sent with every request, so it
                must stay valid for as long as the service runs.
            url: Scribe Fast transcription URL.
            aiohttp_session: Optional session to reuse. One is created and owned
                by the service otherwise.
            timeout_secs: Request timeout in seconds.
            settings: Runtime-updatable settings.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to the parent SegmentedSTTService.
        """
        default_settings = self.Settings(
            model=None, language=Language.EN_US, diarization=False, word_time_offsets=False
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )
        self._api_key = api_key
        self._url = url
        self._session = aiohttp_session
        self._owns_session = aiohttp_session is None
        self._timeout = aiohttp.ClientTimeout(total=timeout_secs)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports metrics.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language to a Scribe locale.

        Args:
            language: The language to convert.

        Returns:
            The Scribe locale. Regional variants Scribe doesn't list resolve to
            the locale of their base language.
        """
        return language_to_zoom_scribe_language(language)

    async def cleanup(self):
        """Close the HTTP session if the service created it."""
        await super().cleanup()
        if self._owns_session and self._session:
            await self._session.close()
            self._session = None

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe one utterance with Scribe Fast.

        Args:
            audio: The utterance as WAV bytes.

        Yields:
            A TranscriptionFrame, or an ErrorFrame if the request fails.
        """
        config: dict[str, Any] = {"language": self._settings.language or "en-US"}
        if self._settings.diarization:
            config["diarization"] = True
        if self._settings.word_time_offsets:
            config["word_time_offsets"] = True
        # Upload the utterance as multipart form data (`file` + `config`).
        form = aiohttp.FormData()
        form.add_field("file", audio, filename="utterance.wav", content_type="audio/wav")
        form.add_field("config", json.dumps(config))

        if self._session is None:
            self._session = aiohttp.ClientSession()
        await self.start_processing_metrics()
        try:
            async with self._session.post(
                self._url, data=form, headers=_headers(self._api_key), timeout=self._timeout
            ) as response:
                body = await response.text()
                status = response.status
            result = json.loads(body) if status == 200 else None
        except Exception as e:
            yield ErrorFrame(error=f"Zoom Scribe Fast request failed: {e}", exception=e)
            return
        finally:
            await self.stop_processing_metrics()

        if result is None:
            yield ErrorFrame(
                error=f"Zoom Scribe Fast error {status}: {body[:500]}",
                category=classify_http_status_code(status),
            )
            return

        text = _clean((result.get("result") or {}).get("text_display"))
        if text:
            language = _frame_language(self._settings.language)
            await self._handle_transcription(text, True, language)
            yield TranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=language,
                result=result,
            )
