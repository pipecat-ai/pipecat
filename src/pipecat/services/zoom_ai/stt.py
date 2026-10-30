#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Zoom Scribe speech-to-text service.

:class:`ZoomAISTTService` streams audio to Scribe Live over a WebSocket. Scribe's
own voice activity detection ends each turn and returns its final transcript.

Settings map to the Scribe API fields of the same name. For what each feature
does, and its availability, pricing and enablement, see
https://developers.zoom.us/docs/ai-services/scribe/ and https://www.zoom.ai/
"""

import asyncio
import copy
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from websockets.exceptions import InvalidStatus
from websockets.protocol import State

from pipecat import version as pipecat_version
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven

# Scribe Live accepts 16 kHz mono PCM16 only.
SCRIBE_SAMPLE_RATE = 16000
# Silence that ends a Scribe Live turn, tuned for voice agents.
DEFAULT_SILENCE_DURATION_MS = 350
# How long stop() waits for Scribe to send the last transcript and session.closed.
SESSION_CLOSE_TIMEOUT_SECS = 2.0
# When input stops mid-turn (e.g. the user muted), Scribe's VAD hears no silence and
# the turn stays open. After this long without input, send enough silence to end it.
INPUT_GAP_SECS = 1.0
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")

# Scribe adds speaker or channel tags to transcripts when diarization is on.
_SPEAKER_TAG = re.compile(r"\[(?:speaker[ _]?\d+|spk_\d+|channel\d+)\]\s*", re.IGNORECASE)


def _clean(text: str | None) -> str:
    return _SPEAKER_TAG.sub("", text or "").strip()


def _tracking_id() -> str:
    """A new ``x-zm-trackingid``, which Zoom support uses to find a session."""
    return f"zoom_ai_community_pipecat_{uuid.uuid4()}"


def _headers(api_key: str, tracking_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": f"pipecat/{pipecat_version()}",
        "x-zm-trackingid": tracking_id,
    }


def _describe_refusal(exception: Exception) -> str | None:
    """Describe Zoom refusing the handshake, if that is what failed.

    Zoom explains a refusal (a bad API key, say) in the JSON response body,
    which the exception's own message leaves out.
    """
    if isinstance(exception, InvalidStatus):
        response = exception.response
        try:
            reason = json.loads(response.body).get("message")
        except (TypeError, ValueError, AttributeError):  # no body, or not a JSON object
            reason = None
        if reason:
            return f"Zoom refused the connection (HTTP {response.status_code}): {reason}"
    return None


def _check_url(url: str) -> str:
    """Return ``url``, warning if it sends the API key and audio unencrypted to a remote host."""
    parsed = urlparse(url)
    if parsed.scheme.lower() == "ws" and parsed.hostname not in _LOOPBACK_HOSTS:
        logger.warning(
            "Zoom Scribe url is not encrypted (ws://); the API key and audio are sent in plaintext"
        )
    return url


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """``base`` with ``extra`` merged in, nested objects merged key by key; ``extra`` wins."""
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


@dataclass(frozen=True)
class _RequestView:
    """What the service acts on, read from the final request (settings plus ``extra``)."""

    diarization: bool
    text_polishing: bool
    language_detection: bool
    silence_duration_ms: int | None

    @classmethod
    def of(cls, request: dict[str, Any]) -> "_RequestView":
        config = request.get("config") or {}
        polishing = (request.get("outputs") or {}).get("text_polishing") or {}
        silence = (request.get("turn_detection") or {}).get("silence_duration_ms")
        return cls(
            diarization=bool(config.get("diarization")),
            text_polishing=bool(polishing.get("enabled")),
            language_detection=bool(config.get("language_detection")),
            silence_duration_ms=silence if isinstance(silence, int) else None,
        )


def _polished_text(payload: dict[str, Any]) -> str | None:
    """Scribe's polished text (``outputs.text_polishing.text``), or None if absent or failed.

    When polishing fails, Scribe still returns the transcription and puts a
    ``[polishing_failed: <reason>]`` marker in place of the polished text.
    """
    polishing = (payload.get("outputs") or {}).get("text_polishing") or {}
    text = polishing.get("text")
    if not isinstance(text, str) or text.lstrip().startswith("[polishing_failed"):
        return None
    return _clean(text)


def _reported_language(
    scribe_language: object, detecting: bool, configured: object
) -> Language | None:
    """The frame language: Scribe's, else None when detecting, else the configured one.

    With language detection on and no language from Scribe, the language isn't known, so
    it is left out rather than guessed.
    """
    if isinstance(scribe_language, str) and scribe_language:
        return _frame_language(scribe_language)
    return None if detecting else _frame_language(configured)


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


_LANGUAGE_MAP = {
    # zoom-scribe-multilingual (and whisper)
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
    Language.KO: "ko-KR",
    Language.KO_KR: "ko-KR",
    Language.PT: "pt-BR",
    Language.PT_BR: "pt-BR",
    Language.PT_PT: "pt-PT",
    Language.ZH: "zh-CN",
    Language.ZH_CN: "zh-CN",
    # whisper only
    Language.BN: "bn-IN",
    Language.BN_IN: "bn-IN",
    Language.CA: "ca-ES",
    Language.CA_ES: "ca-ES",
    Language.CS: "cs-CZ",
    Language.CS_CZ: "cs-CZ",
    Language.DA: "da-DK",
    Language.DA_DK: "da-DK",
    Language.EL: "el-GR",
    Language.EL_GR: "el-GR",
    Language.ET: "et-EE",
    Language.ET_EE: "et-EE",
    Language.FA: "fa-IR",
    Language.FA_IR: "fa-IR",
    Language.FI: "fi-FI",
    Language.FI_FI: "fi-FI",
    Language.FIL: "tl-PH",
    Language.FIL_PH: "tl-PH",
    Language.HE: "he-IL",
    Language.HE_IL: "he-IL",
    Language.HI: "hi-IN",
    Language.HI_IN: "hi-IN",
    Language.HU: "hu-HU",
    Language.HU_HU: "hu-HU",
    Language.ID: "id-ID",
    Language.ID_ID: "id-ID",
    Language.MS: "ms-MY",
    Language.MS_MY: "ms-MY",
    Language.NL: "nl-NL",
    Language.NL_NL: "nl-NL",
    Language.PL: "pl-PL",
    Language.PL_PL: "pl-PL",
    Language.RO: "ro-RO",
    Language.RO_RO: "ro-RO",
    Language.RU: "ru-RU",
    Language.RU_RU: "ru-RU",
    Language.SV: "sv-SE",
    Language.SV_SE: "sv-SE",
    Language.TA: "ta-IN",
    Language.TA_IN: "ta-IN",
    Language.TE: "te-IN",
    Language.TE_IN: "te-IN",
    Language.TH: "th-TH",
    Language.TH_TH: "th-TH",
    Language.TL: "tl-PH",
    Language.TR: "tr-TR",
    Language.TR_TR: "tr-TR",
    Language.UK: "uk-UA",
    Language.UK_UA: "uk-UA",
    Language.VI: "vi-VN",
    Language.VI_VN: "vi-VN",
    Language.YUE: "zh-HK",
    Language.ZH_HK: "zh-HK",
}


def language_to_zoom_ai_language(language: Language) -> str:
    """Convert a Language to a Scribe locale, e.g. ``Language.EN`` to ``"en-US"``.

    A regional variant Scribe doesn't list resolves to the locale of its base
    language, e.g. ``Language.EN_GB`` to ``"en-US"``. For the supported
    languages of each model, see
    https://developers.zoom.us/docs/ai-services/scribe/speech-understanding/

    Args:
        language: The language to convert.

    Returns:
        The Scribe locale.
    """
    return resolve_language(language, _LANGUAGE_MAP, use_base_code=True)


@dataclass
class ZoomAISTTSettings(STTSettings):
    """Settings for :class:`ZoomAISTTService`.

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
        language_detection: Scribe language detection mode
            (``config.language_detection.mode``). ``None`` turns it off. A
            detected language Scribe reports is in ``TranscriptionFrame.language``;
            when Scribe reports none, it is ``None``.
        vocabulary: Words and phrases that customize recognition
            (``config.vocabulary.phrases``).
        text_polishing: Scribe's text polishing (``outputs.text_polishing``).
            The polished text is the transcript, and the recognized text is in
            the Scribe event attached as ``TranscriptionFrame.result``
            (``transcript``); if polishing fails, the recognized text is used.
            Off by default.

    ``extra`` holds Scribe request fields this class has no setting for, shaped
    like the request (e.g. ``{"config": {"profanity_filter": True}}``). It is
    merged into the ``session.update`` over the settings above, except that
    ``audio.format`` stays ``pcm16``.
    """

    silence_duration_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    diarization: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    language_detection: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    vocabulary: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_polishing: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class ZoomAISTTService(WebsocketSTTService):
    """Zoom Scribe Live streaming speech-to-text service.

    Streams 16 kHz PCM16 audio to Scribe Live over a WebSocket. Scribe's own
    voice activity detection decides when the user starts and stops speaking:
    its ``speech_started`` event broadcasts a
    :class:`ProposedUserStartedSpeakingFrame`, and each completed transcript is
    pushed as a :class:`TranscriptionFrame` followed by a
    :class:`ProposedUserStoppedSpeakingFrame`. ``service_metadata_frame()``
    recommends :class:`~pipecat.turns.user_turn_strategies.ExternalUserTurnStrategies`
    so the user aggregator resolves those proposals instead of running local
    VAD. The service pushes final transcripts only. A transcript that arrives
    without a preceding ``speech_started`` (for example right after a
    reconnect) opens its own turn, so it is never dropped; a turn whose
    transcript comes back empty ends with no transcription.

    Example::

        stt = ZoomAISTTService(
            api_key=os.getenv("ZOOM_AI_API_KEY"),
            settings=ZoomAISTTService.Settings(language=Language.EN_US),
        )
    """

    Settings = ZoomAISTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.zoom.us/v2/aiservices/scribe/live",
        sample_rate: int | None = None,
        should_interrupt: bool = True,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Zoom Scribe Live STT service.

        Args:
            api_key: Zoom AI Services API key or JWT. It is sent on every request and
                reconnect, so it must stay valid for as long as the service runs.
            url: Scribe Live WebSocket URL. A plaintext ``ws://`` URL to a host other
                than localhost logs a warning.
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
            language_detection=None,
            vocabulary=None,
            text_polishing=False,
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
        self._url = _check_url(url)
        self._should_interrupt = should_interrupt
        self._resampler = create_stream_resampler()
        self._user_turn_open = False
        self._receive_task = None
        self._session_closed = asyncio.Event()
        self._session_updated = False  # Scribe accepted this session's settings
        self._tracking_id: str | None = None  # the current session's x-zm-trackingid
        self._input_gap_task = None
        self._last_input_time = time.monotonic()
        self._gap_filled = False  # silence already sent for the current input gap
        # What the current session was configured with (settings plus ``extra``).
        self._view = _RequestView.of(self._session_update())

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
        return language_to_zoom_ai_language(language)

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
        self._last_input_time = time.monotonic()
        self._gap_filled = False
        await self._send_audio(audio)
        yield None

    async def _send_keepalive(self, silence: bytes):
        """Send keepalive silence at Scribe's sample rate."""
        await self._send_audio(silence)
        # Keepalive silence is real audio submitted to the service, so it
        # counts toward usage.
        self._record_stt_audio_usage(silence)

    async def _input_gap_handler(self):
        """End a Scribe turn that input stopped in the middle of (e.g. the user muted).

        Scribe's VAD ends a turn after ``silence_duration_ms`` of silence, but no audio is
        not silence. Once input has stopped for ``INPUT_GAP_SECS`` with a turn open, send
        that much silence (plus 100 ms), once per gap.
        """
        while True:
            await asyncio.sleep(INPUT_GAP_SECS / 2)
            if (
                not self._user_turn_open
                or self._gap_filled
                or time.monotonic() - self._last_input_time < INPUT_GAP_SECS
            ):
                continue
            self._gap_filled = True
            window_ms = self._view.silence_duration_ms or 1000
            silence = bytes(self.sample_rate * (window_ms + 100) // 1000 * 2)
            await self._send_audio(silence)
            # Gap silence is real audio submitted to the service, so it counts toward usage.
            self._record_stt_audio_usage(silence)

    async def _send_audio(self, audio: bytes):
        """Send pipeline-rate audio to Scribe, resampled to 16 kHz."""
        if self.sample_rate != SCRIBE_SAMPLE_RATE:
            audio = await self._resampler.resample(audio, self.sample_rate, SCRIBE_SAMPLE_RATE)
        if audio and self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(audio)
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
        if self._settings.model:
            config["model"] = self._settings.model
        if self._settings.language_detection:
            config["language_detection"] = {"mode": self._settings.language_detection}
        if self._settings.diarization:
            config["diarization"] = True
        if self._settings.vocabulary:
            config["vocabulary"] = {"phrases": list(self._settings.vocabulary)}
        if config:
            update["config"] = config
        if self._settings.silence_duration_ms is not None:
            update["turn_detection"] = {"silence_duration_ms": self._settings.silence_duration_ms}
        if self._settings.text_polishing:
            update["outputs"] = {"text_polishing": {"enabled": True}}
        update = _merge(update, self._settings.extra)
        # Fields the service depends on, whatever ``extra`` says.
        update["type"] = "session.update"
        update["audio"] = {**(update.get("audio") or {}), "format": "pcm16"}
        polishing = (update.get("outputs") or {}).get("text_polishing")
        if isinstance(polishing, dict) and polishing.get("enabled"):
            # Live returns only the polished text unless the transcript is asked for too,
            # and the transcript is the fallback when polishing fails.
            polishing["include_transcript"] = True
        return update

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        # Started even when the connection failed: with no socket the receive
        # loop goes straight to its reconnect path, which retries the connect.
        if not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error), name="receive"
            )
        if not self._input_gap_task:
            self._input_gap_task = self.create_task(self._input_gap_handler(), name="input_gap")

    async def _disconnect(self):
        await super()._disconnect()
        if self._input_gap_task:
            await self.cancel_task(self._input_gap_task)
            self._input_gap_task = None
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            self._tracking_id = _tracking_id()
            logger.debug(f"Connecting to Zoom Scribe Live (tracking_id={self._tracking_id})")
            websocket = await self._websocket_connect(
                self._url,
                subprotocols=["live-asr"],
                additional_headers=_headers(self._api_key, self._tracking_id),
            )
            self._websocket = websocket
            self._session_closed.clear()
            self._session_updated = False
            update = self._session_update()
            self._view = _RequestView.of(update)
            await websocket.send(json.dumps(update))
            logger.info(f"{self} connected to Zoom Scribe Live (tracking_id={self._tracking_id})")
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            refusal = _describe_refusal(e)
            await self.push_error(
                error_msg=refusal or f"Unable to connect to Zoom Scribe Live: {e}", exception=e
            )
            await self._call_event_handler("on_connection_error", refusal or f"{e}")

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
        elif kind == "transcription.completed":
            await self._push_final_transcript(event)
        elif kind == "session.updated":
            self._session_updated = True
            # Scribe never rejects diarization; it reports whether labels will come.
            if self._view.diarization and event.get("diarization_enabled") is False:
                logger.warning(f"{self} Scribe can't label speakers for this session")
        elif kind == "error":
            error = event.get("error")
            await self._handle_error_event(error if isinstance(error, dict) else {})
        elif kind == "session.closed":
            logger.debug(f"{self} session closed: {event.get('reason')}")
            self._session_closed.set()

    async def _handle_error_event(self, error: dict[str, Any]):
        message = (
            f"Zoom Scribe Live error {error.get('code')}: {error.get('message')} "
            f"(tracking_id={self._tracking_id})"
        )
        if error.get("fatal") and not self._session_updated:
            # Scribe rejects a session's settings with a fatal error in place of
            # session.updated, then closes; a new session would be rejected the same way.
            await self.push_error(error_msg=message, category=ErrorCategory.INVALID_REQUEST)
        else:
            await self.push_error(error_msg=message)

    async def _push_final_transcript(self, event: dict[str, Any]):
        text = _clean(event.get("transcript"))
        if self._view.text_polishing:
            polished = _polished_text(event)
            if polished is not None:
                text = polished
        if text:
            if not self._user_turn_open:
                # No speech_started for this transcript (e.g. it was lost in a
                # reconnect): open the turn so the aggregator doesn't drop it.
                self._user_turn_open = True
                await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
            language = _reported_language(
                event.get("language"), self._view.language_detection, self._settings.language
            )
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
