#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Live speech-to-text service for Pipecat.

``GeminiSTTService`` streams audio to a Gemini Live transcription model (e.g.
``gemini-3.5-transcribe-live``) over the Gemini Live API, with automatic
language detection, language hints, and adaptation phrases.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.google.stt import language_to_google_stt_language
from pipecat.services.google.utils import update_google_client_http_options
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import GEMINI_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, is_given

try:
    from google.genai import Client
    from google.genai.live import AsyncSession
    from google.genai.types import (
        AudioTranscriptionConfig,
        Blob,
        HttpOptions,
        LiveConnectConfig,
        LiveServerMessage,
        Modality,
        Transcription,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Google AI, you need to `uv add "pipecat-ai[google]"`.')
    raise ImportError(f"Missing module: {e}") from e

# The Gemini transcription config types require google-genai >= 2.9.0. Checked
# at GeminiSTTService construction time so the rest of the Google services stay
# usable with older google-genai versions.
try:
    from google.genai.types import LanguageAuto, LanguageHints
except ImportError:
    LanguageAuto = None
    LanguageHints = None


# Connection management constants
MAX_CONSECUTIVE_FAILURES = 3
CONNECTION_ESTABLISHED_THRESHOLD = 10.0  # seconds


@dataclass
class GeminiSTTSettings(STTSettings):
    """Settings for GeminiSTTService.

    Language configuration maps to the Live API's ``AudioTranscriptionConfig``:
    when ``language``/``languages`` are set they are sent as language hints;
    otherwise the model detects the language automatically. Language hints and
    automatic detection are mutually exclusive — if both are configured, hints
    take precedence.

    Parameters:
        languages: List of ``Language`` enums used as language hints for the
            expected languages in the audio (e.g. ``[Language.ES_ES]``).
        language_auto: Enable automatic language detection. ``None`` (the
            default) auto-detects unless language hints are given. Set to
            ``False`` to disable auto-detection without providing hints.
        adaptation_phrases: Phrases to bias recognition toward, improving
            accuracy for domain-specific terms (e.g. ``["oatmilk"]``).
    """

    languages: list[Language] | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    language_auto: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    adaptation_phrases: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class GeminiSTTService(STTService):
    """Streaming speech-to-text service using Gemini Live transcription models.

    Streams raw PCM audio to a Gemini Live transcription model (e.g.
    ``gemini-3.5-transcribe-live``) over the Gemini Live API and pushes
    interim and final transcription frames as results arrive. Supports
    automatic language detection, language hints, and adaptation phrases.

    The model detects utterance boundaries itself, and when the pipeline's VAD
    signals end of speech the service additionally sends an audio-stream-end
    signal to flush the utterance, so the final transcript is produced promptly
    instead of waiting for the model to decide the utterance ended. Without an
    upstream VAD the model finalizes on its own schedule.

    Audio is sent at the pipeline's input sample rate; the model performs best
    with 16 kHz mono PCM.
    """

    Settings = GeminiSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        http_options: HttpOptions | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = GEMINI_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Gemini STT service.

        Args:
            api_key: Google AI (Gemini) API key.
            http_options: Optional HTTP options passed to the google-genai
                client.
            sample_rate: The sample rate for audio input. If None, will be
                determined from the start frame.
            settings: Runtime-updatable settings. Defaults to the
                ``gemini-3.5-transcribe-live`` model with automatic
                language detection.
            ttfs_p99_latency: P99 latency from speech end to final transcript in
                seconds. Override for your deployment. See
                https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to the parent STTService.

        Raises:
            ImportError: If the installed google-genai version is older than 2.9.0.
        """
        if LanguageAuto is None:
            raise ImportError(
                "GeminiSTTService requires google-genai >= 2.9.0. "
                'Upgrade with `uv add "google-genai>=2.9.0"`.'
            )

        default_settings = self.Settings(
            model="gemini-3.5-transcribe-live",
            language=None,
            languages=[],
            language_auto=None,
            adaptation_phrases=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._http_options = update_google_client_http_options(http_options)
        self._create_client()

        self._session: AsyncSession | None = None
        # The SDK hands out a session through an async context manager. The
        # context is held here so _connect() can await the session being live
        # before it returns.
        self._session_ctx: AbstractAsyncContextManager[AsyncSession] | None = None
        self._receive_task: asyncio.Task | None = None
        self._connection_start_time: float | None = None
        self._consecutive_failures = 0

    def _create_client(self):
        """Create the google-genai client.

        Subclasses (e.g. a Vertex AI variant) can override this to customize
        client construction and authentication.
        """
        self._client = Client(api_key=self._api_key, http_options=self._http_options)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Gemini STT supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert a Language enum to a BCP-47 language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            The BCP-47 language code string.
        """
        return language_to_google_stt_language(language)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Gemini STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Gemini STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def cleanup(self):
        """Release Gemini STT resources."""
        await super().cleanup()
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio data to Gemini for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results arrive via the receive loop).
        """
        await self._send_audio(audio)
        yield None

    async def _send_audio(self, audio: bytes):
        """Send audio to the live session, reconnecting on failure."""
        session = self._session
        if session is None:
            return

        try:
            await session.send_realtime_input(
                audio=Blob(data=audio, mime_type=f"audio/pcm;rate={self.sample_rate}")
            )
        except Exception as e:
            logger.warning(f"{self}: audio send failed, reconnecting: {e}")
            self._session = None
            await self._request_reconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with Gemini-specific handling.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_processing_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._send_finalization_signal()

    async def _send_finalization_signal(self):
        """Prompt a final transcript for the utterance that just ended.

        The model's own activity detection stays enabled; the audio-stream-end
        signal flushes the utterance so it is finalized now instead of when the
        model decides it ended. The stream resumes when the next audio chunk is
        sent.
        """
        if not self._session:
            return
        try:
            await self._session.send_realtime_input(audio_stream_end=True)
        except Exception as e:
            logger.warning(f"{self}: audio stream end failed, reconnecting: {e}")
            self._session = None
            await self._request_reconnect()

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if anything changed.

        All settings (model, languages, adaptation phrases) are connection-time
        configuration, so any change requires a reconnect.
        """
        changed = await super()._update_settings(delta)
        if changed:
            await self._request_reconnect()
        return changed

    def _get_language_codes(self) -> list[str]:
        """Resolve the current language settings to BCP-47 language hint codes.

        Prefers ``languages`` over the single ``language``. Returns an empty
        list when no hints are configured (automatic detection).
        """
        languages = self._settings.languages
        if is_given(languages) and languages:
            return [
                lang if isinstance(lang, str) else self.language_to_service_language(lang)
                for lang in languages
            ]
        language = self._settings.language
        if is_given(language) and language:
            # Stored as a service-specific string by the base class.
            return [str(language)]
        return []

    def _build_live_config(self) -> LiveConnectConfig:
        """Build the Live API connection config from current settings."""
        # __init__ raises on a google-genai too old to provide these, so they
        # are non-None by the time any method runs.
        assert LanguageHints is not None and LanguageAuto is not None

        transcription_kwargs: dict[str, Any] = {}

        language_codes = self._get_language_codes()
        language_auto = self._settings.language_auto
        if not is_given(language_auto):
            language_auto = None
        if language_codes:
            if language_auto:
                logger.warning(
                    f"{self}: language_auto and language hints are mutually exclusive; "
                    "using language hints"
                )
            transcription_kwargs["language_hints"] = LanguageHints(language_codes=language_codes)
        elif language_auto is not False:
            transcription_kwargs["language_auto"] = LanguageAuto()

        adaptation_phrases = self._settings.adaptation_phrases
        if is_given(adaptation_phrases) and adaptation_phrases:
            transcription_kwargs["adaptation_phrases"] = list(adaptation_phrases)

        return LiveConnectConfig(
            response_modalities=[Modality.TEXT],
            input_audio_transcription=AudioTranscriptionConfig(**transcription_kwargs),
        )

    async def _connect(self):
        """Open the Live session, then start receiving on it.

        Returns once the session can accept audio, so callers never have to
        wait for it separately.
        """
        if self._receive_task:
            return

        logger.debug(f"{self}: connecting to Gemini")
        await self._open_session()
        self._receive_task = self.create_task(self._receive_handler())
        self._create_keepalive_task()

    async def _disconnect(self):
        if not self._receive_task:
            return

        logger.debug(f"{self}: disconnecting from Gemini")
        await self._cancel_keepalive_task()
        task, self._receive_task = self._receive_task, None
        await self.cancel_task(task, timeout=1.0)
        await self._close_session()

    async def _open_session(self) -> AsyncSession:
        """Enter the Live session context and record it as usable.

        Returns:
            The session, which is live and able to accept audio.
        """
        model = assert_given(self._settings.model)
        assert model is not None, "GeminiSTTService requires a model"

        config = self._build_live_config()
        self._session_ctx = self._client.aio.live.connect(model=model, config=config)
        self._session = await self._session_ctx.__aenter__()
        self._connection_start_time = time.time()
        await self._call_event_handler("on_connected")
        logger.debug(f"{self}: connected to Gemini")
        return self._session

    async def _close_session(self, exc: BaseException | None = None):
        """Leave the Live session context, if one is open.

        Args:
            exc: The error the session is being closed for, if any. The SDK
                closes the websocket inside the context it yielded from, so
                telling it what went wrong lets it report the cause.
        """
        ctx, self._session_ctx = self._session_ctx, None
        was_connected = self._session is not None
        self._session = None

        if ctx:
            try:
                if exc:
                    await ctx.__aexit__(type(exc), exc, exc.__traceback__)
                else:
                    await ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"{self}: error closing the Gemini session: {e}")

        if was_connected:
            await self._call_event_handler("on_disconnected")

    async def _do_reconnect(self):
        """Disconnect and reconnect to Gemini.

        Called by ``STTService._reconnect()`` inside the reconnecting guard.
        ``_connect()`` returns with the session live, so buffered audio frames
        are replayed only once the new session can accept them.
        """
        await self._disconnect()
        await self._connect()

    async def _receive_handler(self):
        """Receive server messages, reopening the session after transient errors.

        Repeated rapid failures are treated as fatal. Exits cleanly when the
        task is cancelled (i.e. on stop/cancel).
        """
        while True:
            try:
                session = self._session
                if session is None:
                    logger.debug(f"{self}: reopening the Gemini session")
                    session = await self._open_session()

                while True:
                    turn = session.receive()
                    async for message in turn:
                        self._check_and_reset_failure_counter()
                        await self._handle_server_message(message)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self._close_session(e)
                self._consecutive_failures += 1
                if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    error_msg = (
                        f"Max consecutive connection failures "
                        f"({MAX_CONSECUTIVE_FAILURES}) reached: {e}"
                    )
                    await self._call_event_handler("on_connection_error", str(e))
                    await self.push_error(
                        error_msg=error_msg, exception=e, force_treat_as_permanent=True
                    )
                    return
                logger.warning(
                    f"{self}: connection lost, will retry "
                    f"({self._consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}"
                )
                await self.push_error(error_msg=f"connection error: {e}", exception=e)

    def _check_and_reset_failure_counter(self):
        """Reset the failure counter once the connection has proven stable."""
        if (
            self._connection_start_time
            and self._consecutive_failures > 0
            and time.time() - self._connection_start_time >= CONNECTION_ESTABLISHED_THRESHOLD
        ):
            self._consecutive_failures = 0

    async def _handle_server_message(self, message: LiveServerMessage):
        sc = message.server_content
        if not sc:
            return
        if sc.interim_input_transcription:
            await self._handle_interim_transcription(sc.interim_input_transcription, message)
        if sc.input_transcription:
            await self._handle_input_transcription(sc.input_transcription, message)

    async def _handle_interim_transcription(
        self, transcription: Transcription, message: LiveServerMessage
    ):
        """Push an interim transcription frame.

        Interim transcriptions carry the complete utterance text so far, so
        each frame replaces the previous one.
        """
        text = transcription.text
        if not text:
            return
        await self.push_frame(
            InterimTranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                self._language_from_code(transcription.language_code),
                result=message,
            )
        )

    async def _handle_input_transcription(
        self, transcription: Transcription, message: LiveServerMessage
    ):
        """Push a final transcription frame.

        The model streams interim results via ``interim_input_transcription``
        and delivers each completed utterance as a single
        ``input_transcription`` message carrying the full utterance text.
        """
        text = transcription.text
        if not text:
            return

        language = self._language_from_code(transcription.language_code)
        await self.push_frame(
            TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=message,
                finalized=True,
            )
        )
        await self._handle_transcription(text, True, language)
        await self.stop_processing_metrics()

    def _language_from_code(self, language_code: str | None) -> Language | None:
        """Convert a BCP-47 language code from the server to a Language enum."""
        if not language_code:
            return None
        try:
            return Language(language_code)
        except ValueError:
            return None

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    def _is_keepalive_ready(self) -> bool:
        """Check if the session can accept keepalive audio."""
        return self._session is not None

    async def _send_keepalive(self, silence: bytes):
        """Send silent audio to keep the Live session alive.

        Args:
            silence: Silent 16-bit mono PCM audio bytes.
        """
        if self._session:
            await self._session.send_realtime_input(
                audio=Blob(data=silence, mime_type=f"audio/pcm;rate={self.sample_rate}")
            )
