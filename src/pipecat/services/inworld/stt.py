#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inworld AI speech-to-text service implementations."""

import asyncio
import base64
import json
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import aiohttp
from loguru import logger
from pydantic import ValidationError
from websockets.protocol import State

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.inworld.frames import InworldVoiceProfile, InworldVoiceProfileFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService, WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

USER_AGENT = f"pipecat/{pipecat_version()}"


class InworldTurnDetectionMode(StrEnum):
    """Turn detection mode for Inworld realtime STT.

    ``AUTOMATIC`` uses Inworld's voice activity and semantic end-of-turn
    detection. ``MANUAL`` disables Inworld VAD so Pipecat can determine speech
    boundaries and send an explicit ``endTurn`` request after local VAD stops.
    """

    AUTOMATIC = "automatic"
    MANUAL = "manual"


def language_to_inworld_stt_language(language: Language) -> str:
    """Convert a language enum to an Inworld STT language code.

    Args:
        language: The language to convert.

    Returns:
        The corresponding ISO 639 language code. Regional variants fall back
        to their base language code.
    """
    language_map = {
        Language.AR: "ar",
        Language.CS: "cs",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FIL: "fil",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.HU: "hu",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.MK: "mk",
        Language.MS: "ms",
        Language.NL: "nl",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SV: "sv",
        Language.TH: "th",
        Language.TL: "fil",
        Language.TR: "tr",
        Language.VI: "vi",
        Language.YUE: "yue",
        Language.ZH: "zh",
    }
    return resolve_language(language, language_map, use_base_code=True)


@dataclass
class InworldSTTSettings(STTSettings):
    """Settings for :class:`InworldSTTService`.

    Parameters:
        prompts: Terms that bias recognition toward names, jargon, and acronyms.
        enable_voice_profile: Whether to analyze speaker age, emotion, pitch,
            vocal style, and accent. See https://docs.inworld.ai/stt/voice-profiles
        voice_profile_top_n: Maximum labels returned for each Voice Profile category.
    """

    prompts: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_voice_profile: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    voice_profile_top_n: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class InworldRealtimeSTTSettings(InworldSTTSettings):
    """Settings for :class:`InworldRealtimeSTTService`.

    Parameters:
        vad_threshold: Inworld voice activity detection sensitivity in automatic
            turn detection mode. Manual mode always sends ``0`` to disable
            Inworld VAD.
        min_end_of_turn_silence_when_confident: Minimum silence in milliseconds
            before ending a turn when Inworld is confident the turn is complete.
        end_of_turn_confidence_threshold: Confidence threshold for Inworld's
            semantic end-of-turn detection. Lower values end turns more eagerly.
        inactivity_timeout_seconds: Seconds of client silence before Inworld stops
            transcription. ``None`` uses the server default.
    """

    vad_threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_end_of_turn_silence_when_confident: int | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    end_of_turn_confidence_threshold: float | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    inactivity_timeout_seconds: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


def _build_transcribe_config(settings: InworldSTTSettings, sample_rate: int) -> dict[str, Any]:
    """Build the fields shared by Inworld's HTTP and WebSocket STT APIs."""
    model = assert_given(settings.model)
    if not model:
        raise ValueError("Inworld STT model must be specified")

    config: dict[str, Any] = {
        "modelId": model,
        "audioEncoding": "LINEAR16",
        "sampleRateHertz": sample_rate,
        "numberOfChannels": 1,
    }

    language = assert_given(settings.language)
    if language:
        config["language"] = str(language)

    prompts = assert_given(settings.prompts)
    if prompts:
        config["prompts"] = prompts

    enable_voice_profile = assert_given(settings.enable_voice_profile)
    if enable_voice_profile:
        voice_profile_config: dict[str, Any] = {"enableVoiceProfile": True}
        top_n = assert_given(settings.voice_profile_top_n)
        if top_n is not None:
            if top_n < 1:
                raise ValueError("Inworld Voice Profile top_n must be at least 1")
            voice_profile_config["topN"] = top_n
        config["voiceProfileConfig"] = voice_profile_config

    return config


class InworldSTTService(SegmentedSTTService):
    """Speech-to-text service using Inworld's synchronous transcription API.

    The service buffers utterances according to Pipecat VAD events, sends each
    utterance to Inworld as a WAV file, and emits one final transcription frame.
    When Voice Profile analysis is enabled, it also emits an
    :class:`InworldVoiceProfileFrame` before the transcription.
    """

    Settings = InworldSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.inworld.ai",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = None,
        **kwargs,
    ):
        """Initialize the Inworld STT service.

        Args:
            api_key: Inworld API key containing Base64 credentials.
            aiohttp_session: aiohttp client session for HTTP requests.
            base_url: Base URL for the Inworld API.
            sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate.
            settings: Runtime-updatable model, language, and recognition prompts.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to :class:`SegmentedSTTService`.
        """
        default_settings = self.Settings(
            model="inworld/inworld-stt-1",
            language=None,
            prompts=[],
            enable_voice_profile=False,
            voice_profile_top_n=10,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        """Check whether the service can generate processing metrics.

        Returns:
            True, as Inworld STT supports processing metrics.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a language enum to Inworld's STT language format.

        Args:
            language: The language to convert.

        Returns:
            The Inworld ISO 639 language code.
        """
        return language_to_inworld_stt_language(language)

    def _request_payload(self, audio: bytes) -> dict[str, Any]:
        """Build a transcription request payload.

        Args:
            audio: WAV-encoded audio bytes.

        Returns:
            The Inworld transcription request payload.

        Raises:
            ValueError: If no model is configured.
        """
        return {
            "transcribeConfig": _build_transcribe_config(self._settings, self.sample_rate),
            "audioData": {"content": base64.b64encode(audio).decode("ascii")},
        }

    async def _transcribe(self, audio: bytes) -> dict[str, Any]:
        """Send one WAV utterance to Inworld.

        Args:
            audio: WAV-encoded audio bytes.

        Returns:
            The decoded Inworld response.

        Raises:
            RuntimeError: If Inworld returns an unsuccessful status.
        """
        headers = {
            "Authorization": f"Basic {self._api_key}",
            "Content-Type": "application/json",
            "X-Request-Id": str(uuid.uuid4()),
            "X-User-Agent": USER_AGENT,
        }
        async with self._session.post(
            f"{self._base_url}/stt/v1/transcribe",
            json=self._request_payload(audio),
            headers=headers,
        ) as response:
            if not 200 <= response.status < 300:
                error_text = await response.text()
                raise RuntimeError(f"Inworld API error ({response.status}): {error_text}")
            return await response.json()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: str | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe a WAV utterance with Inworld.

        Args:
            audio: WAV-encoded audio bytes produced by :class:`SegmentedSTTService`.

        Yields:
            An optional Voice Profile frame followed by a transcription frame for
            non-empty text, or an error frame on failure.
        """
        await self.start_processing_metrics()
        try:
            result = await self._transcribe(audio)
            timestamp = time_now_iso8601()

            voice_profile_data = result.get("voiceProfile", result.get("voice_profile"))
            if voice_profile_data is not None:
                try:
                    voice_profile = InworldVoiceProfile.model_validate(voice_profile_data)
                    yield InworldVoiceProfileFrame(
                        user_id=self._user_id,
                        timestamp=timestamp,
                        voice_profile=voice_profile,
                    )
                except ValidationError as e:
                    yield ErrorFrame(error=f"Inworld Voice Profile error: {e}", exception=e)

            transcript = result.get("transcription", {}).get("transcript", "").strip()
            if not transcript:
                logger.debug("Inworld returned an empty transcription")
                return

            language_setting = assert_given(self._settings.language)
            language = str(language_setting) if language_setting else None
            await self._handle_transcription(transcript, True, language)
            logger.debug(f"Transcription: [{transcript}]")

            try:
                frame_language = Language(language) if language else None
            except ValueError:
                frame_language = None

            yield TranscriptionFrame(
                transcript,
                self._user_id,
                timestamp,
                frame_language,
                result=result,
            )
        except Exception as e:
            yield ErrorFrame(error=f"Inworld STT error: {e}", exception=e)
        finally:
            await self.stop_processing_metrics()


class InworldRealtimeSTTService(WebsocketSTTService):
    """Speech-to-text service using Inworld's bidirectional WebSocket API.

    The service streams raw LINEAR16 audio and emits interim and final
    transcription frames. Automatic turn detection uses Inworld speech events
    and final transcriptions to propose boundaries through
    :class:`ExternalUserTurnStrategies`. Manual turn detection disables Inworld
    VAD and forwards Pipecat VAD stops as explicit ``endTurn`` requests.

    Voice Profile analysis emits an :class:`InworldVoiceProfileFrame` whenever
    Inworld includes profile data in a streaming transcription result.
    """

    Settings = InworldRealtimeSTTSettings
    TurnDetectionMode = InworldTurnDetectionMode
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.inworld.ai",
        sample_rate: int | None = None,
        turn_detection_mode: InworldTurnDetectionMode = InworldTurnDetectionMode.AUTOMATIC,
        should_interrupt: bool = True,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = None,
        **kwargs,
    ):
        """Initialize the Inworld realtime STT service.

        Args:
            api_key: Inworld API key containing Base64 credentials.
            base_url: Base URL for the Inworld WebSocket API.
            sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate.
            turn_detection_mode: Whether Inworld or Pipecat determines speech
                boundaries. Manual mode requires Pipecat VAD events and sends
                ``endTurn`` when local speech stops.
            should_interrupt: Whether Inworld-proposed turn starts should interrupt
                the current bot response. Passed to the external turn strategies
                recommended in automatic mode.
            settings: Runtime-updatable model, language, recognition, Voice Profile,
                VAD, and end-of-turn settings. Updates reconnect the WebSocket.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to :class:`WebsocketSTTService`.
        """
        self._turn_detection_mode = InworldTurnDetectionMode(turn_detection_mode)

        default_settings = self.Settings(
            model="inworld/inworld-stt-1",
            language=None,
            prompts=[],
            enable_voice_profile=False,
            voice_profile_top_n=10,
            vad_threshold=None,
            min_end_of_turn_silence_when_confident=None,
            end_of_turn_confidence_threshold=None,
            inactivity_timeout_seconds=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

        self._api_key = api_key
        if "://" not in base_url:
            base_url = f"wss://{base_url}"
        self._base_url = (
            base_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
        )
        self._should_interrupt = should_interrupt

        self._connected_event = asyncio.Event()
        self._connected_event.set()
        self._receive_task = None
        self._user_turn_open = False

    @property
    def supports_ttfs(self) -> bool:
        """Check whether Pipecat supplies a distinct speech-end boundary.

        Returns:
            True in manual mode, where local VAD supplies the speech-end
            boundary; False in automatic mode, where Inworld owns it.
        """
        return self._turn_detection_mode is InworldTurnDetectionMode.MANUAL

    def can_generate_metrics(self) -> bool:
        """Check whether the service can generate metrics.

        Returns:
            True, as Inworld realtime STT supports latency and usage metrics.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a language enum to Inworld's STT language format.

        Args:
            language: The language to convert.

        Returns:
            The Inworld ISO 639 language code.
        """
        return language_to_inworld_stt_language(language)

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend turn strategies for the configured endpointing mode.

        Returns:
            STT metadata with external strategies in automatic mode and the
            standard Pipecat strategy selection in manual mode.
        """
        frame = super().service_metadata_frame()
        if self._turn_detection_mode is InworldTurnDetectionMode.AUTOMATIC:
            frame.user_turn_strategies = ExternalUserTurnStrategies(
                enable_interruptions=self._should_interrupt,
            )
        return frame

    async def start(self, frame: StartFrame):
        """Start the service and establish the WebSocket connection.

        Args:
            frame: Frame carrying the negotiated pipeline configuration.
        """
        await super().start(frame)
        await self._connect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Stream one raw LINEAR16 audio chunk to Inworld.

        Args:
            audio: Raw 16-bit mono PCM audio.

        Yields:
            An error frame when no connection is available; responses otherwise
            arrive through the WebSocket receive task.
        """
        await self._connected_event.wait()

        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        if self._websocket and self._websocket.state is State.OPEN:
            message = {
                "audioChunk": {"content": base64.b64encode(audio).decode("ascii")},
            }
            await self.send_with_retry(json.dumps(message), self._report_error)
        else:
            yield ErrorFrame("Inworld realtime STT WebSocket is not connected")
            return

        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and forward manual speech ends to Inworld.

        Args:
            frame: Frame to process.
            direction: Direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if (
            isinstance(frame, VADUserStoppedSpeakingFrame)
            and self._turn_detection_mode is InworldTurnDetectionMode.MANUAL
        ):
            await self._send_end_turn()

    async def _send_end_turn(self):
        """Ask Inworld to finalize the current manually delimited turn."""
        await self._connected_event.wait()

        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        if self._websocket and self._websocket.state is State.OPEN:
            await self.send_with_retry(json.dumps({"endTurn": {}}), self._report_error)
        else:
            await self.push_error(
                error_msg="Unable to end Inworld realtime STT turn: WebSocket is not connected"
            )

    async def _connect(self):
        """Establish the connection and start the receive task."""
        self._connected_event.clear()
        try:
            await self._connect_websocket()
            await super()._connect()
            if self._websocket and not self._receive_task:
                self._receive_task = self.create_task(
                    self._receive_task_handler(self._report_error),
                    name="inworld_stt_receive",
                )
        finally:
            self._connected_event.set()

    async def _disconnect(self):
        """Stop receive processing and close the connection."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply settings and reconnect so Inworld receives a new config.

        Args:
            delta: Runtime settings to apply.

        Returns:
            Mapping of changed setting names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed:
            await self._request_reconnect()
        return changed

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        """Handle local VAD timing according to the turn detection mode.

        Automatic mode ignores local speech-end latency because Inworld owns the
        boundary. Manual mode uses the standard STT timing behavior.

        Args:
            frame: The local VAD stop frame.
        """
        if self._turn_detection_mode is InworldTurnDetectionMode.AUTOMATIC:
            self._user_speaking = False
        else:
            await super()._handle_vad_user_stopped_speaking(frame)

    def _transcribe_config(self) -> dict[str, Any]:
        """Build the initial WebSocket transcription configuration.

        Returns:
            Configuration sent as the first WebSocket message.

        Raises:
            ValueError: If a configured silence or Voice Profile limit is invalid.
        """
        config = _build_transcribe_config(self._settings, self.sample_rate)

        automatic = self._turn_detection_mode is InworldTurnDetectionMode.AUTOMATIC

        end_of_turn_threshold = assert_given(self._settings.end_of_turn_confidence_threshold)
        if end_of_turn_threshold is not None:
            if not 0 <= end_of_turn_threshold <= 1:
                raise ValueError("Inworld end-of-turn confidence threshold must be between 0 and 1")
            if not automatic:
                raise ValueError(
                    "Inworld end-of-turn confidence threshold is only valid in automatic mode"
                )
            config["endOfTurnConfidenceThreshold"] = end_of_turn_threshold

        inactivity_timeout = assert_given(self._settings.inactivity_timeout_seconds)
        if inactivity_timeout is not None:
            if inactivity_timeout < 1:
                raise ValueError("Inworld inactivity timeout must be at least 1 second")
            config["inactivityTimeoutSeconds"] = inactivity_timeout

        inworld_config: dict[str, Any] = {}
        vad_threshold = assert_given(self._settings.vad_threshold)
        if automatic:
            if vad_threshold is not None:
                if not 0 < vad_threshold <= 1:
                    raise ValueError(
                        "Inworld VAD threshold must be greater than 0 and at most 1 in automatic "
                        "mode; use turn_detection_mode=TurnDetectionMode.MANUAL to disable it"
                    )
                inworld_config["vadThreshold"] = vad_threshold
        else:
            if vad_threshold not in (None, 0):
                raise ValueError("Inworld VAD threshold must be 0 or unset in manual mode")
            inworld_config["vadThreshold"] = 0

        min_silence = assert_given(self._settings.min_end_of_turn_silence_when_confident)
        if min_silence is not None:
            if min_silence < 0:
                raise ValueError("Inworld minimum end-of-turn silence cannot be negative")
            if not automatic:
                raise ValueError(
                    "Inworld minimum end-of-turn silence is only valid in automatic mode"
                )
            inworld_config["minEndOfTurnSilenceWhenConfident"] = min_silence

        if inworld_config:
            config["inworldSttV1Config"] = inworld_config

        return config

    async def _connect_websocket(self):
        """Open and configure the Inworld WebSocket."""
        if self._websocket and self._websocket.state is State.OPEN:
            return

        websocket = None
        try:
            config = self._transcribe_config()
            headers = {
                "Authorization": f"Basic {self._api_key}",
                "X-Request-Id": str(uuid.uuid4()),
                "X-User-Agent": USER_AGENT,
            }
            ws_url = f"{self._base_url}/stt/v1/transcribe:streamBidirectional"

            logger.debug("Connecting to Inworld realtime STT")
            websocket = await self._websocket_connect(
                ws_url,
                additional_headers=headers,
            )
            self._websocket = websocket
            await websocket.send(json.dumps({"transcribeConfig": config}))
            await self._call_event_handler("on_connected")
        except Exception as e:
            if websocket is not None:
                try:
                    await websocket.close()
                except Exception:
                    pass
            self._websocket = None
            await self.push_error(
                error_msg=f"Unable to connect to Inworld realtime STT: {e}",
                exception=e,
            )

    async def _disconnect_websocket(self):
        """Close the Inworld transcription stream and WebSocket."""
        websocket = self._websocket
        try:
            if websocket and websocket.state is State.OPEN:
                logger.debug("Disconnecting from Inworld realtime STT")
                await websocket.send(json.dumps({"closeStream": {}}))
        except Exception as e:
            await self.push_error(error_msg=f"Error closing Inworld STT WebSocket: {e}")
        finally:
            if websocket:
                try:
                    await websocket.close()
                except Exception as e:
                    await self.push_error(error_msg=f"Error closing Inworld STT WebSocket: {e}")
            if self._websocket is websocket:
                self._websocket = None
            await self._user_turn_stopped()
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Return the active WebSocket connection."""
        if self._websocket:
            return self._websocket
        raise ConnectionError("Inworld realtime STT WebSocket is not connected")

    async def _receive_messages(self):
        """Receive and process Inworld WebSocket messages."""
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
                await self._process_response(data)
            except json.JSONDecodeError:
                logger.warning(f"Inworld realtime STT returned non-JSON data: {message}")
            except Exception as e:
                logger.exception(f"Error processing Inworld realtime STT message: {e}")

    async def _process_response(self, data: dict[str, Any]):
        """Process one decoded Inworld streaming response.

        Args:
            data: Inworld response envelope.
        """
        result = data.get("result") or data

        error_code = data.get("code", result.get("code"))
        error_message = data.get("message", result.get("message"))
        if error_code is not None or error_message:
            error_code_text = f" ({error_code})" if error_code is not None else ""
            await self.push_error(
                error_msg=f"Inworld realtime STT error{error_code_text}: {error_message}"
            )
            await self._user_turn_stopped()
            return

        if "speechStarted" in result:
            await self._user_turn_started()
        if "speechStopped" in result:
            # This event marks audio silence, not a semantic turn boundary. The
            # final transcription below is the authoritative end-of-turn signal.
            logger.trace("Inworld realtime STT detected speech stop")

        transcription = result.get("transcription")
        if transcription is not None:
            await self._on_transcription(transcription, result)

    async def _user_turn_started(self):
        """Propose an Inworld-owned turn start once."""
        if self._turn_detection_mode is not InworldTurnDetectionMode.AUTOMATIC:
            return
        if self._user_turn_open:
            return
        self._user_turn_open = True
        await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

    async def _user_turn_stopped(self):
        """Propose an Inworld-owned turn stop once."""
        if self._turn_detection_mode is not InworldTurnDetectionMode.AUTOMATIC:
            return
        if not self._user_turn_open:
            return
        self._user_turn_open = False
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: str | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _on_transcription(self, transcription: dict[str, Any], result: dict[str, Any]):
        """Emit frames for a streaming transcription result.

        Args:
            transcription: Interim or final transcription fields.
            result: Full result envelope, including optional Voice Profile data.
        """
        timestamp = time_now_iso8601()
        voice_profile_data = transcription.get(
            "voiceProfile",
            transcription.get(
                "voice_profile",
                result.get("voiceProfile", result.get("voice_profile")),
            ),
        )
        transcript = transcription.get("transcript", "").strip()
        is_final = transcription.get("isFinal", transcription.get("is_final", False))

        if transcript or voice_profile_data is not None:
            await self._user_turn_started()

        if voice_profile_data is not None:
            try:
                voice_profile = InworldVoiceProfile.model_validate(voice_profile_data)
                await self.push_frame(
                    InworldVoiceProfileFrame(
                        user_id=self._user_id,
                        timestamp=timestamp,
                        voice_profile=voice_profile,
                    )
                )
            except ValidationError as e:
                await self.push_error(
                    error_msg=f"Inworld Voice Profile error: {e}",
                    exception=e,
                )

        language_value = transcription.get("language", transcription.get("languageCode"))
        if not language_value:
            language_setting = assert_given(self._settings.language)
            language_value = str(language_setting) if language_setting else None
        try:
            language = Language(language_value) if language_value else None
        except ValueError:
            language = None

        if transcript:
            if is_final:
                logger.debug(f"Final transcription: [{transcript}]")
                await self._handle_transcription(transcript, True, language_value)
                await self.emit_stt_usage_metrics()
                await self.push_frame(
                    TranscriptionFrame(
                        transcript,
                        self._user_id,
                        timestamp,
                        language,
                        result=result,
                        finalized=True,
                    )
                )
            else:
                logger.trace(f"Interim transcription: [{transcript}]")
                await self.push_frame(
                    InterimTranscriptionFrame(
                        transcript,
                        self._user_id,
                        timestamp,
                        language,
                        result=result,
                    )
                )

        if is_final:
            await self._user_turn_stopped()
