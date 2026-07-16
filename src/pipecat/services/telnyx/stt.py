#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Telnyx speech-to-text service.

Provides streaming STT via the Telnyx WebSocket API at
wss://api.telnyx.com/v2/speech-to-text/transcription.

Protocol:
  - Connect with Authorization: Bearer <key> header.
  - Send raw 16-bit PCM audio as binary WebSocket frames.
  - Receive JSON text frames with transcript, is_final, confidence, speech_final.
  - Send {"type": "CloseStream"} to end the session gracefully (Deepgram, Speechmatics, Soniox only).
  - Receive {"errors": [...]} on validation or connection errors.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


def language_to_telnyx_stt_language(language: Language) -> str:
    """Convert a Language enum to a Telnyx STT BCP-47 language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The BCP-47 language code string.
    """
    LANGUAGE_MAP = {
        Language.EN: "en-US",
        Language.EN_US: "en-US",
        Language.EN_GB: "en-GB",
        Language.EN_AU: "en-AU",
        Language.EN_CA: "en-CA",
        Language.EN_IN: "en-IN",
        Language.EN_IE: "en-IE",
        Language.EN_NZ: "en-NZ",
        Language.ES: "es-ES",
        Language.ES_ES: "es-ES",
        Language.ES_MX: "es-MX",
        Language.ES_AR: "es-AR",
        Language.FR: "fr-FR",
        Language.FR_FR: "fr-FR",
        Language.FR_CA: "fr-CA",
        Language.DE: "de-DE",
        Language.DE_DE: "de-DE",
        Language.IT: "it-IT",
        Language.IT_IT: "it-IT",
        Language.PT: "pt-BR",
        Language.PT_BR: "pt-BR",
        Language.PT_PT: "pt-PT",
        Language.NL: "nl-NL",
        Language.NL_NL: "nl-NL",
        Language.PL: "pl-PL",
        Language.PL_PL: "pl-PL",
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
        Language.TR: "tr-TR",
        Language.TR_TR: "tr-TR",
        Language.KO: "ko-KR",
        Language.KO_KR: "ko-KR",
        Language.JA: "ja-JP",
        Language.JA_JP: "ja-JP",
        Language.ZH: "zh-CN",
        Language.ZH_CN: "zh-CN",
        Language.AR: "ar-SA",
        Language.AR_SA: "ar-SA",
        Language.HI: "hi-IN",
        Language.HI_IN: "hi-IN",
        Language.SV: "sv-SE",
        Language.SV_SE: "sv-SE",
        Language.DA: "da-DK",
        Language.DA_DK: "da-DK",
        Language.NO: "nb-NO",
        Language.FI: "fi-FI",
        Language.CS: "cs-CZ",
        Language.CS_CZ: "cs-CZ",
        Language.UK: "uk-UA",
        Language.UK_UA: "uk-UA",
        Language.EL: "el-GR",
        Language.EL_GR: "el-GR",
        Language.HE: "he-IL",
        Language.HE_IL: "he-IL",
        Language.TH: "th-TH",
        Language.TH_TH: "th-TH",
        Language.VI: "vi-VN",
        Language.VI_VN: "vi-VN",
        Language.ID: "id-ID",
        Language.ID_ID: "id-ID",
        Language.MS: "ms-MY",
        Language.MS_MY: "ms-MY",
        Language.FIL: "fil-PH",
        Language.RO: "ro-RO",
        Language.RO_RO: "ro-RO",
        Language.HU: "hu-HU",
        Language.HU_HU: "hu-HU",
        Language.CA: "ca-ES",
        Language.HR: "hr-HR",
        Language.HR_HR: "hr-HR",
        Language.LT: "lt-LT",
        Language.LT_LT: "lt-LT",
        Language.LV: "lv-LV",
        Language.LV_LV: "lv-LV",
        Language.SK: "sk-SK",
        Language.SK_SK: "sk-SK",
        Language.SL: "sl-SI",
        Language.SL_SI: "sl-SI",
        Language.BG: "bg-BG",
        Language.BG_BG: "bg-BG",
        Language.EU: "eu-ES",
    }
    return LANGUAGE_MAP.get(language, str(language))


@dataclass
class TelnyxSTTSettings(STTSettings):
    """Settings for TelnyxSTTService.

    Parameters:
        transcription_engine: Telnyx STT engine (Telnyx, Deepgram, Google, Azure).
        input_format: Audio input encoding (linear16, mulaw, alaw).
        interim_results: Whether to request interim (partial) transcripts.
    """

    transcription_engine: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    input_format: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    interim_results: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class TelnyxSTTService(WebsocketSTTService):
    """Telnyx streaming speech-to-text over WebSocket.

    Sends raw PCM audio as binary frames, receives JSON transcription results.
    The WebSocket stays open for the session. Supports interim and final
    transcripts depending on the engine.

    Example::

        stt = TelnyxSTTService(
            api_key="your-telnyx-api-key",
            settings=TelnyxSTTService.Settings(
                transcription_engine="Telnyx",
                input_format="linear16",
            ),
        )
    """

    Settings = TelnyxSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        transcription_engine: str = "Telnyx",
        input_format: str = "linear16",
        sample_rate: int = 16000,
        language: str = "en-US",
        interim_results: bool = True,
        settings: Settings | None = None,
        **kwargs: Any,
    ):
        """Initialize the Telnyx STT service.

        Args:
            api_key: Telnyx API key for authentication.
            transcription_engine: STT engine (Telnyx, Deepgram, Google, Azure).
            input_format: Audio encoding (linear16, mulaw, alaw).
            sample_rate: Audio sample rate in Hz.
            language: BCP-47 language code (e.g., "en-US").
            interim_results: Whether to request interim (partial) transcripts.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        default_settings = self.Settings(
            transcription_engine=transcription_engine,
            input_format=input_format,
            interim_results=interim_results,
            language=language,
        )

        if settings:
            default_settings.apply_update(settings)

        super().__init__(sample_rate=sample_rate, settings=default_settings, **kwargs)
        self._api_key = api_key
        self._sample_rate = sample_rate
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Telnyx STT BCP-47 language code.

        Args:
            language: The language to convert.

        Returns:
            The BCP-47 language code string.
        """
        return language_to_telnyx_stt_language(language)

    def _build_url(self) -> str:
        return (
            "wss://api.telnyx.com/v2/speech-to-text/transcription"
            f"?transcription_engine={self._settings.transcription_engine}"
            f"&input_format={self._settings.input_format}"
            f"&sample_rate={self._sample_rate}"
            f"&language={self._settings.language}"
            f"&interim_results={'true' if self._settings.interim_results else 'false'}"
        )

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if anything changed."""
        changed = await super()._update_settings(delta)

        if changed:
            await self._disconnect()
            await self._connect()

        return changed

    async def start(self, frame: StartFrame):
        """Start the service and connect to the WebSocket."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and disconnect from the WebSocket."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and disconnect from the WebSocket."""
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, handling VAD events for metrics tracking."""
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_processing_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if (
                self._transcription_engine_supports_finalize()
                and self._websocket
                and self._websocket.state is State.OPEN
            ):
                try:
                    await self._websocket.send(json.dumps({"type": "Finalize"}))
                except Exception as e:
                    logger.warning(f"{self} failed to send finalize: {e}")

    def _transcription_engine_supports_finalize(self) -> bool:
        """Check if the current engine supports the Finalize control message."""
        return self._settings.transcription_engine == "Deepgram"

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

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
            logger.debug("Connecting to Telnyx STT")
            self._websocket = await websocket_connect(
                self._build_url(),
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            if self._websocket:
                logger.debug("Disconnecting from Telnyx STT")
                if self._settings.transcription_engine in ("Deepgram", "Speechmatics", "Soniox"):
                    try:
                        await self._websocket.send(json.dumps({"type": "CloseStream"}))
                    except Exception:
                        pass
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"Telnyx STT: non-JSON message: {message}")
                continue

            if data.get("errors"):
                errors = data["errors"]
                detail = errors[0].get("detail") if errors else "Unknown Telnyx STT error"
                logger.error(f"Telnyx STT error: {detail}")
                await self.push_error(error_msg=detail)
                continue

            if data.get("utterance_end"):
                continue

            is_final = data.get("is_final", False)
            text = data.get("transcript", "").strip()

            if not text:
                continue

            if is_final:
                await self.stop_processing_metrics()
                logger.debug(f"Telnyx final transcript: [{text}]")
                await self._handle_transcription(text, True, self._settings.language)
                await self.push_frame(
                    TranscriptionFrame(
                        text,
                        self._user_id,
                        time_now_iso8601(),
                        self._settings.language,
                        result=data,
                    )
                )
            else:
                logger.trace(f"Telnyx interim transcript: [{text}]")
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text,
                        self._user_id,
                        time_now_iso8601(),
                        self._settings.language,
                        result=data,
                    )
                )

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: str | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to the Telnyx STT WebSocket for transcription.

        Args:
            audio: Raw PCM audio bytes.

        Yields:
            None -- transcription results arrive via WebSocket messages.
        """
        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(audio)
            except Exception as e:
                yield ErrorFrame(error=f"Telnyx STT error: {e}")
                return

        yield None
