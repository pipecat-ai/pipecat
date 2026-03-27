#
# Copyright (c) 2024-2026, Daily
#
import asyncio
import json
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class LokutorTTSSettings(TTSSettings):
    """Settings for Lokutor TTS service."""

    pass


class LokutorTTSService(WebsocketTTSService):
    """Lokutor TTS service implementation."""

    Settings = LokutorTTSSettings
    _settings: LokutorTTSSettings

    class InputParams(BaseModel):
        """Input parameters for Lokutor TTS such as speed and language."""

        language: Optional[Language] = None
        speed: Optional[float] = 1.0
        steps: Optional[int] = 5
        visemes: Optional[bool] = False

    SUPPORTED_VOICES = {"M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"}

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "M1",
        sample_rate: int = 44100,
        params: Optional[InputParams] = None,
        settings: Optional[LokutorTTSSettings] = None,
        base_url: str = "wss://api.lokutor.com/ws",
        **kwargs,
    ):
        if voice_id not in self.SUPPORTED_VOICES:
            raise ValueError(f"Invalid voice_id '{voice_id}'")

        self._api_key = api_key
        self._voice_id = voice_id
        self._params = params or self.InputParams()

        default_settings = self.Settings(
            model=None,
            voice=self._voice_id,
            language=None,
        )

        if params is not None and settings is None:
            default_settings.language = params.language

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._sample_rate = sample_rate
        self._base_url = base_url
        self._websocket = None
        self._receive_task = None

    async def _connect(self):
        await super()._connect()

        try:
            await self._connect_websocket()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Lokutor: {e}") from e

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        if self._websocket and self._websocket.state is State.OPEN:
            return

        logger.debug("Connecting to Lokutor")
        # Lokutor API key is passed as a query parameter, not a header
        url = f"{self._base_url}?api_key={self._api_key}"
        self._websocket = await websocket_connect(url)

        await self._call_event_handler("on_connected")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Lokutor")
                await self._websocket.close()
        except Exception as exc:
            await self.push_error(error_msg=f"Unknown error occurred: {exc}", exception=exc)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        """Keep the websocket connection alive.
        
        Lokutor uses request-response (send request, receive audio), not streaming.
        All message handling happens in run_tts(). This method just keeps the
        background receive task alive to maintain the persistent connection.
        """
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    def _get_websocket(self):
        if self._websocket is None:
            raise ConnectionError("Lokutor websocket not connected")
        return self._websocket

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        await self.start_ttfb_metrics()
        await self.start_tts_usage_metrics(text)
        yield TTSStartedFrame(context_id=context_id)

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            request = {
                "text": text,
                "voice": self._voice_id,
                "speed": self._params.speed,
                "steps": self._params.steps,
                "visemes": self._params.visemes,
            }
            if self._params.language:
                lokutor_lang = language_to_lokutor_language(self._params.language)
                if lokutor_lang:
                    request["lang"] = lokutor_lang

            request_json = json.dumps(request)
            logger.debug(f"Sending request to Lokutor: {request_json}")
            await self._get_websocket().send(request_json)

            while True:
                try:
                    logger.debug("Waiting for message from Lokutor...")
                    message = await asyncio.wait_for(self._get_websocket().recv(), timeout=10.0)
                    logger.debug(
                        f"Received message from Lokutor: {type(message)} {len(message) if isinstance(message, bytes) else message[:100]}"
                    )
                    if isinstance(message, str):
                        # Handle JSON text messages (visemes, EOS, error)
                        try:
                            data = json.loads(message)
                            if isinstance(data, dict):
                                msg_type = data.get("type")
                                if msg_type == "eos":
                                    logger.debug("Received EOS from Lokutor")
                                    break
                                elif msg_type == "error":
                                    error_msg = data.get("message", "Unknown error")
                                    logger.error(f"Lokutor error: {error_msg}")
                                    yield ErrorFrame(error=f"Lokutor error: {error_msg}")
                                    break
                            elif isinstance(data, list):
                                # Viseme data is an array of objects
                                logger.debug(f"Received viseme data: {len(data)} visemes")
                        except json.JSONDecodeError:
                            logger.warning(f"Received unknown text message: {message}")
                    else:
                        # Binary audio data
                        logger.debug(f"Received audio data: {len(message)} bytes")
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(message, self.sample_rate, 1)
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for Lokutor response")
                    yield ErrorFrame(error="Timeout waiting for Lokutor response")
                    break
                except Exception as e:
                    logger.error(f"Error receiving from Lokutor: {e}")
                    yield ErrorFrame(error=f"Error receiving from Lokutor: {e}")
                    break

        except ConnectionError as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame(context_id=context_id)


def language_to_lokutor_language(language):
    if isinstance(language, Language):
        mapping = {
            Language.EN: "en",
            Language.ES: "es",
            Language.FR: "fr",
            Language.PT: "pt",
            Language.KO: "ko",
        }
        return mapping.get(language)
    if hasattr(language, "value"):
        return language.value
    return str(language)


def lokutor_language_to_language(language):
    if not isinstance(language, str):
        return None

    mapping = {
        "en": Language.EN,
        "es": Language.ES,
        "fr": Language.FR,
        "pt": Language.PT,
        "ko": Language.KO,
    }
    return mapping.get(language)
