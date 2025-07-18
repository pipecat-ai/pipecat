#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


"""Smallest TTS service implementation."""

import base64
import json
import uuid
from enum import Enum
from typing import AsyncGenerator, List, Optional, Union

import aiohttp
import websockets
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import (
    AudioContextWordTTSService,
    InterruptibleTTSService,
    TTSService,
)
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.tracing.service_decorators import traced_tts


class SmallestTTSModel(Enum):
    """Supported models for the Smallest API."""

    LIGHTNING_V2 = "lightning-v2"

def language_to_smallest_language(language: Language) -> Optional[str]:
    """Convert a Language enum to a Smallest language string."""
    BASE_LANGUAGES = {
        Language.AR: "ar",
        Language.BN: "bn",
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.IT: "it",
        Language.KN: "kn",
        Language.MR: "mr",
        Language.NL: "nl",
        Language.PL: "pl",
        Language.RU: "ru",
        Language.TA: "ta",
    }

    result = BASE_LANGUAGES.get(language)

    if not result:
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


class SmallestTTSService(InterruptibleTTSService):
    """Smallest TTS Service.

    Provides real-time text-to-speech synthesis using Smallest's WebSocket API.
    Supports streaming audio generation with configurable voice engines and
    language settings.
    """

    class InputParams(BaseModel):
        """Input parameters for Smallest TTS configuration.

        Parameters:
            language: Language for synthesis. Defaults to English.
            speed: Speech speed multiplier. Defaults to 1.0.
            consistency: Consistency level for voice generation. Defaults to 0.5.
            similarity: Similarity level for voice generation. Defaults to 0.
        """
        language: Optional[Language] = Language.EN
        speed: Optional[Union[str, float]] = 1.0
        consistency: Optional[float] = Field(default=0.5, ge=0, le=1)
        similarity: Optional[float] = Field(default=0, ge=0, le=1)
        enhancement: Optional[int] = Field(default=1, ge=0, le=2)
        model: Optional[SmallestTTSModel] = SmallestTTSModel.LIGHTNING_V2.value

    def __init__(
            
        self,
        *,
        api_key: str,
        voice_id: str,
        base_url: str = "wss://waves-api.smallest.ai",
        model: str = SmallestTTSModel.LIGHTNING_V2.value,
        sample_rate: Optional[int] = 24000,
        params: InputParams = InputParams(),
        text_aggregator: Optional[BaseTextAggregator] = None,
        **kwargs,
    ):
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=False,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            text_aggregator=text_aggregator or SkipTagsAggregator([("<spell>", "</spell>")]),
            **kwargs,
        )

        self._api_key = api_key
        self._websocket_url = f"{base_url}/api/v1/{model}/get_speech/stream"
        self._settings = {
            "output_format": {
                "container": "wav",
                "sample_rate": 0,
            },
            "language": language_to_smallest_language(params.language)
            if params.language
            else "en",
            "speed": params.speed,
            "consistency": params.consistency,
            "similarity": params.similarity,
            "enhancement": params.enhancement,
        }
        self.set_model_name(model)
        self.set_voice(voice_id)
        self._websocket = None
        self._receive_task = None
        self._request_id = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Smallest service supports metrics generation.
        """
        return True

    def _build_msg(self, text: str = ""):
        voice_config = {}
        voice_config["mode"] = "id"
        voice_config["id"] = self._voice_id

        msg = {
            "text": text,
            "voice_id": self._voice_id,
            "language": self._settings["language"],
            "speed": self._settings["speed"],
            "consistency": self._settings["consistency"],
            "similarity": self._settings["similarity"],
            "enhancement": self._settings["enhancement"],
        }

        if self._request_id:
            msg["request_id"] = self._request_id

        return msg

    async def start(self, frame: StartFrame):
        """Start the Smallest TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Smallest TTS service.

        Args:
            frame: The end frame containing stop parameters.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Smallest TTS service.

        Args:
            frame: The cancel frame containing cancel parameters.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.open:
                return
        
            logger.debug("Connecting to Smallest")

            if not isinstance(self._websocket_url, str):
                raise ValueError("WebSocket URL is not a string")

            self._websocket = await websockets.connect(
                f"{self._websocket_url}", extra_headers={"Authorization": f"Bearer {self._api_key}"}
            )
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Smallest")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._request_id = None
            self._websocket = None

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        self._request_id = None

    async def _receive_messages(self):
        """Receive messages from Smallest's WebSocket API."""
        async for message in self._get_websocket():
            msg = json.loads(message)

            if msg["status"] == "complete":
                msg_request_id = msg.get("request_id")
                if self._request_id and msg_request_id and msg_request_id == self._request_id:
                    await self.stop_all_metrics()
                    await self.push_frame(TTSStoppedFrame())
                    self._request_id = None
            elif msg["status"] == "chunk":
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(msg["data"]["audio"]),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                await self.push_frame(frame)
            elif msg["status"] == "error":
                logger.error(f"{self} error: {msg}")
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()
                await self.push_error(ErrorFrame(f"{self} error: {msg['error']}"))
                self._request_id = None
            else:
                logger.error(f"{self} error, unknown message type: {msg}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Smallest's API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        try:
            if not self._websocket or self._websocket.closed:
                await self._connect()
            if not self._request_id:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._request_id = str(uuid.uuid4())
            try:
                msg = self._build_msg(text=text)
                await self._get_websocket().send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)
                await self.start_ttfb_metrics()
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(f"{self} error: {str(e)}")

class SmallestHttpTTSService(TTSService):
    """Smallest HTTP TTS Service.

    Provides text-to-speech synthesis using Smallest's HTTP API.
    Supports streaming audio generation with multiple languages.
    Control over voice characteristics like speed, consistency, similarity, enhancement.
    Example::

        tts = SmallestHttpTTSService(
            api_key="your-api-key",
            voice_id="anushka",
            model="lightning-v2",
            params=SmallestHttpTTSService.InputParams(
                language=Language.HI,
                speed=1.2,
                consistency=0.5,
                similarity=0.5,
                enhancement=1,
            )
        )
    """

    class InputParams(BaseModel):
        """Input parameters for Smallest HTTP TTS configuration.

        Parameters:
            language: Language for synthesis. Defaults to English.
            speed: Speech speed multiplier. Defaults to 1.0.
            consistency: Consistency level for voice generation. Defaults to 0.5.
            similarity: Similarity level for voice generation. Defaults to 0.
            enhancement: Enhancement level for voice generation. Defaults to 1.
        """
        language: str = "en"
        speed: float = 1
        consistency: Optional[float] = None
        similarity: Optional[float] = None
        enhancement: Optional[float] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: SmallestTTSModel = SmallestTTSModel.LIGHTNING_V2,
        base_url: str = "https://waves-api.smallest.ai",
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        """Initialize the Smallest HTTP TTS service.

        Args:
            api_key: The API key for the Smallest API.
            voice_id: The voice ID to use for the TTS.
            model: The model to use for the TTS.
            base_url: The base URL for the Smallest API.
            aiohttp_session: The aiohttp session to use for the TTS.
            sample_rate: The sample rate for the TTS.
            params: The parameters for the TTS.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._sample_rate = sample_rate
        self.set_voice(voice_id)
        self.set_model_name(model.value)
        self.base_url = base_url
        self.aiohttp_session = aiohttp_session or aiohttp.ClientSession()
        self._model_url = self._get_model_url()
        self._settings = {
            "language": params.language,
            "speed": params.speed,
            "consistency": params.consistency,
            "similarity": params.similarity,
            "enhancement": params.enhancement,
        }

    def _get_model_url(self) -> str:
        if self._model_name == SmallestTTSModel.LIGHTNING_V2.value:
            return f"{self.base_url}/api/v1/lightning-v2/get_speech"
        else:
            raise ValueError(f"Invalid model name: {self._model_name}")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Smallest service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Smallest HTTP TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

    async def stop(self, frame: EndFrame):
        """Stop the Smallest HTTP TTS service.

        Args:
            frame: The end frame containing stop parameters.
        """
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the Smallest HTTP TTS service.

        Args:
            frame: The cancel frame containing cancel parameters.
        """
        await super().cancel(frame)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using the Smallest API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            payload = {
                "voice_id": self._voice_id,
                "text": text,
                "sample_rate": self._sample_rate,
                **self._settings,
            }

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            
            yield TTSStartedFrame()

            async with self.aiohttp_session.post(
                self._model_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Smallest API error: {error_text}")
                    await self.push_error(ErrorFrame(f"Smallest API error: {error_text}"))
                    return

                result = await response.read()

            await self.start_tts_usage_metrics(text)

            frame = TTSAudioRawFrame(audio=result, sample_rate=self.sample_rate, num_channels=1)
            yield frame

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(f"Error generating TTS: {e}"))
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

