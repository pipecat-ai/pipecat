#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
from typing import Any, AsyncGenerator, Mapping, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import InterruptibleTTSService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio.watchdog_async_iterator import WatchdogAsyncIterator
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import websockets
    from pyneuphonic import Neuphonic, TTSConfig
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Neuphonic, you need to `pip install pipecat-ai[neuphonic]`.")
    raise Exception(f"Missing module: {e}")


def language_to_neuphonic_lang_code(language: Language) -> Optional[str]:
    BASE_LANGUAGES = {
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.NL: "nl",
        Language.AR: "ar",
        Language.FR: "fr",
        Language.PT: "pt",
        Language.RU: "ru",
        Language.HI: "HI",
        Language.ZH: "zh",
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        # Convert enum value to string and get the base language part (e.g. es-ES -> es)
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Look up the base code in our supported languages
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


class NeuphonicTTSService(InterruptibleTTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: Optional[str] = None,
        url: str = "wss://api.neuphonic.com",
        sample_rate: Optional[int] = 22050,
        encoding: str = "pcm_linear",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=False,
            push_stop_frames=True,
            stop_frame_timeout_s=2.0,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or NeuphonicTTSService.InputParams()

        self._api_key = api_key
        self._url = url
        self._settings = {
            "lang_code": self.language_to_service_language(params.language),
            "speed": params.speed,
            "encoding": encoding,
            "sampling_rate": sample_rate,
        }
        self.set_voice(voice_id)

        # Indicates if we have sent TTSStartedFrame. It will reset to False when
        # there's an interruption or TTSStoppedFrame.
        self._started = False
        self._cumulative_time = 0

        self._receive_task = None
        self._keepalive_task = None

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_neuphonic_lang_code(language)

    async def _update_settings(self, settings: Mapping[str, Any]):
        if "voice_id" in settings:
            self.set_voice(settings["voice_id"])

        await super()._update_settings(settings)
        await self._disconnect()
        await self._connect()
        logger.info(f"Switching TTS to settings: [{self._settings}]")

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        if self._websocket:
            msg = {"text": "<STOP>"}
            await self._websocket.send(json.dumps(msg))

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, StartInterruptionFrame)):
            self._started = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # If we received a TTSSpeakFrame and the LLM response included text (it
        # might be that it's only a function calling response) we pause
        # processing more frames until we receive a BotStoppedSpeakingFrame.
        if isinstance(frame, TTSSpeakFrame):
            await self.pause_processing_frames()
        elif isinstance(frame, LLMFullResponseEndFrame) and self._started:
            await self.pause_processing_frames()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.resume_processing_frames()

    async def _connect(self):
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.open:
                return

            logger.debug("Connecting to Neuphonic")

            tts_config = {
                **self._settings,
                "voice_id": self._voice_id,
            }

            query_params = [f"api_key={self._api_key}"]
            for key, value in tts_config.items():
                if value is not None:
                    query_params.append(f"{key}={value}")

            url = f"{self._url}/speak/{self._settings['lang_code']}?{'&'.join(query_params)}"

            self._websocket = await websockets.connect(url)
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Neuphonic")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._started = False
            self._websocket = None

    async def _receive_messages(self):
        async for message in WatchdogAsyncIterator(self._websocket, manager=self.task_manager):
            if isinstance(message, str):
                msg = json.loads(message)
                if msg.get("data", {}).get("audio") is not None:
                    await self.stop_ttfb_metrics()

                    audio = base64.b64decode(msg["data"]["audio"])
                    frame = TTSAudioRawFrame(audio, self.sample_rate, 1)
                    await self.push_frame(frame)

    async def _keepalive_task_handler(self):
        KEEPALIVE_SLEEP = 10 if self.task_manager.task_watchdog_enabled else 3
        while True:
            self.reset_watchdog()
            await asyncio.sleep(KEEPALIVE_SLEEP)
            await self._send_text("")

    async def _send_text(self, text: str):
        if self._websocket:
            msg = {"text": text}
            logger.debug(f"Sending text to websocket: {msg}")
            await self._websocket.send(json.dumps(msg))

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            if not self._websocket or self._websocket.closed:
                await self._connect()

            try:
                if not self._started:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._started = True
                    self._cumulative_time = 0

                await self._send_text(text)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")


class NeuphonicHttpTTSService(TTSService):
    """Neuphonic Text-to-Speech service using HTTP streaming.

    Args:
        api_key: Neuphonic API key
        voice_id: ID of the voice to use
        url: Base URL for the Neuphonic API (default: "https://api.neuphonic.com")
        sample_rate: Sample rate for audio output (default: 22050Hz)
        encoding: Audio encoding format (default: "pcm_linear")
        params: Additional parameters for TTS generation including language and speed
        **kwargs: Additional keyword arguments passed to the parent class
    """

    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: Optional[str] = None,
        url: str = "https://api.neuphonic.com",
        sample_rate: Optional[int] = 22050,
        encoding: str = "pcm_linear",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or NeuphonicHttpTTSService.InputParams()

        self._api_key = api_key
        self._url = url
        self._settings = {
            "lang_code": self.language_to_service_language(params.language),
            "speed": params.speed,
            "encoding": encoding,
            "sampling_rate": sample_rate,
        }
        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)

    async def flush_audio(self):
        pass

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Neuphonic streaming API.

        Args:
            text: The text to convert to speech
        Yields:
            Frames containing audio data and status information
        """
        logger.debug(f"Generating TTS: [{text}]")

        client = Neuphonic(api_key=self._api_key, base_url=self._url.replace("https://", ""))

        sse = client.tts.AsyncSSEClient()

        try:
            await self.start_ttfb_metrics()
            response = sse.send(text, TTSConfig(**self._settings, voice_id=self._voice_id))

            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()

            async for message in response:
                if message.status_code != 200:
                    logger.error(f"{self} error: {message.errors}")
                    yield ErrorFrame(error=f"Neuphonic API error: {message.errors}")

                await self.stop_ttfb_metrics()
                yield TTSAudioRawFrame(message.data.audio, self.sample_rate, 1)
        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            yield TTSStoppedFrame()
