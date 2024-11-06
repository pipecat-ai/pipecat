#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import uuid
from typing import AsyncGenerator, List, Optional, Union

from loguru import logger
from pydantic.main import BaseModel

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
from pipecat.services.ai_services import TTSService, WordTTSService
from pipecat.transcriptions.language import Language

# See .env.example for Cartesia configuration needed
try:
    import websockets
    from cartesia import AsyncCartesia
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Cartesia, you need to `pip install pipecat-ai[cartesia]`. Also, set `CARTESIA_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


def language_to_cartesia_language(language: Language) -> str | None:
    match language:
        case Language.DE:
            return "de"
        case (
            Language.EN
            | Language.EN_US
            | Language.EN_GB
            | Language.EN_AU
            | Language.EN_NZ
            | Language.EN_IN
        ):
            return "en"
        case Language.ES:
            return "es"
        case Language.FR | Language.FR_CA:
            return "fr"
        case Language.JA:
            return "ja"
        case Language.PT | Language.PT_BR:
            return "pt"
        case Language.ZH | Language.ZH_TW:
            return "zh"
    return None


class CartesiaTTSService(WordTTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        speed: Optional[Union[str, float]] = ""
        emotion: Optional[List[str]] = []

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        cartesia_version: str = "2024-06-10",
        url: str = "wss://api.cartesia.ai/tts/websocket",
        model: str = "sonic-english",
        sample_rate: int = 24000,
        encoding: str = "pcm_s16le",
        container: str = "raw",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        # Aggregating sentences still gives cleaner-sounding results and fewer
        # artifacts than streaming one word at a time. On average, waiting for a
        # full sentence should only "cost" us 15ms or so with GPT-4o or a Llama
        # 3 model, and it's worth it for the better audio quality.
        #
        # We also don't want to automatically push LLM response text frames,
        # because the context aggregators will add them to the LLM context even
        # if we're interrupted. Cartesia gives us word-by-word timestamps. We
        # can use those to generate text frames ourselves aligned with the
        # playout timing of the audio!
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=False,
            sample_rate=sample_rate,
            **kwargs,
        )

        self._api_key = api_key
        self._cartesia_version = cartesia_version
        self._url = url
        self._settings = {
            "output_format": {
                "container": container,
                "encoding": encoding,
                "sample_rate": sample_rate,
            },
            "language": self.language_to_service_language(params.language)
            if params.language
            else Language.EN,
            "speed": params.speed,
            "emotion": params.emotion,
        }
        self.set_model_name(model)
        self.set_voice(voice_id)

        self._websocket = None
        self._context_id = None
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        self._model_id = model
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")

    def language_to_service_language(self, language: Language) -> str | None:
        return language_to_cartesia_language(language)

    def _build_msg(
        self, text: str = "", continue_transcript: bool = True, add_timestamps: bool = True
    ):
        voice_config = {}
        voice_config["mode"] = "id"
        voice_config["id"] = self._voice_id

        if self._settings["speed"] or self._settings["emotion"]:
            voice_config["__experimental_controls"] = {}
            if self._settings["speed"]:
                voice_config["__experimental_controls"]["speed"] = self._settings["speed"]
            if self._settings["emotion"]:
                voice_config["__experimental_controls"]["emotion"] = self._settings["emotion"]

        msg = {
            "transcript": text or " ",  # Text must contain at least one character
            "continue": continue_transcript,
            "context_id": self._context_id,
            "model_id": self.model_name,
            "voice": voice_config,
            "output_format": self._settings["output_format"],
            "language": self._settings["language"],
            "add_timestamps": add_timestamps,
        }
        return json.dumps(msg)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        try:
            self._websocket = await websockets.connect(
                f"{self._url}?api_key={self._api_key}&cartesia_version={self._cartesia_version}"
            )
            self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _disconnect(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                await self._websocket.close()
                self._websocket = None

            if self._receive_task:
                self._receive_task.cancel()
                await self._receive_task
                self._receive_task = None

            self._context_id = None
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        self._context_id = None

    async def flush_audio(self):
        if not self._context_id or not self._websocket:
            return
        logger.trace("Flushing audio")
        msg = self._build_msg(text="", continue_transcript=False)
        await self._websocket.send(msg)

    async def _receive_task_handler(self):
        try:
            async for message in self._get_websocket():
                msg = json.loads(message)
                if not msg or msg["context_id"] != self._context_id:
                    continue
                if msg["type"] == "done":
                    await self.stop_ttfb_metrics()
                    # Unset _context_id but not the _context_id_start_timestamp
                    # because we are likely still playing out audio and need the
                    # timestamp to set send context frames.
                    self._context_id = None
                    await self.add_word_timestamps(
                        [("TTSStoppedFrame", 0), ("LLMFullResponseEndFrame", 0)]
                    )
                elif msg["type"] == "timestamps":
                    await self.add_word_timestamps(
                        list(zip(msg["word_timestamps"]["words"], msg["word_timestamps"]["start"]))
                    )
                elif msg["type"] == "chunk":
                    await self.stop_ttfb_metrics()
                    self.start_word_timestamps()
                    frame = TTSAudioRawFrame(
                        audio=base64.b64decode(msg["data"]),
                        sample_rate=self._settings["output_format"]["sample_rate"],
                        num_channels=1,
                    )
                    await self.push_frame(frame)
                elif msg["type"] == "error":
                    logger.error(f"{self} error: {msg}")
                    await self.push_frame(TTSStoppedFrame())
                    await self.stop_all_metrics()
                    await self.push_error(ErrorFrame(f'{self} error: {msg["error"]}'))
                else:
                    logger.error(f"Cartesia error, unknown message type: {msg}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"{self} exception: {e}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            if not self._websocket:
                await self._connect()

            if not self._context_id:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._context_id = str(uuid.uuid4())

            msg = self._build_msg(text=text or " ")  # Text must contain at least one character

            try:
                await self._get_websocket().send(msg)
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


class CartesiaHttpTTSService(TTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        speed: Optional[Union[str, float]] = ""
        emotion: Optional[List[str]] = []

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "sonic-english",
        base_url: str = "https://api.cartesia.ai",
        sample_rate: int = 24000,
        encoding: str = "pcm_s16le",
        container: str = "raw",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._settings = {
            "output_format": {
                "container": container,
                "encoding": encoding,
                "sample_rate": sample_rate,
            },
            "language": self.language_to_service_language(params.language)
            if params.language
            else Language.EN,
            "speed": params.speed,
            "emotion": params.emotion,
        }
        self.set_voice(voice_id)
        self.set_model_name(model)

        self._client = AsyncCartesia(api_key=api_key, base_url=base_url)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        return language_to_cartesia_language(language)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.close()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        try:
            voice_controls = None
            if self._settings["speed"] or self._settings["emotion"]:
                voice_controls = {}
                if self._settings["speed"]:
                    voice_controls["speed"] = self._settings["speed"]
                if self._settings["emotion"]:
                    voice_controls["emotion"] = self._settings["emotion"]

            output = await self._client.tts.sse(
                model_id=self._model_name,
                transcript=text,
                voice_id=self._voice_id,
                output_format=self._settings["output_format"],
                language=self._settings["language"],
                stream=False,
                _experimental_voice_controls=voice_controls,
            )

            await self.stop_ttfb_metrics()

            frame = TTSAudioRawFrame(
                audio=output["audio"],
                sample_rate=self._settings["output_format"]["sample_rate"],
                num_channels=1,
            )
            yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")

        await self.start_tts_usage_metrics(text)
        yield TTSStoppedFrame()
