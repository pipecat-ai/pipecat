#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import uuid
import base64
import asyncio

from typing import AsyncGenerator

from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    Frame,
    AudioRawFrame,
    StartInterruptionFrame,
    StartFrame,
    EndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    LLMFullResponseEndFrame
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transcriptions.language import Language
from pipecat.services.ai_services import AsyncWordTTSService, TTSService

from loguru import logger

# See .env.example for Cartesia configuration needed
try:
    from cartesia import AsyncCartesia
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Cartesia, you need to `pip install pipecat-ai[cartesia]`. Also, set `CARTESIA_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


def language_to_cartesia_language(language: Language) -> str | None:
    match language:
        case Language.DE:
            return "de"
        case Language.EN:
            return "en"
        case Language.ES:
            return "es"
        case Language.FR:
            return "fr"
        case Language.JA:
            return "ja"
        case Language.PT:
            return "pt"
        case Language.ZH:
            return "zh"
    return None


class CartesiaTTSService(AsyncWordTTSService):

    def __init__(
            self,
            *,
            api_key: str,
            voice_id: str,
            cartesia_version: str = "2024-06-10",
            url: str = "wss://api.cartesia.ai/tts/websocket",
            model_id: str = "sonic-english",
            encoding: str = "pcm_s16le",
            sample_rate: int = 16000,
            language: str = "en",
            **kwargs):
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
        super().__init__(aggregate_sentences=True, push_text_frames=False, sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._cartesia_version = cartesia_version
        self._url = url
        self._voice_id = voice_id
        self.set_model_name(model_id)
        self._output_format = {
            "container": "raw",
            "encoding": encoding,
            "sample_rate": sample_rate,
        }
        self._language = language

        self._websocket = None
        self._context_id = None
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        await super().set_model(model)
        logger.debug(f"Switching TTS model to: [{model}]")

    async def set_voice(self, voice: str):
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice_id = voice

    async def set_language(self, language: Language):
        logger.debug(f"Switching TTS language to: [{language}]")
        self._language = language_to_cartesia_language(language)

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
        await self.push_frame(LLMFullResponseEndFrame())
        self._context_id = None

    async def flush_audio(self):
        if not self._context_id or not self._websocket:
            return
        logger.trace("Flushing audio")
        msg = {
            "transcript": "",
            "continue": False,
            "context_id": self._context_id,
            "model_id": self.model_name,
            "voice": {
                "mode": "id",
                "id": self._voice_id
            },
            "output_format": self._output_format,
            "language": self._language,
            "add_timestamps": True,
        }
        await self._websocket.send(json.dumps(msg))

    async def _receive_task_handler(self):
        try:
            async for message in self._get_websocket():
                msg = json.loads(message)
                if not msg or msg["context_id"] != self._context_id:
                    continue
                if msg["type"] == "done":
                    await self.stop_ttfb_metrics()
                    await self.push_frame(TTSStoppedFrame())
                    # Unset _context_id but not the _context_id_start_timestamp
                    # because we are likely still playing out audio and need the
                    # timestamp to set send context frames.
                    self._context_id = None
                    await self.add_word_timestamps([("LLMFullResponseEndFrame", 0)])
                elif msg["type"] == "timestamps":
                    await self.add_word_timestamps(
                        list(zip(msg["word_timestamps"]["words"], msg["word_timestamps"]["start"]))
                    )
                elif msg["type"] == "chunk":
                    await self.stop_ttfb_metrics()
                    self.start_word_timestamps()
                    frame = AudioRawFrame(
                        audio=base64.b64decode(msg["data"]),
                        sample_rate=self._output_format["sample_rate"],
                        num_channels=1
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
                await self.push_frame(TTSStartedFrame())
                await self.start_ttfb_metrics()
                self._context_id = str(uuid.uuid4())

            msg = {
                "transcript": text + " ",
                "continue": True,
                "context_id": self._context_id,
                "model_id": self.model_name,
                "voice": {
                    "mode": "id",
                    "id": self._voice_id
                },
                "output_format": self._output_format,
                "language": self._language,
                "add_timestamps": True,
            }
            try:
                await self._get_websocket().send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                await self.push_frame(TTSStoppedFrame())
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")


class CartesiaHttpTTSService(TTSService):

    def __init__(
            self,
            *,
            api_key: str,
            voice_id: str,
            model_id: str = "sonic-english",
            base_url: str = "https://api.cartesia.ai",
            encoding: str = "pcm_s16le",
            sample_rate: int = 16000,
            language: str = "en",
            **kwargs):
        super().__init__(**kwargs)

        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = {
            "container": "raw",
            "encoding": encoding,
            "sample_rate": sample_rate,
        }
        self._language = language

        self._client = AsyncCartesia(api_key=api_key, base_url=base_url)

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        logger.debug(f"Switching TTS model to: [{model}]")
        self._model_id = model

    async def set_voice(self, voice: str):
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice_id = voice

    async def set_language(self, language: Language):
        logger.debug(f"Switching TTS language to: [{language}]")
        self._language = language_to_cartesia_language(language)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.close()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        await self.push_frame(TTSStartedFrame())
        await self.start_ttfb_metrics()

        try:
            output = await self._client.tts.sse(
                model_id=self._model_id,
                transcript=text,
                voice_id=self._voice_id,
                output_format=self._output_format,
                language=self._language,
                stream=False
            )

            await self.stop_ttfb_metrics()

            frame = AudioRawFrame(
                audio=output["audio"],
                sample_rate=self._output_format["sample_rate"],
                num_channels=1
            )
            yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")

        await self.start_tts_usage_metrics(text)
        await self.push_frame(TTSStoppedFrame())
