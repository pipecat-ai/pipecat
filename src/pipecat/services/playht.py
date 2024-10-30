#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
import json
import struct
import uuid
from typing import AsyncGenerator, Optional

import aiohttp
import websockets
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
from pipecat.services.ai_services import TTSService
from pipecat.transcriptions.language import Language

try:
    from pyht.async_client import AsyncClient
    from pyht.client import TTSOptions
    from pyht.protos.api_pb2 import Format
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use PlayHT, you need to `pip install pipecat-ai[playht]`. Also, set `PLAY_HT_USER_ID` and `PLAY_HT_API_KEY` environment variables."
    )
    raise Exception(f"Missing module: {e}")


def language_to_playht_language(language: Language) -> str | None:
    match language:
        case Language.BG:
            return "BULGARIAN"
        case Language.CA:
            return "CATALAN"
        case Language.CS:
            return "CZECH"
        case Language.DA:
            return "DANISH"
        case Language.DE:
            return "GERMAN"
        case (
            Language.EN
            | Language.EN_US
            | Language.EN_GB
            | Language.EN_AU
            | Language.EN_NZ
            | Language.EN_IN
        ):
            return "ENGLISH"
        case Language.ES:
            return "SPANISH"
        case Language.FR | Language.FR_CA:
            return "FRENCH"
        case Language.EL:
            return "GREEK"
        case Language.HI:
            return "HINDI"
        case Language.HU:
            return "HUNGARIAN"
        case Language.ID:
            return "INDONESIAN"
        case Language.IT:
            return "ITALIAN"
        case Language.JA:
            return "JAPANESE"
        case Language.KO:
            return "KOREAN"
        case Language.MS:
            return "MALAY"
        case Language.NL:
            return "DUTCH"
        case Language.PL:
            return "POLISH"
        case Language.PT | Language.PT_BR:
            return "PORTUGUESE"
        case Language.RU:
            return "RUSSIAN"
        case Language.SV:
            return "SWEDISH"
        case Language.TH:
            return "THAI"
        case Language.TR:
            return "TURKISH"
        case Language.UK:
            return "UKRAINIAN"
    return None


class PlayHTTTSService(TTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0
        seed: Optional[int] = None

    def __init__(
        self,
        *,
        api_key: str,
        user_id: str,
        voice_url: str,
        voice_engine: str = "PlayHT3.0-mini",
        sample_rate: int = 24000,
        output_format: str = "wav",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._user_id = user_id
        self._websocket_url = None
        self._websocket = None
        self._receive_task = None
        self._request_id = None

        self._settings = {
            "sample_rate": sample_rate,
            "language": self.language_to_service_language(params.language)
            if params.language
            else Language.EN,
            "output_format": output_format,
            "voice_engine": voice_engine,
            "speed": params.speed,
            "seed": params.seed,
        }
        self.set_model_name(voice_engine)
        self.set_voice(voice_url)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        # Keep your existing language mapping logic here
        pass

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
            if not self._websocket_url:
                await self._get_websocket_url()

            if not isinstance(self._websocket_url, str):
                raise ValueError("WebSocket URL is not a string")

            self._websocket = await websockets.connect(self._websocket_url)
            self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
            logger.debug("Connected to TTS WebSocket")
        except ValueError as ve:
            logger.error(f"{self} initialization error: {ve}")
            self._websocket = None
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

            self._request_id = None
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    async def _get_websocket_url(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.play.ht/api/v3/websocket-auth",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "X-User-Id": self._user_id,
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status in (200, 201):
                    data = await response.json()
                    if "websocket_url" in data and isinstance(data["websocket_url"], str):
                        self._websocket_url = data["websocket_url"]
                    else:
                        raise ValueError("Invalid or missing WebSocket URL in response")
                else:
                    raise Exception(f"Failed to get WebSocket URL: {response.status}")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        self._request_id = None

    async def _receive_task_handler(self):
        try:
            header_size = 78  # Size of the WAV header + extra bytes we want to skip
            header_received = False
            async for message in self._get_websocket():
                if isinstance(message, bytes):
                    chunk_size = len(message)

                    # Skip the WAV header
                    if not header_received and chunk_size == header_size:
                        header_received = True
                        continue

                    await self.stop_ttfb_metrics()
                    frame = TTSAudioRawFrame(message, self._settings["sample_rate"], 1)
                    await self.push_frame(frame)
                else:
                    logger.debug(f"Received text message: {message}")
                    try:
                        msg = json.loads(message)
                        if "request_id" in msg and msg["request_id"] == self._request_id:
                            await self.push_frame(TTSStoppedFrame())
                            header_received = False  # Reset for the next audio stream
                            self._request_id = None
                        elif "error" in msg:
                            logger.error(f"{self} error: {msg}")
                            await self.push_error(ErrorFrame(f'{self} error: {msg["error"]}'))
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON message: {message}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"{self} exception in receive task: {e}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            # Reconnect if the websocket is closed
            if not self._websocket or self._websocket.closed:
                await self._connect()

            if not self._request_id:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._request_id = str(uuid.uuid4())

            tts_command = {
                "text": text,
                "voice": self._voice_id,
                "voice_engine": self._settings["voice_engine"],
                "output_format": self._settings["output_format"],
                "sample_rate": self._settings["sample_rate"],
                "language": self._settings["language"],
                "speed": self._settings["speed"],
                "seed": self._settings["seed"],
                "request_id": self._request_id,
            }

            try:
                await self._get_websocket().send(json.dumps(tts_command))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()
                return

            # The actual audio frames will be handled in _receive_task_handler
            yield None

        except Exception as e:
            logger.error(f"{self} error generating TTS: {e}")
            yield ErrorFrame(f"{self} error: {str(e)}")


class PlayHTHttpTTSService(TTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0
        seed: Optional[int] = None

    def __init__(
        self,
        *,
        api_key: str,
        user_id: str,
        voice_url: str,
        voice_engine: str = "PlayHT3.0-mini",
        sample_rate: int = 24000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._user_id = user_id
        self._api_key = api_key

        self._client = AsyncClient(
            user_id=self._user_id,
            api_key=self._api_key,
        )
        self._settings = {
            "sample_rate": sample_rate,
            "language": self.language_to_service_language(params.language)
            if params.language
            else Language.EN,
            "format": Format.FORMAT_WAV,
            "voice_engine": voice_engine,
            "speed": params.speed,
            "seed": params.seed,
        }
        self.set_model_name(voice_engine)
        self.set_voice(voice_url)
        self._options = TTSOptions(
            voice=self._voice_id,
            language=self._settings["language"],
            sample_rate=self._settings["sample_rate"],
            format=self._settings["format"],
            speed=self._settings["speed"],
            seed=self._settings["seed"],
        )

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        return language_to_playht_language(language)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            b = bytearray()
            in_header = True

            await self.start_ttfb_metrics()

            playht_gen = self._client.tts(
                text, voice_engine=self._settings["voice_engine"], options=self._options
            )

            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()
            async for chunk in playht_gen:
                # skip the RIFF header.
                if in_header:
                    b.extend(chunk)
                    if len(b) <= 36:
                        continue
                    else:
                        fh = io.BytesIO(b)
                        fh.seek(36)
                        (data, size) = struct.unpack("<4sI", fh.read(8))
                        while data != b"data":
                            fh.read(size)
                            (data, size) = struct.unpack("<4sI", fh.read(8))
                        in_header = False
                else:
                    if len(chunk):
                        await self.stop_ttfb_metrics()
                        frame = TTSAudioRawFrame(chunk, self._settings["sample_rate"], 1)
                        yield frame
            yield TTSStoppedFrame()
        except Exception as e:
            logger.exception(f"{self} error generating TTS: {e}")
