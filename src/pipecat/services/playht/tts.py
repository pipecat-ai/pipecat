#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import json
import struct
import uuid
from typing import AsyncGenerator, Optional

import aiohttp
import websockets
from loguru import logger
from pydantic import BaseModel

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
from pipecat.services.tts_service import InterruptibleTTSService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from pyht.async_client import AsyncClient
    from pyht.client import Format, TTSOptions
    from pyht.client import Language as PlayHTLanguage
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use PlayHT, you need to `pip install pipecat-ai[playht]`.")
    raise Exception(f"Missing module: {e}")


def language_to_playht_language(language: Language) -> Optional[str]:
    BASE_LANGUAGES = {
        Language.AF: "afrikans",
        Language.AM: "amharic",
        Language.AR: "arabic",
        Language.BN: "bengali",
        Language.BG: "bulgarian",
        Language.CA: "catalan",
        Language.CS: "czech",
        Language.DA: "danish",
        Language.DE: "german",
        Language.EL: "greek",
        Language.EN: "english",
        Language.ES: "spanish",
        Language.FR: "french",
        Language.GL: "galician",
        Language.HE: "hebrew",
        Language.HI: "hindi",
        Language.HR: "croatian",
        Language.HU: "hungarian",
        Language.ID: "indonesian",
        Language.IT: "italian",
        Language.JA: "japanese",
        Language.KO: "korean",
        Language.MS: "malay",
        Language.NL: "dutch",
        Language.PL: "polish",
        Language.PT: "portuguese",
        Language.RU: "russian",
        Language.SQ: "albanian",
        Language.SR: "serbian",
        Language.SV: "swedish",
        Language.TH: "thai",
        Language.TL: "tagalog",
        Language.TR: "turkish",
        Language.UK: "ukrainian",
        Language.UR: "urdu",
        Language.XH: "xhosa",
        Language.ZH: "mandarin",
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


class PlayHTTTSService(InterruptibleTTSService):
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
        voice_engine: str = "Play3.0-mini",
        sample_rate: Optional[int] = None,
        output_format: str = "wav",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or PlayHTTTSService.InputParams()

        self._api_key = api_key
        self._user_id = user_id
        self._websocket_url = None
        self._receive_task = None
        self._request_id = None

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "english",
            "output_format": output_format,
            "voice_engine": voice_engine,
            "speed": params.speed,
            "seed": params.seed,
        }
        self.set_model_name(voice_engine)
        self.set_voice(voice_url)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_playht_language(language)

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

            logger.debug("Connecting to PlayHT")

            if not self._websocket_url:
                await self._get_websocket_url()

            if not isinstance(self._websocket_url, str):
                raise ValueError("WebSocket URL is not a string")

            self._websocket = await websockets.connect(self._websocket_url)
        except ValueError as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from PlayHT")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._request_id = None
            self._websocket = None

    async def _get_websocket_url(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.play.ht/api/v4/websocket-auth",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "X-User-Id": self._user_id,
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status in (200, 201):
                    data = await response.json()
                    # Handle the new response format with multiple URLs
                    if "websocket_urls" in data:
                        # Select URL based on voice_engine
                        if self._settings["voice_engine"] in data["websocket_urls"]:
                            self._websocket_url = data["websocket_urls"][
                                self._settings["voice_engine"]
                            ]
                        else:
                            raise ValueError(
                                f"Unsupported voice engine: {self._settings['voice_engine']}"
                            )
                    else:
                        raise ValueError("Invalid response: missing websocket_urls")
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

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                # Skip the WAV header message
                if message.startswith(b"RIFF"):
                    continue
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(message, self.sample_rate, 1)
                await self.push_frame(frame)
            else:
                logger.debug(f"Received text message: {message}")
                try:
                    msg = json.loads(message)
                    if msg.get("type") == "start":
                        # Handle start of stream
                        logger.debug(f"Started processing request: {msg.get('request_id')}")
                    elif msg.get("type") == "end":
                        # Handle end of stream
                        if "request_id" in msg and msg["request_id"] == self._request_id:
                            await self.push_frame(TTSStoppedFrame())
                            self._request_id = None
                    elif "error" in msg:
                        logger.error(f"{self} error: {msg}")
                        await self.push_error(ErrorFrame(f"{self} error: {msg['error']}"))
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON message: {message}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

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
                "sample_rate": self.sample_rate,
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
        voice_engine: str = "Play3.0-mini",
        protocol: str = "http",  # Options: http, ws
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or PlayHTHttpTTSService.InputParams()

        self._user_id = user_id
        self._api_key = api_key

        self._client = AsyncClient(
            user_id=self._user_id,
            api_key=self._api_key,
        )

        # Check if voice_engine contains protocol information (backward compatibility)
        if "-http" in voice_engine:
            # Extract the base engine name
            voice_engine = voice_engine.replace("-http", "")
            protocol = "http"
        elif "-ws" in voice_engine:
            # Extract the base engine name
            voice_engine = voice_engine.replace("-ws", "")
            protocol = "ws"

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "english",
            "format": Format.FORMAT_WAV,
            "voice_engine": voice_engine,
            "protocol": protocol,
            "speed": params.speed,
            "seed": params.seed,
        }
        self.set_model_name(voice_engine)
        self.set_voice(voice_url)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._settings["sample_rate"] = self.sample_rate

    def _create_options(self) -> TTSOptions:
        language_str = self._settings["language"]
        playht_language = None
        if language_str:
            # Convert string to PlayHT Language enum
            for lang in PlayHTLanguage:
                if lang.value == language_str:
                    playht_language = lang
                    break

        return TTSOptions(
            voice=self._voice_id,
            language=playht_language,
            sample_rate=self.sample_rate,
            format=self._settings["format"],
            speed=self._settings["speed"],
            seed=self._settings["seed"],
        )

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_playht_language(language)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            options = self._create_options()

            await self.start_ttfb_metrics()

            playht_gen = self._client.tts(
                text,
                voice_engine=self._settings["voice_engine"],
                protocol=self._settings["protocol"],
                options=options,
            )

            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            b = bytearray()
            in_header = True
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
                elif len(chunk) > 0:
                    await self.stop_ttfb_metrics()
                    frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                    yield frame
        except Exception as e:
            logger.error(f"{self} error generating TTS: {e}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
