#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
from typing import AsyncGenerator

from loguru import logger

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
from pipecat.services.websocket_service import WebsocketService
from pipecat.transcriptions.language import Language

# See .env.example for LMNT configuration needed
try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use LMNT, you need to `pip install pipecat-ai[lmnt]`. Also, set `LMNT_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


def language_to_lmnt_language(language: Language) -> str | None:
    BASE_LANGUAGES = {
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.KO: "ko",
        Language.PT: "pt",
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


class LmntTTSService(TTSService, WebsocketService):
    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        sample_rate: int = 24000,
        language: Language = Language.EN,
        **kwargs,
    ):
        TTSService.__init__(
            self,
            push_stop_frames=True,
            sample_rate=sample_rate,
            **kwargs,
        )
        WebsocketService.__init__(self)

        self._api_key = api_key
        self._voice_id = voice_id
        self._settings = {
            "sample_rate": sample_rate,
            "language": self.language_to_service_language(language),
            "format": "raw",  # Use raw format for direct PCM data
        }
        self._started = False

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        return language_to_lmnt_language(language)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, StartInterruptionFrame)):
            self._started = False

    async def _connect(self):
        await self._connect_websocket()

        self._receive_task = self.get_event_loop().create_task(
            self._receive_task_handler(self.push_error)
        )

    async def _disconnect(self):
        await self._disconnect_websocket()

        if self._receive_task:
            self._receive_task.cancel()
            await self._receive_task
            self._receive_task = None

    async def _connect_websocket(self):
        """Connect to LMNT websocket."""
        try:
            logger.debug("Connecting to LMNT")

            # Build initial connection message
            init_msg = {
                "X-API-Key": self._api_key,
                "voice": self._voice_id,
                "format": self._settings["format"],
                "sample_rate": self._settings["sample_rate"],
                "language": self._settings["language"],
            }

            # Connect to LMNT's websocket directly
            self._websocket = await websockets.connect("wss://api.lmnt.com/v1/ai/speech/stream")

            # Send initialization message
            await self._websocket.send(json.dumps(init_msg))

        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _disconnect_websocket(self):
        """Disconnect from LMNT websocket."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from LMNT")
                # Send EOF message before closing
                await self._websocket.send(json.dumps({"eof": True}))
                await self._websocket.close()
                self._websocket = None

            self._started = False
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive messages from LMNT websocket."""
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                # Raw audio data
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=message,
                    sample_rate=self._settings["sample_rate"],
                    num_channels=1,
                )
                await self.push_frame(frame)
            else:
                try:
                    msg = json.loads(message)
                    if "error" in msg:
                        logger.error(f"{self} error: {msg['error']}")
                        await self.push_frame(TTSStoppedFrame())
                        await self.stop_all_metrics()
                        await self.push_error(ErrorFrame(f"{self} error: {msg['error']}"))
                        return
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON message: {message}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio from text."""
        logger.debug(f"Generating TTS: [{text}]")

        try:
            if not self._websocket:
                await self._connect()

            try:
                if not self._started:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._started = True

                # Send text to LMNT
                await self._get_websocket().send(json.dumps({"text": text}))
                # Force synthesis
                await self._get_websocket().send(json.dumps({"flush": True}))
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
