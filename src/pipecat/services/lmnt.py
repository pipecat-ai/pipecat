#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import AsyncGenerator

from loguru import logger
from tenacity import AsyncRetrying, RetryCallState, stop_after_attempt, wait_exponential

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

# See .env.example for LMNT configuration needed
try:
    from lmnt.api import Speech
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


class LmntTTSService(TTSService):
    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        sample_rate: int = 24000,
        language: Language = Language.EN,
        **kwargs,
    ):
        # Let TTSService produce TTSStoppedFrames after a short delay of
        # no activity.
        super().__init__(push_stop_frames=True, sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._settings = {
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": sample_rate,
            },
            "language": self.language_to_service_language(language),
        }

        self.set_voice(voice_id)

        self._speech = None
        self._connection = None
        self._receive_task = None
        # Indicates if we have sent TTSStartedFrame. It will reset to False when
        # there's an interruption or TTSStoppedFrame.
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
        await self._connect_lmnt()

        self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())

    async def _disconnect(self):
        await self._disconnect_lmnt()

        if self._receive_task:
            self._receive_task.cancel()
            await self._receive_task
            self._receive_task = None

    async def _connect_lmnt(self):
        try:
            logger.debug("Connecting to LMNT")

            self._speech = Speech()
            self._connection = await self._speech.synthesize_streaming(
                self._voice_id,
                format="raw",
                sample_rate=self._settings["output_format"]["sample_rate"],
                language=self._settings["language"],
            )
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._connection = None

    async def _disconnect_lmnt(self):
        try:
            await self.stop_all_metrics()

            if self._connection:
                logger.debug("Disconnecting from LMNT")
                await self._connection.socket.close()
                self._connection = None
            if self._speech:
                await self._speech.close()
                self._speech = None

            self._started = False
        except Exception as e:
            logger.error(f"{self} error closing connection: {e}")

    async def _receive_messages(self):
        async for msg in self._connection:
            if "error" in msg:
                logger.error(f'{self} error: {msg["error"]}')
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()
                await self.push_error(ErrorFrame(f'{self} error: {msg["error"]}'))
            elif "audio" in msg:
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=msg["audio"],
                    sample_rate=self._settings["output_format"]["sample_rate"],
                    num_channels=1,
                )
                await self.push_frame(frame)
            else:
                logger.error(f"{self}: LMNT error, unknown message type: {msg}")

    async def _reconnect_websocket(self, retry_state: RetryCallState):
        logger.warning(f"{self} reconnecting (attempt: {retry_state.attempt_number})")
        await self._disconnect_lmnt()
        await self._connect_lmnt()

    async def _receive_task_handler(self):
        while True:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=4, max=10),
                    before_sleep=self._reconnect_websocket,
                    reraise=True,
                ):
                    with attempt:
                        await self._receive_messages()
            except asyncio.CancelledError:
                break
            except Exception as e:
                message = f"{self} error receiving messages: {e}"
                logger.error(message)
                await self.push_error(ErrorFrame(message, fatal=True))
                break

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            if not self._connection:
                await self._connect()

            if not self._started:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._started = True

            try:
                await self._connection.append_text(text)
                await self._connection.flush()
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
