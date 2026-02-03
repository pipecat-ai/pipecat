import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import asyncio
import json
import urllib.parse
from typing import AsyncGenerator, Optional

import websockets
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.services.telnyx.utils import (
    DEFAULT_SAMPLE_RATE,
    STT_WS_URL,
    create_wav_header,
    get_api_key,
)


class TelnyxSTTService(STTService):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        input_format: str = "wav",
        transcription_engine: str = "telnyx",
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate or DEFAULT_SAMPLE_RATE, **kwargs)
        self._api_key = get_api_key(api_key)
        self._input_format = input_format
        self._transcription_engine = transcription_engine
        self._ws = None
        self._receive_task = None
        self._wav_header_sent = False

    def _build_url(self) -> str:
        params = {
            "transcription_engine": self._transcription_engine,
            "input_format": self._input_format,
        }
        return f"{STT_WS_URL}?{urllib.parse.urlencode(params)}"

    @override
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, EndFrame):
            await self._disconnect()

    @override
    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    @override
    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    @override
    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        if self._ws:
            return

        headers = {"Authorization": f"Bearer {self._api_key}"}

        self._ws = await websockets.connect(self._build_url(), extra_headers=headers)
        logger.info("Telnyx STT Connected")
        self._receive_task = asyncio.create_task(self._receive_messages())

    async def _disconnect(self):
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._wav_header_sent = False
            logger.info("Telnyx STT Disconnected")

    async def _receive_messages(self):
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)

                    if "error" in data:
                        logger.error(f"Telnyx STT API Error: {data}")
                        await self.push_frame(ErrorFrame(error=f"Telnyx Error: {data}"))
                        continue

                    transcript = data.get("transcript")
                    if transcript:
                        logger.debug(f"STT Transcript: {transcript}")
                        await self.push_frame(TranscriptionFrame(
                            text=transcript,
                            user_id="",
                            timestamp=""
                        ))

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from Telnyx: {message}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in Telnyx receive loop: {e}")
            await self.push_frame(ErrorFrame(error=f"Receive error: {e}"))

    @override
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._ws:
            await self._connect()
            if not self._ws:
                yield None
                return

        try:
            if not self._wav_header_sent:
                header = create_wav_header(sample_rate=self.sample_rate)
                await self._ws.send(header)
                self._wav_header_sent = True
                logger.debug("Sent WAV header to Telnyx")

            await self._ws.send(audio)

        except Exception as e:
            logger.error(f"Telnyx STT send error: {e}")
            await self._disconnect()

        yield None
