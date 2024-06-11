#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import time

from typing import AsyncGenerator

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    SystemFrame,
    TranscriptionFrame)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AIService, TTSService

from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)

from loguru import logger


class DeepgramTTSService(TTSService):

    def __init__(
            self,
            *,
            aiohttp_session: aiohttp.ClientSession,
            api_key: str,
            voice: str = "aura-helios-en",
            **kwargs):
        super().__init__(**kwargs)

        self._voice = voice
        self._api_key = api_key
        self._aiohttp_session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        base_url = "https://api.deepgram.com/v1/speak"
        request_url = f"{base_url}?model={self._voice}&encoding=linear16&container=none&sample_rate=16000"
        headers = {"authorization": f"token {self._api_key}"}
        body = {"text": text}

        try:
            await self.start_ttfb_metrics()
            async with self._aiohttp_session.post(request_url, headers=headers, json=body) as r:
                if r.status != 200:
                    text = await r.text()
                    logger.error(f"Error getting audio (status: {r.status}, error: {text})")
                    yield ErrorFrame(f"Error getting audio (status: {r.status}, error: {text})")
                    return

                async for data in r.content:
                    await self.stop_ttfb_metrics()
                    frame = AudioRawFrame(audio=data, sample_rate=16000, num_channels=1)
                    yield frame
        except Exception as e:
            logger.error(f"Deepgram exception: {e}")


class DeepgramSTTService(AIService):
    def __init__(self,
                 api_key: str,
                 live_options: LiveOptions = LiveOptions(
                     encoding="linear16",
                     language="en-US",
                     model="nova-2-conversationalai",
                     sample_rate=16000,
                     channels=1,
                     interim_results=True,
                     smart_format=True,
                 ),
                 **kwargs):
        super().__init__(**kwargs)

        self._live_options = live_options

        self._client = DeepgramClient(api_key)
        self._connection = self._client.listen.asynclive.v("1")
        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_message)

        self._create_push_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start()
            await self.push_frame(frame)
        elif isinstance(frame, CancelFrame):
            await self._stop()
            self._push_frame_task.cancel()
            await self.push_frame(frame)
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame)
        elif isinstance(frame, EndFrame):
            await self._stop()
            await self._push_queue.put((frame, direction))
            await self._push_frame_task
        elif isinstance(frame, AudioRawFrame):
            await self._connection.send(frame.audio)
        else:
            await self._push_queue.put((frame, direction))

    async def _start(self):
        if not await self._connection.start(self._live_options):
            logger.error("Unable to connect to Deepgram")

    async def _stop(self):
        await self._connection.finish()

    def _create_push_task(self):
        self._push_frame_task = self.get_event_loop().create_task(self._push_frame_task_handler())
        self._push_queue = asyncio.Queue()

    async def _push_frame_task_handler(self):
        running = True
        while running:
            try:
                (frame, direction) = await self._push_queue.get()
                await self.push_frame(frame, direction)
                running = not isinstance(frame, EndFrame)
            except asyncio.CancelledError:
                break

    async def _on_message(self, *args, **kwargs):
        result = kwargs["result"]
        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript
        if len(transcript) > 0:
            if is_final:
                await self._push_queue.put((TranscriptionFrame(transcript, "", int(time.time_ns() / 1000000)), FrameDirection.DOWNSTREAM))
            else:
                await self._push_queue.put((InterimTranscriptionFrame(transcript, "", int(time.time_ns() / 1000000)), FrameDirection.DOWNSTREAM))
