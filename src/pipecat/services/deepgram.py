#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import time

from pipecat.frames.frames import (
    AudioRawFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
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

    async def run_tts(self, text: str):
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
                    await self.push_error(ErrorFrame(f"Error getting audio (status: {r.status}, error: {text})"))
                    return

                async for data in r.content:
                    await self.stop_ttfb_metrics()
                    frame = AudioRawFrame(audio=data, sample_rate=16000, num_channels=1)
                    await self.push_service_frame(frame)
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

    async def start(self, frame: StartFrame):
        await super().start(frame)

        if not await self._connection.start(self._live_options):
            logger.error("Unable to connect to Deepgram")

    async def stop(self):
        await super().stop()
        await self._connection.finish()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame of audio data, either buffering or transcribing it."""
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            await self._connection.send(frame.audio)
        else:
            await self.push_service_frame(frame, direction)

    async def _on_message(self, *args, **kwargs):
        result = kwargs["result"]
        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript
        if len(transcript) > 0:
            if is_final:
                await self.push_service_frame(TranscriptionFrame(transcript, "", int(time.time_ns() / 1000000)))
            else:
                await self.push_service_frame(InterimTranscriptionFrame(transcript, "", int(time.time_ns() / 1000000)))
