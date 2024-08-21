#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import AsyncGenerator

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AsyncAIService, TTSService
from pipecat.utils.time import time_now_iso8601

# See .env.example for Deepgram configuration needed
try:
    from deepgram import (
        DeepgramClient,
        DeepgramClientOptions,
        LiveOptions,
        LiveTranscriptionEvents,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Deepgram, you need to `pip install pipecat-ai[deepgram]`. Also, set `DEEPGRAM_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class DeepgramTTSService(TTSService):

    def __init__(
            self,
            *,
            api_key: str,
            aiohttp_session: aiohttp.ClientSession,
            voice: str = "aura-helios-en",
            base_url: str = "https://api.deepgram.com/v1/speak",
            sample_rate: int = 16000,
            encoding: str = "linear16",
            **kwargs):
        super().__init__(**kwargs)

        self._voice = voice
        self._api_key = api_key
        self._base_url = base_url
        self._sample_rate = sample_rate
        self._encoding = encoding
        self._aiohttp_session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        return True

    async def set_voice(self, voice: str):
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice = voice

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        base_url = self._base_url
        request_url = f"{base_url}?model={self._voice}&encoding={self._encoding}&container=none&sample_rate={self._sample_rate}"
        headers = {"authorization": f"token {self._api_key}"}
        body = {"text": text}

        try:
            await self.start_ttfb_metrics()
            async with self._aiohttp_session.post(request_url, headers=headers, json=body) as r:
                if r.status != 200:
                    response_text = await r.text()
                    # If we get a a "Bad Request: Input is unutterable", just print out a debug log.
                    # All other unsuccesful requests should emit an error frame. If not specifically
                    # handled by the running PipelineTask, the ErrorFrame will cancel the task.
                    if "unutterable" in response_text:
                        logger.debug(f"Unutterable text: [{text}]")
                        return

                    logger.error(
                        f"{self} error getting audio (status: {r.status}, error: {response_text})")
                    yield ErrorFrame(f"Error getting audio (status: {r.status}, error: {response_text})")
                    return

                await self.start_tts_usage_metrics(text)

                await self.push_frame(TTSStartedFrame())
                async for data in r.content:
                    await self.stop_ttfb_metrics()
                    frame = AudioRawFrame(audio=data, sample_rate=self._sample_rate, num_channels=1)
                    yield frame
                await self.push_frame(TTSStoppedFrame())
        except Exception as e:
            logger.exception(f"{self} exception: {e}")


class DeepgramSTTService(AsyncAIService):
    def __init__(self,
                 *,
                 api_key: str,
                 url: str = "",
                 live_options: LiveOptions = LiveOptions(
                     encoding="linear16",
                     language="en-US",
                     model="nova-2-conversationalai",
                     sample_rate=16000,
                     channels=1,
                     interim_results=True,
                     smart_format=True,
                 ),
                 mute_during_speech=False,
                 **kwargs):
        super().__init__(**kwargs)

        self._live_options = live_options

        self._client = DeepgramClient(
            api_key, config=DeepgramClientOptions(url=url, options={"keepalive": "true"}))
        self._connection = self._client.listen.asynclive.v("1")
        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
        self.mute_during_speech = mute_during_speech
        self.bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # print(f"Is Frame BotStartedSpeakingFrame: {isinstance(Frame, BotStartedSpeakingFrame)}")
        # print(f"Frame: {frame}")
        if isinstance(frame, BotStartedSpeakingFrame):
            print("Bot Speaking")
            self.bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            print("Bot Stopped Speaking")
            self.bot_speaking = False
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, AudioRawFrame):
            # print(f"AUDIO RAW FRAME: {frame}")
            if not (self.mute_during_speech and self.bot_speaking):
                await self._connection.send(frame.audio)
            else:
                print("Bot Speaking")
        else:
            await self.queue_frame(frame, direction)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if await self._connection.start(self._live_options):
            logger.debug(f"{self}: Connected to Deepgram")
        else:
            logger.error(f"{self}: Unable to connect to Deepgram")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._connection.finish()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._connection.finish()

    async def _on_message(self, *args, **kwargs):
        # print(f"ON MESSAGE: {args}, {kwargs}")
        result = kwargs["result"]
        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript
        if len(transcript) > 0:
            if is_final:
                await self.queue_frame(TranscriptionFrame(transcript, "", time_now_iso8601()))
            else:
                await self.queue_frame(InterimTranscriptionFrame(transcript, "", time_now_iso8601()))
