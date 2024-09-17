#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp

from typing import AsyncGenerator

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TranscriptionFrame)
from pipecat.services.ai_services import STTService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from loguru import logger


# See .env.example for Deepgram configuration needed
try:
    from deepgram import (
        AsyncListenWebSocketClient,
        DeepgramClient,
        DeepgramClientOptions,
        LiveTranscriptionEvents,
        LiveOptions,
        LiveResultResponse
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


class DeepgramSTTService(STTService):
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
                     punctuate=True,
                     profanity_filter=True,
                 ),
                 **kwargs):
        super().__init__(**kwargs)

        self._live_options = live_options

        self._client = DeepgramClient(
            api_key, config=DeepgramClientOptions(url=url, options={"keepalive": "true"}))
        self._connection: AsyncListenWebSocketClient = self._client.listen.asyncwebsocket.v("1")
        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_message)

    async def set_model(self, model: str):
        await super().set_model(model)
        logger.debug(f"Switching STT model to: [{model}]")
        self._live_options.model = model
        await self._disconnect()
        await self._connect()

    async def set_language(self, language: Language):
        logger.debug(f"Switching STT language to: [{language}]")
        self._live_options.language = language
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self.start_processing_metrics()
        await self._connection.send(audio)
        await self.stop_processing_metrics()
        yield None

    async def _connect(self):
        if await self._connection.start(self._live_options):
            logger.debug(f"{self}: Connected to Deepgram")
        else:
            logger.error(f"{self}: Unable to connect to Deepgram")

    async def _disconnect(self):
        if self._connection.is_connected:
            await self._connection.finish()
            logger.debug(f"{self}: Disconnected from Deepgram")

    async def _on_message(self, *args, **kwargs):
        result: LiveResultResponse = kwargs["result"]
        if len(result.channel.alternatives) == 0:
            return
        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript
        language = None
        if result.channel.alternatives[0].languages:
            language = result.channel.alternatives[0].languages[0]
            language = Language(language)
        if len(transcript) > 0:
            if is_final:
                await self.push_frame(TranscriptionFrame(transcript, "", time_now_iso8601(), language))
            else:
                await self.push_frame(InterimTranscriptionFrame(transcript, "", time_now_iso8601(), language))
