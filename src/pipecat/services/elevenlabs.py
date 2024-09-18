#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json

from typing import Any, AsyncGenerator, List, Literal, Mapping, Tuple
from pydantic import BaseModel

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame)
from pipecat.metrics.metrics import TTSUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AsyncWordTTSService

from loguru import logger

# See .env.example for ElevenLabs configuration needed
try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use ElevenLabs, you need to `pip install pipecat-ai[elevenlabs]`. Also, set `ELEVENLABS_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


def sample_rate_from_output_format(output_format: str) -> int:
    match output_format:
        case "pcm_16000":
            return 16000
        case "pcm_22050":
            return 22050
        case "pcm_24000":
            return 24000
        case "pcm_44100":
            return 44100
    return 16000


def calculate_word_times(
        alignment_info: Mapping[str, Any], cumulative_time: float
) -> List[Tuple[str, float]]:
    zipped_times = list(zip(alignment_info["chars"], alignment_info["charStartTimesMs"]))

    words = "".join(alignment_info["chars"]).split(" ")

    # Calculate start time for each word. We do this by finding a space character
    # and using the previous word time, also taking into account there might not
    # be a space at the end.
    times = []
    for (i, (a, b)) in enumerate(zipped_times):
        if a == " " or i == len(zipped_times) - 1:
            t = cumulative_time + (zipped_times[i - 1][1] / 1000.0)
            times.append(t)

    word_times = list(zip(words, times))

    return word_times


class ElevenLabsTTSService(AsyncWordTTSService):
    class InputParams(BaseModel):
        output_format: Literal["pcm_16000", "pcm_22050", "pcm_24000", "pcm_44100"] = "pcm_16000"

    def __init__(
            self,
            *,
            api_key: str,
            voice_id: str,
            model: str = "eleven_turbo_v2_5",
            url: str = "wss://api.elevenlabs.io",
            params: InputParams = InputParams(),
            **kwargs):
        # Aggregating sentences still gives cleaner-sounding results and fewer
        # artifacts than streaming one word at a time. On average, waiting for a
        # full sentence should only "cost" us 15ms or so with GPT-4o or a Llama
        # 3 model, and it's worth it for the better audio quality.
        #
        # We also don't want to automatically push LLM response text frames,
        # because the context aggregators will add them to the LLM context even
        # if we're interrupted. ElevenLabs gives us word-by-word timestamps. We
        # can use those to generate text frames ourselves aligned with the
        # playout timing of the audio!
        #
        # Finally, ElevenLabs doesn't provide information on when the bot stops
        # speaking for a while, so we want the parent class to send TTSStopFrame
        # after a short period not receiving any audio.
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=False,
            push_stop_frames=True,
            stop_frame_timeout_s=2.0,
            **kwargs
        )

        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._url = url
        self._params = params
        self._sample_rate = sample_rate_from_output_format(params.output_format)

        # Websocket connection to ElevenLabs.
        self._websocket = None
        # Indicates if we have sent TTSStartedFrame. It will reset to False when
        # there's an interruption or TTSStoppedFrame.
        self._started = False
        self._cumulative_time = 0

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        logger.debug(f"Switching TTS model to: [{model}]")
        self._model = model
        await self._disconnect()
        await self._connect()

    async def set_voice(self, voice: str):
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice_id = voice
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

    async def flush_audio(self):
        if self._websocket:
            msg = {"text": " ", "flush": True}
            await self._websocket.send(json.dumps(msg))

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, StartInterruptionFrame)):
            self._started = False
            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("LLMFullResponseEndFrame", 0)])

    async def _connect(self):
        try:
            voice_id = self._voice_id
            model = self._model
            output_format = self._params.output_format
            url = f"{
                self._url}/v1/text-to-speech/{voice_id}/stream-input?model_id={model}&output_format={output_format}"
            self._websocket = await websockets.connect(url)
            self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
            self._keepalive_task = self.get_event_loop().create_task(self._keepalive_task_handler())

            # According to ElevenLabs, we should always start with a single space.
            msg = {
                "text": " ",
                "xi_api_key": self._api_key,
            }
            await self._websocket.send(json.dumps(msg))
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _disconnect(self):
        try:
            await self.stop_all_metrics(self._model)

            if self._websocket:
                await self._websocket.send(json.dumps({"text": ""}))
                await self._websocket.close()
                self._websocket = None

            if self._receive_task:
                self._receive_task.cancel()
                await self._receive_task
                self._receive_task = None

            if self._keepalive_task:
                self._keepalive_task.cancel()
                await self._keepalive_task
                self._keepalive_task = None

            self._started = False
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    async def _receive_task_handler(self):
        try:
            async for message in self._websocket:
                msg = json.loads(message)
                if msg.get("audio"):
                    await self.stop_ttfb_metrics(self._model)
                    self.start_word_timestamps()

                    audio = base64.b64decode(msg["audio"])
                    frame = AudioRawFrame(audio, self._sample_rate, 1)
                    await self.push_frame(frame)

                if msg.get("alignment"):
                    word_times = calculate_word_times(msg["alignment"], self._cumulative_time)
                    await self.add_word_timestamps(word_times)
                    self._cumulative_time = word_times[-1][1]
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"{self} exception: {e}")

    async def _keepalive_task_handler(self):
        while True:
            try:
                await asyncio.sleep(10)
                await self._send_text("")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self} exception: {e}")

    async def _send_text(self, text: str):
        if self._websocket:
            msg = {"text": text + " "}
            await self._websocket.send(json.dumps(msg))

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            if not self._websocket:
                await self._connect()

            try:
                if not self._started:
                    await self.push_frame(TTSStartedFrame())
                    await self.start_ttfb_metrics()
                    self._started = True
                    self._cumulative_time = 0

                await self._send_text(text)
                await self.start_tts_usage_metrics(TTSUsageMetricsData(processor=self.name, model=self._model, value=len(text)))
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                await self.push_frame(TTSStoppedFrame())
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
