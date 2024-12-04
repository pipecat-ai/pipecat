#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import uuid
from typing import AsyncGenerator, List, Optional, Union

from loguru import logger
from pydantic.main import BaseModel

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import TTSService, WordTTSService
from pipecat.transcriptions.language import Language

from pipecat.audio import audio_io

try:
    import websockets

    import riva.client
    from riva.client.argparse_utils import add_connection_argparse_parameters
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Fastpitch, you need to `pip install pipecat-ai[fastpitch]`. Also, set `NVIDIA_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")

# https://build.nvidia.com/nvidia/fastpitch-hifigan-tts/api
# {
#     "en-US": {
#         "voices": [
#             "English-US.Female-1",
#             "English-US.Male-1",
#             "English-US.Female-Neutral",
#             "English-US.Male-Neutral",
#             "English-US.Female-Angry",
#             "English-US.Male-Angry",
#             "English-US.Female-Calm",
#             "English-US.Male-Calm",
#             "English-US.Female-Fearful",
#             "English-US.Female-Happy",
#             "English-US.Male-Happy",
#             "English-US.Female-Sad"
#         ]
#     }
# }    

class FastpitchTTSService(WordTTSService):
    class InputParams(BaseModel):
        language: Optional[str] = "en-US"

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "English-US.Female-1",
        sample_rate_hz: int = 44100,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        # Aggregating sentences still gives cleaner-sounding results and fewer
        # artifacts than streaming one word at a time. On average, waiting for a
        # full sentence should only "cost" us 15ms or so with GPT-4o or a Llama
        # 3 model, and it's worth it for the better audio quality.
        #
        # We also don't want to automatically push LLM response text frames,
        # because the context aggregators will add them to the LLM context even
        # if we're interrupted. Fastpitch gives us word-by-word timestamps. We
        # can use those to generate text frames ourselves aligned with the
        # playout timing of the audio!
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=False,
            sample_rate=sample_rate,
            **kwargs,
        )

        self._api_key = api_key

        self.set_model_name("fastpitch-hifigan-tts")
        self.set_voice(voice_id)

        self._websocket = None
        self._context_id = None
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        self._model_id = model
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")

    def language_to_service_language(self, language: Language) -> str | None:
        return language_to_fastpitch_language(language)

    def _build_msg(
        self, text: str = "", continue_transcript: bool = True, add_timestamps: bool = True
    ):
        msg = {
            "transcript": text or " ",  # Text must contain at least one character
            "continue": continue_transcript,
            "context_id": self._context_id,
            "model_id": self.model_name,
            "output_format": self._settings["output_format"],
            "language": self._settings["language"],
            "add_timestamps": add_timestamps,
        }
        return json.dumps(msg)

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
            self._websocket = await websockets.connect(
                f"{self._url}?api_key={self._api_key}&fastpitch_version={self._fastpitch_version}"
            )
            self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
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

            self._context_id = None
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        self._context_id = None

    async def flush_audio(self):
        if not self._context_id or not self._websocket:
            return
        logger.trace("Flushing audio")
        msg = self._build_msg(text="", continue_transcript=False)
        await self._websocket.send(msg)

    async def _receive_task_handler(self):
        try:
            async for message in self._get_websocket():
                msg = json.loads(message)
                if not msg or msg["context_id"] != self._context_id:
                    continue
                if msg["type"] == "done":
                    await self.stop_ttfb_metrics()
                    # Unset _context_id but not the _context_id_start_timestamp
                    # because we are likely still playing out audio and need the
                    # timestamp to set send context frames.
                    self._context_id = None
                    await self.add_word_timestamps(
                        [("TTSStoppedFrame", 0), ("LLMFullResponseEndFrame", 0), ("Reset", 0)]
                    )
                elif msg["type"] == "timestamps":
                    await self.add_word_timestamps(
                        list(zip(msg["word_timestamps"]["words"], msg["word_timestamps"]["start"]))
                    )
                elif msg["type"] == "chunk":
                    await self.stop_ttfb_metrics()
                    self.start_word_timestamps()
                    frame = TTSAudioRawFrame(
                        audio=base64.b64decode(msg["data"]),
                        sample_rate=self._settings["output_format"]["sample_rate"],
                        num_channels=1,
                    )
                    await self.push_frame(frame)
                elif msg["type"] == "error":
                    logger.error(f"{self} error: {msg}")
                    await self.push_frame(TTSStoppedFrame())
                    await self.stop_all_metrics()
                    await self.push_error(ErrorFrame(f'{self} error: {msg["error"]}'))
                else:
                    logger.error(f"Fastpitch error, unknown message type: {msg}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"{self} exception: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # If we received a TTSSpeakFrame and the LLM response included text (it
        # might be that it's only a function calling response) we pause
        # processing more frames until we receive a BotStoppedSpeakingFrame.
        if isinstance(frame, TTSSpeakFrame):
            await self.pause_processing_frames()
        elif isinstance(frame, LLMFullResponseEndFrame) and self._context_id:
            await self.pause_processing_frames()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.resume_processing_frames()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            if not self._websocket:
                await self._connect()

            if not self._context_id:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._context_id = str(uuid.uuid4())

            msg = self._build_msg(text=text or " ")  # Text must contain at least one character

            try:
                await self._get_websocket().send(msg)
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

class FastpitchHttpTTSService(TTSService):
    class InputParams(BaseModel):
        language: Optional[str] = "en-US"    

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "English-US.Female-1",
        sample_rate_hz: int = 44100,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate_hz, **kwargs)
        self._api_key = api_key

        self.set_model_name("fastpitch-hifigan-tts")
        self.set_voice(voice_id)

        self.voice_id = voice_id
        self.sample_rate_hz = sample_rate_hz
        self.language_code=params.language
        self.nchannels = 1
        self.sampwidth = 2
        self.sound_stream = None
        self.quality=None 

        # "function-id" is hard-coded in the example curl request
        # if this should be a configurable thing, we can update that later
        metadata=[["function-id", "0149dedb-2be8-4195-b9a0-e57e0e14f972"], 
            ["authorization", f"Bearer {api_key}"]]
        auth = riva.client.Auth(
            None, 
            True,
            "grpc.nvcf.nvidia.com:443", 
            metadata
        )

        self.service = riva.client.SpeechSynthesisService(auth)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        if self.sound_stream:
            self.sound_stream.close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        if self.sound_stream:
            self.sound_stream.close()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        try:
            self.sound_stream = audio_io.SoundCallBack(
                None, nchannels=self.nchannels, 
                sampwidth=self.sampwidth, 
                framerate=self.sample_rate_hz
            )
            
            custom_dictionary_input = {}
            responses = self.service.synthesize_online(
                text, 
                self.voice_id, 
                self.language_code, 
                sample_rate_hz=self.sample_rate_hz,
                audio_prompt_file=None, 
                quality=20 if self.quality is None else self.quality,
                custom_dictionary=custom_dictionary_input
            )

            for resp in responses:
                if self.sound_stream is not None:
                    self.sound_stream(resp.audio)
            await self.stop_ttfb_metrics()

            frame = TTSAudioRawFrame(
                audio=resp.audio,
                sample_rate=self.sample_rate_hz,
                num_channels=self.nchannels,
            )
            yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")

        await self.start_tts_usage_metrics(text)
        yield TTSStoppedFrame()
