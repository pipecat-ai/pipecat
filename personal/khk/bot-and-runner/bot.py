import asyncio
import sys
import os
import argparse
import json
import struct
import math
import time

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVILLMConfig,
    RTVIProcessor,
    RTVISetup,
    RTVITTSConfig)
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TranscriptionFrame,
    MetricsFrame,
    EndFrame
)

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

class DebugLogger(FrameProcessor):
    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        logger.debug(frame)
        await self.push_frame(frame, direction)


class TranscriptionTimingLogger(FrameProcessor):
    def __init__(self, avt):
        super().__init__(name="Transcription")
        self._avt = avt
 
    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                elapsed = time.time() - self._avt.last_transition_ts
                logger.debug(f"Transcription TTFB: {elapsed}")
                # await self.push_frame(MetricsFrame(ttfb={self.name: elapsed}))
                await self.push_frame(MetricsFrame(ttfb=[{"processor": self.name, "value": elapsed}]))

            await self.push_frame(frame, direction)
        except Exception as e:
            logger.debug(f"Exception {e}")

class AudioVolumeTimer(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.last_transition_ts = 0
        self._prev_volume = -80
        self._speech_volume_threshold = -50

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            volume = self.calculate_volume(frame)
            # print(f"Audio volume: {volume:.2f} dB")
            if (volume >= self._speech_volume_threshold and
                    self._prev_volume < self._speech_volume_threshold):
                # logger.debug("transition above speech volume threshold")
                self.last_transition_ts = time.time()
            elif (volume < self._speech_volume_threshold and
                    self._prev_volume >= self._speech_volume_threshold):
                # logger.debug("transition below non-speech volume threshold")
                self.last_transition_ts = time.time()
            self._prev_volume = volume

        await self.push_frame(frame, direction)

    def calculate_volume(self, frame: AudioRawFrame) -> float:
        if frame.num_channels != 1:
            raise ValueError(f"Expected 1 channel, got {frame.num_channels}")

        # Unpack audio data into 16-bit integers
        fmt = f"{len(frame.audio) // 2}h"
        audio_samples = struct.unpack(fmt, frame.audio)

        # Calculate RMS
        sum_squares = sum(sample**2 for sample in audio_samples)
        rms = math.sqrt(sum_squares / len(audio_samples))

        # Convert RMS to decibels (dB)
        # Reference: maximum value for 16-bit audio is 32767
        if rms > 0:
            db = 20 * math.log10(rms / 32767)
        else:
            db = -96  # Minimum value (almost silent)

        return db

async def main(room_url, token, bot_config):
    transport = DailyTransport(
        room_url,
        token,
        "Realtime AI",
        DailyParams(
            audio_out_enabled=True,
            # transcription_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                stop_secs=float(os.getenv("VAD_STOP_SECS", "0.3"))
            )),
            vad_audio_passthrough=True
        ))

    avt = AudioVolumeTimer()
    tl = TranscriptionTimingLogger(avt)

    stt = DeepgramSTTService(
        name="STT",
        api_key=os.getenv("DEEPGRAM_API_KEY", ""),
        url=os.getenv("DEEPGRAM_STT_BASE_URL", "wss://api.deepgram.com")
    )

    rtai = RTVIProcessor(
        transport=transport,
        setup=RTVISetup(config=RTVIConfig(**bot_config)),
        llm_api_key=os.getenv("OPENAI_API_KEY", ""),
        tts_api_key=os.getenv("CARTESIA_API_KEY", ""))

    runner = PipelineRunner()

    pipeline = Pipeline([transport.input(), avt, stt, tl, rtai])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            # report_only_initial_ttfb=True
        ))

    @ transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        transport.capture_participant_transcription(participant["id"])
        logger.info("First participant joined")

    @ transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.queue_frame(EndFrame())
        logger.info("Partcipant left. Exiting.")

    @ transport.event_handler("on_call_state_updated")
    async def on_call_state_updated(transport, state):
        logger.info("Call state %s " % state)
        if state == "left":
            await task.queue_frame(EndFrame())

    await runner.run(task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    parser.add_argument("-c", type=str, help="Bot configuration blob")
    config = parser.parse_args()

    bot_config = json.loads(config.c) if config.c else {}
    # bot_config = {"llm":{"model":"llama3-70b-8192","messages":[{"role":"system","content":"You are Chatbot, a friendly, helpful robot. Your output will be converted to audio so don't include special characters other than '!' or '?' in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by saying hello."}]},"tts":{"voice":"79a125e8-cd45-4c13-8a67-188112f4dd22"}}

    if config.u and config.t and bot_config:
        asyncio.run(main(config.u, config.t, bot_config))
    else:
        logger.error("Room URL and Token are required")
