#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from loguru import logger
from runner import configure
from pipecat.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.transports.services.daily import DailyParams, DailyTransport, DailyTransportMessageFrame
from pipecat.services.openai import OpenAILLMService, OpenAILLMContext, OpenAILLMContextFrame
from pipecat.services.deepgram import DeepgramTTSService, DeepgramSTTService
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.logger import FrameLogger
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMMessagesFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    AudioRawFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    LLMFullResponseStartFrame,
    TTSStoppedFrame
)
import asyncio
import aiohttp
import os
import sys
import json
from dataclasses import dataclass, field
from typing import List
import math
import struct
import time

from dotenv import load_dotenv
load_dotenv(override=True)


logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class GreedyLLMAggregator(FrameProcessor):
    def __init__(self, context: OpenAILLMContext = None, **kwargs):
        super().__init__(**kwargs)
        self.context: OpenAILLMContext = context if context else OpenAILLMContext()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        logger.debug(f"{frame}")

        try:
            if isinstance(frame, InterimTranscriptionFrame):
                return

            if isinstance(frame, TranscriptionFrame):
                # append transcribed text to last "user" frame
                if self.context.messages and self.context.messages[-1]["role"] == "user":
                    last_frame = self.context.messages.pop()
                else:
                    last_frame = {"role": "user", "content": ""}

                last_frame["content"] += " " + frame.text
                self.context.messages.append(last_frame)

                oai_context_frame = OpenAILLMContextFrame(context=self.context)
                logger.debug(f"pushing frame {oai_context_frame}")
                await self.push_frame(oai_context_frame)
                return

            await self.push_frame(frame, direction)
        except Exception as e:
            logger.debug(f"error: {e}")


class ClearableDeepgramTTSService(DeepgramTTSService):
    def __init___(self, **kwargs):
        super().__init(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            self._current_sentence = ""


@dataclass
class BufferedSentence:
    audio_frames: List[AudioRawFrame] = field(default_factory=list)
    text_frame: TextFrame = None


class VADGate(FrameProcessor):

    def __init__(
            self,
            vad_analyzer: VADAnalyzer = None,
            context: OpenAILLMContext = None,
            **kwargs):
        super().__init__(**kwargs)
        self.vad_analyzer = vad_analyzer
        self.context = context

        self._audio_pusher_task = None
        self._expect_text_frame_next = False
        self._sentences: List[BufferedSentence] = []

    # queue output from tts one sentence at a time. associate a buffer of audio frames with the content of
    # each text frame.
    #
    # start a coroutine to service the queue and send sentences down the pipeline when possible.
    # 1. do not send anything when we are not in VADState.QUIET
    # 2. if we are in VADState.QUIET, send a sentence, estimate how long it will take for that sentence
    #    to output, sleep until it's time to send another sentence
    # 3. each time we send a sentence, append it to the conversation context
    # 3. when the sentence buffer becomes empty, cancel the coroutine
    # 4. if we get a new LLMFullResponse, treat that as a cancellation, too

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        try:

            # A TTSService will emit a series of AudioRawFrame objects, then a TTSStoppedFrame,
            # then a TextFrame.

            if self._expect_text_frame_next:
                self._expect_text_frame_next = False
                if isinstance(frame, TextFrame):
                    self._sentences[-1].text_frame = frame
                else:
                    logger.debug(f"expected a text frame, but received {frame}")
                    await self.push_frame(frame, direction)
                return
            else:
                if isinstance(frame, TextFrame):
                    logger.error(f"XXXXXXXXXXXXXXXXXXX received a text frame, wasn't expecting it.")

            if isinstance(frame, AudioRawFrame):
                # if our buffer is empty or has a "finished" sentence at the end,
                # then we need to start buffering a new sentence
                if not self._sentences or self._sentences[-1].text_frame:
                    self._sentences.append(BufferedSentence())
                self._sentences[-1].audio_frames.append(frame)
                await self.maybe_start_audio_pusher_task()
                return

            if isinstance(frame, TTSStoppedFrame):
                self._expect_text_frame_next = True
                await self.push_frame(frame, direction)
                return

            # There are two ways we can be interrupted. During greedy inference, a new
            # LLM response can start. Or, during playout, we can get a traditional
            # user interruption frame.
            if (isinstance(frame, LLMFullResponseStartFrame) or
                    isinstance(frame, StartInterruptionFrame)):
                logger.debug(f"{frame} - Handle interruption in VADGate")
                self._sentences = []
                if self._audio_pusher_task:
                    self._audio_pusher_task.cancel()
                    self._audio_pusher_task = None
                await self.push_frame(frame, direction)
                return

            await self.push_frame(frame, direction)
        except Exception as e:
            logger.debug(f"error: {e}")

    async def maybe_start_audio_pusher_task(self):
        try:
            if self._audio_pusher_task:
                return
            self._audio_pusher_task = self.get_event_loop().create_task(self.push_audio())

        except Exception as e:
            logger.debug(f"Exception {e}")

    async def push_audio(self):
        try:
            while True:
                if not self._sentences:
                    await asyncio.sleep(0.01)
                    continue

                if self.vad_analyzer._vad_state != VADState.QUIET:
                    await asyncio.sleep(0.01)
                    continue

                # we only want to push completed sentence buffers
                if not self._sentences[0].text_frame:
                    await asyncio.sleep(0.01)
                    continue

                s = self._sentences.pop(0)
                if not s.audio_frames:
                    continue
                sample_rate = s.audio_frames[0].sample_rate
                duration = 0
                logger.debug(f"Pushing {len(s.audio_frames)} audio frames")
                for frame in s.audio_frames:
                    await self.push_frame(frame)
                    # assume linear16 encoding (2 bytes per sample). todo: add some more
                    # metadata to AudioRawFrame, maybe
                    duration += (len(frame.audio) / 2 / frame.num_channels) / sample_rate
                await asyncio.sleep(duration - 20 / 1000)
                if self.context:
                    logger.debug(f"Appending assistant message to context: [{s.text_frame.text}]")
                    self.context.messages.append(
                        {"role": "assistant", "content": s.text_frame.text}
                    )
                await self.push_frame(s.text_frame)

        except Exception as e:
            logger.debug(f"Exception {e}")


class TranscriptionLogger(FrameProcessor):
    def __init__(self, avt):
        super().__init__()
        self._avt = avt

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            logger.debug(f"Transcription TTF: {time.time() - self._avt.last_transition_ts}")
        await self.push_frame(frame, direction)


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
        fmt = f"{len(frame.audio)//2}h"
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


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=2.500)),
                vad_audio_passthrough=True
            )
        )

        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            url="ws://localhost:8080"
        )

        tts = ClearableDeepgramTTSService(
            aiohttp_session=session,
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            voice="aura-asteria-en",
            base_url="http://0.0.0.0:8080/v1/speak"
        )

        llm = OpenAILLMService(
            # To use OpenAI
            # api_key=os.getenv("OPENAI_API_KEY"),
            # model="gpt-4o"
            # Or, to use a local vLLM (or similar) api server
            # model="meta-llama/Meta-Llama-3-70B-Instruct",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            # model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
            base_url="http://0.0.0.0:8000/v1"
        )

        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant in an audio conversation.

Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Be concise in your answers to basic questions. If you are asked to elaborate or tell a story, provide a longer response.
""",
            },
        ]

        ctx = OpenAILLMContext()
        greedy = GreedyLLMAggregator(name="greedy", context=ctx)
        gate = VADGate(name="gate", vad_analyzer=transport.input().vad_analyzer(), context=ctx)
        avt = AudioVolumeTimer()
        tl = TranscriptionLogger(avt)

        pipeline = Pipeline([
            transport.input(),   # Transport user input
            avt,
            stt,
            tl,
            greedy,
            llm,                 # LLM
            tts,                 # TTS
            gate,
            transport.output(),  # Transport bot output
            # FrameLogger()
        ])

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True, enable_metrics=True))

        # When a participant joins, start transcription for that participant so the
        # bot can "hear" and respond to them.
        # @ transport.event_handler("on_participant_joined")
        # async def on_participant_joined(transport, participant):
        #    transport.capture_participant_transcription(participant["id"])

        # When the first participant joins, the bot should introduce itself.
        @ transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            messages.append(
                {"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMMessagesFrame(messages)])

        # Handle "latency-ping" messages. The client will send app messages that look like
        # this:
        #   { "latency-ping": { ts: <client-side timestamp> }}
        #
        # We want to send an immediate pong back to the client from this handler function.
        # Also, we will push a frame into the top of the pipeline and send it after the
        #
        @ transport.event_handler("on_app_message")
        async def on_app_message(transport, message, sender):
            try:
                if "latency-ping" in message:
                    logger.debug(f"Received latency ping app message: {message}")
                    ts = message["latency-ping"]["ts"]
                    # Send immediately
                    transport.output().send_message(DailyTransportMessageFrame(
                        message={"latency-pong-msg-handler": {"ts": ts}},
                        participant_id=sender))
                    # And push to the pipeline for the Daily transport.output to send
                    await tma_in.push_frame(
                        DailyTransportMessageFrame(
                            message={"latency-pong-pipeline-delivery": {"ts": ts}},
                            participant_id=sender))
            except Exception as e:
                logger.debug(f"message handling error: {e} - {message}")

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
