#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os
from dataclasses import dataclass

from dotenv import load_dotenv
from google.genai.types import Content, Part
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


marker = "|----|"
system_message = f"""
You are a helpful LLM in a WebRTC call. Your goals are to be helpful and brief in your responses.

You are expert at transcribing audio to text. You will receive a mixture of audio and text input. When
asked to transcribe what the user said, output an exact, word-for-word transcription.

Your output will be converted to audio so don't include special characters in your answers.

Each time you answer, you should respond in three parts.

1. Transcribe exactly what the user said.
2. Output the separator field '{marker}'.
3. Respond to the user's input in a helpful, creative way using only simple text and punctuation.

Example:

User: How many ounces are in a pound?

You: How many ounces are in a pound?
{marker}
There are 16 ounces in a pound.
"""


@dataclass
class MagicDemoTranscriptionFrame(Frame):
    text: str


class UserAudioCollector(FrameProcessor):
    def __init__(self, context, user_context_aggregator):
        super().__init__()
        self._context = context
        self._user_context_aggregator = user_context_aggregator
        self._audio_frames = []
        self._start_secs = 0.2  # this should match VAD start_secs (hardcoding for now)
        self._user_speaking = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # We could gracefully handle both audio input and text/transcription input ...
            # but let's leave that as an exercise to the reader. :-)
            return
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            self._context.add_audio_frames_message(audio_frames=self._audio_frames)
            await self._user_context_aggregator.push_frame(LLMRunFrame())

        elif isinstance(frame, InputAudioRawFrame):
            if self._user_speaking:
                self._audio_frames.append(frame)
            else:
                # Append the audio frame to our buffer. Treat the buffer as a ring buffer, dropping the oldest
                # frames as necessary. Assume all audio frames have the same duration.
                self._audio_frames.append(frame)
                frame_duration = len(frame.audio) / 16 * frame.num_channels / frame.sample_rate
                buffer_duration = frame_duration * len(self._audio_frames)
                while buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration

        await self.push_frame(frame, direction)


class TranscriptExtractor(FrameProcessor):
    def __init__(self, context):
        super().__init__()
        self._context = context
        self._accumulator = ""
        self._processing_llm_response = False
        self._accumulating_transcript = False

    def reset(self):
        self._accumulator = ""
        self._processing_llm_response = False
        self._accumulating_transcript = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._processing_llm_response = True
            self._accumulating_transcript = True
        elif isinstance(frame, TextFrame) and self._processing_llm_response:
            if self._accumulating_transcript:
                text = frame.text
                split_index = text.find(marker)
                if split_index < 0:
                    self._accumulator += frame.text
                    # do not push this frame
                    return
                else:
                    self._accumulating_transcript = False
                    self._accumulator += text[:split_index]
                    frame.text = text[split_index + len(marker) :]
            await self.push_frame(frame)
            return
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self.push_frame(MagicDemoTranscriptionFrame(text=self._accumulator.strip()))
            self.reset()

        await self.push_frame(frame, direction)


class TranscriptionContextFixup(FrameProcessor):
    def __init__(self, context):
        super().__init__()
        self._context = context
        self._transcript = "THIS IS A TRANSCRIPT"

    def swap_user_audio(self):
        if not self._transcript:
            return
        message = self._context.messages[-2]
        last_part = message.parts[-1]
        if (
            message.role == "user"
            and last_part.inline_data
            and last_part.inline_data.mime_type == "audio/wav"
        ):
            self._context.messages[-2] = Content(role="user", parts=[Part(text=self._transcript)])

    def add_transcript_back_to_inference_output(self):
        if not self._transcript:
            return
        message = self._context.messages[-1]
        last_part = message.parts[-1]
        if message.role == "model" and last_part.text:
            self._context.messages[-1].parts[-1].text += f"\n\n{marker}\n{self._transcript}\n"

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, MagicDemoTranscriptionFrame):
            self._transcript = frame.text
        elif isinstance(frame, LLMFullResponseEndFrame) or isinstance(frame, InterruptionFrame):
            self.swap_user_audio()
            self.add_transcript_back_to_inference_output()
            self._transcript = ""

        await self.push_frame(frame, direction)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash",
        # turn on thinking if you want it
        # params=GoogleLLMService.InputParams(extra={"thinking_config": {"thinking_budget": 4096}}),
    )

    tts = GoogleTTSService(
        voice_id="en-US-Chirp3-HD-Charon",
        params=GoogleTTSService.InputParams(language=Language.EN_US),
        credentials=os.getenv("GOOGLE_TEST_CREDENTIALS"),
    )

    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": "Start by saying hello.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)
    audio_collector = UserAudioCollector(context, context_aggregator.user())
    pull_transcript_out_of_llm_output = TranscriptExtractor(context)
    fixup_context_messages = TranscriptionContextFixup(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            audio_collector,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            pull_transcript_out_of_llm_output,
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
            fixup_context_messages,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
