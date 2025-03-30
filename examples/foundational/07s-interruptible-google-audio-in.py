#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from dataclasses import dataclass

import aiohttp
import google.ai.generativelanguage as glm
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

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
            await self._user_context_aggregator.push_frame(
                self._user_context_aggregator.get_context_frame()
            )
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


class TanscriptionContextFixup(FrameProcessor):
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
            self._context.messages[-2] = glm.Content(
                role="user", parts=[glm.Part(text=self._transcript)]
            )

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
        elif isinstance(frame, LLMFullResponseEndFrame) or isinstance(
            frame, StartInterruptionFrame
        ):
            self.swap_user_audio()
            self.add_transcript_back_to_inference_output()
            self._transcript = ""

        await self.push_frame(frame, direction)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                # No transcription at all. just audio input to Gemini!
                # transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-001")

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

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)
        audio_collector = UserAudioCollector(context, context_aggregator.user())
        pull_transcript_out_of_llm_output = TranscriptExtractor(context)
        fixup_context_messages = TanscriptionContextFixup(context)

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
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
