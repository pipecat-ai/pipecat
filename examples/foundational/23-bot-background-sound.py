#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure_with_args

from pipecat.audio.mixers.soundfile_mixer import SoundfileMixer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotInterruptionFrame,
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ControlFrame,
    Frame,
    InputAudioRawFrame,
    LLMTextFrame,
    MetricsFrame,
    MixerEnableFrame,
    MixerUpdateSettingsFrame,
    TextFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="INFO")


class DebugProcessor(FrameProcessor):
    """A processor for debugging frames in the pipeline."""

    def __init__(self, name, **kwargs):  # noqa: D107
        self._name = name
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):  # noqa: D102
        await super().process_frame(frame, direction)
        if not (
            isinstance(frame, InputAudioRawFrame)
            or isinstance(frame, TTSAudioRawFrame)
            or isinstance(frame, BotSpeakingFrame)
            or isinstance(frame, BotStartedSpeakingFrame)
            or isinstance(frame, MetricsFrame)
            or isinstance(frame, LLMTextFrame)
        ):
            logger.info(f"{self._name}: {frame} {direction}")
        await self.push_frame(frame, direction)


@dataclass
class StartHoldMusicFrame(ControlFrame):
    """Starts hold music."""

    pass


class HoldMusicProcessor(FrameProcessor):
    """A processor to play hold music."""

    def __init__(self, **kwargs):  # noqa: D107
        super().__init__(**kwargs)
        self._play_hold_music = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):  # noqa: D102
        await super().process_frame(frame, direction)

        if isinstance(frame, StartHoldMusicFrame):
            self._play_hold_music = True

        if isinstance(frame, BotStoppedSpeakingFrame) and self._play_hold_music:
            await self.push_frame(
                MixerUpdateSettingsFrame({"volume": 1, "sound": "office", "loop": False})
            )
            await self.push_frame(MixerEnableFrame(True))
            # await self.queue_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)
        elif isinstance(frame, BotSpeakingFrame):
            await self.push_frame(MixerEnableFrame(False))
        await self.push_frame(frame, direction)


async def main():
    """Main function to run the bot background sound."""
    async with aiohttp.ClientSession() as session:
        parser = argparse.ArgumentParser(description="Bot Background Sound")
        parser.add_argument("-i", "--input", type=str, required=True, help="Input audio file")

        (room_url, token, args) = await configure_with_args(session, parser)

        soundfile_mixer = SoundfileMixer(
            sound_files={"office": args.input},
            default_sound="office",
            volume=0,
        )

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                audio_out_mixer=soundfile_mixer,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)
        dp = DebugProcessor("post-llm")
        hold_music_processor = HoldMusicProcessor()

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                context_aggregator.user(),  # User responses
                llm,  # LLM
                dp,  # Debug processor
                hold_music_processor,  # Hold music
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])

            # Kick off the conversation.
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([context_aggregator.user().get_context_frame()])
            await task.queue_frame(TextFrame("I'm going to play some hold music."))
            await task.queue_frame(StartHoldMusicFrame())

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
