#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
from typing import Any

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.bey.video import BeyVideoService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.daily.utils import (
    DailyRESTHelper,
    DailyMeetingTokenParams,
    DailyMeetingTokenProperties,
)


load_dotenv(override=True)

# Ege stock avatar
# Ref: https://docs.bey.dev/get-started/avatars/default
AVATAR_ID = "b9be11b8-89fb-4227-8f86-4a881393cbdb"

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=False,
        video_out_enabled=False,
        video_out_is_live=False,
        microphone_out_enabled=False,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=False,
        video_out_enabled=False,
        video_out_is_live=False,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    if not isinstance(transport, DailyTransport):
        raise ValueError("This example only supports Daily transport")
        # TODO: Support Small WebRTC transport
    async with aiohttp.ClientSession() as session:
        stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))

        tts = OpenAITTSService(api_key=os.getenv("OPENAI_API_KEY"))

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        daily_rest_helper = DailyRESTHelper(
            daily_api_key=os.getenv("DAILY_API_KEY"),
            aiohttp_session=session,
        )

        video_bot_name = "My Video Bot"

        bey_video = BeyVideoService(
            api_key=os.getenv("BEY_API_KEY"),
            avatar_id=AVATAR_ID,
            bot_name=video_bot_name,
            # we stream audio to a video bot in the Daily room, so we need this
            transport_client=transport._client,
            # video bot joins the room remotely on demand, we need these to manage it
            rest_helper=daily_rest_helper,
            session=session,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Your output will be converted to audio so don't include special characters in your answers. Be succinct and respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                bey_video,  # Bey Video Avatar
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
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

        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant: dict[str, Any]):
            logger.info(f"Participant joined: {participant['info']['userName']}")
            
            if participant["info"]["userName"] == video_bot_name:
                await transport.update_subscriptions(
                    participant_settings={
                        participant["id"]: {"media": {"microphone": "unsubscribed"}}
                    }
                )
                return

            # Kick off the conversation.
            messages.append(
                {
                    "role": "system",
                    "content": "Please introduce yourself to the user.",
                }
            )
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
