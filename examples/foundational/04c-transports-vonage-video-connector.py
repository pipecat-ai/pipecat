#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example of using AWS Nova Sonic LLM service with Vonage Video Connector transport."""

import asyncio
import json
import os
import sys
from typing import Any, Callable

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    UserTurnStoppedMessage,
)
from pipecat.runner.vonage import configure
from pipecat.services import aws_nova_sonic
from pipecat.services.aws_nova_sonic.aws import AWSNovaSonicLLMService
from pipecat.transports.vonage.video_connector import (
    VonageVideoConnectorTransport,
    VonageVideoConnectorTransportParams,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main() -> None:
    """Main entry point for the nova sonic vonage video connector example."""
    (application_id, session_id, token) = await configure()

    system_instruction = (
        "You are a friendly assistant. The user and you will engage in a spoken dialog exchanging "
        "the transcripts of a natural real-time conversation. Keep your responses short, generally "
        "two or three sentences for chatty scenarios. "
        f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}"
    )
    transport = VonageVideoConnectorTransport(
        application_id,
        session_id,
        token,
        VonageVideoConnectorTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            publisher_name="Bot",
        ),
    )

    llm = AWSNovaSonicLLMService(
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
        region=os.getenv("AWS_REGION", ""),
        session_token=os.getenv("AWS_SESSION_TOKEN", ""),
        voice_id="tiffany",
    )
    context = LLMContext(
        messages=[
            {"role": "system", "content": f"{system_instruction}"},
            {
                "role": "user",
                "content": "Tell me a fun fact!",
            },
        ],
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(pipeline)

    # Handle client connection event
    event_handler: Callable[[str], Callable[[Any], Any]] = transport.event_handler

    @event_handler("on_client_connected")
    async def on_client_connected(transport: VonageVideoConnectorTransport, client: object) -> None:
        logger.info(f"Client connected")
        await task.queue_frames([LLMRunFrame()])
        # HACK: for now, we need this special way of triggering the first assistant response in AWS
        # Nova Sonic. Note that this trigger requires a special corresponding bit of text in the
        # system instruction. In the future, simply queueing the context frame should be sufficient.
        await llm.trigger_assistant_response()  # type: ignore[no-untyped-call]

    runner = PipelineRunner()

    await asyncio.gather(runner.run(task))


if __name__ == "__main__":
    asyncio.run(main())
