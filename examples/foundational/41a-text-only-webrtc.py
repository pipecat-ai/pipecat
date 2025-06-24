#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import (
    ActionResult,
    RTVIAction,
    RTVIActionArgument,
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
    RTVIServerMessageFrame,
)
from pipecat.services.openai import OpenAIContextAggregatorPair
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


# This is an example of a text-only chatbot using small webrtc tranport.
# It uses the small webrtc transport prebuilt web UI.
# https://github.com/pipecat-ai/small-webrtc-prebuilt


def create_action_llm_append_to_messages(context_aggregator: OpenAIContextAggregatorPair):
    async def action_llm_append_to_messages_handler(
        rtvi: RTVIProcessor, service: str, arguments: dict[str, any]
    ) -> ActionResult:
        run_immediately = arguments["run_immediately"] if "run_immediately" in arguments else True
        logger.info(f"run_immediately: {run_immediately}")
        if run_immediately:
            await rtvi.interrupt_bot()
            # We just interrupted the bot so it should be fine to use the
            # context directly instead of through frame.
            if "messages" in arguments and arguments["messages"]:
                frame = LLMMessagesAppendFrame(messages=arguments["messages"])
                await rtvi.push_frame(frame)

            frame = context_aggregator.user().get_context_frame()
            await rtvi.push_frame(frame)
        return True

    action_llm_append_to_messages = RTVIAction(
        service="llm",
        action="append_to_messages",
        result="bool",
        arguments=[
            RTVIActionArgument(name="messages", type="array"),
            RTVIActionArgument(name="run_immediately", type="bool"),
        ],
        handler=action_llm_append_to_messages_handler,
    )
    return action_llm_append_to_messages


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "webrtc": lambda: TransportParams(),
}


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot")

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    action_llm_append_to_messages = create_action_llm_append_to_messages(context_aggregator)
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    rtvi.register_action(action_llm_append_to_messages)

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            context_aggregator.user(),
            llm,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()

        # This block is frontend UI specific
        # These messages are intended for small webrtc UI to only handle text
        # https://github.com/pipecat-ai/small-webrtc-prebuilt
        messages = {
            "show_text_container": True,
            "show_video_container": False,
            "show_debug_container": False,
        }

        rtvi_frame = RTVIServerMessageFrame(data=messages)
        await task.queue_frames([rtvi_frame])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")
        # Kick off the conversation.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")

    @transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info(f"Client closed connection")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
