#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.aws_nova_sonic import AWSNovaSonicLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

# Load environment variables
load_dotenv(override=True)


async def fetch_weather_from_api(params: FunctionCallParams):
    temperature = 75 if params.arguments["format"] == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": params.arguments["format"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


weather_function = FunctionSchema(
    name="get_current_weather",
    description="Get the current weather",
    properties={
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
        "format": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "The temperature unit to use. Infer this from the users location.",
        },
    },
    required=["location", "format"],
)

# Create tools schema
tools = ToolsSchema(standard_tools=[weather_function])


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot")

    # Specify initial system instruction.
    # HACK: note that, for now, we need to inject a special bit of text into this instruction to
    # allow the first assistant response to be programmatically triggered (which happens in the
    # on_client_connected handler, below)
    system_instruction = (
        "You are a friendly assistant. The user and you will engage in a spoken dialog exchanging "
        "the transcripts of a natural real-time conversation. Keep your responses short, generally "
        "two or three sentences for chatty scenarios. "
        f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}"
    )

    # Create the AWS Nova Sonic LLM service
    llm = AWSNovaSonicLLMService(
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        region=os.getenv("AWS_REGION"),  # as of 2025-05-06, us-east-1 is the only supported region
        session_token=os.getenv("AWS_SESSION_TOKEN"),
        voice_id="tiffany",  # matthew, tiffany, amy
        # you could choose to pass instruction here rather than via context
        # system_instruction=system_instruction
        # you could choose to pass tools here rather than via context
        # tools=tools
    )

    # Register function for function calls
    # you can either register a single function for all function calls, or specific functions
    # llm.register_function(None, fetch_weather_from_api)
    llm.register_function("get_current_weather", fetch_weather_from_api)

    # Set up context and context management.
    # AWSNovaSonicService will adapt OpenAI LLM context objects with standard message format to
    # what's expected by Nova Sonic.
    context = OpenAILLMContext(
        messages=[
            {"role": "system", "content": f"{system_instruction}"},
            {
                "role": "user",
                "content": "Tell me a fun fact!",
            },
        ],
        tools=tools,
    )
    context_aggregator = llm.create_context_aggregator(context)

    # Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    # Configure the pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Handle client connection event
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        # HACK: for now, we need this special way of triggering the first assistant response in AWS
        # Nova Sonic. Note that this trigger requires a special corresponding bit of text in the
        # system instruction. In the future, simply queueing the context frame should be sufficient.
        await llm.trigger_assistant_response()

    # Handle client disconnection events
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    # Run the pipeline
    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
