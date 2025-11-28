#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

import aiohttp
from bedrock_agentcore import BedrockAgentCoreApp
from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import DailyRunnerArguments, RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyLogLevel, DailyParams, DailyTransport

app = BedrockAgentCoreApp()

load_dotenv(override=True)


async def get_public_ip():
    """Retrieve public IP from AWS metadata service or external service."""
    try:
        # Fallback to external service
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.ipify.org", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return await response.text()
    except Exception:
        pass

    return None


async def fetch_weather_from_api(params: FunctionCallParams):
    await params.result_callback({"conditions": "nice", "temperature": "75"})


async def fetch_restaurant_recommendation(params: FunctionCallParams):
    await params.result_callback({"name": "The Golden Dragon"})


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
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    public_ip = await get_public_ip()
    if public_ip:
        logger.info(f"Public IP address: {public_ip}")
    else:
        logger.warning("Could not retrieve public IP address")

    yield {"status": "initializing", "ip": public_ip}

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # You can also register a function_name of None to get all functions
    # sent to the same callback with an additional function_name parameter.
    llm.register_function("get_current_weather", fetch_weather_from_api)
    llm.register_function("get_restaurant_recommendation", fetch_restaurant_recommendation)

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

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
                "description": "The temperature unit to use. Infer this from the user's location.",
            },
        },
        required=["location", "format"],
    )
    restaurant_function = FunctionSchema(
        name="get_restaurant_recommendation",
        description="Get a restaurant recommendation",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        required=["location"],
    )
    tools = ToolsSchema(standard_tools=[weather_function, restaurant_function])

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = LLMContext(messages, tools)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
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
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    task_id = app.add_async_task("voice_agent")

    await runner.run(task)

    app.complete_async_task(task_id)

    yield {"status": "completed"}


async def bot(runner_args: RunnerArguments):
    """Bot entry point for running locally and on Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    async for result in run_bot(transport, runner_args):
        pass  # Consume the stream


@app.entrypoint
async def agentcore_bot(payload, context):
    """Bot entry point for running on Amazon Bedrock AgentCore Runtime."""
    room_url = payload.get("roomUrl")
    transport = await create_transport(
        DailyRunnerArguments(room_url=room_url),
        transport_params,
    )
    if isinstance(transport, DailyTransport):
        transport.set_log_level(DailyLogLevel.Info)
        turn_username = os.getenv("TURN_USERNAME")
        turn_credential = os.getenv("TURN_CREDENTIAL")
        transport._client._client.set_ice_config(
            {
                "placement": "replace",
                "iceServers": [
                    {
                        "urls": [
                            "turn:turn.cloudflare.com:80?transport=tcp",
                            "turns:turn.cloudflare.com:443?transport=tcp",
                        ],
                        "username": turn_username,
                        "credential": turn_credential,
                    },
                ],
            }
        )

    async for result in run_bot(transport, RunnerArguments()):
        yield result


if __name__ == "__main__":
    # NOTE: ideally we shouldn't have to branch for local dev vs AgentCore, but
    # local AgentCore container-based dev doesn't seem to be working, or at
    # least not for this project.
    if os.getenv("PIPECAT_LOCAL_DEV") == "1":
        # Running locally
        from pipecat.runner.run import main

        main()
    else:
        # Running on AgentCore Runtime
        app.run()
