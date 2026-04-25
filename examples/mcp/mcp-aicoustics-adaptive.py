#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice assistant with LLM-controlled audio enhancement.

Demonstrates how an LLM can dynamically adjust ai-coustics audio enhancement
in response to user feedback during a call. The LLM receives a
`set_audio_enhancement_level` tool and uses it whenever the user reports audio
quality issues. The tool pushes a `FilterUpdateSettingsFrame` into the pipeline,
which the transport's input stage forwards to the `AICFilter` instance.

Required env vars:
    AIC_LICENSE_KEY  – ai-coustics SDK license key
    ANTHROPIC_API_KEY       – Anthropic API key
    DEEPGRAM_API_KEY        – Deepgram STT key
    CARTESIA_API_KEY        – Cartesia TTS key

Optional env vars:
    AIC_MODEL_ID     – Enhancement model ID (default: quail-vf-2.1-l-16khz)
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.filters.aic_filter import AICFilter
from pipecat.frames.frames import FilterUpdateSettingsFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

_DEFAULT_ENHANCEMENT_LEVEL = 0.5
_MODEL_ID = os.getenv("AIC_MODEL_ID", "quail-vf-2.1-l-16khz")

aic_filter = AICFilter(
    license_key=os.getenv("AIC_LICENSE_KEY", ""),
    model_id=_MODEL_ID,
    enhancement_level=_DEFAULT_ENHANCEMENT_LEVEL,
)
aic_vad = aic_filter.create_vad_analyzer(speech_hold_duration=0.05, sensitivity=6.0)

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=aic_filter,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=aic_filter,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=aic_filter,
    ),
}

_set_enhancement_schema = FunctionSchema(
    name="set_audio_enhancement_level",
    description=(
        "Adjust the ai-coustics audio enhancement strength for the caller's microphone. "
        "Use this when the user reports audio quality issues such as background noise, "
        "echo, or difficulty being heard. Higher values apply stronger enhancement."
    ),
    properties={
        "level": {
            "type": "number",
            "description": "Enhancement strength between 0.0 (off) and 1.0 (maximum).",
        },
        "reason": {
            "type": "string",
            "description": "Brief reason for the change, for logging purposes.",
        },
    },
    required=["level"],
)

_SYSTEM_PROMPT = f"""\
You are a helpful voice assistant.

You have a `set_audio_enhancement_level` tool that controls the ai-coustics audio \
enhancement applied to the caller's microphone input. The current level is \
{_DEFAULT_ENHANCEMENT_LEVEL}.

Use the tool proactively when:
- The user says they can't be heard, the audio is noisy, or asks you to improve the sound quality.
- You detect repeated misunderstandings that may be caused by poor audio.
- The user asks to "boost", "improve", "fix", or "turn up" audio quality.

After adjusting, briefly confirm the change in one sentence.

Your output will be spoken aloud. Avoid bullet points, emojis, or markdown formatting.
"""


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = AnthropicLLMService(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        settings=AnthropicLLMService.Settings(
            system_instruction=_SYSTEM_PROMPT,
        ),
    )

    # task is defined below; capture it via a mutable cell so the handler closure can
    # reference it before the variable is assigned.
    task_ref: list[PipelineTask] = []

    async def set_audio_enhancement_level(params: FunctionCallParams):
        level = float(params.arguments["level"])
        reason = params.arguments.get("reason", "")
        if task_ref:
            await task_ref[0].queue_frames(
                [FilterUpdateSettingsFrame(settings={"enhancement_level": level})]
            )
        logger.info(f"Audio enhancement → {level}" + (f" ({reason})" if reason else ""))
        await params.result_callback(f"Audio enhancement level set to {level}.")

    llm.register_function("set_audio_enhancement_level", set_audio_enhancement_level)

    tools = ToolsSchema(standard_tools=[_set_enhancement_schema])
    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=aic_vad),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
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
    task_ref.append(task)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
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
