#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""TTSSpeakFrame timing and transcript ordering example.

Demonstrates two common patterns for injecting hardcoded speech into a voice
agent without breaking the LLM context / transcript ordering:

    1. Intro / pre-roll. Say something before the agent speaks, and make sure
       it shows up in the LLM context in the right place.
    2. Tool-call filler. Say something while a function call is in flight,
       without the filler audio overlapping the post-tool LLM response and
       without the filler text landing on the wrong turn in the transcript.

Key techniques shown:

    - ``TTSSpeakFrame(text, append_to_context=True)`` — the TTS service commits
      the spoken text to the assistant aggregator after the audio drains, so
      turn ordering in the transcript matches the audio.
    - ``pause_frame_processing=True`` on the TTS service — stops the TTS from
      processing the next LLM response while the filler is still speaking,
      which is what keeps the audio and the transcript aligned during tool
      calls.
    - A system-prompt nudge asking the LLM not to acknowledge before a tool
      call, so you don't get double acknowledgements (one from the LLM, one
      from ``on_function_calls_started``).

Notes:

    - Do NOT call ``asyncio.sleep`` to add pauses around TTS. Use
      ``FrameProcessorPauseFrame`` / ``FrameProcessorResumeUrgentFrame`` if you
      need a synthetic gap. ``asyncio.sleep`` does not interact with the
      frame-processing system and will desync audio and transcript.
    - The base ``TTSService`` defaults ``pause_frame_processing`` to ``False``.
      Many wrappers (ElevenLabs, Rime, Deepgram, Groq, Azure, ...) hardcode it
      to ``True`` in their ``super().__init__()`` calls, so you don't need to
      opt in. ``OpenAITTSService`` inherits the base default (``False``), so we
      pass it explicitly below.
    - ``CartesiaTTSService`` is the odd one: it hardcodes
      ``pause_frame_processing=False`` AND does not accept the kwarg via the
      constructor (you'll get ``TypeError: got multiple values for keyword
      argument 'pause_frame_processing'``). If you're on Cartesia, set it after
      construction: ``tts._pause_frame_processing = True``.

Requirements:
    - OpenAI API key

    Environment variables (.env):
        OPENAI_API_KEY=...
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
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
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


SYSTEM_INSTRUCTION = """You are a helpful assistant in a voice conversation. Your \
responses will be spoken aloud, so avoid emojis, bullet points, or other formatting \
that can't be spoken. Keep responses brief.

IMPORTANT: When you are about to call a tool, do NOT say an acknowledgement like \
"Let me check on that" or "One moment" before the call. The system plays its own \
filler audio while the tool runs, so if you also acknowledge you will be heard twice."""


async def fetch_weather_from_api(params: FunctionCallParams):
    await params.result_callback({"conditions": "sunny", "temperature": "75"})


transport_params = {
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting TTSSpeakFrame timing demo")

    stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))

    # pause_frame_processing=True keeps filler audio and the post-tool LLM
    # response from overlapping. OpenAI TTS inherits the base default of False,
    # so we opt in explicitly here.
    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAITTSService.Settings(voice="ballad"),
        pause_frame_processing=True,
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(system_instruction=SYSTEM_INSTRUCTION),
    )

    llm.register_function("get_current_weather", fetch_weather_from_api)

    # Tool-call filler. Fires once per function-call batch. append_to_context=True
    # makes the filler text show up in the transcript in the correct turn order,
    # because the TTS service commits it only after the audio drains.
    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Gotcha, one sec.", append_to_context=True))

    weather_function = FunctionSchema(
        name="get_current_weather",
        description="Get the current weather for a location",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        required=["location"],
    )
    tools = ToolsSchema(standard_tools=[weather_function])

    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
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
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

        # Intro / pre-roll. append_to_context=True makes this line land in the
        # LLM context before the first user message, in the correct turn order.
        # No LLMFullResponseStart/End wrap needed.
        await tts.queue_frame(
            TTSSpeakFrame(
                "Hi, I'm Paul, your virtual agent. Ask me about the weather anywhere.",
                append_to_context=True,
            )
        )

        # Kick off the LLM so it's ready to respond to the first user turn.
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
