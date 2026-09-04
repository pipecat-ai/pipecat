#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A two-tier voice agent: a cascade frontend delegating to a backend LLM.

The frontend keeps the conversation moving with a small, fast model and no
tools of its own. Anything that needs tools or careful reasoning it hands to
a ``BackendLLMWorker`` running Claude, through the ``delegate`` tool, and
relays its answer. This is the same backend worker and job contract
``OpenAILiveLLMService`` uses for client delegation.

The frontend here is a cascade pipeline; ``backend-llm-realtime-frontend.py``
puts a speech-to-speech model in the same role, against the same backend.

Architecture::

    Main worker (transport + STT + fast LLM + TTS, ``delegate`` tool)
      └── job → BackendLLMWorker (Claude + tools)

Requirements:

- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import FunctionCallResultProperties, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi import (
    RTVIFunctionCallReportLevel,
    RTVIObserverParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.llm import (
    BackendLLMWorker,
    BackendOutput,
    delegate_to_backend,
    render_transcript_request,
)
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

BACKEND_NAME = "backend"

FRONTEND_INSTRUCTIONS = """You are a friendly, concise voice assistant. Your responses are spoken
aloud, so keep them to one or two natural sentences without any formatting.

Answer simple conversational questions yourself. Whenever the user asks for
current information, such as the weather or a restaurant recommendation, or
asks you to look something up, call the delegate tool. The backend reads the
conversation, so you don't need to word the request — hand off as soon as
you know it is for the backend. While it runs, keep the conversation going;
when the result comes back, relay it in your own words."""

BACKEND_INSTRUCTIONS = """You are the backend of a voice assistant. Each message you receive
is the recent voice conversation between the user and the assistant, as a
transcript. Work out what is being asked from it and answer that. The
transcript may contain transcription errors; use the most likely intent.

Use the available tools to answer questions about the weather and
restaurants. Reply with the verified result in concise, conversational plain
text that the assistant can say to the user — no Markdown, no raw JSON — and
never claim an action completed without a tool result confirming it."""

transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    temperature = 75 if format == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": format,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


async def get_restaurant_recommendation(params: FunctionCallParams, location: str):
    """Get a restaurant recommendation.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
    """
    await params.result_callback({"name": "The Golden Dragon"})


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline
        ),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-5.4-mini", system_instruction=FRONTEND_INSTRUCTIONS
        ),
    )

    # The backend is handed the conversation and works out the request from
    # it, so it can read a short reply or a correction for itself. Only what
    # it hasn't seen: its own context keeps the rest. The count belongs to
    # this session.
    delegated_through = 0

    @tool_options(cancel_on_interruption=False)
    async def delegate(params: FunctionCallParams):
        """Hand the conversation to the backend, for anything needing tools, current information or careful reasoning."""
        nonlocal delegated_through
        messages = params.context.get_messages()
        conversation = messages[delegated_through:]
        first, delegated_through = delegated_through == 0, len(messages)
        logger.info(f"Delegating to the backend: {len(conversation)} new message(s)")

        async def on_update(output: BackendOutput):
            # The final answer comes back as this tool's result, below, so it
            # is skipped here. Everything else is recorded as an intermediate
            # result, and `speakable` decides whether the frontend says it now
            # or merely knows it: running the LLM is what gives this pipeline a
            # voice, the way the commentary channel does for a speech-to-speech
            # frontend.
            if output.is_final:
                return
            logger.info(f"Backend update (speakable={output.speakable}): {output.text!r}")
            await params.result_callback(
                {"text": output.text},
                properties=FunctionCallResultProperties(is_final=False, run_llm=output.speakable),
            )

        text = await delegate_to_backend(
            params.pipeline_worker,
            BACKEND_NAME,
            request=render_transcript_request(conversation, first=first),
            on_update=on_update,
            timeout_secs=120,
        )
        logger.info(f"Backend result: {text!r}")
        await params.result_callback(text)

    # The frontend's only tool is the handoff; the real tools live in the backend.
    context = LLMContext(tools=[delegate])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        name="frontend",
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        # Whether a client sees the handoff as a tool call is the app's call:
        # it is a real tool call, though it reports the handoff rather than the
        # work behind it. This example hides it. (The backend's own calls run in
        # its worker's pipeline, which this observer doesn't watch.)
        rtvi_observer_params=RTVIObserverParams(
            function_call_report_level={"delegate": RTVIFunctionCallReportLevel.DISABLED},
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    backend = BackendLLMWorker(
        name=BACKEND_NAME,
        # Thinking summaries stream back to the frontend as "thought" updates.
        llm=AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(
                system_instruction=BACKEND_INSTRUCTIONS,
                thinking=AnthropicLLMService.ThinkingConfig(type="adaptive", display="summarized"),
            ),
        ),
        context=LLMContext(tools=[get_current_weather, get_restaurant_recommendation]),
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker, backend)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {"role": "developer", "content": "Greet the user and ask how you can help."}
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
