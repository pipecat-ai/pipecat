#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
from pathlib import Path
from typing import Optional

from loguru import logger
from utils import load_module_from_path, print_begin_test, print_end_test

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

SCRIPT_DIR = Path(__file__).resolve().parent

FOUNDATIONAL_DIR = SCRIPT_DIR.parent.parent / "examples" / "foundational"

EVAL_PROMPT = ""

PIPELINE_IDLE_TIMEOUT_SECS = 30


class EvalResult:
    def __init__(self, queue: asyncio.Queue) -> None:
        self._queue = queue

    async def assert_eval(self, params: FunctionCallParams):
        reasoning = params.arguments["reasoning"]
        logger.debug(f"🧠 EVAL REASONING: {reasoning}")
        await self._queue.put(params.arguments["result"])
        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        await params.result_callback(None)


async def run_example_pipeline(example_file: str):
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL")

    if not room_url:
        logger.error("DAILY_SAMPLE_ROOM_URL is not defined")
        return

    script_path = FOUNDATIONAL_DIR / example_file

    module = load_module_from_path(script_path)

    transport = DailyTransport(
        room_url,
        None,
        "Pipecat",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    await module.run_example(transport, argparse.Namespace(), True)


async def run_eval_pipeline(prompt: str, eval: Optional[str], queue: asyncio.Queue):
    logger.info(f"Starting eval bot")

    eval_result = EvalResult(queue)

    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL")

    if not room_url:
        logger.error("DAILY_SAMPLE_ROOM_URL is not defined")
        await queue.put(False)
        return

    transport = DailyTransport(
        room_url,
        None,
        "Pipecat Eval",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=2.0)),
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    llm.register_function("assert_eval", eval_result.assert_eval)

    eval_function = FunctionSchema(
        name="assert_eval",
        description="Called when the user answers a question.",
        properties={
            "result": {
                "type": "boolean",
                "description": "The result of the eval",
            },
            "reasoning": {
                "type": "string",
                "description": "Why the answer was considered correct or invalid",
            },
        },
        required=["result", "reasoning"],
    )
    tools = ToolsSchema(standard_tools=[eval_function])

    # See if we need to include an eval prompt.
    eval_prompt = ""
    if eval:
        eval_prompt = f"The answer is correct if the user says [{eval}]."

    messages = [
        {
            "role": "system",
            "content": f"You are an LLM eval, be extremly brief. Your goal is to only ask one question: {prompt}. Call the eval function only if the user answers the question and check if the answer is correct. {eval_prompt}",
        },
    ]

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
        idle_timeout_secs=PIPELINE_IDLE_TIMEOUT_SECS,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    @task.event_handler("on_idle_timeout")
    async def on_pipeline_idle_timeout(task):
        await queue.put(False)

    runner = PipelineRunner()

    await runner.run(task)


async def run_eval(example_file: str, prompt: str, eval: Optional[str] = None):
    print_begin_test(example_file)

    # Create result queue and run eval.
    queue = asyncio.Queue()

    try:
        await asyncio.gather(
            run_example_pipeline(example_file),
            run_eval_pipeline(prompt, eval, queue),
        )
    except asyncio.CancelledError:
        pass

    try:
        result = await asyncio.wait_for(queue.get(), timeout=1.0)
    except asyncio.TimeoutError:
        result = False

    print_end_test(example_file, result)
