#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

from loguru import logger
from utils import (
    EvalResult,
    load_module_from_path,
    print_begin_test,
    print_end_test,
    print_test_results,
)

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

sys.path.insert(0, os.path.abspath(FOUNDATIONAL_DIR))

EVAL_PROMPT = ""

PIPELINE_IDLE_TIMEOUT_SECS = 30


class EvalRunner:
    def __init__(self, pattern: str = ""):
        self._pattern = f".*{pattern}.*" if pattern else ""
        self._total_success = 0
        self._tests: List[EvalResult] = []
        self._queue = asyncio.Queue()

    async def assert_eval(self, params: FunctionCallParams):
        reasoning = params.arguments["reasoning"]
        logger.debug(f"🧠 EVAL REASONING: {reasoning}")
        await self._queue.put(params.arguments["result"])
        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        await params.result_callback(None)

    async def assert_eval_false(self):
        await self._queue.put(False)

    async def run_eval(self, example_file: str, prompt: str, eval: Optional[str] = None):
        if not re.match(self._pattern, example_file):
            return

        print_begin_test(example_file)

        start_time = time.time()

        try:
            await asyncio.wait(
                [
                    asyncio.create_task(run_example_pipeline(example_file)),
                    asyncio.create_task(run_eval_pipeline(self, prompt, eval)),
                ],
                timeout=90,
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"ERROR: Unable to run {example_file}: {e}")

        try:
            result = await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            result = False

        if result:
            self._total_success += 1

        eval_time = time.time() - start_time

        self._tests.append(EvalResult(name=example_file, result=result, time=eval_time))

        print_end_test(example_file, result, eval_time)

    def print_results(self):
        print_test_results(self._tests, self._total_success)


async def run_example_pipeline(example_file: str):
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL")

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


async def run_eval_pipeline(eval_runner: EvalRunner, prompt: str, eval: Optional[str]):
    logger.info(f"Starting eval bot")

    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL")

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

    llm.register_function("assert_eval", eval_runner.assert_eval)

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
            "content": f"You are an LLM eval, be extremly brief. Your goal is to only ask one question: {prompt}. Tell the user to simply say the answer. Call the eval function only if the user answers the question and check if the answer is correct (words as numbers are valid). {eval_prompt}",
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
        await eval_runner.assert_eval_false()

    runner = PipelineRunner()

    await runner.run(task)
