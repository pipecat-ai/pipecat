#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
import os
import re
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import aiofiles
from deepgram import LiveOptions
from loguru import logger
from PIL.ImageFile import ImageFile
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
from pipecat.frames.frames import EndTaskFrame, LLMRunFrame, OutputImageRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.daily.transport import DailyParams, DailyTransport

SCRIPT_DIR = Path(__file__).resolve().parent

PIPELINE_IDLE_TIMEOUT_SECS = 60
EVAL_TIMEOUT_SECS = 120

EvalPrompt = str | Tuple[str, ImageFile]


class EvalRunner:
    def __init__(
        self,
        *,
        examples_dir: Path,
        pattern: str = "",
        record_audio: bool = False,
        name: Optional[str] = None,
        log_level: str = "DEBUG",
    ):
        self._examples_dir = examples_dir
        self._pattern = f".*{pattern}.*" if pattern else ""
        self._record_audio = record_audio
        self._log_level = log_level
        self._total_success = 0
        self._tests: List[EvalResult] = []
        self._queue = asyncio.Queue()

        # We to save runner files.
        name = name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._runs_dir = os.path.join(SCRIPT_DIR, "test-runs", name)
        self._logs_dir = os.path.join(self._runs_dir, "logs")
        self._recordings_dir = os.path.join(self._runs_dir, "recordings")
        os.makedirs(self._logs_dir, exist_ok=True)
        os.makedirs(self._recordings_dir, exist_ok=True)

    async def assert_eval(self, params: FunctionCallParams):
        result = params.arguments["result"]
        reasoning = params.arguments["reasoning"]
        logger.debug(f"🧠 EVAL REASONING(result: {result}): {reasoning}")
        await self._queue.put(result)
        await params.result_callback(None)
        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    async def assert_eval_false(self):
        await self._queue.put(False)

    async def run_eval(
        self,
        example_file: str,
        prompt: EvalPrompt,
        eval: str,
        user_speaks_first: bool = False,
    ):
        if not re.match(self._pattern, example_file):
            return

        # Store logs
        filename = self._log_file_name(example_file)
        log_file_id = logger.add(filename, level=self._log_level)

        print_begin_test(example_file)

        script_path = self._examples_dir / example_file

        start_time = time.time()

        try:
            tasks = [
                asyncio.create_task(run_example_pipeline(script_path)),
                asyncio.create_task(
                    run_eval_pipeline(self, example_file, prompt, eval, user_speaks_first)
                ),
            ]
            _, pending = await asyncio.wait(tasks, timeout=EVAL_TIMEOUT_SECS)
            if pending:
                logger.error(f"ERROR: Eval timeout expired, cancelling pending tasks...")
                # Both pipeline idle timeouts should have worked and both tasks
                # should have exited already, but if we got here something went
                # wrong so we perform an abrupt asyncio task cancellation, which
                # will not cleanup things nicely.
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        except Exception as e:
            logger.error(f"ERROR: Unable to run {example_file}: {e}")

        try:
            result = await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            result = False

        if result:
            self._total_success += 1

        eval_time = time.time() - start_time

        self._tests.append(EvalResult(name=example_file, result=result, time=eval_time))

        print_end_test(example_file, result, eval_time)

        logger.remove(log_file_id)

    def print_results(self):
        print_test_results(self._tests, self._total_success, self._runs_dir)

    async def save_audio(self, name: str, audio: bytes, sample_rate: int, num_channels: int):
        if len(audio) > 0:
            filename = self._recording_file_name(name)
            logger.debug(f"Saving {name} audio to {filename}")
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(2)
                    wf.setnchannels(num_channels)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio)
                async with aiofiles.open(filename, "wb") as file:
                    await file.write(buffer.getvalue())
        else:
            logger.warning(f"There's no audio to save for {name}")

    def _base_file_name(self, example_file: str):
        base_name = os.path.splitext(example_file)[0]
        return f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _log_file_name(self, example_file: str):
        base_name = self._base_file_name(example_file)
        return os.path.join(self._logs_dir, f"{base_name}.log")

    def _recording_file_name(self, example_file: str):
        base_name = self._base_file_name(example_file)
        return os.path.join(self._recordings_dir, f"{base_name}.wav")


async def run_example_pipeline(script_path: Path):
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL")

    module = load_module_from_path(script_path)

    transport = DailyTransport(
        room_url,
        None,
        "Pipecat",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    runner_args = RunnerArguments()
    runner_args.pipeline_idle_timeout_secs = PIPELINE_IDLE_TIMEOUT_SECS

    await module.run_bot(transport, runner_args)


async def run_eval_pipeline(
    eval_runner: EvalRunner,
    example_file: str,
    prompt: EvalPrompt,
    eval: str,
    user_speaks_first: bool = False,
):
    logger.info(f"Starting eval bot")

    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL")

    transport = DailyTransport(
        room_url,
        None,
        "Pipecat Eval",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=2.0)),
        ),
    )

    # We disable smart formatting because some times if the user says "3 + 2 is
    # 5" (in audio) this can be converted to "32 is 5".
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            language="multi",
            smart_format=False,
        ),
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="97f4b8fb-f2fe-444b-bb9a-c109783a857a",  # Nathan
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    llm.register_function("assert_eval", eval_runner.assert_eval)

    eval_function = FunctionSchema(
        name="assert_eval",
        description="Called when the user answers a question.",
        properties={
            "result": {
                "type": "boolean",
                "description": "Whether the answer is correct or not",
            },
            "reasoning": {
                "type": "string",
                "description": "Why the answer was considered correct or invalid",
            },
        },
        required=["result", "reasoning"],
    )
    tools = ToolsSchema(standard_tools=[eval_function])

    # Load example prompt depending on image.
    example_prompt = ""
    example_image: Optional[ImageFile] = None
    if isinstance(prompt, str):
        example_prompt = prompt
    elif isinstance(prompt, tuple):
        example_prompt, example_image = prompt

    eval_prompt = f"The answer is correct if it matches: {eval}."
    common_system_prompt = (
        "The user might say things other than the answer and that's allowed. "
        f"You should only call the eval function with your assessment when the user actually answers the question. {eval_prompt}"
    )
    if user_speaks_first:
        system_prompt = f"You are an LLM eval, be extremly brief. You will start the conversation by saying: '{example_prompt}'. {common_system_prompt}"
    else:
        system_prompt = f"You are an LLM eval, be extremly brief. Your goal is to first ask one question: {example_prompt}. {common_system_prompt}"

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    context = LLMContext(messages, tools)
    context_aggregator = LLMContextAggregatorPair(context)

    audio_buffer = AudioBufferProcessor()

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            audio_buffer,
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
        ),
        idle_timeout_secs=PIPELINE_IDLE_TIMEOUT_SECS,
    )

    @audio_buffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        await eval_runner.save_audio(example_file, audio, sample_rate, num_channels)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        if example_image:
            await task.queue_frame(
                OutputImageRawFrame(
                    image=example_image.tobytes(),
                    size=example_image.size,
                    format="RGB",
                )
            )
        await audio_buffer.start_recording()

        # Default behavior is for the bot to speak first
        # If the eval bot speaks first, we append the prompt to the messages
        if user_speaks_first:
            messages.append(
                {"role": "user", "content": f"Start by saying this exactly: '{prompt}'"}
            )
            await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    @task.event_handler("on_idle_timeout")
    async def on_pipeline_idle_timeout(task):
        await eval_runner.assert_eval_false()

    # TODO(aleix): We should handle SIGINT and SIGTERM so we can cancel both the
    # eval and the example.
    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)
