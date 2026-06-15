#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice agent + sensor-controller worker, both as plain PipelineTasks.

Two ``PipelineWorker`` instances run side by side:

- The **voice agent** is built inline in ``run_bot`` — a standard
  transport + STT + LLM + TTS pipeline. Its LLM has a single tool,
  ``ask_controller(question)``, which forwards the user's request to
  the controller over the bus and speaks back the response.
- The **sensor controller** (``build_sensor_controller``) is a
  ``PipelineWorker`` whose pipeline runs a simulated temperature sensor
  (see ``sensor.py``) alongside its own LLM. The worker LLM has tool
  access to read the current reading, inspect rolling stats, and
  mutate the simulated sensor's target temperature and response rate.

The worker does **not** subclass ``LLMWorker`` and is **not** bridged.
The voice agent and the controller communicate exclusively through
``BusJobRequestMessage`` / ``BusJobResponseMessage``. The controller
collects responses by listening to the assistant aggregator's
``on_assistant_turn_stopped`` event and pairing each LLM completion
with the in-flight job id.

Requirements:

- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- DAILY_API_KEY (for Daily transport)

Example voice exchange::

    User: What's the temperature?
    Controller: 22.1°C, holding steady.

    User: Make it warmer.
    Controller: I set the target to 26°C. Give it about 20 seconds.

    User: Is it stable yet?
    Controller: It's at 25.4°C and still climbing — almost there.

    User: Why is it slow?
    Controller: The response rate is 5%. I sped it up to 20%; it'll settle faster now.
"""

import os

from dotenv import load_dotenv
from loguru import logger
from sensor import SensorReader, SensorStats

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.bus import BusJobRequestMessage
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMMessagesAppendFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


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


def build_sensor_controller() -> PipelineWorker:
    """Build the controller worker as a plain :class:`PipelineWorker`.

    The pipeline shape is::

        SensorReader -> SensorStats -> user_agg -> llm -> assistant_agg

    ``SensorReader`` runs an autonomous tick loop that emits a
    :class:`SensorReadingFrame` every second; ``SensorStats`` consumes
    those readings and exposes rolling statistics. The LLM has four
    direct tools that read or mutate the sensor.

    Jobs arrive via the ``on_job_request`` event handler. The handler
    stores the active ``job_id``, then queues an
    :class:`LLMMessagesAppendFrame` with the user's question and runs
    the LLM. When the assistant turn finishes (signalled by the
    assistant aggregator's ``on_assistant_turn_stopped`` event), the
    handler sends a :class:`BusJobResponseMessage` carrying the LLM's
    answer back to the voice agent.
    """
    sensor = SensorReader()
    stats = SensorStats()

    async def get_current_reading(params: FunctionCallParams):
        """Read the sensor's current temperature in degrees Celsius."""
        await params.result_callback({"temperature": round(sensor.current, 2)})

    async def get_stats(params: FunctionCallParams):
        """Rolling minimum, maximum, average, and trend of the temperature."""
        await params.result_callback(
            {
                "min": round(stats.min, 2),
                "max": round(stats.max, 2),
                "avg": round(stats.avg, 2),
                "trend": stats.trend,
            }
        )

    async def set_target_temperature(params: FunctionCallParams, target_celsius: float):
        """Adjust the target temperature; the sensor will drift toward it.

        Args:
            target_celsius (float): The new target temperature in degrees Celsius.
        """
        sensor.set_target(target_celsius)
        await params.result_callback({"ok": True, "new_target": target_celsius})

    async def set_response_rate(params: FunctionCallParams, rate: float):
        """Set how aggressively the sensor approaches the target.

        Args:
            rate (float): Response rate between 0.01 (slow) and 0.3 (fast).
        """
        sensor.set_response_rate(rate)
        await params.result_callback({"ok": True, "new_rate": rate})

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a temperature sensor controller. You manage a single "
                "thermometer and answer the user's questions about it. Use the "
                "provided tools to read the current temperature, inspect rolling "
                "statistics, change the target temperature, or change how fast "
                "the sensor responds. When the user asks for a vague change "
                "('make it warmer', 'cooler'), pick a sensible target and call "
                "set_target_temperature. Always answer in one or two short "
                "sentences — your reply is spoken aloud."
            ),
        ),
    )
    context = LLMContext(
        tools=[
            get_current_reading,
            get_stats,
            set_target_temperature,
            set_response_rate,
        ]
    )
    aggregators = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            sensor,
            stats,
            aggregators.user(),
            llm,
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(pipeline, name="sensor-controller")

    # The controller handles one job at a time (the LLM pipeline can only
    # run one turn at a time). ``state["job_id"]`` pairs the in-flight
    # job with the next ``on_assistant_turn_stopped`` event.
    state: dict[str, str | None] = {"job_id": None}

    @worker.event_handler("on_job_request")
    async def on_request(_task, message: BusJobRequestMessage):
        question = message.payload["question"]
        logger.info(f"Controller: received question '{question}'")
        state["job_id"] = message.job_id
        await worker.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": question}],
                run_llm=True,
            )
        )

    @aggregators.assistant().event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(_aggregator, message: AssistantTurnStoppedMessage):
        # The aggregator fires this event on every ``LLMFullResponseEndFrame``,
        # including the tool-call round that precedes the tool result and has
        # no spoken text. Skip those so we only forward the LLM's final
        # response to the voice agent.
        if not message.content:
            return
        if state["job_id"] is None:
            return
        job_id, state["job_id"] = state["job_id"], None
        logger.info(f"Controller: answering job {job_id[:8]}")
        await worker.send_job_response(job_id, response={"answer": message.content})

    return worker


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting sensor-controller bot")

    # Voice agent: standard transport + STT + LLM + TTS pipeline. The
    # only tool the voice LLM has is ``ask_controller`` — it does not
    # know anything about temperatures, trends, or response rates.
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline
        ),
    )

    @tool_options(timeout_secs=60)
    async def ask_controller(params: FunctionCallParams, question: str):
        """Ask the temperature sensor controller anything about the sensor.

        Forward the user's request verbatim and speak back the answer.

        Args:
            question (str): The user's question or instruction to forward to the controller.
        """
        logger.info(f"Voice agent: forwarding to controller: '{question}'")
        async with params.pipeline_worker.job(
            "sensor-controller", payload={"question": question}, timeout=30
        ) as t:
            pass
        await params.result_callback(t.response["answer"])

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a friendly voice assistant with access to a temperature "
                "sensor controller. For ANY request about the temperature — "
                "reading it, adjusting it, checking trends, changing how fast it "
                "responds — call the ask_controller tool. Forward the user's "
                "request verbatim. Then speak the controller's answer back. "
                "Keep responses brief; do not add extra commentary."
            ),
        ),
    )
    context = LLMContext(tools=[ask_controller])
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
        name="voice-agent",
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Greet the user and let them know you can read or adjust a "
                    "temperature sensor on their behalf."
                ),
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(build_sensor_controller(), worker)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
