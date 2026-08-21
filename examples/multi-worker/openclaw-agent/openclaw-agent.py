#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice front end for an OpenClaw agent.

Two loops run side by side. The **voice loop** is an ordinary transport +
STT + LLM + TTS pipeline that answers the user itself. The **agent loop**
is a ``PipelineWorker`` driving an OpenClaw agent over its Gateway
websocket. They meet only at the bus, over jobs.

Architecture::

    Voice loop (transport + LLM + send/stop/status tools)
      └── job → Agent loop (OpenClawGatewayService)
                  └── websocket → OpenClaw Gateway

The voice LLM makes one judgment per turn: answer the user itself, or
forward what they said to the agent. It does **not** decide whether that
input starts a task or redirects the running one — the agent loop owns
that, because it is a property of the Gateway rather than of the
conversation. So there is one delegation tool, and a follow-up shouted
mid-task goes through it exactly like a fresh request.

An agent run takes minutes, so ``send_to_agent`` dispatches and returns.
The voice loop keeps taking turns while the agent works, and speaks each
outcome when it lands.

Requirements:

- An OpenClaw Gateway to talk to, and ``OPENCLAW_TOKEN`` for it
- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- DAILY_API_KEY (for Daily transport)
"""

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger
from openclaw_worker import WORKER_NAME, build_openclaw_worker

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.bus.messages import BusJobResponseMessage, BusJobUpdateMessage
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import (
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
    LLMRunFrame,
)
from pipecat.pipeline.job_context import JobStatus
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
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

SPOKEN_OUTPUT_INSTRUCTION = (
    "Use plain spoken text only: no markdown, bullets, code fences, or emoji."
)

SYSTEM_PROMPT = f"""\
You are the voice of a coding agent. You handle the conversation; the agent
handles the work.

Answer directly whenever you can: greetings, small talk, and anything you
already know from this conversation. Do not call a tool for those.

Call send_to_agent for anything you cannot answer immediately yourself:
reading or writing files, running commands, investigating a bug, anything
that needs the codebase or the machine. Call it BOTH to start new work and
to pass along a correction, follow-up, or change of mind while the agent is
working. Just forward what the user said. The agent decides whether that
input starts a task or redirects the one it is running; you do not.

Say nothing when you call send_to_agent. The agent reports back within a beat
and you will be told what to say: a short acknowledgement of one to four words
when it takes the work up, or what happened when it does something else. Do not
add filler or a call to action.

To stop the agent, call stop_agent. To answer "what's it doing?" or "is that
done yet?", call agent_status, which is free and touches nothing.

Results arrive later in a developer message. Summarize each one
conversationally in a sentence or two, keeping any codes, numbers, and names
accurate, and let the user ask for more.

{SPOKEN_OUTPUT_INSTRUCTION}
"""

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


@dataclass
class ActiveTask:
    """What the agent is working on, as the voice loop understands it."""

    job_id: str
    request: str
    started_at: float


class VoiceLoopWorker(PipelineWorker):
    """The media path and the voice loop.

    It keeps one handle on the agent's task, learned from the agent loop, so it
    can stop the work, say what is running, and narrate each outcome honestly.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the worker."""
        super().__init__(*args, **kwargs)
        self._active: ActiveTask | None = None

    @property
    def active_task(self) -> ActiveTask | None:
        """The agent's current task, or None while it is idle."""
        return self._active

    async def stop_active_task(self, reason: str) -> str | None:
        """Cancel the agent's task, returning its job id, or None if idle.

        Args:
            reason: Why the user wants it stopped.

        Returns:
            The cancelled job id, or None when nothing was running.
        """
        if self._active is None:
            return None
        logger.info(f"Voice loop: stopping job {self._active.job_id}: '{reason}'")
        await self.cancel_job_group(self._active.job_id, reason=reason)
        return self._active.job_id

    async def on_job_update(self, message: BusJobUpdateMessage):
        """Record the task the agent has accepted.

        Args:
            message: The update from the agent loop.
        """
        await super().on_job_update(message)
        update = message.update or {}
        if message.source == WORKER_NAME and update.get("kind") == "started":
            self._active = ActiveTask(
                job_id=message.job_id,
                request=str(update.get("request", "")),
                started_at=time.monotonic(),
            )
            # The acknowledgement waits for this rather than riding on the tool
            # result, so what the user hears is the agent actually picking the
            # work up. A follow-up produces no start, and is narrated as a
            # redirect instead, so exactly one of the two is ever spoken.
            await self._say(
                "The user's request is now with the agent. Acknowledge it in one to "
                "four words, like 'On it.' Say nothing else."
            )

    async def on_job_response(self, message: BusJobResponseMessage):
        """Turn one agent outcome into something the voice loop can say.

        Args:
            message: The response from the agent loop.
        """
        await super().on_job_response(message)
        if message.source != WORKER_NAME:
            return

        response = message.response or {}
        kind = response.get("kind")
        cancelled = message.status == JobStatus.CANCELLED
        failed = not cancelled and (message.status != JobStatus.COMPLETED or kind == "error")
        steering = not cancelled and not failed and kind == "steering"

        if self._active and self._active.job_id == message.job_id and not steering:
            self._active = None

        if cancelled:
            note = "The agent's task was stopped. Tell the user it is cancelled."
        elif failed:
            note = (
                "The agent could not finish the task. Tell the user it failed. "
                f"Reason: {response.get('error', message.status)}"
            )
        elif steering:
            # The Gateway aborts the running turn and starts a replacement, so
            # do not say the note was added to work already in progress.
            note = (
                "The agent has switched to the user's update and is working on that "
                "now. Acknowledge it in a few words. Do not imply it is still "
                "working on the earlier version."
            )
        else:
            note = (
                "A result from the agent is ready. Turn it into one concise spoken "
                "answer, keeping any codes, numbers, and names accurate. If it says "
                "it could not work something out, say so plainly. Do not add a "
                f"follow-up question or a call to action. Result: {response.get('answer', '')}"
            )

        await self._say(note)

    async def _say(self, note: str):
        """Hand the voice loop something to say on its next turn."""
        await self.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "developer", "content": f"{note} {SPOKEN_OUTPUT_INSTRUCTION}"}],
                run_llm=True,
            )
        )


@tool_options(cancel_on_interruption=False, timeout_secs=5)
async def send_to_agent(params: FunctionCallParams, user_input: str):
    """Send the user's request to the agent, or add to what it is doing.

    Use this both to start new work and to pass along a follow-up, correction,
    or change of mind while the agent is already working. Forward what the user
    said; the agent decides whether that starts a task or redirects the running
    one. The result arrives later and you will be asked to speak it.

    Args:
        user_input (str): What the user wants done or wants to add, keeping the
            details that matter.
    """
    worker = params.pipeline_worker

    job_id = await worker.request_job(WORKER_NAME, name="run", payload={"input": user_input})
    logger.info(f"Voice loop: forwarded as job {job_id[:8]}: '{user_input}'")

    # Say nothing yet. The agent loop reports back within a beat, and what it
    # reports decides what the user hears: a start gets the quick "on it", a
    # redirect gets told as a redirect. Only the agent loop knows which of the
    # two this was.
    await params.result_callback(
        {"status": "sent"},
        properties=FunctionCallResultProperties(run_llm=False),
    )


@tool_options(cancel_on_interruption=False, timeout_secs=5)
async def stop_agent(params: FunctionCallParams, reason: str):
    """Stop what the agent is working on right now.

    This is preemptive: it halts the work rather than queueing another
    instruction. If the agent is not working on anything, say so.

    Args:
        reason (str): Why the user wants it stopped, briefly.
    """
    job_id = await params.pipeline_worker.stop_active_task(reason)
    if job_id is None:
        await params.result_callback(
            {"status": "nothing_running"},
            properties=FunctionCallResultProperties(run_llm=True),
        )
        return
    # The agent loop answers the cancelled job; narrate from there.
    await params.result_callback(
        {"status": "stopping"},
        properties=FunctionCallResultProperties(run_llm=False),
    )


@tool_options(cancel_on_interruption=False)
async def agent_status(params: FunctionCallParams):
    """Say whether the agent is working, on what, and for how long.

    Read-only and instant: it reads what the voice loop already knows and does
    not touch the agent.
    """
    active = params.pipeline_worker.active_task
    await params.result_callback(
        {
            "status": "working" if active else "idle",
            "asked": active.request if active else None,
            "running_for_secs": round(time.monotonic() - active.started_at, 1) if active else None,
        },
        properties=FunctionCallResultProperties(run_llm=True),
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting OpenClaw voice front end")

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline
        ),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(system_instruction=SYSTEM_PROMPT),
    )

    context = LLMContext(tools=[send_to_agent, stop_agent, agent_status])
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

    worker = VoiceLoopWorker(
        pipeline,
        name="voice-loop",
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        # A user waiting on the agent is quiet, and quiet reads as idle. Decide
        # what to do about it below rather than letting the timeout end the
        # session on its own.
        cancel_on_idle_timeout=False,
    )

    @worker.event_handler("on_idle_timeout")
    async def on_idle_timeout(worker):
        if worker.active_task:
            logger.info("Voice loop: quiet, but the agent is working; staying up")
            return
        logger.info("Voice loop: idle with nothing running; ending the session")
        await runner.cancel()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Greet the user, and tell them you have a coding agent behind you "
                    "that they can put to work."
                ),
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(build_openclaw_worker(), worker)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
