#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent loop: a ``PipelineWorker`` that drives one OpenClaw agent.

The pipeline is two processors long::

    OpenClawGatewayService -> OpenClawAggregator

The processor turns the Gateway's websocket traffic into frames, and the
collector folds one run's frames into the single answer a voice loop can speak.

This worker owns the decision the voice loop does not make: whether forwarded
input starts a task or redirects the one already running. All the Gateway's
particulars live here, and the voice loop just forwards.
"""

import asyncio
import os
import time

from loguru import logger

from pipecat.bus import BusJobRequestMessage
from pipecat.frames.frames import Frame
from pipecat.pipeline.job_context import JobStatus
from pipecat.pipeline.job_decorator import job
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openclaw.client import (
    DEFAULT_GATEWAY_URL,
    DEFAULT_SESSION_KEY,
)
from pipecat.services.openclaw.frames import (
    OpenClawAbortFrame,
    OpenClawEndFrame,
    OpenClawSendFrame,
    OpenClawStartedFrame,
    OpenClawSteerFrame,
    OpenClawTextFrame,
)
from pipecat.services.openclaw.gateway import OpenClawGatewayService

WORKER_NAME = "openclaw-agent"

# What an agent has to be told to produce an answer someone can listen to. The
# client sends the caller's message verbatim, so framing is the caller's job,
# and this is the caller.
#
# The Gateway labels a programmatic sender in the message envelope and marks
# that label untrusted, which an agent will otherwise remark on in its answer.
# Asking it not to describe the plumbing is as far as this goes: whether the
# sender deserves more trust than the Gateway affords it is the operator's call
# to make, in their own session with the agent.
SPOKEN_ANSWER_INSTRUCTION = (
    "This arrived from someone speaking out loud, and your answer is read back to "
    "them by a voice bot. Do not describe the sender, the channel, or how the "
    "message reached you; answer what was asked. Use plain spoken text: no "
    "markdown, bullets, code fences, links, or emoji. Give one concise answer for "
    "someone who is listening rather than reading, and say so plainly if you "
    "cannot work it out. Search your tools before concluding you cannot do "
    "something: an agent that discloses its tools progressively only sees a few "
    "of them until it looks."
)


class OpenClawAggregator(FrameProcessor):
    """Folds one run's frames into the answer the voice loop speaks.

    An agent answers in pieces over minutes. A spoken turn wants it whole, so
    the deltas are gathered here and delivered once the run reaches a terminal
    frame.

    Event handlers available:

    - on_started: Called with the run id when a run begins.
    - on_result: Called with how a run ended and what it produced.

    Example::

        @aggregator.event_handler("on_result")
        async def on_result(aggregator, status, text):
            ...
    """

    def __init__(self, **kwargs):
        """Initialize the aggregator.

        Args:
            **kwargs: Additional arguments passed to the frame processor.
        """
        super().__init__(**kwargs)
        self._parts: list[str] = []
        # Synchronous, so a run's outcome reaches the job waiting on it before
        # the next frame moves.
        self._register_event_handler("on_started", sync=True)
        self._register_event_handler("on_result", sync=True)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Gather a run's text and report how it ended.

        Args:
            frame: The frame to process.
            direction: The direction the frame is travelling.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, OpenClawStartedFrame):
            self._parts = []
            await self._call_event_handler("on_started", frame.run_id)
        elif isinstance(frame, OpenClawTextFrame):
            self._parts.append(frame.text)
        elif isinstance(frame, OpenClawEndFrame):
            # A run that answered reports the answer, falling back to what it
            # streamed. A run that didn't reports why, which the text carries
            # on its own.
            answer = frame.text
            if frame.status == "completed":
                answer = frame.text or "".join(self._parts).strip()
            await self._call_event_handler("on_result", frame.status, answer)

        await self.push_frame(frame, direction)


class OpenClawAgentWorker(PipelineWorker):
    """Bus worker that owns one OpenClaw agent's state.

    Forwarded input starts a task when the agent is idle and redirects the
    running one when it is not. That decision lives here rather than in the
    voice loop, because it is a property of the backend: a Gateway session
    carries one turn, and ``sessions.steer`` is how a second instruction
    reaches a turn already in flight.

    Steering is not merging. The Gateway aborts the running turn and starts a
    replacement carrying the follow-up, so the reply says ``redirected`` and
    the voice loop is careful not to imply the earlier work continues.
    """

    def __init__(
        self,
        service: OpenClawGatewayService,
        *,
        name: str = WORKER_NAME,
        idle_timeout_secs: float | None = None,
        **kwargs,
    ):
        """Initialize the worker.

        Args:
            service: The Gateway service the pipeline drives.
            name: The name the voice loop addresses jobs to.
            idle_timeout_secs: Off by default. Idleness is measured in bot and
                user speaking frames, and this pipeline carries neither, so a
                timeout here would fire while the conversation was going and
                take the whole runner down with it.
            **kwargs: Additional arguments passed to the pipeline worker.
        """
        aggregator = OpenClawAggregator()
        aggregator.add_event_handler("on_started", self._on_run_started)
        aggregator.add_event_handler("on_result", self._on_run_result)
        pipeline = Pipeline([service, aggregator])
        super().__init__(pipeline, name=name, idle_timeout_secs=idle_timeout_secs, **kwargs)
        # The run in flight, and what the job handler is waiting on.
        self._result: asyncio.Future | None = None
        # Whether the run this job asked for has actually begun. A run the user
        # stopped can still report itself once the next one is under way, and
        # that report belongs to neither job.
        self._started = False

    @job(name="run")
    async def run_agent(self, message: BusJobRequestMessage):
        """Start a task, or redirect the one already running.

        Args:
            message: The job carrying the user's input.
        """
        user_input = str((message.payload or {}).get("input", ""))

        if self._busy:
            logger.info(f"Agent loop: redirecting onto '{user_input}'")
            await self.queue_frame(OpenClawSteerFrame(message=user_input))
            await self.send_job_response(
                message.job_id,
                {"kind": "steering", "status": "redirected", "input": user_input},
                urgent=True,
            )
            return

        logger.info(f"Agent loop: starting '{user_input}'")
        result = asyncio.get_running_loop().create_future()
        self._result = result
        self._started = False
        await self.send_job_update(
            message.job_id,
            {"kind": "started", "request": user_input, "started_at": time.time()},
            urgent=True,
        )
        await self.queue_frame(
            OpenClawSendFrame(message=f"{user_input}\n\n{SPOKEN_ANSWER_INSTRUCTION}")
        )

        try:
            status, text = await result
        except asyncio.CancelledError:
            # Stopped by the voice loop. The bus answers the requester CANCELLED
            # on its own; the run itself has to be told.
            await self.queue_frame(OpenClawAbortFrame(reason="the user asked to stop"))
            self._release(result)
            raise

        self._release(result)
        logger.info(f"Agent loop: run {status} for job {message.job_id[:8]}")
        if status == "failed":
            await self.send_job_response(
                message.job_id,
                {"kind": "error", "error": text},
                status=JobStatus.ERROR,
                urgent=True,
            )
        else:
            # Urgent, so a finished answer preempts queued bus traffic and the
            # voice loop can speak it promptly.
            await self.send_job_response(
                message.job_id, {"kind": "final", "status": status, "answer": text}, urgent=True
            )

    @property
    def _busy(self) -> bool:
        """Whether a run is in flight."""
        return self._result is not None and not self._result.done()

    def _release(self, result: asyncio.Future):
        """Free the agent, unless the next task already claimed it.

        A job settled a moment before the next one arrived leaves the handler
        still to resume, and by then the slot may belong to someone else.
        """
        if self._result is result:
            self._result = None

    async def _on_run_started(self, aggregator: OpenClawAggregator, run_id: str):
        """Note that the waiting job's own run is now under way."""
        self._started = True

    async def _on_run_result(self, aggregator: OpenClawAggregator, status: str, text: str):
        """Hand a finished run back to the job that is waiting on it.

        A stopped run reports itself whenever the Gateway gets round to it,
        which can be after the next job has been accepted. Anything arriving
        before that job's own run starts belongs to the run before it.
        """
        if not self._started:
            logger.debug(f"Agent loop: ignoring a {status} run from before this job")
            return
        if self._result and not self._result.done():
            self._result.set_result((status, text))


def build_openclaw_worker(name: str = WORKER_NAME) -> OpenClawAgentWorker:
    """Build the agent loop from the environment.

    Args:
        name: The name the voice loop addresses jobs to.

    Returns:
        The worker, ready to hand to the runner.
    """
    service = OpenClawGatewayService(
        url=os.getenv("OPENCLAW_GATEWAY_URL", DEFAULT_GATEWAY_URL),
        token=os.getenv("OPENCLAW_TOKEN"),
        session_key=os.getenv("OPENCLAW_SESSION_KEY", DEFAULT_SESSION_KEY),
    )
    return OpenClawAgentWorker(service, name=name)
