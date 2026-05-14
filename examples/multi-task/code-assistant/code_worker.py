#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Code worker that explores a codebase using Claude Agent SDK."""

import asyncio

from loguru import logger

from pipecat.bus import (
    BusCancelTaskMessage,
    BusEndTaskMessage,
    BusJobRequestMessage,
)
from pipecat.pipeline.base_task import BaseTask
from pipecat.pipeline.job_context import JobStatus

try:
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use CodeWorker, you need to `pip install claude-agent-sdk`.")
    raise Exception(f"Missing module: {e}")


class CodeWorker(BaseTask):
    """Bus-only task that answers code questions using Claude Agent SDK.

    Maintains a persistent Claude SDK session so follow-up questions
    share context. Questions are queued and processed sequentially. The
    worker has no Pipecat pipeline — it consumes job requests from the
    bus and replies with job responses.
    """

    def __init__(self, name: str, *, project_path: str):
        """Initialize the CodeWorker.

        Args:
            name: Unique task name.
            project_path: Filesystem path the Claude SDK should explore.
        """
        super().__init__(name)

        self._project_path = project_path
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

        self._claude_options = ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            system_prompt=(
                f"You are a code assistant. The project is at: {self._project_path}\n\n"
                "Answer the user's question by exploring the codebase. Use Read to "
                "view files, Glob to find files by pattern, and Bash to run commands "
                "like grep or find. Be thorough but concise in your answer. "
                "Focus on what the user asked. Respond with a clear, spoken-friendly "
                "summary (no markdown, no bullet points, no code blocks)."
            ),
            allowed_tools=["Read", "Bash", "Glob", "Grep"],
            model="sonnet",
            max_turns=10,
        )

    async def start(self) -> None:
        """Launch the Claude SDK worker loop alongside the standard task start."""
        await super().start()
        self._worker_task = self.create_task(self._worker_loop(), f"{self.name}::worker")

    async def stop(self) -> None:
        """Cancel the worker loop before tearing down the task."""
        if self._worker_task:
            await self.cancel_task(self._worker_task)
            self._worker_task = None
        await super().stop()

    async def on_job_request(self, message: BusJobRequestMessage) -> None:
        """Enqueue an incoming job for the worker loop."""
        await super().on_job_request(message)
        logger.info(f"Worker '{self.name}': queued '{message.payload['question']}'")
        self._queue.put_nowait(message)

    async def _handle_task_end(self, message: BusEndTaskMessage) -> None:
        """Signal the run loop to finish on a graceful end."""
        await super()._handle_task_end(message)
        self._finished_event.set()

    async def _handle_task_cancel(self, message: BusCancelTaskMessage) -> None:
        """Signal the run loop to finish on cancellation."""
        await super()._handle_task_cancel(message)
        self._finished_event.set()

    async def _worker_loop(self):
        try:
            async with ClaudeSDKClient(options=self._claude_options) as client:
                while True:
                    message = await self._queue.get()
                    question = message.payload["question"]
                    logger.info(f"Worker '{self.name}': researching '{question}'")

                    try:
                        answer = ""
                        await client.query(prompt=question)
                        async for msg in client.receive_response():
                            if type(msg).__name__ == "AssistantMessage":
                                for block in msg.content:
                                    if type(block).__name__ == "TextBlock":
                                        answer += block.text

                        logger.info(f"Worker '{self.name}': completed ({len(answer)} chars)")
                        await self.send_job_response(message.job_id, {"answer": answer})

                    except Exception as e:
                        logger.error(f"Worker '{self.name}': error: {e}")
                        await self.send_job_response(
                            message.job_id, {"error": str(e)}, status=JobStatus.ERROR
                        )
        except Exception as e:
            logger.error(f"Worker '{self.name}': failed to start Claude SDK: {e}")
