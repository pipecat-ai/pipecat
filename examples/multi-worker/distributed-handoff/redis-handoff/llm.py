#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM worker — run on Machine B (or locally alongside ``main.py``).

A standalone process that runs one LLM worker (greeter or support)
attached to the same Redis-backed `WorkerBus` as the main worker.
Multiple instances can run on different machines.

Usage::

    python llm.py greeter --redis-url redis://localhost:6379
    python llm.py support --redis-url redis://localhost:6379

Requirements:

- OPENAI_API_KEY
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from redis.asyncio import Redis

from pipecat.bus.network.redis import RedisBus
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.workers.llm import LLMWorker, LLMWorkerActivationArgs, tool
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

WORKER_CONFIG = {
    "greeter": {
        "system_instruction": (
            "You are a friendly greeter for Acme Corp. The available products "
            "are: the Acme Rocket Boots, the Acme Invisible Paint, and the Acme "
            "Tornado Kit. Ask which one they'd like to learn more about. "
            "When the user picks a product or asks a question about one, "
            "immediately call the transfer_to_agent tool with agent 'support'. "
            "Do not answer product questions yourself. If the user says goodbye, "
            "call the end_conversation tool. Do not mention transferring — just do it "
            "seamlessly. Keep responses brief — this is a voice conversation."
        ),
        "watch": ["support"],
    },
    "support": {
        "system_instruction": (
            "You are a support agent for Acme Corp. You know about three "
            "products: Acme Rocket Boots (jet-powered boots, $299, run up "
            "to 60 mph), Acme Invisible Paint (makes anything invisible for "
            "24 hours, $49 per can), and Acme Tornado Kit (portable tornado "
            "generator, $199, batteries included). Answer the user's questions "
            "about these products. If the user wants to browse other products "
            "or start over, call the transfer_to_agent tool with agent "
            "'greeter'. If the user says goodbye, call the end_conversation "
            "tool. Do not mention transferring — just do it seamlessly. "
            "Keep responses brief — this is a voice conversation."
        ),
        "watch": ["greeter"],
    },
}


class AcmeLLMTask(LLMWorker):
    """LLM worker for Acme Corp with transfer and end tools."""

    def __init__(self, name: str, *, system_instruction: str, watch: list[str]):
        """Initialize the AcmeLLMTask.

        Args:
            name: Unique worker name (``"greeter"`` or ``"support"``).
            system_instruction: System prompt for this LLM role.
            watch: Sibling worker names this worker will watch via the
                registry so it knows when they become available for
                handoff.
        """
        llm = OpenAILLMService(
            name=f"{name}::OpenAILLMService",
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(system_instruction=system_instruction),
        )
        super().__init__(name, llm=llm, bridged=())
        self._watch = watch

    async def start(self) -> None:
        """Register watches for sibling workers once ready."""
        await super().start()
        await self.watch_workers(*self._watch)

    @tool(cancel_on_interruption=False)
    async def transfer_to_agent(self, params: FunctionCallParams, agent: str, reason: str):
        """Transfer the user to another agent.

        Args:
            agent (str): The agent to transfer to (e.g. 'greeter', 'support').
            reason (str): Why the user is being transferred.
        """
        logger.info(f"Task '{self.name}': transferring to '{agent}' ({reason})")
        await self.activate_worker(
            agent,
            args=LLMWorkerActivationArgs(messages=[{"role": "developer", "content": reason}]),
            deactivate_self=True,
            result_callback=params.result_callback,
        )

    @tool
    async def end_conversation(self, params: FunctionCallParams, reason: str):
        """End the conversation when the user says goodbye.

        Args:
            reason (str): Why the conversation is ending.
        """
        logger.info(f"Task '{self.name}': ending conversation ({reason})")
        await self.end(
            reason=reason,
            messages=[{"role": "developer", "content": reason}],
            result_callback=params.result_callback,
        )


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="LLM worker (greeter or support)")
    parser.add_argument("worker", choices=list(WORKER_CONFIG), help="Which worker to run")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--channel", default="pipecat:acme", help="Redis pub/sub channel")
    args = parser.parse_args()

    redis = Redis.from_url(args.redis_url)
    bus = RedisBus(redis=redis, channel=args.channel)

    config = WORKER_CONFIG[args.worker]
    worker = AcmeLLMTask(
        args.worker,
        system_instruction=config["system_instruction"],
        watch=config["watch"],
    )

    runner = WorkerRunner(bus=bus, handle_sigint=True)
    logger.info(f"Starting {args.worker} worker, waiting for activation...")
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main_async())
