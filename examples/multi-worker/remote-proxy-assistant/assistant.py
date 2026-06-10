#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Remote assistant LLM server.

Runs a FastAPI server that accepts WebSocket connections from a
``main.py``-style client. Each connection spins up a
`WebSocketProxyServer` bridging the socket to a local
`WorkerRunner` and an `LLMWorker` that handles the conversation.

Usage::

    python assistant.py
    python assistant.py --port 9000

Requirements:

- OPENAI_API_KEY
"""

import argparse
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from loguru import logger

from pipecat.bus import BusFrameMessage
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.workers.llm import LLMWorker, tool
from pipecat.workers.proxy.websocket import WebSocketProxyServer
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

app = FastAPI()


class AcmeAssistant(LLMWorker):
    """Handles greetings, product questions, and conversation end."""

    def __init__(self):
        """Initialize the AcmeAssistant LLM worker."""
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(
                system_instruction=(
                    "You are a friendly assistant for Acme Corp. You know about three "
                    "products: Acme Rocket Boots (jet-powered boots, $299, run up to "
                    "60 mph), Acme Invisible Paint (makes anything invisible for 24 hours, "
                    "$49 per can), and Acme Tornado Kit (portable tornado generator, $199, "
                    "batteries included). Greet the user, help them with product questions, "
                    "and call end_conversation when the user says goodbye. "
                    "Keep responses brief, this is a voice conversation."
                ),
            ),
        )
        super().__init__("assistant", llm=llm, bridged=())

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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle a WebSocket connection from the main bot's proxy."""
    await websocket.accept()

    runner = WorkerRunner(handle_sigint=False)

    proxy = WebSocketProxyServer(
        "gateway",
        websocket=websocket,
        worker_name="assistant",
        remote_worker_name="acme",
        forward_messages=(BusFrameMessage,),
    )

    @proxy.event_handler("on_client_connected")
    async def on_client_connected(proxy, client):
        logger.info("WebSocket client connected")

    @proxy.event_handler("on_client_disconnected")
    async def on_client_disconnected(proxy, client):
        logger.info("WebSocket client disconnected")
        await runner.cancel()

    assistant = AcmeAssistant()

    await runner.add_workers(proxy, assistant)

    logger.info("Assistant server ready, waiting for activation")
    await runner.run()
    logger.info("Assistant server session ended")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote assistant LLM server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
