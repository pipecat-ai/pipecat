#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Serve the Deepgram Voice Agent API from Pipecat.

Clients written for Deepgram's Voice Agent API connect to
``ws://localhost:8765/v1/agent/converse`` unchanged: each connection runs a
Pipecat pipeline built from the client's ``Settings`` message.

Environment:

- ``DEEPGRAM_API_KEY`` (required): listen and speak.
- ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``: the ``open_ai`` think provider.
- ``ANTHROPIC_API_KEY``: the ``anthropic`` think provider.

Install and run::

    pip install "pipecat-ai[deepgram-voice-agent,openai,anthropic]"
    python examples/deepgram-voice-agent/server.py --port 8765

The ``deepgram-voice-agent`` extra includes ``lameenc``, for ``mp3`` output.
"""

import argparse
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from pipecat.extensions.deepgram_voice_agent.server import DeepgramVoiceAgentServer
from pipecat.extensions.deepgram_voice_agent.services import ProviderCredentials

load_dotenv(override=True)

server = DeepgramVoiceAgentServer(
    ProviderCredentials(
        deepgram_api_key=os.environ["DEEPGRAM_API_KEY"],
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
)

app = FastAPI()
app.add_api_websocket_route("/v1/agent/converse", server.handle_websocket)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepgram Voice Agent API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
