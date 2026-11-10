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
- ``CARTESIA_API_KEY`` / ``ELEVENLABS_API_KEY``: those speak providers, when the
  client's ``endpoint`` doesn't carry its own key.
- ``GOOGLE_API_KEY``: the ``google`` think provider.

Install and run::

    pip install "pipecat-ai[deepgram,openai,anthropic,cartesia,elevenlabs,google,runner]"
    python examples/deepgram-voice-agent/server.py --port 8765
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
        cartesia_api_key=os.getenv("CARTESIA_API_KEY"),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
)

app = FastAPI()
server.add_routes(app)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepgram Voice Agent API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    # Like the Voice Agent API, rely on the client's KeepAlive messages rather
    # than pinging it: a client that isn't reading its socket can't answer.
    uvicorn.run(app, host=args.host, port=args.port, ws_ping_interval=None)
