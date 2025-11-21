# SPDX-License-Identifier: BSD-2-Clause
"""
Vonage Audio connector with OpenAI.

The example:
- Runs a Pipecat voice assistant using OpenAI STT/LLM/TTS.
- Exposes a WebSocket server using VonageAudioConnectorTransport.
- Once the server is ready, it calls the Vonage Video API "Audio Connector"
  to connect an existing routed session to this WebSocket endpoint.

Requirements:
    - OpenAI API Key
    - Vonage API Key
    - Vonage API Secret
    - Vonage Session Id
    - Websocket Server WS URI (ngrok)

    Environment variables (.env file):
        OPENAI_API_KEY
        VONAGE_API_KEY
        VONAGE_API_SECRET
        VONAGE_SESSION_ID
        VONAGE_AUDIO_WS_URI   (e.g. wss://<your-ngrok-domain>/ws)

Note:
    Start a Vonage Video API session (routed) in your app, and make sure
    VONAGE_SESSION_ID matches that session.

The example focuses on:
- Wiring Vonage Audio Connector → Pipecat pipeline.
- Using OpenAI for STT + LLM + TTS.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from opentok import Client as OpenTokClient  # Vonage Video SDK

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.serializers.vonage import VonageFrameSerializer
from pipecat.services.openai import OpenAILLMService, OpenAISTTService, OpenAITTSService
from pipecat.transports.network.websocket_server import WebsocketServerParams
from pipecat.transports.vonage.audio_connector import VonageAudioConnectorTransport

logger.remove(0)
logger.add(sys.stderr, level="INFO")


SYSTEM_INSTRUCTION = (
    "You are a friendly voice assistant. "
    "The user and you will talk through a phone or browser call. "
    "Keep responses short (1–2 sentences) and easy to speak aloud."
)


async def connect_audio_connector(
    *,
    api_key: str,
    api_secret: str,
    session_id: str,
    ws_uri: str,
    audio_rate: int,
    api_base: str,
) -> None:
    """
    Call the Vonage Audio Connector "connect" API using the OpenTok SDK:

    POST /v2/project/{apiKey}/connect
    {
      \"sessionId\": \"...\",
      \"token\": \"...\",
      \"websocket\": { \"uri\": \"wss://...\", ... }
    }
    """
    logger.info(
        "Connecting Vonage Audio Connector to WebSocket: "
        f"session_id={session_id}, ws_uri={ws_uri}, audioRate={audio_rate}"
    )

    # The OpenTok SDK is synchronous, so run it in a thread.
    def _call_connect() -> object:
        try:
            ot = OpenTokClient(api_key, api_secret, api_url=api_base)
        except TypeError:
            # Older SDKs may not accept api_url
            ot = OpenTokClient(api_key, api_secret)

        token = ot.generate_token(session_id)

        ws_opts = {
            "uri": ws_uri,
            "audioRate": audio_rate,
            "bidirectional": True,
        }

        resp = ot.connect_audio_to_websocket(session_id, token, ws_opts)
        logger.info(f"Audio Connector connect() response (repr): {resp!r}")
        return resp

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _call_connect)
    except Exception as exc:
        logger.error(f"Failed to connect Vonage Audio Connector: {exc}")
        raise


async def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Vonage Audio Connector + OpenAI foundational example"
    )
    parser.add_argument(
        "--host", default=os.getenv("VONAGE_WS_HOST", "0.0.0.0"), help="WebSocket bind host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("VONAGE_WS_PORT", "8005")),
        help="WebSocket bind port",
    )
    args = parser.parse_args()

    host = args.host
    port = args.port

    # --- OpenAI services -----------------------------------------------------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY is not set. Please set it in your .env.")
        sys.exit(1)

    stt = OpenAISTTService(
        api_key=openai_api_key,
        model="gpt-4o-transcribe",
        prompt="You will hear a human speaking conversational English.",
    )

    tts = OpenAITTSService(
        api_key=openai_api_key,
        voice="alloy",  # any supported OpenAI voice
        instructions="Ignore literal '\\n' characters when speaking.",
    )

    llm = OpenAILLMService(api_key=openai_api_key)

    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # --- Vonage / Audio Connector config ------------------------------------
    vonage_api_key = os.getenv("VONAGE_API_KEY")
    vonage_api_secret = os.getenv("VONAGE_API_SECRET")
    vonage_session_id = os.getenv("VONAGE_SESSION_ID")

    if not (vonage_api_key and vonage_api_secret and vonage_session_id):
        logger.error(
            "VONAGE_API_KEY, VONAGE_API_SECRET, and VONAGE_SESSION_ID "
            "must be set in .env for this example."
        )
        sys.exit(1)

    api_base = os.getenv("OPENTOK_API_URL", "https://api.opentok.com")

    # Where the Audio Connector will connect:
    ws_uri = os.getenv("VONAGE_AUDIO_WS_URI")
    if not ws_uri:
        # Expose a public wss:// URL (e.g. ngrok or your own domain).
        logger.error(
            "VONAGE_AUDIO_WS_URI not set "
            "please set this environment variable to a public wss://URL (e.g. ngrok)."
        )
        sys.exit(1)

    audio_rate = int(os.getenv("VONAGE_AUDIO_RATE", "16000"))

    # --- Serializer & transport ---------------------------------------------
    serializer = VonageFrameSerializer(
        VonageFrameSerializer.InputParams(
            auto_hang_up=False,
            send_clear_audio_event=True,
        )
    )

    transport = VonageAudioConnectorTransport(
        host=host,
        port=port,
        params=WebsocketServerParams(
            serializer=serializer,
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_analyzer=SileroVADAnalyzer(),
            session_timeout=60 * 5,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=24_000,
            enable_metrics=False,
            enable_usage_metrics=False,
        ),
    )

    # --- Event handlers ------------------------------------------------------

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("Vonage Audio Connector WebSocket client connected.")
        # Optional: send a small intro prompt to prime the LLM
        messages.append(
            {"role": "system", "content": "Please briefly introduce yourself to the caller."}
        )
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("Vonage Audio Connector WebSocket client disconnected.")
        await task.cancel()

    @transport.event_handler("on_websocket_ready")
    async def on_websocket_ready(_client):
        """
        Called when the WebSocket server is ready to accept incoming connections.

        We use this to trigger the Audio Connector "connect" call from the same file,
        so this foundational example remains single-file and self-contained.
        """
        logger.info("WebSocket server ready – calling Audio Connector connect()")
        await connect_audio_connector(
            api_key=vonage_api_key,
            api_secret=vonage_api_secret,
            session_id=vonage_session_id,
            ws_uri=ws_uri,
            audio_rate=audio_rate,
            api_base=api_base,
        )

    # --- Run -----------------------------------------------------------------
    runner = PipelineRunner()
    logger.info(f"Starting Vonage Audio Connector example on ws://{host}:{port}")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
