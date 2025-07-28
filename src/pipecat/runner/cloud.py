#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Cloud-compatible development server for running Pipecat bots.

This module provides a FastAPI-based development server that can run bots
structured for Pipecat Cloud deployment. The runner enables you to run Pipecat
bots locally or deployed without requiring any code changes. It supports
multiple transport types and handles room/token management automatically.

It requires the `pipecatcloud` package for proper session argument types.

Install with::

    pip install pipecat-ai[pipecatcloud]

All bots must implement a `bot(session_args)` async function as the entry point.
The server automatically discovers and executes this function when connections
are established.

Single transport example::

    async def bot(session_args):
        if isinstance(session_args, DailySessionArguments):
            transport = DailyTransport(
                session_args.room_url,
                session_args.token,
                "Bot",
                DailyParams(...)
            )
        # Your bot logic here
        await run_pipeline(transport)

    if __name__ == "__main__":
        from pipecat.runner.cloud import main
        main()

Multiple transport example::

    async def bot(session_args):
        # Type-safe transport detection
        if isinstance(session_args, DailySessionArguments):
            transport = setup_daily_transport(session_args)  # Your application code
        elif isinstance(session_args, SmallWebRTCSessionArguments):
            transport = setup_webrtc_transport(session_args)  # Your application code
        elif isinstance(session_args, WebSocketSessionArguments):
            transport = setup_telephony_transport(session_args)  # Your application code

        # Your bot implementation
        await run_pipeline(transport)

Supported transports:

- Daily - Creates rooms and tokens, runs bot as participant
- WebRTC - Provides local WebRTC interface with prebuilt UI
- Telephony - Handles webhook and WebSocket connections for Twilio, Telnyx, Plivo

To run locally:

- Daily: `python bot.py -t daily`
- WebRTC: `python bot.py -t webrtc`
- Telephony: `python bot.py -t twilio -x your_username.ngrok.io`
"""

import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from loguru import logger

try:
    from pipecatcloud.agent import DailySessionArguments, WebSocketSessionArguments
except ImportError:
    raise ImportError(
        "pipecatcloud package is required for cloud-compatible bots. "
        "Install with: pip install pipecat-ai[pipecatcloud]"
    )


# Define WebRTC type locally until it's added to pipecatcloud
@dataclass
class SmallWebRTCSessionArguments:
    """Small WebRTC session arguments for local development.

    This will be replaced by pipecatcloud.agent.SmallWebRTCSessionArguments
    when WebRTC support is added to Pipecat Cloud.
    """

    webrtc_connection: Any
    session_id: Optional[str] = None


load_dotenv(override=True)
os.environ["LOCAL_RUN"] = "1"


def _get_bot_module():
    """Get the bot module from the calling script."""
    import importlib.util

    # Get the main module (the file that was executed)
    main_module = sys.modules["__main__"]

    # Check if it has a bot function
    if hasattr(main_module, "bot"):
        return main_module

    # Try to import 'bot' module from current directory
    try:
        import bot  # type: ignore[import-untyped]

        return bot
    except ImportError:
        pass

    # Look for any .py file in current directory that has a bot function
    cwd = os.getcwd()
    for filename in os.listdir(cwd):
        if filename.endswith(".py") and filename != "server.py":
            try:
                module_name = filename[:-3]  # Remove .py extension
                spec = importlib.util.spec_from_file_location(
                    module_name, os.path.join(cwd, filename)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "bot"):
                    return module
            except Exception:
                continue

    raise ImportError(
        "Could not find 'bot' function. Make sure your bot file has a 'bot' function."
    )


async def _run_telephony_bot(websocket: WebSocket):
    """Run a bot for telephony transports."""
    bot_module = _get_bot_module()

    # Just pass the WebSocket - let the bot handle parsing
    session_args = WebSocketSessionArguments(websocket=websocket, session_id=None)

    await bot_module.bot(session_args)


def _create_server_app(transport_type: str, host: str = "localhost", proxy: str = None):
    """Create FastAPI app with transport-specific routes."""
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Set up transport-specific routes
    if transport_type == "webrtc":
        _setup_webrtc_routes(app)
    elif transport_type == "daily":
        _setup_daily_routes(app)
    elif transport_type in ["twilio", "telnyx", "plivo"]:
        _setup_telephony_routes(app, transport_type, proxy)
    else:
        logger.warning(f"Unknown transport type: {transport_type}")

    return app


def _setup_webrtc_routes(app: FastAPI):
    """Set up WebRTC-specific routes."""
    try:
        from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

        from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection
    except ImportError as e:
        logger.error(f"WebRTC transport dependencies not installed.")
        return

    # Store connections by pc_id
    pcs_map: Dict[str, SmallWebRTCConnection] = {}

    # Mount the frontend
    app.mount("/client", SmallWebRTCPrebuiltUI)

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        """Redirect root requests to client interface."""
        return RedirectResponse(url="/client/")

    @app.post("/api/offer")
    async def offer(request: dict, background_tasks: BackgroundTasks):
        """Handle WebRTC offer requests and manage peer connections."""
        pc_id = request.get("pc_id")

        if pc_id and pc_id in pcs_map:
            pipecat_connection = pcs_map[pc_id]
            logger.info(f"Reusing existing connection for pc_id: {pc_id}")
            await pipecat_connection.renegotiate(
                sdp=request["sdp"],
                type=request["type"],
                restart_pc=request.get("restart_pc", False),
            )
        else:
            pipecat_connection = SmallWebRTCConnection()
            await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

            @pipecat_connection.event_handler("closed")
            async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
                """Handle WebRTC connection closure and cleanup."""
                logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
                pcs_map.pop(webrtc_connection.pc_id, None)

            bot_module = _get_bot_module()
            session_args = SmallWebRTCSessionArguments(
                webrtc_connection=pipecat_connection,
                session_id=None,
            )
            background_tasks.add_task(bot_module.bot, session_args)

        answer = pipecat_connection.get_answer()
        pcs_map[answer["pc_id"]] = pipecat_connection
        return answer

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage FastAPI application lifecycle and cleanup connections."""
        yield
        coros = [pc.disconnect() for pc in pcs_map.values()]
        await asyncio.gather(*coros)
        pcs_map.clear()

    app.router.lifespan_context = lifespan


def _setup_daily_routes(app: FastAPI):
    """Set up Daily-specific routes."""

    @app.get("/")
    async def start_agent():
        """Launch a Daily bot and redirect to room."""
        print("Starting bot with Daily transport")

        import aiohttp

        from pipecat.runner.daily import configure

        async with aiohttp.ClientSession() as session:
            room_url, token = await configure(session)

            # Start the bot in the background
            bot_module = _get_bot_module()
            session_args = DailySessionArguments(
                room_url=room_url, token=token, body={}, session_id=None
            )
            asyncio.create_task(bot_module.bot(session_args))
            return RedirectResponse(room_url)

    @app.post("/connect")
    async def rtvi_connect():
        """Launch a Daily bot and return connection info for RTVI clients."""
        print("Starting bot with Daily transport")

        import aiohttp

        from pipecat.runner.daily import configure

        async with aiohttp.ClientSession() as session:
            room_url, token = await configure(session)

            # Start the bot in the background
            bot_module = _get_bot_module()
            session_args = DailySessionArguments(
                room_url=room_url, token=token, body={}, session_id=None
            )
            asyncio.create_task(bot_module.bot(session_args))
            return {"room_url": room_url, "token": token}


def _setup_telephony_routes(app: FastAPI, transport_type: str, proxy: str):
    """Set up telephony-specific routes."""
    # XML response templates
    XML_TEMPLATES = {
        "twilio": f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{proxy}/ws"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>""",
        "telnyx": f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{proxy}/ws" bidirectionalMode="rtp"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>""",
        "plivo": f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">wss://{proxy}/ws</Stream>
</Response>""",
    }

    @app.post("/")
    async def start_call():
        """Handle telephony webhook and return XML response."""
        logger.debug(f"POST {transport_type.upper()} XML")
        xml_content = XML_TEMPLATES.get(transport_type, "<Response></Response>")
        return HTMLResponse(content=xml_content, media_type="application/xml")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Handle WebSocket connections for telephony."""
        await websocket.accept()
        logger.debug("WebSocket connection accepted")
        await _run_telephony_bot(websocket)

    @app.get("/")
    async def start_agent():
        """Simple status endpoint for telephony transports."""
        return {"status": f"Bot started with {transport_type}"}


def main():
    """Start the cloud-compatible development server.

    Parses command-line arguments and starts a FastAPI server configured
    for the specified transport type. The server will discover and run
    any bot() function found in the current directory.

    Command-line arguments:
        --host: Server host address (default: localhost)
        --port: Server port (default: 7860)
        -t/--transport: Transport type (daily, webrtc, twilio, telnyx, plivo)
        -x/--proxy: Public proxy hostname for telephony webhooks
        -v/--verbose: Increase logging verbosity

    The bot file must contain a `bot(session_args)` function as the entry point.
    """
    parser = argparse.ArgumentParser(description="Pipecat Cloud-Compatible Development Server")
    parser.add_argument("--host", type=str, default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument(
        "-t",
        "--transport",
        type=str,
        choices=["daily", "webrtc", "twilio", "telnyx", "plivo"],
        default="webrtc",
        help="Transport type",
    )
    parser.add_argument("--proxy", "-x", help="Public proxy host name")
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Increase logging verbosity"
    )

    args = parser.parse_args()

    # Log level
    logger.remove()
    logger.add(sys.stderr, level="TRACE" if args.verbose else "DEBUG")

    # Print startup message
    if args.transport == "webrtc":
        print()
        print(f"🚀 WebRTC server starting at http://{args.host}:{args.port}/client")
        print(f"   Open this URL in your browser to connect!")
        print()
    elif args.transport == "daily":
        print()
        print(f"🚀 Daily server starting at http://{args.host}:{args.port}")
        print(f"   Open this URL in your browser to start a session!")
        print()

    # Create the app with transport-specific setup
    app = _create_server_app(args.transport, args.host, args.proxy)

    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
