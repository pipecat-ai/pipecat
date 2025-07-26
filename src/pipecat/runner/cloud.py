#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Cloud-compatible development server for running Pipecat bots.

This module provides a FastAPI-based development server that can run bots
structured for Pipecat Cloud deployment. It supports multiple transport types
and handles room/token management automatically.

All bots must implement a `bot(session_args)` async function as the entry point.
The server automatically discovers and executes this function when connections
are established.

Bot function signature::

    async def bot(session_args):
        # session_args contains transport-specific connection information

        # For Daily: session_args.room_url, session_args.token, session_args.body
        # For WebRTC: session_args.webrtc_connection
        # For Telephony: session_args.websocket

        # Create transport based on session_args attributes
        if hasattr(session_args, 'room_url'):
            # Daily transport setup
            from pipecat.transports.services.daily import DailyTransport, DailyParams
            transport = DailyTransport(session_args.room_url, session_args.token, "Bot", DailyParams(...))
        elif hasattr(session_args, 'webrtc_connection'):
            # WebRTC transport setup
            from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
            from pipecat.transports.base_transport import TransportParams
            transport = SmallWebRTCTransport(TransportParams(...), webrtc_connection=session_args.webrtc_connection)

        # Run your bot logic
        await run_bot_logic(transport)

Supported transports:

- Daily - Creates rooms and tokens, runs bot as participant
- WebRTC - Provides local WebRTC interface with prebuilt UI
- Telephony - Handles webhook and WebSocket connections for Twilio, Telnyx, Plivo

Example::

    async def bot(session_args):
        # Detect transport type from session_args
        if hasattr(session_args, "room_url"):
            # Daily
            transport = create_daily_transport(session_args)
        elif hasattr(session_args, "webrtc_connection"):
            # WebRTC
            transport = create_webrtc_transport(session_args)

        # Your bot implementation
        await run_pipeline(transport)

    if __name__ == "__main__":
        from pipecat.runner.cloud import main
        main()

Then run: `python bot.py -t daily` or `python bot.py -t webrtc`
"""

import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger

from pipecat.runner.utils import setup_websocket_routes

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
        import bot

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


async def _run_telephony_bot(transport_type: str, websocket, call_info):
    """Run a bot for telephony transports."""
    bot_module = _get_bot_module()

    class WebSocketSessionArgs:
        def __init__(self, transport_type, websocket, call_info):
            self.transport_type = transport_type
            self.websocket = websocket
            self.call_info = call_info
            self.body = {}
            self.handle_sigint = False

    session_args = WebSocketSessionArgs(transport_type, websocket, call_info)
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

    # Add transport-specific routes
    if transport_type == "webrtc":
        # WebRTC setup
        try:
            from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

            from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection
        except ImportError as e:
            logger.error(f"WebRTC transport dependencies not installed.")
            return app

        # Store connections by pc_id
        pcs_map: Dict[str, SmallWebRTCConnection] = {}

        # Mount the frontend at /
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

                # Run bot directly
                bot_module = _get_bot_module()

                class WebRTCSessionArgs:
                    def __init__(self, webrtc_connection):
                        self.transport_type = "webrtc"
                        self.webrtc_connection = webrtc_connection
                        self.body = {}
                        self.handle_sigint = False

                session_args = WebRTCSessionArgs(pipecat_connection)
                background_tasks.add_task(bot_module.bot, session_args)

            answer = pipecat_connection.get_answer()
            pcs_map[answer["pc_id"]] = pipecat_connection
            return answer

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Manage FastAPI application lifecycle and cleanup connections."""
            yield  # Run app
            coros = [pc.disconnect() for pc in pcs_map.values()]
            await asyncio.gather(*coros)
            pcs_map.clear()

        app.router.lifespan_context = lifespan

    elif transport_type in ["twilio", "telnyx", "plivo"]:
        # Create a wrapper function for telephony
        async def telephony_runner(transport_type_inner: str, **kwargs):
            if "websocket" in kwargs and "call_info" in kwargs:
                await _run_telephony_bot(
                    transport_type_inner, kwargs["websocket"], kwargs["call_info"]
                )

        setup_websocket_routes(app, telephony_runner, transport_type, proxy)

    # Add general routes
    @app.get("/")
    async def start_agent():
        """Launch a bot and redirect appropriately."""
        print(f"Starting bot with {transport_type} transport")

        if transport_type == "daily":
            # Create Daily room and start bot
            import aiohttp

            from pipecat.runner.daily import configure

            async with aiohttp.ClientSession() as session:
                room_url, token = await configure(session)

                # Start the bot in the background to join the room
                bot_module = _get_bot_module()

                class DailySessionArgs:
                    def __init__(self, room_url, token):
                        self.room_url = room_url
                        self.token = token
                        self.body = {}
                        self.handle_sigint = False

                session_args = DailySessionArgs(room_url, token)
                asyncio.create_task(bot_module.bot(session_args))
                return RedirectResponse(room_url)

        elif transport_type == "webrtc":
            return RedirectResponse("/client/")
        else:
            return {"status": f"Bot started with {transport_type}"}

    @app.post("/connect")
    async def rtvi_connect():
        """Launch a bot and return connection info for RTVI clients."""
        print(f"Starting bot with {transport_type} transport")

        if transport_type == "daily":
            import aiohttp

            from pipecat.runner.daily import configure

            async with aiohttp.ClientSession() as session:
                room_url, token = await configure(session)

                # Start the bot in the background
                bot_module = _get_bot_module()

                class DailySessionArgs:
                    def __init__(self, room_url, token):
                        self.room_url = room_url
                        self.token = token
                        self.body = {}
                        self.handle_sigint = False

                session_args = DailySessionArgs(room_url, token)
                asyncio.create_task(bot_module.bot(session_args))
                return {"transport": "daily", "room_url": room_url, "token": token}

        elif transport_type == "webrtc":
            return {"transport": "webrtc", "client_url": "/client/"}
        else:
            return {
                "error": f"RTVI connect not supported for {transport_type} transport. Use Daily or WebRTC."
            }

    return app


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

    args = parser.parse_args()

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
