#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cloud-compatible development server - simplified without subprocesses."""

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

# Import the common transport utility functions
from .transport_utilities import setup_websocket_routes

load_dotenv(override=True)
os.environ["LOCAL_RUN"] = "1"


def get_bot_module():
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


async def run_bot_directly(transport_type: str, **kwargs):
    """Run a bot directly in the same process - no subprocess needed."""
    if transport_type == "webrtc":
        if "webrtc_connection" in kwargs:
            # Direct WebRTC connection
            bot_module = get_bot_module()

            class WebRTCSessionArgs:
                def __init__(self, webrtc_connection):
                    self.transport_type = "webrtc"
                    self.webrtc_connection = webrtc_connection
                    self.body = {}
                    self.handle_sigint = False

            session_args = WebRTCSessionArgs(kwargs["webrtc_connection"])
            await bot_module.bot(session_args)

    elif transport_type in ["twilio", "telnyx", "plivo"]:
        if "websocket" in kwargs:
            # Direct WebSocket connection
            bot_module = get_bot_module()

            class WebSocketSessionArgs:
                def __init__(self, transport_type, websocket, call_info):
                    self.transport_type = transport_type
                    self.websocket = websocket
                    self.call_info = call_info
                    self.body = {}
                    self.handle_sigint = False

            session_args = WebSocketSessionArgs(
                transport_type, kwargs["websocket"], kwargs["call_info"]
            )
            await bot_module.bot(session_args)


def create_server_app(transport_type: str, host: str = "0.0.0.0", proxy: str = None):
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
        # Direct WebRTC setup
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
                background_tasks.add_task(
                    run_bot_directly, "webrtc", webrtc_connection=pipecat_connection
                )

            answer = pipecat_connection.get_answer()

            if host and host != "0.0.0.0":
                from .transport_utilities import smallwebrtc_sdp_munging

                answer["sdp"] = smallwebrtc_sdp_munging(answer["sdp"], host)

            # Updating the peer connection inside the map
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
        setup_websocket_routes(app, run_bot_directly, transport_type, proxy)

    # Add general routes
    @app.get("/")
    async def start_agent():
        """Launch a bot and redirect appropriately."""
        print(f"Starting bot with {transport_type} transport")

        if transport_type == "daily":
            # Create Daily room and start bot
            import aiohttp

            from .daily_runner import configure

            async with aiohttp.ClientSession() as session:
                room_url, token = await configure(session)

                # Start the bot in the background to join the room
                bot_module = get_bot_module()

                class DailySessionArgs:
                    def __init__(self, room_url, token):
                        self.room_url = room_url
                        self.token = token
                        self.body = {}
                        self.handle_sigint = False

                session_args = DailySessionArgs(room_url, token)

                # Run bot in background task
                asyncio.create_task(bot_module.bot(session_args))

                # Redirect user to the room
                return RedirectResponse(room_url)

        elif transport_type == "livekit":
            # Create LiveKit room and start bot
            from .livekit_runner import configure

            url, token, room_name = await configure()

            # Start the bot in the background to join the room
            bot_module = get_bot_module()

            class LiveKitSessionArgs:
                def __init__(self, url, token, room_name):
                    self.url = url
                    self.token = token
                    self.room_name = room_name
                    self.body = {}
                    self.handle_sigint = False

            session_args = LiveKitSessionArgs(url, token, room_name)

            # Run bot in background task
            asyncio.create_task(bot_module.bot(session_args))

            # Redirect user to the room
            return RedirectResponse(url)

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

            from .daily_runner import configure

            async with aiohttp.ClientSession() as session:
                room_url, token = await configure(session)

                # Start the bot in the background
                bot_module = get_bot_module()

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
            # RTVI only supports Daily and WebRTC
            return {
                "error": f"RTVI connect not supported for {transport_type} transport. Use Daily or WebRTC."
            }

    return app


def main():
    """Main entry point for cloud-compatible server."""
    parser = argparse.ArgumentParser(description="Pipecat Cloud-Compatible Development Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument(
        "-t",
        "--transport",
        type=str,
        choices=["daily", "livekit", "webrtc", "twilio", "telnyx", "plivo"],
        default="webrtc",
        help="Transport type",
    )
    parser.add_argument("--proxy", "-x", help="Public proxy host name")

    args = parser.parse_args()

    # Create the app with transport-specific setup
    app = create_server_app(args.transport, args.host, args.proxy)

    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
