#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Direct execution runner for local-only examples."""

import argparse
import asyncio
import os
import sys
from typing import Callable, Dict, Mapping, Optional

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import RedirectResponse
from loguru import logger

# Import the common transport utility functions
from .transport_utilities import setup_websocket_routes

load_dotenv(override=True)


def run_webrtc(
    run: Callable, args: argparse.Namespace, transport_params: Mapping[str, Callable] = {}
):
    """Run using WebRTC transport with FastAPI server."""
    try:
        from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

        from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
        from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection
    except ImportError as e:
        logger.error(f"WebRTC transport dependencies not installed.")
        logger.debug(f"Import error: {e}")
        return

    logger.info("Running with SmallWebRTCTransport...")

    app = FastAPI()

    # Store connections by pc_id (like the working version)
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

            # Run function with SmallWebRTC transport arguments (like working version)
            params = transport_params[args.transport]()
            transport = SmallWebRTCTransport(params=params, webrtc_connection=pipecat_connection)

            class MockArgs:
                def __init__(self):
                    self.transport = "webrtc"

            background_tasks.add_task(run, transport, MockArgs(), False)

        answer = pipecat_connection.get_answer()

        if args.esp32 and args.host:
            from .transport_utilities import smallwebrtc_sdp_munging

            answer["sdp"] = smallwebrtc_sdp_munging(answer["sdp"], args.host)

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
    uvicorn.run(app, host=args.host, port=args.port)


def run_twilio(
    run: Callable, args: argparse.Namespace, transport_params: Mapping[str, Callable] = {}
):
    """Run using Twilio transport with FastAPI WebSocket server."""
    try:
        from pipecat.serializers.twilio import TwilioFrameSerializer
        from pipecat.transports.network.fastapi_websocket import (
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )
    except ImportError as e:
        logger.error(f"Twilio transport dependencies not installed.")
        logger.debug(f"Import error: {e}")
        return

    logger.info("Running with FastAPIWebsocketTransport (Twilio)...")

    app = FastAPI()

    # Twilio WebSocket handler
    async def twilio_runner(transport_type: str, **kwargs):
        if "websocket" in kwargs and "call_info" in kwargs:
            call_info = kwargs["call_info"]

            params = transport_params["twilio"]()
            params.add_wav_header = False
            params.serializer = TwilioFrameSerializer(
                stream_sid=call_info["stream_sid"],
                call_sid=call_info["call_sid"],
                account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
                auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            )

            transport = FastAPIWebsocketTransport(websocket=kwargs["websocket"], params=params)

            class MockArgs:
                def __init__(self):
                    self.transport = "twilio"

            await run(transport, MockArgs(), False)

    setup_websocket_routes(app, twilio_runner, "twilio", args.proxy)
    uvicorn.run(app, host=args.host, port=args.port)


def run_telnyx(
    run: Callable, args: argparse.Namespace, transport_params: Mapping[str, Callable] = {}
):
    """Run using Telnyx transport with FastAPI WebSocket server."""
    try:
        from pipecat.serializers.telnyx import TelnyxFrameSerializer
        from pipecat.transports.network.fastapi_websocket import (
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )
    except ImportError as e:
        logger.error(f"Telnyx transport dependencies not installed.")
        logger.debug(f"Import error: {e}")
        return

    logger.info("Running with FastAPIWebsocketTransport (Telnyx)...")

    app = FastAPI()

    # Telnyx WebSocket handler
    async def telnyx_runner(transport_type: str, **kwargs):
        if "websocket" in kwargs and "call_info" in kwargs:
            call_info = kwargs["call_info"]

            params = transport_params["telnyx"]()
            params.add_wav_header = False
            params.serializer = TelnyxFrameSerializer(
                stream_id=call_info["stream_id"],
                call_control_id=call_info["call_control_id"],
                outbound_encoding=call_info["outbound_encoding"],
                inbound_encoding="PCMU",
            )

            transport = FastAPIWebsocketTransport(websocket=kwargs["websocket"], params=params)

            class MockArgs:
                def __init__(self):
                    self.transport = "telnyx"

            await run(transport, MockArgs(), False)

    setup_websocket_routes(app, telnyx_runner, "telnyx", args.proxy)
    uvicorn.run(app, host=args.host, port=args.port)


def run_plivo(
    run: Callable, args: argparse.Namespace, transport_params: Mapping[str, Callable] = {}
):
    """Run using Plivo transport with FastAPI WebSocket server."""
    try:
        from pipecat.serializers.plivo import PlivoFrameSerializer
        from pipecat.transports.network.fastapi_websocket import (
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )
    except ImportError as e:
        logger.error(f"Plivo transport dependencies not installed.")
        logger.debug(f"Import error: {e}")
        return

    logger.info("Running with FastAPIWebsocketTransport (Plivo)...")

    app = FastAPI()

    # Plivo WebSocket handler
    async def plivo_runner(transport_type: str, **kwargs):
        if "websocket" in kwargs and "call_info" in kwargs:
            call_info = kwargs["call_info"]

            params = transport_params["plivo"]()
            params.add_wav_header = False
            params.serializer = PlivoFrameSerializer(
                stream_id=call_info["stream_id"],
                call_id=call_info["call_id"],
            )

            transport = FastAPIWebsocketTransport(websocket=kwargs["websocket"], params=params)

            class MockArgs:
                def __init__(self):
                    self.transport = "plivo"

            await run(transport, MockArgs(), False)

    setup_websocket_routes(app, plivo_runner, "plivo", args.proxy)
    uvicorn.run(app, host=args.host, port=args.port)


def run_daily(
    run: Callable, args: argparse.Namespace, transport_params: Mapping[str, Callable] = {}
):
    """Run using Daily.co transport."""
    try:
        from pipecat.runner.daily import configure
        from pipecat.transports.services.daily import DailyParams, DailyTransport
    except ImportError as e:
        logger.error(f"Daily transport dependencies not installed.")
        logger.debug(f"Import error: {e}")
        return

    logger.info("Running with DailyTransport...")

    async def run_daily_impl():
        async with aiohttp.ClientSession() as session:
            (room_url, token) = await configure(session)
            params: DailyParams = transport_params[args.transport]()
            transport = DailyTransport(room_url, token, "Pipecat", params=params)
            await run(transport, args, True)

    asyncio.run(run_daily_impl())


def run_livekit(
    run: Callable, args: argparse.Namespace, transport_params: Mapping[str, Callable] = {}
):
    """Run using LiveKit transport."""
    try:
        from pipecat.runner.livekit import configure
        from pipecat.transports.services.livekit import LiveKitParams, LiveKitTransport
    except ImportError as e:
        logger.error(f"LiveKit transport dependencies not installed.")
        logger.debug(f"Import error: {e}")
        return

    logger.info("Running with LiveKitTransport...")

    async def run_livekit_impl():
        (url, token, room_name) = await configure()
        params: LiveKitParams = transport_params[args.transport]()
        transport = LiveKitTransport(url=url, token=token, room_name=room_name, params=params)
        await run(transport, args, True)

    asyncio.run(run_livekit_impl())


def run_main(
    run: Callable, args: argparse.Namespace, transport_params: Mapping[str, Callable] = {}
):
    """Run the application with the specified transport type."""
    if args.transport not in transport_params:
        logger.error(f"Transport '{args.transport}' not supported by this application.")
        return

    match args.transport:
        case "daily":
            run_daily(run, args, transport_params)
        case "livekit":
            run_livekit(run, args, transport_params)
        case "plivo":
            run_plivo(run, args, transport_params)
        case "telnyx":
            run_telnyx(run, args, transport_params)
        case "twilio":
            run_twilio(run, args, transport_params)
        case "webrtc":
            run_webrtc(run, args, transport_params)


def main(
    run: Callable,
    *,
    parser: Optional[argparse.ArgumentParser] = None,
    transport_params: Mapping[str, Callable] = {},
):
    """Main entry point for running Pipecat applications with transport selection."""
    if not parser:
        parser = argparse.ArgumentParser(description="Pipecat Bot Runner")

    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    parser.add_argument(
        "--transport",
        "-t",
        type=str,
        choices=list(transport_params.keys())
        if transport_params
        else ["daily", "livekit", "plivo", "telnyx", "twilio", "webrtc"],
        default="webrtc",
        help="The transport this application should use",
    )
    parser.add_argument(
        "--proxy", "-x", help="A public proxy host name (no protocol, e.g. proxy.example.com)"
    )
    parser.add_argument(
        "--esp32", action="store_true", default=False, help="Perform SDP munging for the ESP32"
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()

    if args.esp32 and args.host == "localhost":
        logger.error("For ESP32, you need to specify `--host IP` so we can do SDP munging.")
        return

    # Log level
    logger.remove(0)
    logger.add(sys.stderr, level="TRACE" if args.verbose else "DEBUG")

    run_main(run, args, transport_params)
