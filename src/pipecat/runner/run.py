#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat runner with support for multiple transport types.

This module provides a unified interface for running Pipecat applications
across different transport types including Daily.co, WebRTC, and Twilio. It
handles setup, configuration, and lifecycle management for each transport type.

Example usage:
    DailyTransport::

        python bot.py --transport daily

    LiveKitTransport::

        python bot.py --transport livekit

    Plivo::

        python bot.py --transport plivo --proxy username.ngrok.io
        # Note: Concurrently, run an ngrok tunnel to your local server:
        # ngrok http 7860

    SmallWebRTCTransport::

        python bot.py --transport webrtc

    Telnyx::

        python bot.py --transport telnyx --proxy username.ngrok.io
        # Note: Concurrently, run an ngrok tunnel to your local server:
        # ngrok http 7860

    Twilio::

        python bot.py --transport twilio --proxy username.ngrok.io
        # Note: Concurrently, run an ngrok tunnel to your local server:
        # ngrok http 7860
"""

import argparse
import asyncio
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Mapping, Optional

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from loguru import logger

from pipecat.transports.base_transport import BaseTransport, TransportParams

# Load environment variables
load_dotenv(override=True)


def get_install_command(transport: str) -> str:
    """Get the pip install command for a specific transport.

    Args:
        transport: The transport name.

    Returns:
        The pip install command string.
    """
    install_map = {
        "daily": "pip install pipecat-ai[daily]",
        "livekit": "pip install pipecat-ai[livekit]",
        "webrtc": "pip install pipecat-ai[webrtc]",
        "twilio": "pip install pipecat-ai[websocket]",
        "telnyx": "pip install pipecat-ai[websocket]",
        "plivo": "pip install pipecat-ai[websocket]",
    }
    return install_map.get(transport, f"pip install pipecat-ai[{transport}]")


def get_transport_client_id(transport: BaseTransport, client: Any) -> str:
    """Get client identifier from transport-specific client object.

    Args:
        transport: The transport instance.
        client: Transport-specific client object.

    Returns:
        Client identifier string, empty if transport not supported.
    """
    # Import conditionally to avoid dependency issues
    try:
        from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

        if isinstance(transport, SmallWebRTCTransport):
            return client.pc_id
    except ImportError:
        pass

    try:
        from pipecat.transports.services.daily import DailyTransport

        if isinstance(transport, DailyTransport):
            return client["id"]
    except ImportError:
        pass

    logger.warning(f"Unable to get client id from unsupported transport {type(transport)}")
    return ""


async def maybe_capture_participant_camera(
    transport: BaseTransport, client: Any, framerate: int = 0
):
    """Capture participant camera video if transport supports it.

    Args:
        transport: The transport instance.
        client: Transport-specific client object.
        framerate: Video capture framerate. Defaults to 0 (auto).
    """
    try:
        from pipecat.transports.services.daily import DailyTransport

        if isinstance(transport, DailyTransport):
            await transport.capture_participant_video(
                client["id"], framerate=framerate, video_source="camera"
            )
    except ImportError:
        pass


async def maybe_capture_participant_screen(
    transport: BaseTransport, client: Any, framerate: int = 0
):
    """Capture participant screen video if transport supports it.

    Args:
        transport: The transport instance.
        client: Transport-specific client object.
        framerate: Video capture framerate. Defaults to 0 (auto).
    """
    try:
        from pipecat.transports.services.daily import DailyTransport

        if isinstance(transport, DailyTransport):
            await transport.capture_participant_video(
                client["id"], framerate=framerate, video_source="screenVideo"
            )
    except ImportError:
        pass


def smallwebrtc_sdp_cleanup_ice_candidates(text: str, pattern: str) -> str:
    """Clean up ICE candidates in SDP text for SmallWebRTC.

    Args:
        text: SDP text to clean up.
        pattern: Pattern to match for candidate filtering.

    Returns:
        Cleaned SDP text with filtered ICE candidates.
    """
    result = []
    lines = text.splitlines()
    for line in lines:
        if re.search("a=candidate", line):
            if re.search(pattern, line) and not re.search("raddr", line):
                result.append(line)
        else:
            result.append(line)
    return "\r\n".join(result)


def smallwebrtc_sdp_cleanup_fingerprints(text: str) -> str:
    """Remove unsupported fingerprint algorithms from SDP text.

    Args:
        text: SDP text to clean up.

    Returns:
        SDP text with sha-384 and sha-512 fingerprints removed.
    """
    result = []
    lines = text.splitlines()
    for line in lines:
        if not re.search("sha-384", line) and not re.search("sha-512", line):
            result.append(line)
    return "\r\n".join(result)


def smallwebrtc_sdp_munging(sdp: str, host: str) -> str:
    """Apply SDP modifications for SmallWebRTC compatibility.

    Args:
        sdp: Original SDP string.
        host: Host address for ICE candidate filtering.

    Returns:
        Modified SDP string with fingerprint and ICE candidate cleanup.
    """
    sdp = smallwebrtc_sdp_cleanup_fingerprints(sdp)
    sdp = smallwebrtc_sdp_cleanup_ice_candidates(sdp, host)
    return sdp


def run_daily(
    run: Callable,
    args: argparse.Namespace,
    transport_params: Mapping[str, Callable] = {},
):
    """Run using Daily.co transport.

    Args:
        run: The function to run.
        args: Parsed command-line arguments.
        transport_params: Mapping of transport names to parameter factory functions.
    """
    try:
        from pipecat.runner.daily_runner import configure
        from pipecat.transports.services.daily import DailyParams, DailyTransport
    except ImportError as e:
        logger.error(
            f"Daily transport dependencies not installed. Install with: {get_install_command('daily')}"
        )
        logger.debug(f"Import error: {e}")
        return

    logger.info("Running with DailyTransport...")

    async def run_daily_impl():
        async with aiohttp.ClientSession() as session:
            (room_url, token) = await configure(session)

            # Run function with DailyTransport transport arguments.
            params: DailyParams = transport_params[args.transport]()
            transport = DailyTransport(room_url, token, "Pipecat", params=params)
            await run(transport, args, True)

    asyncio.run(run_daily_impl())


def run_livekit(
    run: Callable,
    args: argparse.Namespace,
    transport_params: Mapping[str, Callable] = {},
):
    """Run using LiveKit transport.

    Args:
        run: The function to run.
        args: Parsed command-line arguments.
        transport_params: Mapping of transport names to parameter factory functions.
    """
    try:
        from pipecat.runner.livekit_runner import configure
        from pipecat.transports.services.livekit import LiveKitParams, LiveKitTransport
    except ImportError as e:
        logger.error(
            f"LiveKit transport dependencies not installed. Install with: {get_install_command('livekit')}"
        )
        logger.debug(f"Import error: {e}")
        return

    logger.info("Running with LiveKitTransport...")

    async def run_livekit_impl():
        (url, token, room_name) = await configure()

        # Run function with LiveKit transport arguments.
        params: LiveKitParams = transport_params[args.transport]()
        transport = LiveKitTransport(url=url, token=token, room_name=room_name, params=params)
        await run(transport, args, True)

    asyncio.run(run_livekit_impl())


def run_webrtc(
    run: Callable,
    args: argparse.Namespace,
    transport_params: Mapping[str, Callable] = {},
):
    """Run using WebRTC transport with FastAPI server.

    Args:
        run: The function to run.
        args: Parsed command-line arguments.
        transport_params: Mapping of transport names to parameter factory functions.
    """
    try:
        from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

        from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
        from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection
    except ImportError as e:
        logger.error(
            f"WebRTC transport dependencies not installed. Install with: {get_install_command('webrtc')}"
        )
        logger.debug(f"Import error: {e}")
        return

    logger.info("Running with SmallWebRTCTransport...")

    app = FastAPI()

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
        """Handle WebRTC offer requests and manage peer connections.

        Args:
            request: WebRTC offer request containing SDP and connection details.
            background_tasks: FastAPI background tasks for running applications.

        Returns:
            WebRTC answer with connection details.
        """
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
                """Handle WebRTC connection closure and cleanup.

                Args:
                    webrtc_connection: The closed WebRTC connection.
                """
                logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
                pcs_map.pop(webrtc_connection.pc_id, None)

            # Run function with SmallWebRTC transport arguments.
            params: TransportParams = transport_params[args.transport]()
            transport = SmallWebRTCTransport(params=params, webrtc_connection=pipecat_connection)
            background_tasks.add_task(run, transport, args, False)

        answer = pipecat_connection.get_answer()

        if args.esp32 and args.host:
            answer["sdp"] = smallwebrtc_sdp_munging(answer["sdp"], args.host)

        # Updating the peer connection inside the map
        pcs_map[answer["pc_id"]] = pipecat_connection

        return answer

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage FastAPI application lifecycle and cleanup connections.

        Args:
            app: The FastAPI application instance.

        Yields:
            Control to the FastAPI application runtime.
        """
        yield  # Run app
        coros = [pc.disconnect() for pc in pcs_map.values()]
        await asyncio.gather(*coros)
        pcs_map.clear()

    uvicorn.run(app, host=args.host, port=args.port)


def run_twilio(
    run: Callable,
    args: argparse.Namespace,
    transport_params: Mapping[str, Callable] = {},
):
    """Run using Twilio transport with FastAPI WebSocket server.

    Args:
        run: The function to run.
        args: Parsed command-line arguments.
        transport_params: Mapping of transport names to parameter factory functions.
    """
    try:
        from pipecat.serializers.twilio import TwilioFrameSerializer
        from pipecat.transports.network.fastapi_websocket import (
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )
    except ImportError as e:
        logger.error(
            f"Twilio transport dependencies not installed. Install with: {get_install_command('twilio')}"
        )
        logger.debug(f"Import error: {e}")
        return

    logger.info("Running with FastAPIWebsocketTransport (Twilio)...")

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for testing
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/")
    async def start_call():
        """Handle Twilio webhook and return TwiML response.

        Returns:
            TwiML XML response directing call to WebSocket stream.
        """
        logger.debug("POST TwiML")

        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{args.proxy}/ws"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>
        """
        return HTMLResponse(content=xml_content, media_type="application/xml")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Handle Twilio WebSocket connections for voice streaming.

        Args:
            websocket: The WebSocket connection from Twilio.
        """
        await websocket.accept()

        logger.debug("WebSocket connection accepted")

        # Reading Twilio data.
        start_data = websocket.iter_text()
        await start_data.__anext__()
        call_data = json.loads(await start_data.__anext__())
        print(call_data, flush=True)
        stream_sid = call_data["start"]["streamSid"]
        call_sid = call_data["start"]["callSid"]

        # Create websocket transport and update params.
        params: FastAPIWebsocketParams = transport_params[args.transport]()
        params.add_wav_header = False
        params.serializer = TwilioFrameSerializer(
            stream_sid=stream_sid,
            call_sid=call_sid,
            account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        )
        transport = FastAPIWebsocketTransport(websocket=websocket, params=params)
        await run(transport, args, False)

    uvicorn.run(app, host=args.host, port=args.port)


def run_telnyx(
    run: Callable,
    args: argparse.Namespace,
    transport_params: Mapping[str, Callable] = {},
):
    """Run using Telnyx transport with FastAPI WebSocket server.

    Args:
        run: The function to run.
        args: Parsed command-line arguments.
        transport_params: Mapping of transport names to parameter factory functions.
    """
    try:
        from pipecat.serializers.telnyx import TelnyxFrameSerializer
        from pipecat.transports.network.fastapi_websocket import (
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )
    except ImportError as e:
        logger.error(
            f"Telnyx transport dependencies not installed. Install with: {get_install_command('telnyx')}"
        )
        logger.debug(f"Import error: {e}")
        return

    logger.info("Running with FastAPIWebsocketTransport (Telnyx)...")

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for testing
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/")
    async def start_call():
        """Handle Telnyx webhook and return TeXML response.

        Returns:
            TeXML XML response directing call to WebSocket stream.
        """
        logger.debug("POST TeXML")

        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{args.proxy}/ws" bidirectionalMode="rtp"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>
        """
        return HTMLResponse(content=xml_content, media_type="application/xml")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Handle Telnyx WebSocket connections for voice streaming.

        Args:
            websocket: The WebSocket connection from Telnyx.
        """
        await websocket.accept()

        logger.debug("WebSocket connection accepted")

        # Reading Telnyx data.
        start_data = websocket.iter_text()
        await start_data.__anext__()
        call_data = json.loads(await start_data.__anext__())
        print(call_data, flush=True)
        stream_id = call_data["stream_id"]
        call_control_id = call_data["start"]["call_control_id"]
        outbound_encoding = call_data["start"]["media_format"]["encoding"]

        # Create websocket transport and update params.
        params: FastAPIWebsocketParams = transport_params[args.transport]()
        params.add_wav_header = False
        params.serializer = TelnyxFrameSerializer(
            stream_id=stream_id,
            call_control_id=call_control_id,
            outbound_encoding=outbound_encoding,
            inbound_encoding="PCMU",  # You might want to make this configurable
        )
        transport = FastAPIWebsocketTransport(websocket=websocket, params=params)
        await run(transport, args, False)

    uvicorn.run(app, host=args.host, port=args.port)


def run_plivo(
    run: Callable,
    args: argparse.Namespace,
    transport_params: Mapping[str, Callable] = {},
):
    """Run using Plivo transport with FastAPI WebSocket server.

    Args:
        run: The function to run.
        args: Parsed command-line arguments.
        transport_params: Mapping of transport names to parameter factory functions.
    """
    try:
        from pipecat.serializers.plivo import PlivoFrameSerializer
        from pipecat.transports.network.fastapi_websocket import (
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )
    except ImportError as e:
        logger.error(
            f"Plivo transport dependencies not installed. Install with: {get_install_command('plivo')}"
        )
        logger.debug(f"Import error: {e}")
        return

    logger.info("Running with FastAPIWebsocketTransport (Plivo)...")

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for testing
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/")
    async def start_call():
        """Handle Plivo webhook and return Plivo XML response.

        Returns:
            Plivo XML response directing call to WebSocket stream.
        """
        logger.debug("POST Plivo XML")

        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">wss://{args.proxy}/ws</Stream>
</Response>
        """
        return HTMLResponse(content=xml_content, media_type="application/xml")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Handle Plivo WebSocket connections for voice streaming.

        Args:
            websocket: The WebSocket connection from Plivo.
        """
        await websocket.accept()

        logger.debug("WebSocket connection accepted")

        # Reading Plivo data.
        start_data = websocket.iter_text()
        start_message = json.loads(await start_data.__anext__())

        logger.debug(f"Received start message: {start_message}")

        # Extract stream_id and call_id from the start event
        start_info = start_message.get("start", {})
        stream_id = start_info.get("streamId")
        call_id = start_info.get("callId")

        if not stream_id:
            logger.error("No streamId found in start message")
            await websocket.close()
            return

        logger.info(f"WebSocket connection accepted for stream: {stream_id}, call: {call_id}")

        # Create websocket transport and update params.
        params: FastAPIWebsocketParams = transport_params[args.transport]()
        params.add_wav_header = False
        params.serializer = PlivoFrameSerializer(
            stream_id=stream_id,
            call_id=call_id,
        )
        transport = FastAPIWebsocketTransport(websocket=websocket, params=params)
        await run(transport, args, False)

    uvicorn.run(app, host=args.host, port=args.port)


def run_main(
    run: Callable,
    args: argparse.Namespace,
    transport_params: Mapping[str, Callable] = {},
):
    """Run the application with the specified transport type.

    Args:
        run: The function to run.
        args: Parsed command-line arguments.
        transport_params: Mapping of transport names to parameter factory functions.
    """
    if args.transport not in transport_params:
        logger.error(f"Transport '{args.transport}' not supported by this application.")
        logger.info(
            f"To add {args.transport} support, install with: {get_install_command(args.transport)}"
        )
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
    """Main entry point for running Pipecat applications with transport selection.

    Args:
        run: The function to run.
        parser: Optional argument parser. If None, creates a default one.
        transport_params: Mapping of transport names to parameter factory functions.
    """
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
        choices=["daily", "livekit", "plivo", "telnyx", "twilio", "webrtc"],
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

    # Import the bot file
    run_main(run, args, transport_params)
