#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat development runner.

This development runner executes Pipecat bots and provides the supporting
infrastructure they need - creating Daily rooms and tokens, managing WebRTC
connections, and setting up telephony webhook/WebSocket infrastructure. It
supports multiple transport types with a unified interface.

Install with::

    pip install pipecat-ai[runner]

All bots must implement a `bot(runner_args)` async function as the entry point.
The server automatically discovers and executes this function when connections
are established.

Single transport example::

    async def bot(runner_args: RunnerArguments):
        transport = DailyTransport(
            runner_args.room_url,
            runner_args.token,
            "Bot",
            DailyParams(...)
        )
        # Your bot logic here
        await run_pipeline(transport)

    if __name__ == "__main__":
        from pipecat.runner.run import main
        main()

Multiple transport example::

    async def bot(runner_args: RunnerArguments):
        # Type-safe transport detection
        if isinstance(runner_args, DailyRunnerArguments):
            transport = setup_daily_transport(runner_args)  # Your application code
        elif isinstance(runner_args, SmallWebRTCRunnerArguments):
            transport = setup_webrtc_transport(runner_args)  # Your application code
        elif isinstance(runner_args, WebSocketRunnerArguments):
            transport = setup_telephony_transport(runner_args)  # Your application code

        # Your bot implementation
        await run_pipeline(transport)

Supported transports:

- Daily - Creates rooms and tokens, runs bot as participant
- WebRTC - Provides local WebRTC interface with prebuilt UI
- Telephony - Handles webhook and WebSocket connections for Twilio, Telnyx, Plivo, Exotel

To run locally:

- WebRTC: `python bot.py -t webrtc`
- ESP32: `python bot.py -t webrtc --esp32 --host 192.168.1.100`
- Daily (server): `python bot.py -t daily`
- Daily (direct, testing only): `python bot.py -d`
- Telephony: `python bot.py -t twilio -x your_username.ngrok.io`
- Exotel: `python bot.py -t exotel` (no proxy needed, but ngrok connection to HTTP 7860 is required)
"""

import argparse
import asyncio
import mimetypes
import os
import sys
import uuid
from contextlib import asynccontextmanager
from http import HTTPMethod
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import aiohttp
from fastapi.responses import FileResponse, Response
from loguru import logger

from pipecat.runner.types import (
    DailyRunnerArguments,
    SmallWebRTCRunnerArguments,
    WebSocketRunnerArguments,
)

try:
    import uvicorn
    from dotenv import load_dotenv
    from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, RedirectResponse
except ImportError as e:
    logger.error(f"Runner dependencies not available: {e}")
    logger.error("To use Pipecat runners, install with: pip install pipecat-ai[runner]")
    raise ImportError(
        "Runner dependencies required. Install with: pip install pipecat-ai[runner]"
    ) from e


load_dotenv(override=True)
os.environ["ENV"] = "local"

TELEPHONY_TRANSPORTS = ["twilio", "telnyx", "plivo", "exotel"]

RUNNER_DOWNLOADS_FOLDER: Optional[str] = None
RUNNER_HOST: str = "localhost"
RUNNER_PORT: int = 7860


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
    # (excluding server.py).
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
    runner_args = WebSocketRunnerArguments(websocket=websocket)

    await bot_module.bot(runner_args)


def _create_server_app(
    *,
    transport_type: str,
    host: str = "localhost",
    proxy: str,
    esp32_mode: bool = False,
    whatsapp_enabled: bool = False,
    folder: Optional[str] = None,
):
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
        _setup_webrtc_routes(app, esp32_mode=esp32_mode, host=host, folder=folder)
        if whatsapp_enabled:
            _setup_whatsapp_routes(app)
    elif transport_type == "daily":
        _setup_daily_routes(app)
    elif transport_type in TELEPHONY_TRANSPORTS:
        _setup_telephony_routes(app, transport_type=transport_type, proxy=proxy)
    else:
        logger.warning(f"Unknown transport type: {transport_type}")

    return app


def _setup_webrtc_routes(
    app: FastAPI, *, esp32_mode: bool = False, host: str = "localhost", folder: Optional[str] = None
):
    """Set up WebRTC-specific routes."""
    try:
        from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

        from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
        from pipecat.transports.smallwebrtc.request_handler import (
            IceCandidate,
            SmallWebRTCPatchRequest,
            SmallWebRTCRequest,
            SmallWebRTCRequestHandler,
        )
    except ImportError as e:
        logger.error(f"WebRTC transport dependencies not installed: {e}")
        return

    class IceConfig(TypedDict):
        iceServers: List[IceServer]

    class StartBotResult(TypedDict, total=False):
        sessionId: str
        iceConfig: Optional[IceConfig]

    # In-memory store of active sessions: session_id -> session info
    active_sessions: Dict[str, Dict[str, Any]] = {}

    # Mount the frontend
    app.mount("/client", SmallWebRTCPrebuiltUI)

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        """Redirect root requests to client interface."""
        return RedirectResponse(url="/client/")

    @app.get("/files/{filename:path}")
    async def download_file(filename: str):
        """Handle file downloads."""
        if not folder:
            logger.warning(f"Attempting to dowload {filename}, but downloads folder not setup.")
            return

        file_path = Path(folder) / filename
        if not os.path.exists(file_path):
            raise HTTPException(404)

        media_type, _ = mimetypes.guess_type(file_path)

        return FileResponse(path=file_path, media_type=media_type, filename=filename)

    # Initialize the SmallWebRTC request handler
    small_webrtc_handler: SmallWebRTCRequestHandler = SmallWebRTCRequestHandler(
        esp32_mode=esp32_mode, host=host
    )

    @app.post("/api/offer")
    async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
        """Handle WebRTC offer requests via SmallWebRTCRequestHandler."""

        # Prepare runner arguments with the callback to run your bot
        async def webrtc_connection_callback(connection):
            bot_module = _get_bot_module()
            runner_args = SmallWebRTCRunnerArguments(webrtc_connection=connection)
            background_tasks.add_task(bot_module.bot, runner_args)

        # Delegate handling to SmallWebRTCRequestHandler
        answer = await small_webrtc_handler.handle_web_request(
            request=request,
            webrtc_connection_callback=webrtc_connection_callback,
        )
        return answer

    @app.patch("/api/offer")
    async def ice_candidate(request: SmallWebRTCPatchRequest):
        """Handle WebRTC new ice candidate requests."""
        logger.debug(f"Received patch request: {request}")
        await small_webrtc_handler.handle_patch_request(request)
        return {"status": "success"}

    @app.post("/start")
    async def rtvi_start(request: Request):
        """Mimic Pipecat Cloud's /start endpoint."""
        # Parse the request body
        try:
            request_data = await request.json()
            logger.debug(f"Received request: {request_data}")
        except Exception as e:
            logger.error(f"Failed to parse request body: {e}")
            request_data = {}

        # Store session info immediately in memory, replicate the behavior expected on Pipecat Cloud
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = request_data

        result: StartBotResult = {"sessionId": session_id}
        if request_data.get("enableDefaultIceServers"):
            result["iceConfig"] = IceConfig(
                iceServers=[IceServer(urls="stun:stun.l.google.com:19302")]
            )

        return result

    @app.api_route(
        "/sessions/{session_id}/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    )
    async def proxy_request(
        session_id: str, path: str, request: Request, background_tasks: BackgroundTasks
    ):
        """Mimic Pipecat Cloud's proxy."""
        active_session = active_sessions.get(session_id)
        if not active_session:
            return Response(content="Invalid or not-yet-ready session_id", status_code=404)

        if path.endswith("api/offer"):
            # Parse the request body and convert to SmallWebRTCRequest
            try:
                request_data = await request.json()
                if request.method == HTTPMethod.POST.value:
                    webrtc_request = SmallWebRTCRequest(
                        sdp=request_data["sdp"],
                        type=request_data["type"],
                        pc_id=request_data.get("pc_id"),
                        restart_pc=request_data.get("restart_pc"),
                        request_data=request_data,
                    )
                    return await offer(webrtc_request, background_tasks)
                elif request.method == HTTPMethod.PATCH.value:
                    patch_request = SmallWebRTCPatchRequest(
                        pc_id=request_data["pc_id"],
                        candidates=[IceCandidate(**c) for c in request_data.get("candidates", [])],
                    )
                    return await ice_candidate(patch_request)
            except Exception as e:
                logger.error(f"Failed to parse WebRTC request: {e}")
                return Response(content="Invalid WebRTC request", status_code=400)

        logger.info(f"Received request for path: {path}")
        return Response(status_code=200)

    @asynccontextmanager
    async def smallwebrtc_lifespan(app: FastAPI):
        """Manage FastAPI application lifecycle and cleanup connections."""
        yield
        await small_webrtc_handler.close()

    # Add the SmallWebRTC lifespan to the app
    _add_lifespan_to_app(app, smallwebrtc_lifespan)


def _add_lifespan_to_app(app: FastAPI, new_lifespan):
    """Add a new lifespan context manager to the app, combining with existing if present.

    Args:
        app: The FastAPI application instance
        new_lifespan: The new lifespan context manager to add
    """
    if hasattr(app.router, "lifespan_context") and app.router.lifespan_context is not None:
        # If there's already a lifespan context, combine them
        existing_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def combined_lifespan(app: FastAPI):
            async with existing_lifespan(app):
                async with new_lifespan(app):
                    yield

        app.router.lifespan_context = combined_lifespan
    else:
        # No existing lifespan, use the new one
        app.router.lifespan_context = new_lifespan


def _setup_whatsapp_routes(app: FastAPI):
    """Set up WebRTC-specific routes."""
    WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET")
    WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
    WHATSAPP_WEBHOOK_VERIFICATION_TOKEN = os.getenv("WHATSAPP_WEBHOOK_VERIFICATION_TOKEN")

    if not all(
        [
            WHATSAPP_APP_SECRET,
            WHATSAPP_PHONE_NUMBER_ID,
            WHATSAPP_TOKEN,
            WHATSAPP_WEBHOOK_VERIFICATION_TOKEN,
        ]
    ):
        logger.error(
            """Missing required environment variables for WhatsApp transport:
    WHATSAPP_APP_SECRET
    WHATSAPP_PHONE_NUMBER_ID
    WHATSAPP_TOKEN
    WHATSAPP_WEBHOOK_VERIFICATION_TOKEN
            """
        )
        return

    try:
        from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

        from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
        from pipecat.transports.smallwebrtc.request_handler import (
            SmallWebRTCRequest,
            SmallWebRTCRequestHandler,
        )
        from pipecat.transports.whatsapp.api import WhatsAppWebhookRequest
        from pipecat.transports.whatsapp.client import WhatsAppClient
    except ImportError as e:
        logger.error(f"WhatsApp transport dependencies not installed: {e}")
        return

    # Global WhatsApp client instance
    whatsapp_client: Optional[WhatsAppClient] = None

    @app.get(
        "/whatsapp",
        summary="Verify WhatsApp webhook",
        description="Handles WhatsApp webhook verification requests from Meta",
    )
    async def verify_webhook(request: Request):
        """Verify WhatsApp webhook endpoint.

        This endpoint is called by Meta's WhatsApp Business API to verify
        the webhook URL during setup. It validates the verification token
        and returns the challenge parameter if successful.
        """
        if whatsapp_client is None:
            logger.error("WhatsApp client is not initialized")
            raise HTTPException(status_code=503, detail="Service unavailable")

        params = dict(request.query_params)
        logger.debug(f"Webhook verification request received with params: {list(params.keys())}")

        try:
            result = await whatsapp_client.handle_verify_webhook_request(
                params=params, expected_verification_token=WHATSAPP_WEBHOOK_VERIFICATION_TOKEN
            )
            logger.info("Webhook verification successful")
            return result
        except ValueError as e:
            logger.warning(f"Webhook verification failed: {e}")
            raise HTTPException(status_code=403, detail="Verification failed")

    @app.post(
        "/whatsapp",
        summary="Handle WhatsApp webhook events",
        description="Processes incoming WhatsApp messages and call events",
    )
    async def whatsapp_webhook(
        body: WhatsAppWebhookRequest,
        background_tasks: BackgroundTasks,
        request: Request,
        x_hub_signature_256: str = Header(None),
    ):
        """Handle incoming WhatsApp webhook events.

        For call events, establishes WebRTC connections and spawns bot instances
        in the background to handle real-time communication.
        """
        if whatsapp_client is None:
            logger.error("WhatsApp client is not initialized")
            raise HTTPException(status_code=503, detail="Service unavailable")

        # Validate webhook object type
        if body.object != "whatsapp_business_account":
            logger.warning(f"Invalid webhook object type: {body.object}")
            raise HTTPException(status_code=400, detail="Invalid object type")

        logger.debug(f"Processing WhatsApp webhook: {body.model_dump()}")

        async def connection_callback(connection: SmallWebRTCConnection):
            """Handle new WebRTC connections from WhatsApp calls.

            Called when a WebRTC connection is established for a WhatsApp call.
            Spawns a bot instance to handle the conversation.

            Args:
                connection: The established WebRTC connection
            """
            bot_module = _get_bot_module()
            runner_args = SmallWebRTCRunnerArguments(webrtc_connection=connection)
            background_tasks.add_task(bot_module.bot, runner_args)

        try:
            # Process the webhook request
            raw_body = await request.body()
            result = await whatsapp_client.handle_webhook_request(
                body, connection_callback, sha256_signature=x_hub_signature_256, raw_body=raw_body
            )
            logger.debug(f"Webhook processed successfully: {result}")
            return {"status": "success", "message": "Webhook processed successfully"}
        except ValueError as ve:
            logger.warning(f"Invalid webhook request format: {ve}")
            raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
        except Exception as e:
            logger.error(f"Internal error processing webhook: {e}")
            raise HTTPException(status_code=500, detail="Internal server error processing webhook")

    @asynccontextmanager
    async def whatsapp_lifespan(app: FastAPI):
        """Manage WhatsApp client lifecycle and cleanup connections."""
        nonlocal whatsapp_client

        # Initialize WhatsApp client with persistent HTTP session
        async with aiohttp.ClientSession() as session:
            whatsapp_client = WhatsAppClient(
                whatsapp_token=WHATSAPP_TOKEN,
                whatsapp_secret=WHATSAPP_APP_SECRET,
                phone_number_id=WHATSAPP_PHONE_NUMBER_ID,
                session=session,
            )
            logger.info("WhatsApp client initialized successfully")

            try:
                yield  # Run the application
            finally:
                # Cleanup all active calls on shutdown
                logger.info("Cleaning up WhatsApp client resources...")
                if whatsapp_client:
                    await whatsapp_client.terminate_all_calls()
                logger.info("WhatsApp cleanup completed")

    # Add the WhatsApp lifespan to the app
    _add_lifespan_to_app(app, whatsapp_lifespan)


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

            # Start the bot in the background with empty body for GET requests
            bot_module = _get_bot_module()
            runner_args = DailyRunnerArguments(room_url=room_url, token=token)
            asyncio.create_task(bot_module.bot(runner_args))
            return RedirectResponse(room_url)

    async def _handle_rtvi_request(request: Request):
        """Common handler for both /start and /connect endpoints.

        Expects POST body like::

            {
                "createDailyRoom": true,
                "dailyRoomProperties": { "start_video_off": true },
                "body": { "custom_data": "value" }
            }
        """
        print("Starting bot with Daily transport")

        # Parse the request body
        try:
            request_data = await request.json()
            logger.debug(f"Received request: {request_data}")
        except Exception as e:
            logger.error(f"Failed to parse request body: {e}")
            request_data = {}

        # Extract the body data that should be passed to the bot
        # This mimics Pipecat Cloud's behavior
        bot_body = request_data.get("body", {})

        # Log the extracted body data for debugging
        if bot_body:
            logger.info(f"Extracted body data for bot: {bot_body}")
        else:
            logger.debug("No body data provided in request")

        from pipecat.runner.daily import configure

        async with aiohttp.ClientSession() as session:
            room_url, token = await configure(session)

            # Start the bot in the background with extracted body data
            bot_module = _get_bot_module()
            runner_args = DailyRunnerArguments(room_url=room_url, token=token, body=bot_body)
            asyncio.create_task(bot_module.bot(runner_args))
            # Match PCC /start endpoint response format:
            return {"dailyRoom": room_url, "dailyToken": token}

    @app.post("/start")
    async def rtvi_start(request: Request):
        """Launch a Daily bot and return connection info for RTVI clients."""
        return await _handle_rtvi_request(request)

    @app.post("/connect")
    async def rtvi_connect(request: Request):
        """Launch a Daily bot and return connection info for RTVI clients.

        .. deprecated:: 0.0.78
            Use /start instead. This endpoint will be removed in a future version.
        """
        logger.warning(
            "DEPRECATED: /connect endpoint is deprecated. Please use /start instead. "
            "This endpoint will be removed in a future version."
        )
        return await _handle_rtvi_request(request)


def _setup_telephony_routes(app: FastAPI, *, transport_type: str, proxy: str):
    """Set up telephony-specific routes."""
    # XML response templates (Exotel doesn't use XML webhooks)
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
        if transport_type == "exotel":
            # Exotel doesn't use POST webhooks - redirect to proper documentation
            logger.debug("POST Exotel endpoint - not used")
            return {
                "error": "Exotel doesn't use POST webhooks",
                "websocket_url": f"wss://{proxy}/ws",
                "note": "Configure the WebSocket URL above in your Exotel App Bazaar Voicebot Applet",
            }
        else:
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


async def _run_daily_direct():
    """Run Daily bot with direct connection (no FastAPI server)."""
    try:
        from pipecat.runner.daily import configure
    except ImportError as e:
        logger.error("Daily transport dependencies not installed.")
        return

    logger.info("Running with direct Daily connection...")

    async with aiohttp.ClientSession() as session:
        room_url, token = await configure(session)

        # Direct connections have no request body, so use empty dict
        runner_args = DailyRunnerArguments(room_url=room_url, token=token)
        runner_args.handle_sigint = True

        # Get the bot module and run it directly
        bot_module = _get_bot_module()

        print(f"📞 Joining Daily room: {room_url}")
        print("   (Direct connection - no web server needed)")
        print()

        await bot_module.bot(runner_args)


def _validate_and_clean_proxy(proxy: str) -> str:
    """Validate and clean proxy hostname, removing protocol if present."""
    if not proxy:
        return proxy

    original_proxy = proxy

    # Strip common protocols
    if proxy.startswith(("http://", "https://")):
        proxy = proxy.split("://", 1)[1]
        logger.warning(
            f"Removed protocol from proxy URL. Using '{proxy}' instead of '{original_proxy}'. "
            f"The --proxy argument expects only the hostname (e.g., 'mybot.ngrok.io')."
        )

    # Remove trailing slashes
    proxy = proxy.rstrip("/")

    return proxy


def runner_downloads_folder() -> Optional[str]:
    """Returns the folder where files are stored for later download."""
    return RUNNER_DOWNLOADS_FOLDER


def runner_host() -> str:
    """Returns the host name of this runner."""
    return RUNNER_HOST


def runner_port() -> int:
    """Returns the port of this runner."""
    return RUNNER_PORT


def main():
    """Start the Pipecat development runner.

    Parses command-line arguments and starts a FastAPI server configured
    for the specified transport type. The runner will discover and run
    any bot() function found in the current directory.

    Command-line arguments:

    Args:
        --host: Server host address (default: localhost)
        --port: Server port (default: 7860)
        -t/--transport: Transport type (daily, webrtc, twilio, telnyx, plivo, exotel)
        -x/--proxy: Public proxy hostname for telephony webhooks
        --esp32: Enable SDP munging for ESP32 compatibility (requires --host with IP address)
        -d/--direct: Connect directly to Daily room (automatically sets transport to daily)
        -v/--verbose: Increase logging verbosity

    The bot file must contain a `bot(runner_args)` function as the entry point.
    """
    global RUNNER_DOWNLOADS_FOLDER, RUNNER_HOST, RUNNER_PORT

    parser = argparse.ArgumentParser(description="Pipecat Development Runner")
    parser.add_argument("--host", type=str, default=RUNNER_HOST, help="Host address")
    parser.add_argument("--port", type=int, default=RUNNER_PORT, help="Port number")
    parser.add_argument(
        "-t",
        "--transport",
        type=str,
        choices=["daily", "webrtc", *TELEPHONY_TRANSPORTS],
        default="webrtc",
        help="Transport type",
    )
    parser.add_argument("--proxy", "-x", help="Public proxy host name")
    parser.add_argument(
        "--esp32",
        action="store_true",
        default=False,
        help="Enable SDP munging for ESP32 compatibility (requires --host with IP address)",
    )
    parser.add_argument(
        "-d",
        "--direct",
        action="store_true",
        default=False,
        help="Connect directly to Daily room (automatically sets transport to daily)",
    )
    parser.add_argument("-f", "--folder", type=str, help="Path to downloads folder")
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Increase logging verbosity"
    )
    parser.add_argument(
        "--whatsapp",
        action="store_true",
        default=False,
        help="Ensure requried WhatsApp environment variables are present",
    )

    args = parser.parse_args()

    # Validate and clean proxy hostname
    if args.proxy:
        args.proxy = _validate_and_clean_proxy(args.proxy)

    # Auto-set transport to daily if --direct is used without explicit transport
    if args.direct and args.transport == "webrtc":  # webrtc is the default
        args.transport = "daily"
    elif args.direct and args.transport != "daily":
        logger.error("--direct flag only works with Daily transport (-t daily)")
        return

    # Validate ESP32 requirements
    if args.esp32 and args.host == "localhost":
        logger.error("For ESP32, you need to specify `--host IP` so we can do SDP munging.")
        return

    if args.transport in TELEPHONY_TRANSPORTS and not args.proxy:
        logger.error(f"For telephony transports, you need to specify `--proxy PROXY`.")
        return

    # Log level
    logger.remove()
    logger.add(sys.stderr, level="TRACE" if args.verbose else "DEBUG")

    # Handle direct Daily connection (no FastAPI server)
    if args.direct:
        print()
        print("🚀 Connecting directly to Daily room...")
        print()

        # Run direct Daily connection
        asyncio.run(_run_daily_direct())
        return

    # Print startup message for server-based transports
    if args.transport == "webrtc":
        print()
        if args.esp32:
            print(f"🚀 Bot ready! (ESP32 mode)")
        elif args.whatsapp:
            print(f"🚀 Bot ready! (WhatsApp)")
        else:
            print(f"🚀 Bot ready!")
        print(f"   → Open http://{args.host}:{args.port}/client in your browser")
        print()
    elif args.transport == "daily":
        print()
        print(f"🚀 Bot ready!")
        print(f"   → Open http://{args.host}:{args.port} in your browser to start a session")
        print()

    RUNNER_DOWNLOADS_FOLDER = args.folder
    RUNNER_HOST = args.host
    RUNNER_PORT = args.port

    # Create the app with transport-specific setup
    app = _create_server_app(
        transport_type=args.transport,
        host=args.host,
        proxy=args.proxy,
        esp32_mode=args.esp32,
        whatsapp_enabled=args.whatsapp,
        folder=args.folder,
    )

    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
