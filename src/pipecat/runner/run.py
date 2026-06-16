#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat development runner.

This development runner executes Pipecat bots and provides the supporting
infrastructure they need - creating Daily rooms and tokens, managing WebRTC
connections, and setting up telephony webhook/WebSocket infrastructure. It
supports multiple transport types with a unified interface.

Install with::

    uv add "pipecat-ai[runner]"

All bots must implement a `bot(runner_args)` async function as the entry point.
The server automatically discovers and executes this function when connections
are established.

By default the runner starts a single FastAPI server that supports WebRTC, Daily,
and telephony transports simultaneously. Clients declare which transport they want
via the ``transport`` field in the ``/start`` request body (default: ``"webrtc"``).

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

The ``/start`` endpoint accepts::

    {
        "transport": "webrtc",        // "webrtc" | "daily" | "twilio" | "telnyx" |
                                      // "plivo" | "exotel" — default: "webrtc"

        // WebRTC-specific
        "enableDefaultIceServers": false,
        "body": {...},

        // Daily-specific
        "createDailyRoom": true,
        "dailyRoomProperties": {...},
        "dailyMeetingTokenProperties": {...},
        "body": {...}
    }

To run locally:

- All transports (default): ``python bot.py``
- WebRTC only: ``python bot.py -t webrtc``
- ESP32: ``python bot.py -t webrtc --esp32 --host 192.168.1.100``
- Daily only: ``python bot.py -t daily``
- Daily (direct, testing only): ``python bot.py -d``
- Telephony: ``python bot.py -t twilio -x your_username.ngrok.io``
- Exotel: ``python bot.py -t exotel`` (no proxy needed, but ngrok connection to HTTP 7860 is required)
- WhatsApp: ``python bot.py --whatsapp``
"""

import argparse
import asyncio
import base64
import hashlib
import hmac
import importlib.util
import json
import mimetypes
import os
import secrets
import sys
import time
import uuid
from contextlib import asynccontextmanager
from http import HTTPMethod
from pathlib import Path
from typing import Any, TypedDict

import aiohttp
from fastapi.responses import FileResponse, Response
from loguru import logger

from pipecat.runner.types import (
    DailyRunnerArguments,
    EvalRunnerArguments,
    RunnerArguments,
    SmallWebRTCRunnerArguments,
    VonageRunnerArguments,
    WebSocketRunnerArguments,
)
from pipecat.runner.vonage import configure as configure_vonage
from pipecat.utils.security.allowed_origins import is_origin_allowed

try:
    import uvicorn
    from dotenv import load_dotenv
    from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, WebSocket
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
except ImportError as e:
    logger.error(f"Runner dependencies not available: {e}")
    logger.error('To use Pipecat runners, install with: uv add "pipecat-ai[runner]"')
    raise ImportError(
        'Runner dependencies required. Install with: uv add "pipecat-ai[runner]"'
    ) from e


load_dotenv(override=True)
os.environ["ENV"] = "local"

TELEPHONY_TRANSPORTS = ["twilio", "telnyx", "plivo", "exotel"]
TRANSPORT_ROUTE_DEPENDENCIES = {
    "daily": ("daily",),
    "webrtc": ("aiortc",),
    "telephony": ("fastapi", "websockets"),
    "websocket": ("fastapi", "websockets"),
}
TRANSPORT_INSTALL_HINTS = {
    "daily": "install pipecat-ai[daily]",
    "webrtc": "install pipecat-ai[webrtc]",
    "telephony": "install pipecat-ai[websocket]",
    "websocket": "install pipecat-ai[websocket]",
}

# Mirror Pipecat Cloud's 4-hour max session limit so dev rooms get cleaned up.
PIPECAT_ROOM_EXP_HOURS = 4.0

RUNNER_DOWNLOADS_FOLDER: str | None = None
RUNNER_HOST: str = "localhost"
RUNNER_PORT: int = 7860

# Per-process HMAC secret for WebSocket token authentication. Auto-generated so
# tokens from one runner instance cannot be replayed against another.
_WS_AUTH_SECRET: bytes = secrets.token_bytes(32)

app: FastAPI = FastAPI()
"""The FastAPI application instance.

Import this to add custom routes from other packages before calling
:func:`main`::

    from pipecat.runner.run import app, main

    @app.get("/my-route")
    async def my_route():
        return {"hello": "world"}

    if __name__ == "__main__":
        main()
"""


def _is_module_available(module: str) -> bool:
    """Check whether a module can be imported without importing it.

    Args:
        module: Fully-qualified module name to check.

    Returns:
        ``True`` if Python can resolve the module, ``False`` otherwise.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _transport_route_dependencies(transport: str) -> tuple[str, ...]:
    """Return module dependencies required for a transport route.

    Args:
        transport: Transport name from the runner request or CLI.

    Returns:
        Module names required to enable the transport route.
    """
    if transport in TELEPHONY_TRANSPORTS:
        return TRANSPORT_ROUTE_DEPENDENCIES["telephony"]
    return TRANSPORT_ROUTE_DEPENDENCIES.get(transport, ())


def _transport_routes_enabled(transport: str) -> bool:
    """Return whether a transport route can run in this environment.

    Args:
        transport: Transport name from the runner request or CLI.

    Returns:
        ``True`` if the requested transport is enabled.
    """
    return all(_is_module_available(module) for module in _transport_route_dependencies(transport))


def _runner_url(args: argparse.Namespace) -> str:
    """Return the browser URL for the runner prebuilt client."""
    return f"http://{args.host}:{args.port}"


def _transport_status_lists() -> tuple[list[str], list[str]]:
    """Return enabled and disabled transport labels for the startup banner."""
    transports = ["daily", "webrtc", "telephony", "websocket"]
    enabled = []
    disabled = []

    for label in transports:
        if _transport_routes_enabled(label):
            enabled.append(label)
        else:
            disabled.append(f"{label} ({TRANSPORT_INSTALL_HINTS[label]})")

    return enabled, disabled


def _format_transport_status(labels: list[str]) -> str:
    """Format a startup banner transport status list."""
    return ", ".join(labels) if labels else "none"


def _generate_ws_token(ttl: int = 300) -> str:
    """Return a signed, self-expiring WebSocket session token.

    The token is ``<base64url-payload>.<hmac-sha256-hex>`` where the payload
    encodes ``{"exp": unix_timestamp, "jti": random_nonce}``. Valid for ``ttl``
    seconds (default 5 min). The nonce ensures uniqueness within the same second.
    """
    payload = (
        base64.urlsafe_b64encode(
            json.dumps({"exp": int(time.time()) + ttl, "jti": secrets.token_hex(8)}).encode()
        )
        .decode()
        .rstrip("=")
    )
    sig = hmac.new(_WS_AUTH_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _verify_and_consume_ws_token(used: set[str], token: str) -> bool:
    """Validate a WebSocket session token and mark it as used (one-time use).

    Args:
        used: Set of already-consumed tokens (mutated on success).
        token: Token string obtained from :func:`_generate_ws_token`.

    Returns:
        ``True`` if the token has a valid signature, has not expired, and has
        not been used before. Adds the token to ``used`` on success so replay
        attempts are rejected.
    """
    try:
        payload, sig = token.rsplit(".", 1)
    except ValueError:
        return False
    expected = hmac.new(_WS_AUTH_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return False
    padded = payload + "=" * (-len(payload) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(padded))
    except Exception:
        return False
    if time.time() > data.get("exp", 0):
        return False
    if token in used:
        return False
    used.add(token)
    return True


def _extract_ws_token(websocket) -> str | None:
    """Extract a WebSocket session token from the connection handshake.

    Checks, in order:

    1. ``Authorization: Bearer <token>`` request header.
    2. ``?token=<token>`` query parameter.

    Returns the token string, or ``None`` if not present.
    """
    auth = websocket.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return websocket.query_params.get("token")


def _print_security_status(args: argparse.Namespace):
    """Print security status lines (auth + origin restriction)."""
    if args.ws_auth == "token":
        print("   → WebSocket auth:  token (HMAC, call /start to obtain a token)")
    if args.allowed_origins:
        print(f"   → Allowed origins: {', '.join(args.allowed_origins)}")
    else:
        print("   → Allowed origins: all (no restriction)")


def _print_startup_message(args: argparse.Namespace):
    """Print connection information for the development runner."""
    print()
    if args.transport is None:
        enabled, disabled = _transport_status_lists()
        print("🚀 Bot ready!")
        print(f"   → Open: {_runner_url(args)}")
        print(f"   → Enabled transports: {_format_transport_status(enabled)}")
        if disabled:
            print(f"   → Disabled transports: {_format_transport_status(disabled)}")
        _print_security_status(args)
    elif args.transport == "webrtc":
        if args.esp32:
            print("🚀 Bot ready! (ESP32 mode)")
        elif args.whatsapp:
            print("🚀 Bot ready! (WhatsApp)")
        else:
            print("🚀 Bot ready! (WebRTC)")
        if _transport_routes_enabled("webrtc"):
            print(f"   → Open: {_runner_url(args)}")
        else:
            print(f"   → WebRTC disabled ({TRANSPORT_INSTALL_HINTS['webrtc']})")
    elif args.transport == "daily":
        print("🚀 Bot ready! (Daily)")
        if not _transport_routes_enabled("daily"):
            print(f"   → Daily disabled ({TRANSPORT_INSTALL_HINTS['daily']})")
        else:
            print(f"   → Open: {_runner_url(args)}")
            if args.dialin:
                print(
                    f"   → Daily dial-in webhook: "
                    f"http://{args.host}:{args.port}/daily-dialin-webhook"
                )
                print("   → Configure this URL in your Daily phone number settings")
    elif args.transport in TELEPHONY_TRANSPORTS:
        print(f"🚀 Bot ready! ({args.transport.capitalize()})")
        if not _transport_routes_enabled(args.transport):
            print(f"   → Telephony disabled ({TRANSPORT_INSTALL_HINTS['telephony']})")
        else:
            print(f"   → Open: {_runner_url(args)}")
            if args.proxy:
                print(f"   → XML webhook: http://{args.host}:{args.port}/")
            print(f"   → WebSocket:   ws://{args.host}:{args.port}/ws")
            _print_security_status(args)
    elif args.transport == "websocket":
        print("🚀 Bot ready! (WebSocket)")
        if not _transport_routes_enabled("websocket"):
            print(f"   → WebSocket disabled ({TRANSPORT_INSTALL_HINTS['websocket']})")
        else:
            print(f"   → Open: {_runner_url(args)}")
            scheme = "wss" if args.host != "localhost" else "ws"
            print(f"   → WebSocket:   {scheme}://{args.host}:{args.port}/ws-client")
            _print_security_status(args)
    elif args.transport == "vonage":
        print()
        print("🚀 Bot ready!")
    print()


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
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "bot"):
                    return module
            except Exception:
                continue

    raise ImportError(
        "Could not find 'bot' function. Make sure your bot file has a 'bot' function."
    )


async def _run_telephony_bot(websocket: WebSocket, args: argparse.Namespace):
    """Run a bot for telephony transports."""
    bot_module = _get_bot_module()

    # Just pass the WebSocket - let the bot handle parsing
    runner_args = WebSocketRunnerArguments(websocket=websocket, session_id=str(uuid.uuid4()))
    runner_args.cli_args = args

    await bot_module.bot(runner_args)


async def _run_websocket_bot(websocket: WebSocket, args: argparse.Namespace):
    """Run a bot for plain WebSocket transport."""
    bot_module = _get_bot_module()

    runner_args = WebSocketRunnerArguments(
        websocket=websocket,
        transport_type="websocket",
        session_id=str(uuid.uuid4()),
    )
    runner_args.cli_args = args

    await bot_module.bot(runner_args)


def _setup_websocket_routes(app: FastAPI, args: argparse.Namespace, ws_used_tokens: set[str]):
    """Set up the plain WebSocket route at ``/ws-client``.

    When ``args.ws_auth == "token"``, connections must present a valid HMAC
    session token obtained via ``POST /start``. The token may be supplied as:

    - ``Authorization: Bearer <token>`` header
    - ``?token=<token>`` query parameter
    - URL path segment: ``/ws-client/<token>``

    Invalid or missing tokens are rejected with WebSocket close code 4003.
    """
    if not _transport_routes_enabled("websocket"):
        return

    async def _handle_plain_ws(websocket: WebSocket, path_token: str | None = None):
        if args.ws_auth == "token":
            token = path_token or _extract_ws_token(websocket)
            if not token or not _verify_and_consume_ws_token(ws_used_tokens, token):
                logger.warning("WebSocket connection rejected: invalid or missing token")
                await websocket.close(code=4003)
                return
        origin = websocket.headers.get("origin", "")
        if not is_origin_allowed(origin, args.allowed_origins):
            logger.warning(f"WebSocket connection rejected: origin '{origin}' not allowed")
            await websocket.close(code=4003)
            return
        await websocket.accept()
        logger.debug("Plain WebSocket connection accepted")
        await _run_websocket_bot(websocket, args)

    @app.websocket("/ws-client")
    async def websocket_client_endpoint(websocket: WebSocket):
        """Handle plain WebSocket connections (non-telephony)."""
        await _handle_plain_ws(websocket)

    @app.websocket("/ws-client/{token}")
    async def websocket_client_endpoint_with_token(websocket: WebSocket, token: str):
        """Handle plain WebSocket connections with token in the URL path."""
        await _handle_plain_ws(websocket, path_token=token)


def _configure_server_app(args: argparse.Namespace):
    """Configure the module-level FastAPI app with routes for all transports."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # FastAPI returns 422 Unprocessable Entity for Pydantic validation failures by default, but
    # swallows the raw request body in the error response. This handler overrides that behavior to
    # log both the validation errors and the raw body, making it much easier to debug malformed
    # payloads from any transport (WhatsApp, WebRTC, telephony, etc.).
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        body = await request.body()
        logger.error(f"422 Validation error on {request.url.path}: {exc.errors()}")
        logger.error(
            "Raw body: %s",
            body.decode(errors="replace")[:5000],
        )
        return JSONResponse(status_code=422, content=jsonable_encoder({"detail": exc.errors()}))

    # Shared session store: session_id -> body data. Used by the WebRTC /start
    # flow and the /sessions/{session_id}/... proxy routes.
    active_sessions: dict[str, dict[str, Any]] = {}

    # Consumed WebSocket tokens (one-time use). Shared across both WebSocket
    # endpoint families (/ws and /ws-client).
    ws_used_tokens: set[str] = set()

    _setup_frontend_routes(app)
    _setup_webrtc_routes(app, args, active_sessions)
    _setup_daily_routes(app, args)
    _setup_telephony_routes(app, args, ws_used_tokens)
    _setup_websocket_routes(app, args, ws_used_tokens)
    _setup_unified_start_route(app, args, active_sessions)

    if args.whatsapp:
        _setup_whatsapp_routes(app, args)


def _setup_unified_start_route(
    app: FastAPI, args: argparse.Namespace, active_sessions: dict[str, dict[str, Any]]
):
    """Register the unified POST /start and GET /status endpoints.

    Handles WebRTC, Daily, and telephony transport start flows. Clients specify
    which transport they want via the ``transport`` field in the request body.
    When ``-t`` was passed on the command line, requests for any other transport
    are rejected with HTTP 400.
    """
    ALL_TRANSPORTS = ["webrtc", "daily", *TELEPHONY_TRANSPORTS, "websocket"]

    @app.get("/status")
    async def status():
        """Return the transports supported by this runner instance."""
        transports = [args.transport] if args.transport is not None else ALL_TRANSPORTS
        return {"status": "ready", "transports": transports}

    class IceServer(TypedDict, total=False):
        urls: str | list[str]

    class IceConfig(TypedDict):
        iceServers: list[IceServer]

    class StartBotResult(TypedDict, total=False):
        sessionId: str
        iceConfig: IceConfig | None
        dailyRoom: str | None
        dailyToken: str | None
        wsUrl: str | None
        token: str | None

    @app.post("/start")
    async def start_agent(request: Request):
        """Start a bot session.

        Accepts::

            {
                "transport": "webrtc",        // "webrtc" | "daily" | "twilio" | "telnyx" |
                                              // "plivo" | "exotel" — default: "webrtc"

                // WebRTC-specific
                "enableDefaultIceServers": false,
                "body": {...},

                // Daily-specific
                "createDailyRoom": true,
                "dailyRoomProperties": {...},
                "dailyMeetingTokenProperties": {...},
                "body": {...}
            }
        """
        try:
            request_data = await request.json()
            logger.debug(f"Received request: {request_data}")
        except Exception as e:
            logger.error(f"Failed to parse request body: {e}")
            request_data = {}

        # Determine transport: explicit field → legacy Daily hint → CLI default → webrtc
        transport = request_data.get("transport")
        if transport is None and request_data.get("createDailyRoom", False):
            transport = "daily"
        if transport is None:
            transport = args.transport or "webrtc"

        # Enforce restriction when -t was explicitly set on the command line
        if args.transport is not None and transport != args.transport:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Transport '{transport}' is not allowed. "
                    f"Server is configured for '{args.transport}' only (-t {args.transport})."
                ),
            )

        if not _transport_routes_enabled(transport):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Transport '{transport}' is disabled in this runner environment. "
                    "Check the startup banner for enabled transports."
                ),
            )

        if transport == "webrtc":
            # WebRTC: register the session; the bot starts when the WebRTC offer arrives.
            session_id = str(uuid.uuid4())
            active_sessions[session_id] = request_data.get("body", {})

            result = StartBotResult(
                sessionId=session_id,
            )
            if request_data.get("enableDefaultIceServers"):
                result["iceConfig"] = IceConfig(
                    iceServers=[IceServer(urls=["stun:stun.l.google.com:19302"])]
                )
            return result

        elif transport == "daily":
            create_daily_room = request_data.get("createDailyRoom", False)
            body = request_data.get("body", {})
            daily_room_properties_dict = request_data.get("dailyRoomProperties", None)
            daily_token_properties_dict = request_data.get("dailyMeetingTokenProperties", None)

            bot_module = _get_bot_module()

            existing_room_url = os.getenv("DAILY_ROOM_URL")
            session_id = str(uuid.uuid4())
            result: StartBotResult | None = None

            if create_daily_room or existing_room_url:
                from pipecat.runner.daily import configure
                from pipecat.transports.daily.utils import (
                    DailyMeetingTokenProperties,
                    DailyRoomProperties,
                )

                async with aiohttp.ClientSession() as session:
                    room_properties = None
                    if daily_room_properties_dict:
                        daily_room_properties_dict.setdefault(
                            "exp", time.time() + PIPECAT_ROOM_EXP_HOURS * 3600
                        )
                        daily_room_properties_dict.setdefault("eject_at_room_exp", True)
                        try:
                            room_properties = DailyRoomProperties(**daily_room_properties_dict)
                            logger.debug(f"Using custom room properties: {room_properties}")
                        except Exception as e:
                            logger.error(f"Failed to parse dailyRoomProperties: {e}")

                    token_properties = None
                    if daily_token_properties_dict:
                        try:
                            token_properties = DailyMeetingTokenProperties(
                                **daily_token_properties_dict
                            )
                            logger.debug(f"Using custom token properties: {token_properties}")
                        except Exception as e:
                            logger.error(f"Failed to parse dailyMeetingTokenProperties: {e}")

                    room_url, token = await configure(
                        session,
                        room_exp_duration=PIPECAT_ROOM_EXP_HOURS,
                        room_properties=room_properties,
                        token_properties=token_properties,
                    )
                    runner_args = DailyRunnerArguments(
                        room_url=room_url, token=token, body=body, session_id=session_id
                    )
                    result = StartBotResult(
                        dailyRoom=room_url,
                        dailyToken=token,
                        sessionId=session_id,
                    )
            else:
                runner_args = RunnerArguments(body=body, session_id=session_id)

            runner_args.cli_args = args
            asyncio.create_task(bot_module.bot(runner_args))
            return result

        elif transport in TELEPHONY_TRANSPORTS:
            # Telephony: the bot starts when the provider connects to /ws.
            # Return the WebSocket URL so the caller knows where to point their provider.
            scheme = "wss" if args.host != "localhost" else "ws"
            result = StartBotResult(wsUrl=f"{scheme}://{args.host}:{args.port}/ws")
            if args.ws_auth == "token":
                result["token"] = _generate_ws_token()
            return result

        elif transport == "websocket":
            # Plain WebSocket: the bot starts when the client connects to /ws-client.
            scheme = "wss" if args.host != "localhost" else "ws"
            session_id = str(uuid.uuid4())
            token = _generate_ws_token() if args.ws_auth == "token" else None
            return StartBotResult(
                wsUrl=f"{scheme}://{args.host}:{args.port}/ws-client",
                sessionId=session_id,
                token=token,
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown transport '{transport}'.",
            )


def _resolve_download_path(folder: str, filename: str) -> Path:
    """Resolve a download path and ensure it stays within the downloads folder."""
    allowed_base = Path(folder).resolve()
    file_path = (allowed_base / filename).resolve()

    if not file_path.is_relative_to(allowed_base):
        raise HTTPException(status_code=403, detail="Access denied")

    return file_path


def _setup_frontend_routes(app: FastAPI):
    """Mount the prebuilt frontend UI and root redirect for all transports."""
    try:
        from pipecat_ai_prebuilt.frontend import PipecatPrebuiltUI
    except ImportError as e:
        logger.error(f"Prebuilt frontend not available: {e}")
        return

    app.mount("/client", PipecatPrebuiltUI)

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        """Redirect root requests to client interface."""
        return RedirectResponse(url="/client/")


def _setup_webrtc_routes(
    app: FastAPI, args: argparse.Namespace, active_sessions: dict[str, dict[str, Any]]
):
    """Set up WebRTC-specific routes."""
    if not _transport_routes_enabled("webrtc"):
        return

    try:
        from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
        from pipecat.transports.smallwebrtc.request_handler import (
            IceCandidate,
            SmallWebRTCPatchRequest,
            SmallWebRTCRequest,
            SmallWebRTCRequestHandler,
        )
    except ImportError as e:
        logger.warning(f"WebRTC routes disabled after dependency check passed: {e}")
        return

    @app.get("/files/{filename:path}")
    async def download_file(filename: str):
        """Handle file downloads."""
        if not args.folder:
            logger.warning(f"Attempting to download {filename}, but downloads folder not setup.")
            raise HTTPException(404)

        file_path = _resolve_download_path(args.folder, filename)
        if not file_path.exists():
            raise HTTPException(404)

        media_type, _ = mimetypes.guess_type(file_path)

        return FileResponse(path=file_path, media_type=media_type, filename=file_path.name)

    # Initialize the SmallWebRTC request handler
    small_webrtc_handler: SmallWebRTCRequestHandler = SmallWebRTCRequestHandler(
        esp32_mode=args.esp32, host=args.host
    )

    @app.post("/api/offer")
    async def offer(
        request: SmallWebRTCRequest,
        background_tasks: BackgroundTasks,
        session_id: str | None = None,
    ):
        """Handle WebRTC offer requests via SmallWebRTCRequestHandler."""
        # When called via the /sessions/{session_id}/api/offer proxy the
        # session_id is threaded through; for direct /api/offer calls we mint
        # one so bots see a stable identifier in either path.
        resolved_session_id = session_id or str(uuid.uuid4())

        # Prepare runner arguments with the callback to run your bot
        async def webrtc_connection_callback(connection: SmallWebRTCConnection):
            bot_module = _get_bot_module()

            runner_args = SmallWebRTCRunnerArguments(
                webrtc_connection=connection,
                body=request.request_data,
                session_id=resolved_session_id,
            )
            runner_args.cli_args = args
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

    @app.api_route(
        "/sessions/{session_id}/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    )
    async def proxy_request(
        session_id: str, path: str, request: Request, background_tasks: BackgroundTasks
    ):
        """Mimic Pipecat Cloud's proxy."""
        active_session = active_sessions.get(session_id)
        if active_session is None:
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
                        request_data=request_data.get("request_data")
                        or request_data.get("requestData")
                        or active_session,
                    )
                    return await offer(webrtc_request, background_tasks, session_id=session_id)
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


def _setup_whatsapp_routes(app: FastAPI, args: argparse.Namespace):
    """Set up WhatsApp-specific routes."""
    required_vars = [
        "WHATSAPP_APP_SECRET",
        "WHATSAPP_PHONE_NUMBER_ID",
        "WHATSAPP_TOKEN",
        "WHATSAPP_WEBHOOK_VERIFICATION_TOKEN",
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        missing_list = "\n    ".join(missing)
        logger.error(
            f"""Missing required environment variables for WhatsApp transport:
    {missing_list}
            """
        )
        return

    WHATSAPP_APP_SECRET = os.environ["WHATSAPP_APP_SECRET"]
    WHATSAPP_PHONE_NUMBER_ID = os.environ["WHATSAPP_PHONE_NUMBER_ID"]
    WHATSAPP_TOKEN = os.environ["WHATSAPP_TOKEN"]
    WHATSAPP_WEBHOOK_VERIFICATION_TOKEN = os.environ["WHATSAPP_WEBHOOK_VERIFICATION_TOKEN"]

    try:
        from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
        from pipecat.transports.whatsapp.api import WhatsAppConnectCall, WhatsAppWebhookRequest
        from pipecat.transports.whatsapp.client import WhatsAppClient
    except ImportError as e:
        logger.error(f"WhatsApp transport dependencies not installed: {e}")
        return

    # Global WhatsApp client instance
    whatsapp_client: WhatsAppClient | None = None

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

        async def connection_callback(connection: SmallWebRTCConnection, call: WhatsAppConnectCall):
            """Handle new WebRTC connections from WhatsApp calls.

            Called when a WebRTC connection is established for a WhatsApp call.
            Spawns a bot instance to handle the conversation.

            Args:
                connection: The established WebRTC connection.
                call: The WhatsApp call metadata (caller phone number, call ID,
                    direction, timestamp, etc.), passed as ``runner_args.body``.
            """
            bot_module = _get_bot_module()
            runner_args = SmallWebRTCRunnerArguments(
                webrtc_connection=connection,
                session_id=str(uuid.uuid4()),
                body=call,
            )
            runner_args.cli_args = args
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


def _setup_daily_routes(app: FastAPI, args: argparse.Namespace):
    """Set up Daily-specific routes."""
    if not _transport_routes_enabled("daily"):
        return

    @app.get("/daily")
    async def create_room_and_start_agent():
        """Launch a Daily bot and redirect to room."""
        logger.debug("Starting bot with Daily transport and redirecting to Daily room")

        from pipecat.runner.daily import configure

        async with aiohttp.ClientSession() as session:
            room_url, token = await configure(session, room_exp_duration=PIPECAT_ROOM_EXP_HOURS)

            # Start the bot in the background with empty body for GET requests
            bot_module = _get_bot_module()
            runner_args = DailyRunnerArguments(
                room_url=room_url, token=token, session_id=str(uuid.uuid4())
            )
            runner_args.cli_args = args
            asyncio.create_task(bot_module.bot(runner_args))
            return RedirectResponse(room_url)

    if args.dialin:

        @app.post("/daily-dialin-webhook")
        async def handle_dialin_webhook(request: Request):
            """Handle incoming Daily PSTN dial-in webhook.

            This endpoint mimics Pipecat Cloud's dial-in webhook handler.
            It receives Daily webhook data, creates a SIP-enabled room, and starts the bot.

            Expected webhook payload::

                {
                    "From": "+15551234567",
                    "To": "+15559876543",
                    "callId": "uuid-call-id",
                    "callDomain": "uuid-call-domain",
                    "sipHeaders": {...}  // optional
                }

            Returns::

                {
                    "dailyRoom": "https://...",
                    "dailyToken": "...",
                    "sessionId": "uuid"
                }
            """
            logger.debug("Received Daily dial-in webhook")

            try:
                data = await request.json()
                logger.debug(f"Webhook data: {data}")
            except Exception as e:
                logger.error(f"Failed to parse webhook data: {e}")
                raise HTTPException(status_code=400, detail="Invalid JSON payload")

            # Handle webhook verification test (sent by Daily when configuring webhook)
            if data.get("test") or data.get("Test"):
                logger.debug("Webhook verification test received")
                return {"status": "OK"}

            # Validate required fields
            if not all(key in data for key in ["From", "To", "callId", "callDomain"]):
                raise HTTPException(
                    status_code=400,
                    detail="Missing required fields: From, To, callId, callDomain",
                )

            from pipecat.runner.daily import configure
            from pipecat.runner.types import DailyDialinRequest, DialinSettings

            # Create Daily room with SIP capabilities
            async with aiohttp.ClientSession() as session:
                try:
                    room_config = await configure(
                        session,
                        sip_caller_phone=data.get("From"),
                        room_exp_duration=PIPECAT_ROOM_EXP_HOURS,
                    )
                except Exception as e:
                    logger.error(f"Failed to create Daily room: {e}")
                    raise HTTPException(
                        status_code=500, detail=f"Failed to create Daily room: {str(e)}"
                    )

            # Get Daily API URL from environment, fallback to production
            daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

            # Get Daily API key from environment
            daily_api_key = os.getenv("DAILY_API_KEY")
            if not daily_api_key:
                logger.error("DAILY_API_KEY not found in environment")
                raise HTTPException(
                    status_code=500, detail="DAILY_API_KEY not configured on server"
                )

            # Prepare dial-in settings matching Pipecat Cloud structure
            dialin_settings = DialinSettings(
                call_id=data.get("callId"),
                call_domain=data.get("callDomain"),
                To=data.get("To"),
                From=data.get("From"),
                sip_headers=data.get("sipHeaders"),
            )

            # Create request body matching Pipecat Cloud payload
            request_body = DailyDialinRequest(
                dialin_settings=dialin_settings,
                daily_api_key=daily_api_key,
                daily_api_url=daily_api_url,
            )

            # Generate session ID for both the runner args and the response
            session_id = str(uuid.uuid4())

            # Start bot with dial-in context
            bot_module = _get_bot_module()
            runner_args = DailyRunnerArguments(
                room_url=room_config.room_url,
                token=room_config.token,
                body=request_body.model_dump(),
                session_id=session_id,
            )
            runner_args.cli_args = args

            asyncio.create_task(bot_module.bot(runner_args))

            # Return response matching Pipecat Cloud format
            return {
                "dailyRoom": room_config.room_url,
                "dailyToken": room_config.token,
                "sessionId": session_id,
            }


def _setup_telephony_routes(app: FastAPI, args: argparse.Namespace, ws_used_tokens: set[str]):
    """Set up telephony-specific routes.

    The WebSocket endpoint (``/ws``) is always registered so providers can
    connect directly. The XML webhook (``POST /``) is only registered when a
    specific telephony transport is chosen via ``-t`` because the XML template
    is provider-specific and requires a proxy hostname (``--proxy``).

    When ``args.ws_auth == "token"``, connections must present a valid HMAC
    session token obtained via ``POST /start``. The token may be supplied as:

    - ``Authorization: Bearer <token>`` header
    - ``?token=<token>`` query parameter
    - URL path segment: ``/ws/<token>`` (recommended for telephony providers)

    Invalid or missing tokens are rejected with WebSocket close code 4003.
    """
    if not _transport_routes_enabled("telephony"):
        return

    if args.transport in TELEPHONY_TRANSPORTS:
        # XML response templates (Exotel doesn't use XML webhooks)
        XML_TEMPLATES = {
            "twilio": f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{args.proxy}/ws"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>""",
            "telnyx": f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{args.proxy}/ws" bidirectionalMode="rtp"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>""",
            "plivo": f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">wss://{args.proxy}/ws</Stream>
</Response>""",
        }

        @app.post("/")
        async def start_call():
            """Handle telephony webhook and return XML response."""
            if args.transport == "exotel":
                # Exotel doesn't use POST webhooks - redirect to proper documentation
                logger.debug("POST Exotel endpoint - not used")
                return {
                    "error": "Exotel doesn't use POST webhooks",
                    "websocket_url": f"wss://{args.proxy}/ws",
                    "note": "Configure the WebSocket URL above in your Exotel App Bazaar Voicebot Applet",
                }
            else:
                logger.debug(f"POST {args.transport.upper()} XML")
                xml_content = XML_TEMPLATES.get(args.transport, "<Response></Response>")
                return HTMLResponse(content=xml_content, media_type="application/xml")

    async def _handle_telephony_ws(websocket: WebSocket, path_token: str | None = None):
        if args.ws_auth == "token":
            token = path_token or _extract_ws_token(websocket)
            if not token or not _verify_and_consume_ws_token(ws_used_tokens, token):
                logger.warning("WebSocket connection rejected: invalid or missing token")
                await websocket.close(code=4003)
                return
        origin = websocket.headers.get("origin", "")
        if not is_origin_allowed(origin, args.allowed_origins):
            logger.warning(f"WebSocket connection rejected: origin '{origin}' not allowed")
            await websocket.close(code=4003)
            return
        await websocket.accept()
        logger.debug("WebSocket connection accepted")
        await _run_telephony_bot(websocket, args)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Handle WebSocket connections for telephony."""
        await _handle_telephony_ws(websocket)

    @app.websocket("/ws/{token}")
    async def websocket_endpoint_with_token(websocket: WebSocket, token: str):
        """Handle WebSocket connections for telephony with token in the URL path."""
        await _handle_telephony_ws(websocket, path_token=token)


async def _run_daily_direct(args: argparse.Namespace):
    """Run Daily bot with direct connection (no FastAPI server)."""
    try:
        from pipecat.runner.daily import configure
    except ImportError as e:
        logger.error("Daily transport dependencies not installed.")
        return

    logger.info("Running with direct Daily connection...")

    async with aiohttp.ClientSession() as session:
        room_url, token = await configure(session, room_exp_duration=PIPECAT_ROOM_EXP_HOURS)

        # Direct connections have no request body, so use empty dict
        runner_args = DailyRunnerArguments(
            room_url=room_url, token=token, session_id=str(uuid.uuid4())
        )
        runner_args.handle_sigint = True
        runner_args.cli_args = args

        # Get the bot module and run it directly
        bot_module = _get_bot_module()

        print(f"📞 Joining Daily room: {room_url}")
        print("   (Direct connection - no web server needed)")
        print()

        await bot_module.bot(runner_args)


async def _run_eval(args: argparse.Namespace):
    """Run a bot with the eval transport (no FastAPI server).

    The eval transport is a ``SingleClientWebsocketServerTransport`` speaking RTVI that
    hosts its own local WebSocket server for the harness to connect to. The
    dev runner here just constructs ``EvalRunnerArguments`` and invokes the bot
    function directly — no FastAPI routes are needed.
    """
    logger.info("Running with eval transport...")

    runner_args = EvalRunnerArguments(host=args.host, port=args.port, session_id=str(uuid.uuid4()))
    runner_args.handle_sigint = True
    runner_args.cli_args = args

    # A bot may need session data it would normally receive in the /start request
    # body (e.g. a vision bot's image path). The eval transport has no such
    # endpoint, so the body is read from a JSON file passed with --runner-body.
    if args.runner_body:
        runner_args.body = json.loads(Path(args.runner_body).read_text())

    bot_module = _get_bot_module()
    await bot_module.bot(runner_args)


async def _run_vonage():
    """Run Vonage bot (no FastAPI server)."""
    logger.info("Running Vonage transport...")

    application_id, session_id, token = await configure_vonage()
    runner_args = VonageRunnerArguments(
        application_id=application_id, vonage_session_id=session_id, token=token
    )
    runner_args.handle_sigint = True

    # Get the bot module and run it directly
    bot_module = _get_bot_module()

    print(f"Joining Vonage session: {runner_args.vonage_session_id}")
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


def runner_downloads_folder() -> str | None:
    """Returns the folder where files are stored for later download."""
    return RUNNER_DOWNLOADS_FOLDER


def runner_host() -> str:
    """Returns the host name of this runner."""
    return RUNNER_HOST


def runner_port() -> int:
    """Returns the port of this runner."""
    return RUNNER_PORT


def main(parser: argparse.ArgumentParser | None = None):
    """Start the Pipecat development runner.

    Parses command-line arguments and starts a FastAPI server that supports
    WebRTC, Daily, and telephony transports simultaneously. Clients declare
    which transport to use via the ``transport`` field in the ``/start`` body.

    When ``-t`` is provided, the server restricts ``/start`` to that transport
    only and displays transport-specific startup information.

    The runner discovers and runs any ``bot(runner_args)`` function found in the
    calling module.

    Command-line arguments:
       - --host: Server host address (default: localhost)
       - --port: Server port (default: 7860)
       - -t/--transport: Restrict to a single transport and set as default for /start
         (daily, webrtc, websocket, twilio, telnyx, plivo, exotel). Omit to support
         all transports.
       - -x/--proxy: Public proxy hostname for telephony webhooks
       - -d/--direct: Connect directly to Daily room (automatically sets transport to daily)
       - -f/--folder: Path to downloads folder
       - --dialin/--no-dialin: Mount the Daily PSTN dial-in webhook for -t daily
         (on by default; --no-dialin disables it)
       - --esp32: Enable SDP munging for ESP32 compatibility (requires --host with IP address)
       - --whatsapp: Ensure required WhatsApp environment variables are present
       - -v/--verbose: Increase logging verbosity

    Args:
        parser: Optional custom argument parser. If provided, default runner
            arguments are added to it so bots can define their own CLI
            arguments. Custom arguments should not conflict with the default
            ones. Custom args are accessible via `runner_args.cli_args`.

    """
    global RUNNER_DOWNLOADS_FOLDER, RUNNER_HOST, RUNNER_PORT

    if not parser:
        parser = argparse.ArgumentParser(description="Pipecat Development Runner")
    parser.add_argument("--host", type=str, default=RUNNER_HOST, help="Host address")
    parser.add_argument("--port", type=int, default=RUNNER_PORT, help="Port number")
    parser.add_argument(
        "-t",
        "--transport",
        type=str,
        choices=["daily", "eval", "vonage", "webrtc", "websocket", *TELEPHONY_TRANSPORTS],
        default=None,
        help=(
            "Restrict the server to a single transport and set it as the default for /start. "
            "Omit to support all transports simultaneously (default behaviour)."
        ),
    )
    parser.add_argument("-x", "--proxy", help="Public proxy host name")
    parser.add_argument(
        "-d",
        "--direct",
        action="store_true",
        default=False,
        help="Connect directly to Daily room (automatically sets transport to daily)",
    )
    parser.add_argument("-f", "--folder", type=str, help="Path to downloads folder")
    parser.add_argument(
        "--runner-body",
        type=str,
        default=None,
        help="Path to a JSON file with the runner args body (e.g. a vision bot's image path under -t eval)",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity"
    )
    parser.add_argument(
        "--dialin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mount the Daily PSTN dial-in webhook for -t daily. On by default (a local "
            "stand-in for Pipecat Cloud's dial-in handler); use --no-dialin to disable."
        ),
    )
    parser.add_argument(
        "--esp32",
        action="store_true",
        default=False,
        help="Enable SDP munging for ESP32 compatibility (requires --host with IP address)",
    )
    parser.add_argument(
        "--whatsapp",
        action="store_true",
        default=False,
        help="Ensure required WhatsApp environment variables are present",
    )
    parser.add_argument(
        "--ws-auth",
        dest="ws_auth",
        choices=["none", "token"],
        default=os.getenv("PIPECAT_WEBSOCKET_AUTH", "none"),
        help=(
            "WebSocket authentication mode. 'token' requires clients to call /start "
            "and obtain a signed HMAC session token before connecting to /ws or "
            "/ws-client. Defaults to the PIPECAT_WEBSOCKET_AUTH environment variable "
            "or 'none'."
        ),
    )
    _env_origins = [
        o.strip() for o in os.getenv("PIPECAT_ALLOWED_ORIGINS", "").split(",") if o.strip()
    ]
    parser.add_argument(
        "--allowed-origins",
        dest="allowed_origins",
        nargs="*",
        default=_env_origins,
        help=(
            "Allowed origins for HTTP and WebSocket connections (e.g. https://example.com). "
            "Omit or leave empty to allow all origins. "
            "Defaults to the PIPECAT_ALLOWED_ORIGINS environment variable "
            "(comma-separated)."
        ),
    )

    args = parser.parse_args()

    # Validate and clean proxy hostname
    if args.proxy:
        args.proxy = _validate_and_clean_proxy(args.proxy)

    # --direct implies Daily transport
    if args.direct:
        if args.transport is None or args.transport == "daily":
            args.transport = "daily"
        else:
            logger.error("--direct flag only works with Daily transport (-t daily)")
            return

    # Validate ESP32 requirements
    if args.esp32 and args.host == "localhost":
        logger.error("For ESP32, you need to specify `--host IP` so we can do SDP munging.")
        return

    # The dial-in webhook is mounted only inside _setup_daily_routes, so --dialin is a
    # no-op for non-Daily transports; nothing to validate here.

    # Log level
    logger.remove()
    logger.add(sys.stderr, level="TRACE" if args.verbose else "DEBUG")

    # Handle direct Daily connection (no FastAPI server)
    if args.direct:
        print()
        print("🚀 Connecting directly to Daily room...")
        print()

        # Run direct Daily connection
        asyncio.run(_run_daily_direct(args))
        return

    # Handle eval transport (no FastAPI server — the WebSocket server transport
    # runs its own WS server)
    if args.transport == "eval":
        print()
        print(f"🚀 Bot ready! (eval transport on ws://{args.host}:{args.port})")
        print()
        asyncio.run(_run_eval(args))
        return

    # Print startup message
    _print_startup_message(args)
    if args.transport == "vonage":
        asyncio.run(_run_vonage())
        print()
        return

    RUNNER_DOWNLOADS_FOLDER = args.folder
    RUNNER_HOST = args.host
    RUNNER_PORT = args.port

    # Configure the app with transport-specific routes
    _configure_server_app(args)

    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
