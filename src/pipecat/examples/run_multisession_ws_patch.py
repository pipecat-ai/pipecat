#
# PATCHED VERSION: Adds FastAPI startup/shutdown events for live captions WebSocket server
#

import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
from pipecat.transports.services.daily import DailyParams, DailyTransport

# Load environment variables
load_dotenv(override=True)

# --- Live captions WebSocket server global state ---
ws_server = None

# Ensure repository root (two levels up from this file) is on sys.path so that
# the top-level 'examples' package can be imported regardless of invocation.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.foundational.ollama_chatbot_rag_multi import run_example, transport_params
from examples.foundational.ws_handler import _ws_handler


def get_transport_client_id(transport: BaseTransport, client: Any) -> str:
    if isinstance(transport, SmallWebRTCTransport):
        return client.pc_id
    elif isinstance(transport, DailyTransport):
        return client["id"]
    logger.warning(f"Unable to get client id from unsupported transport {type(transport)}")
    return ""


# ... (other helper functions unchanged) ...


def run_example_webrtc(
    run_example: Callable,
    args: argparse.Namespace,
    params: TransportParams,
):
    logger.info("Running example with SmallWebRTCTransport...")

    from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

    app = FastAPI()

    # CORS setup for browser clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "https://ai.alexcovo.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store connections by pc_id
    pcs_map: Dict[str, SmallWebRTCConnection] = {}

    ice_servers = [
        IceServer(urls="stun:96.242.77.142:3478"),
        IceServer(
            urls="turn:96.242.77.142:3478?transport=udp",
            username="alex",
            credential="supersecret",
        ),
    ]

    # Mount the frontend at /
    app.mount("/client", SmallWebRTCPrebuiltUI)
    # Mount custom web UI if present
    web_dir = Path(__file__).resolve().parents[3] / "examples" / "foundational" / "web"
    if web_dir.exists():
        app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        # Prefer /web if UI folder exists, else fall back to /client/
        if any(r.mount_path == "/web" for r in app.router.routes if hasattr(r, "mount_path")):
            return RedirectResponse(url="/web/")
        return RedirectResponse(url="/client/")

    @app.post("/api/offer")
    async def offer(request: Request, background_tasks: BackgroundTasks):
        try:
            body = await request.json()
        except Exception:
            body = {}
        logger.debug(f"/api/offer payload received: {body}")
        if "sdp" not in body:
            return {}

        pc_id = body.get("pc_id")

        if pc_id and pc_id in pcs_map:
            pipecat_connection = pcs_map[pc_id]
            await pipecat_connection.renegotiate(
                sdp=body["sdp"],
                type=body["type"],
                restart_pc=body.get("restart_pc", False),
            )
        else:
            pipecat_connection = SmallWebRTCConnection(ice_servers)
            await pipecat_connection.initialize(sdp=body["sdp"], type=body["type"])

            @pipecat_connection.event_handler("closed")
            async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
                logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
                pcs_map.pop(webrtc_connection.pc_id, None)

            # Always use the correct TransportParams object, not a dict
            webrtc_params = transport_params["webrtc"]()  # Call the lambda to get the params object
            transport = SmallWebRTCTransport(
                params=webrtc_params, webrtc_connection=pipecat_connection
            )
            background_tasks.add_task(run_example, transport, args, False)

        answer = pipecat_connection.get_answer()
        pcs_map[answer["pc_id"]] = pipecat_connection
        return answer

    # --- FastAPI startup/shutdown events for ws_server ---
    import websockets

    global ws_server

    @app.on_event("startup")
    async def startup_event():
        global ws_server
        if _ws_handler is not None and ws_server is None:
            ws_server = await websockets.serve(_ws_handler, "0.0.0.0", 9876)
            logger.info("Live text WebSocket at ws://0.0.0.0:9876")

    @app.on_event("shutdown")
    async def shutdown_event():
        global ws_server
        if ws_server is not None:
            ws_server.close()
            await ws_server.wait_closed()
            ws_server = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield  # Run app
        coros = [pc.disconnect() for pc in pcs_map.values()]
        await asyncio.gather(*coros)
        pcs_map.clear()

    uvicorn.run(app, host=args.host, port=args.port)


# ... (rest of the file unchanged) ...

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-session Pipecat backend")
    parser.add_argument("--stateless", action="store_true", help="Use stateless RAG prompt construction")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server")
    parser.add_argument("--port", type=int, default=27880, help="Port for HTTP server")
    parser.add_argument("--engine", choices=["ollama", "mlx"], default="ollama", help="Language model backend")
    args = parser.parse_args()
    run_example_webrtc(run_example, args, transport_params)
