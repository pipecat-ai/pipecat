#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import importlib.util
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

# Load environment variables
load_dotenv(override=True)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("pipecat-server")

app = FastAPI()

# Store connections by pc_id
pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = ["stun:stun.l.google.com:19302"]

# Mount the frontend at /
app.mount("/client", SmallWebRTCPrebuiltUI)

# Store the run_bot function from the imported module
run_bot_func = None


def import_run_bot_from_file(file_path: str) -> Callable[[Any], Any]:
    """Dynamically import the run_bot function from a given file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Bot file not found: {file_path}")

    # Extract module name without extension
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Get the run_bot function
    if not hasattr(module, "run_bot"):
        raise AttributeError(f"No run_bot function found in {file_path}")

    return module.run_bot


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    global run_bot_func

    if not run_bot_func:
        raise RuntimeError("No bot file has been loaded")

    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"], type=request["type"], restart_pc=request.get("restart_pc", False)
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        background_tasks.add_task(run_bot_func, pipecat_connection)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.close() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC demo")
    parser.add_argument("bot_file", nargs="?", help="Path to the bot file")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Import the run_bot function from the specified file
    try:
        run_bot_func = import_run_bot_from_file(args.bot_file)
        logger.info(f"Successfully loaded bot from {args.bot_file}")
    except Exception as e:
        logger.error(f"Error loading bot file: {e}")
        sys.exit(1)

    uvicorn.run(app, host=args.host, port=args.port)
