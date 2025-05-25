#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Callable, Dict, Mapping, Optional

import aiohttp
import uvicorn
from daily_runner import configure
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse
from loguru import logger
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
from pipecat.transports.services.daily import DailyParams, DailyTransport

# Load environment variables
load_dotenv(override=True)


def run_example_daily(
    run_example: Callable,
    args: argparse.Namespace,
    params: DailyParams,
):
    logger.info("Running example with DailyTransport...")

    async def run():
        async with aiohttp.ClientSession() as session:
            (room_url, token) = await configure(session)

            # Run example function with DailyTransport transport arguments.
            transport = DailyTransport(room_url, token, "Pipecat", params=params)
            await run_example(transport, args)

    asyncio.run(run())


def run_example_webrtc(
    run_example: Callable,
    args: argparse.Namespace,
    params: TransportParams,
):
    logger.info("Running example with SmallWebRTCTransport...")

    app = FastAPI()

    # Store connections by pc_id
    pcs_map: Dict[str, SmallWebRTCConnection] = {}

    ice_servers = [
        IceServer(
            urls="stun:stun.l.google.com:19302",
        )
    ]

    # Mount the frontend at /
    app.mount("/client", SmallWebRTCPrebuiltUI)

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/client/")

    @app.post("/api/offer")
    async def offer(request: dict, background_tasks: BackgroundTasks):
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
            pipecat_connection = SmallWebRTCConnection(ice_servers)
            await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

            @pipecat_connection.event_handler("closed")
            async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
                logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
                pcs_map.pop(webrtc_connection.pc_id, None)

            # Run example function with SmallWebRTC transport arguments.
            transport = SmallWebRTCTransport(params=params, webrtc_connection=pipecat_connection)
            background_tasks.add_task(run_example, transport, args)

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

    uvicorn.run(app, host=args.host, port=args.port)


def run_main(
    run_example: Callable,
    args: argparse.Namespace,
    transport_params: Mapping[str, Callable] = {},
):
    params = transport_params[args.transport]()
    match args.transport:
        case "daily":
            run_example_daily(run_example, args, params)
        case "webrtc":
            run_example_webrtc(run_example, args, params)


def main(
    run_example: Callable,
    *,
    parser: Optional[argparse.ArgumentParser] = None,
    transport_params: Mapping[str, Callable] = {},
):
    if not parser:
        parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host", default="localhost", help="Host for WebRTC HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for WebRTC HTTP server (default: 7860)"
    )
    parser.add_argument(
        "--transport",
        "-t",
        type=str,
        choices=["daily", "webrtc"],
        default="webrtc",
        help="Transport",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    # Log level
    logger.remove(0)
    logger.add(sys.stderr, level="TRACE" if args.verbose else "DEBUG")

    # Import the bot file
    run_main(run_example, args, transport_params)
