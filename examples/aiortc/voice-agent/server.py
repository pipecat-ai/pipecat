import argparse
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from aiortc_bot import run_bot
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse

from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

# Load environment variables
load_dotenv(override=True)

logger = logging.getLogger("pc")

app = FastAPI()

pcs = set()


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pipecat_connection = SmallWebRTCConnection()
    await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

    pcs.add(pipecat_connection)

    @pipecat_connection.on("closed")
    async def handle_disconnected():
        logger.info("Discarding the peer connection.")
        pcs.discard(pipecat_connection)

    background_tasks.add_task(run_bot, pipecat_connection)

    return pipecat_connection.get_answer()


@app.get("/")
async def serve_index():
    return FileResponse("index.html")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC demo")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    uvicorn.run(app, host=args.host, port=args.port)
