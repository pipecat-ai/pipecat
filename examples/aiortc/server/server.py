import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRelay
from fastapi import BackgroundTasks, FastAPI

from pipecat.transports.webrtc.webrtc_connection import PipecatWebRTCConnection

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

app = FastAPI()


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pipecat_connection = PipecatWebRTCConnection()

    # TODO need to do how we are going to fix this
    await run_new_bot(pipecat_connection)

    await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

    pcs.add(pipecat_connection)

    @pipecat_connection.on("closed")
    def handle_disconnected():
        logger.info("Discarding the peer connection.")
        pcs.discard(pipecat_connection)

    # background_tasks.add_task(run_new_bot, pipecat_connection)

    return pipecat_connection.get_answer()


async def run_new_bot(pipecat_connection: PipecatWebRTCConnection):
    logger.info("Setting up media handling for the bot")

    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    recorder = MediaBlackhole()
    await recorder.start()

    def handle_track(track: MediaStreamTrack):
        if track.kind == "audio":
            pipecat_connection.pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pipecat_connection.pc.addTrack(relay.subscribe(track))

    @pipecat_connection.on("connected")
    def on_connected():
        logger.info("Peer connection established.")

    @pipecat_connection.on("disconnected")
    async def on_disconnected():
        logger.info("Peer connection lost.")

    @pipecat_connection.on("track-started")
    def on_track_started(track: MediaStreamTrack):
        logger.info(f"Processing new track: {track.kind}")
        handle_track(track)

    @pipecat_connection.on("track-ended")
    async def on_track_ended(track):
        logger.info(f"Track ended: {track.kind}")
        await recorder.stop()

    # Checking in case already had some existent track
    for track in pipecat_connection.tracks():
        logger.info(f"handling existent track: {track.kind}")
        handle_track(track)


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

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
