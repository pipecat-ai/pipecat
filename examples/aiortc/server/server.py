import argparse
import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRelay

from pipecat.utils.event_emitter import EventEmitter

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

app = FastAPI()


class PipecatWebRTCConnection(EventEmitter):

    def __init__(self):
        super().__init__()
        self.pc = RTCPeerConnection()
        self.pc_id = "PeerConnection(%s)" % uuid.uuid4()
        self._setup_listeners()
        self._tracks = set()

    def _log_info(self, msg, *args):
        logger.info(self.pc_id + " " + msg, *args)

    def _setup_listeners(self):
        @self.pc.on("datachannel")
        def on_datachannel(channel):
            @channel.on("message")
            def on_message(message):
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send("pong" + message[4:])

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            self._log_info("Connection state is %s", self.pc.connectionState)
            self.emit(self.pc.connectionState)
            if self.pc.connectionState == "failed":
                await self.close()

        @self.pc.on("track")
        def on_track(track):
            self._log_info("Track %s received", track.kind)
            self._tracks.add(track)
            self.emit("track-started", track)

            @track.on("ended")
            async def on_ended():
                self._log_info("Track %s ended", track.kind)
                self._tracks.discard(track)
                self.emit("track-ended", track)

    async def initialize(self, sdp: str, type: str):
        offer = RTCSessionDescription(sdp=sdp, type=type)

        logger.info("create_peer_connection 00")
        await self.pc.setRemoteDescription(offer)
        logger.info("create_peer_connection 01")
        answer = await self.pc.createAnswer()
        logger.info("create_peer_connection 02")
        await self.pc.setLocalDescription(answer)
        logger.info("create_peer_connection 03")
        return self.pc

    async def close(self):
        if self.pc:
            await self.pc.close()

    def get_answer(self):
        return {"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type}

    def tracks(self):
        return self._tracks


class PipecatWebRTCUtil:
    @staticmethod
    async def create_new_webrtc_connection(sdp: str, type: str):
        connection = PipecatWebRTCConnection()
        await connection.create_peer_connection(RTCSessionDescription(sdp=sdp, type=type))
        return connection


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pipecat_connection = PipecatWebRTCConnection()
    await run_new_bot(pipecat_connection)

    await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

    pcs.add(pipecat_connection)
    @pipecat_connection.on("closed")
    def handle_disconnected():
        logger.info("Discarding the peer connection.")
        pcs.discard(pipecat_connection)


    #background_tasks.add_task(run_new_bot, pipecat_connection)

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
    parser.add_argument("--port", type=int, default=7860, help="Port for HTTP server (default: 7860)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
