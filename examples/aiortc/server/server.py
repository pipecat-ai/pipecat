import argparse
import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRelay

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

app = FastAPI()


class PipecatWebRTCUtil:
    async def create_new_webrtc_connection(sdp:str, type:str):
        # Create WebRTC connection instance and process the offer
        pipecat_webrtc_connection = PipecatWebRTCConnection()
        await pipecat_webrtc_connection.create_peer_connection(RTCSessionDescription(sdp=sdp, type=type))
        return pipecat_webrtc_connection

class PipecatWebRTCConnection:
    def __init__(self):
        self.pc = None

    async def create_peer_connection(self, offer:RTCSessionDescription):
        pc = RTCPeerConnection()
        self.pc = pc
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        pcs.add(pc)

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

        log_info("Created")

        # prepare local media
        player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
        # TODO: in case we wish to record the audio
        # recorder = MediaRecorder(args.record_to)
        recorder = MediaBlackhole()

        @pc.on("datachannel")
        def on_datachannel(channel):
            @channel.on("message")
            def on_message(message):
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send("pong" + message[4:])

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            log_info("Track %s received", track.kind)

            if track.kind == "audio":
                pc.addTrack(player.audio)
                recorder.addTrack(track)
            elif track.kind == "video":
                # TODO: we probably don't need the mediarelay here, but keeping it here for now
                pc.addTrack(relay.subscribe(track))

            @track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)
                await recorder.stop()

        # handle offer
        await pc.setRemoteDescription(offer)
        await recorder.start()

        # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return pc

    async def close_connection(self):
        if self.pc:
            await self.pc.close()
            pcs.discard(self.pc)

    def get_answer(self):
        return {
            "sdp": self.pc.localDescription.sdp,
            "type": self.pc.localDescription.type
        }


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pipecat_connection = await PipecatWebRTCUtil.create_new_webrtc_connection(sdp=request["sdp"], type=request["type"])
    # This background task will run after sending the response
    background_tasks.add_task(run_new_bot, pipecat_connection)
    # Send the answer back to the client
    return pipecat_connection.get_answer()


async def run_new_bot(webrtc_connection: PipecatWebRTCConnection):
    logger.info("Should run the bot here passing the peer connection")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles FastAPI startup and shutdown."""
    yield  # Run app
    coros = [webrtc_connection.close_connection() for webrtc_connection in pcs]
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
