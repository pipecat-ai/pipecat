import uuid

from aiortc import RTCPeerConnection, RTCSessionDescription
from loguru import logger

from pipecat.utils.event_emitter import EventEmitter


class PipecatWebRTCConnection(EventEmitter):
    def __init__(self):
        super().__init__()
        self.pc = RTCPeerConnection()
        self.pc_id = "PeerConnection(%s)" % uuid.uuid4()
        self._setup_listeners()
        self._tracks = set()

    def _setup_listeners(self):
        @self.pc.on("datachannel")
        def on_datachannel(channel):
            # TODO: we should probably refactor and remove it from here
            @channel.on("message")
            def on_message(message):
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send("pong to: " + message)

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", self.pc.connectionState)
            self.emit(self.pc.connectionState)
            if self.pc.connectionState == "failed":
                await self.close()

        @self.pc.on("track")
        def on_track(track):
            logger.info("Track %s received", track.kind)
            self._tracks.add(track)
            self.emit("track-started", track)

            @track.on("ended")
            async def on_ended():
                logger.info("Track %s ended", track.kind)
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
        return {
            "sdp": self.pc.localDescription.sdp,
            "type": self.pc.localDescription.type,
            "pc_id": self.pc_id,
        }

    def tracks(self):
        return self._tracks
