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
            logger.info(f"Connection state is {self.pc.connectionState}")
            await self.emit(self.pc.connectionState)
            if self.pc.connectionState == "failed":
                await self.close()

        @self.pc.on("track")
        def on_track(track):
            logger.info(f"Track {track.kind} received")
            self._tracks.add(track)
            self.emit("track-started", track)

            @track.on("ended")
            async def on_ended():
                logger.info(f"Track {track.kind} ended")
                self._tracks.discard(track)
                self.emit("track-ended", track)

    async def initialize(self, sdp: str, type: str):
        offer = RTCSessionDescription(sdp=sdp, type=type)
        await self.pc.setRemoteDescription(offer)

        # For some reason, aiortc is not respecting the SDP for the transceivers to be sendrcv
        # so we are basically forcing it to act this way
        self.force_transceivers_to_send_recv()

        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        return self.pc

    def force_transceivers_to_send_recv(self):
        for transceiver in self.pc.getTransceivers():
            transceiver.direction = "sendrecv"
            #logger.info(
            #    f"Transceiver: {transceiver}, Mid: {transceiver.mid}, Direction: {transceiver.direction}"
            #)
            #logger.info(f"Sender track: {transceiver.sender.track}")

    def replace_audio_track(self, track):
        logger.info(f"Replacing audio track {track.kind}")
        # Transceivers always appear in creation-order for both peers
        # For now we are only considering that we are going to have 02 transceivers,
        # one for audio and one for video
        audio_transceiver = self.pc.getTransceivers()[0]
        audio_transceiver.sender.replaceTrack(track)

    def replace_video_track(self, track):
        logger.info(f"Replacing video track {track.kind}")
        # Transceivers always appear in creation-order for both peers
        # For now we are only considering that we are going to have 02 transceivers,
        # one for audio and one for video
        video_transceiver = self.pc.getTransceivers()[1]
        video_transceiver.sender.replaceTrack(track)

    async def close(self):
        if self.pc:
            await self.pc.close()

    def get_answer(self):
        return {
            "sdp": self.pc.localDescription.sdp,
            "type": self.pc.localDescription.type,
            "pc_id": self.pc_id,
        }

    def is_connected(self):
        return self.pc.connectionState == "connected"

    def tracks(self):
        return self._tracks
