import json
import uuid
from enum import Enum
from typing import Any

from aiortc import RTCPeerConnection, RTCSessionDescription
from loguru import logger

from pipecat.utils.event_emitter import EventEmitter

SIGNALLING_TYPE = "signalling"


class SignallingMessage(Enum):
    RENEGOTIATE = "renegotiate"
    SEND_KEY_FRAME = "sendKeyFrame"


class SmallWebRTCConnection(EventEmitter):
    def __init__(self):
        super().__init__()
        self.pc = RTCPeerConnection()
        self.pc_id = "PeerConnection(%s)" % uuid.uuid4()
        self._setup_listeners()
        self._tracks = set()
        self._data_channel = None

    def _setup_listeners(self):
        @self.pc.on("datachannel")
        def on_datachannel(channel):
            self._data_channel = channel

            @channel.on("message")
            async def on_message(message):
                try:
                    json_message = json.loads(message)
                    await self.emit("appMessage", json_message)
                except Exception as e:
                    logger.exception(f"Error parsing JSON message {message}, {e}")

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state is {self.pc.connectionState}")
            await self.emit(self.pc.connectionState)
            if self.pc.connectionState == "failed":
                await self.close()

        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"Track {track.kind} received")
            self._tracks.add(track)
            await self.emit("track-started", track)

            @track.on("ended")
            async def on_ended():
                logger.info(f"Track {track.kind} ended")
                self._tracks.discard(track)
                await self.emit("track-ended", track)

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
            # logger.info(
            #    f"Transceiver: {transceiver}, Mid: {transceiver.mid}, Direction: {transceiver.direction}"
            # )
            # logger.info(f"Sender track: {transceiver.sender.track}")

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

    def audio_input_track(self):
        # Transceivers always appear in creation-order for both peers
        # For now we are only considering that we are going to have 02 transceivers,
        # one for audio and one for video
        audio_transceiver = self.pc.getTransceivers()[0]
        if not audio_transceiver:
            return None

        return audio_transceiver.receiver.track

    def video_input_track(self):
        # Transceivers always appear in creation-order for both peers
        # For now we are only considering that we are going to have 02 transceivers,
        # one for audio and one for video
        video_transceiver = self.pc.getTransceivers()[1]
        if not video_transceiver:
            return None

        return video_transceiver.receiver.track

    def tracks(self):
        return self._tracks

    def send_app_message(self, message: Any):
        if self._data_channel:
            json_message = json.dumps(message)
            self._data_channel.send(json_message)

    def request_key_frame(self):
        self.send_app_message(
            {"type": SIGNALLING_TYPE, "message": SignallingMessage.SEND_KEY_FRAME.value}
        )
