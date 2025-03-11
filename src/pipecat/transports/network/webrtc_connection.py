import json
import uuid
from enum import Enum
from typing import Any, Optional

from aiortc import RTCPeerConnection, RTCSessionDescription
from loguru import logger

from pipecat.utils.event_emitter import EventEmitter

SIGNALLING_TYPE = "signalling"


class SignallingMessage(Enum):
    RENEGOTIATE = "renegotiate"


class SmallWebRTCConnection(EventEmitter):
    def __init__(self):
        super().__init__()
        self.answer: Optional[RTCSessionDescription] = None
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

        self.answer = await self.pc.createAnswer()

        return self.pc

    async def connect(self):
        await self.pc.setLocalDescription(self.answer)

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
        transceivers = self.pc.getTransceivers()
        if len(transceivers) > 0 and transceivers[0].sender:
            transceivers[0].sender.replaceTrack(track)
        else:
            logger.warning("Audio transceiver not found. Cannot replace audio track.")

    def replace_video_track(self, track):
        logger.info(f"Replacing video track {track.kind}")
        # Transceivers always appear in creation-order for both peers
        # For now we are only considering that we are going to have 02 transceivers,
        # one for audio and one for video
        transceivers = self.pc.getTransceivers()
        if len(transceivers) > 1 and transceivers[1].sender:
            transceivers[1].sender.replaceTrack(track)
        else:
            logger.warning("Video transceiver not found. Cannot replace video track.")

    async def close(self):
        if self.pc:
            await self.pc.close()

    def get_answer(self):
        if not self.answer:
            return None

        return {
            "sdp": self.answer.sdp,
            "type": self.answer.type,
            "pc_id": self.pc_id,
        }

    def is_connected(self):
        return self.pc.connectionState == "connected"

    def audio_input_track(self):
        # Transceivers always appear in creation-order for both peers
        # For now we are only considering that we are going to have 02 transceivers,
        # one for audio and one for video
        transceivers = self.pc.getTransceivers()
        if len(transceivers) == 0 or not transceivers[0].receiver:
            logger.warning("No audio transceiver is available")
            return None

        return transceivers[0].receiver.track

    def video_input_track(self):
        # Transceivers always appear in creation-order for both peers
        # For now we are only considering that we are going to have 02 transceivers,
        # one for audio and one for video
        transceivers = self.pc.getTransceivers()
        if len(transceivers) <= 1 or not transceivers[1].receiver:
            logger.warning("No video transceiver is available")
            return None

        return transceivers[1].receiver.track

    def tracks(self):
        return self._tracks

    def send_app_message(self, message: Any):
        if self._data_channel:
            json_message = json.dumps(message)
            self._data_channel.send(json_message)

    def renegotiate(self):
        self.send_app_message(
            {"type": SIGNALLING_TYPE, "message": SignallingMessage.RENEGOTIATE.value}
        )
