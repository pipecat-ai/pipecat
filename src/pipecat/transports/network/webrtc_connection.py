import asyncio
import json
import time
import uuid
from enum import Enum
from typing import Any, Optional

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from loguru import logger

from pipecat.utils.event_emitter import EventEmitter

SIGNALLING_TYPE = "signalling"


class SignallingMessage(Enum):
    RENEGOTIATE = "renegotiate"


class SmallWebRTCConnection(EventEmitter):
    def __init__(self, ice_servers=None):
        super().__init__()
        if ice_servers:
            self.ice_servers = [RTCIceServer(urls=server) for server in ice_servers]
        else:
            self.ice_servers = []
        self._connect_invoked = False
        self._initialize()

    def _initialize(self):
        logger.info("Initializing new peer connection")
        rtc_config = RTCConfiguration(iceServers=self.ice_servers)

        self.answer: Optional[RTCSessionDescription] = None
        self.pc = RTCPeerConnection(rtc_config)
        self.pc_id = "PeerConnection(%s)" % uuid.uuid4()
        self._setup_listeners()
        self._tracks = set()
        self._data_channel = None
        self._renegotiation_in_progress = False
        self._last_received_time = None

    def _setup_listeners(self):
        @self.pc.on("datachannel")
        def on_datachannel(channel):
            self._data_channel = channel

            @channel.on("message")
            async def on_message(message):
                try:
                    # aiortc does not provide any way so we can be aware when we are disconnected,
                    # so we are using this keep alive message as a way to implement that
                    if isinstance(message, str) and message.startswith("ping"):
                        self._last_received_time = time.time()
                    else:
                        json_message = json.loads(message)
                        await self.emit("appMessage", json_message)
                except Exception as e:
                    logger.exception(f"Error parsing JSON message {message}, {e}")

        # Despite the fact that aiortc provides this listener, they don't have a status for "disconnected"
        # So, in case we loose connection, this event will not be triggered
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            await self._handle_new_connection_state()

        # Despite the fact that aiortc provides this listener, they don't have a status for "disconnected"
        # So, in case we loose connection, this event will not be triggered
        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(
                f"Ice connection state is {self.pc.iceConnectionState}, connection is {self.pc.connectionState}"
            )

        @self.pc.on("icegatheringstatechange")
        async def on_icegatheringstatechange():
            logger.info(f"Ice gathering state is {self.pc.iceGatheringState}")

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

    async def _create_answer(self, sdp: str, type: str):
        offer = RTCSessionDescription(sdp=sdp, type=type)
        await self.pc.setRemoteDescription(offer)

        # For some reason, aiortc is not respecting the SDP for the transceivers to be sendrcv
        # so we are basically forcing it to act this way
        self.force_transceivers_to_send_recv()

        # this answer does not contain the ice candidates, which will be gathered later, after the setLocalDescription
        logger.info(f"Creating answer")
        local_answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(local_answer)
        logger.info(f"Setting the answer after the local description is created")
        self.answer = self.pc.localDescription

    async def initialize(self, sdp: str, type: str):
        await self._create_answer(sdp, type)

    async def connect(self):
        self._connect_invoked = True
        # If we already connected, trigger again the connected event
        if self.is_connected():
            await self.emit("connected", self)
            # We are renegotiating here, because likely we have loose the first video frames
            # and aiortc does not handle that pretty well.
            self.ask_to_renegotiate()

    async def renegotiate(self, sdp: str, type: str, restart_pc: bool = False):
        logger.info(f"Renegotiating {self.pc_id}")

        if restart_pc:
            await self.emit("disconnected", self)
            logger.info("Closing old peer connection")
            # removing the listeners to prevent the bot from closing
            self.pc.remove_all_listeners()
            await self.close()
            # we are initializing a new peer connection in this case.
            self._initialize()

        await self._create_answer(sdp, type)

        # Maybe we should refactor to receive a message from the client side when the renegotiation is completed.
        # or look at the peer connection listeners
        # but this is good enough for now for testing.
        async def delayed_task():
            await asyncio.sleep(2)
            self._renegotiation_in_progress = False

        asyncio.create_task(delayed_task())

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

    async def _handle_new_connection_state(self):
        state = self.pc.connectionState
        logger.info(f"Connection state changed to: {state}")
        await self.emit(state, self)
        if state == "failed":
            logger.warning("Connection failed, closing peer connection.")
            await self.close()

    # Despite the fact that aiortc provides this listener, they don't have a status for "disconnected"
    # So, there is no advantage in looking at self.pc.connectionState
    # That is why we are trying to keep our own state
    def is_connected(self):
        # If the small webrtc transport has never invoked to connect
        # we are acting like if we are not connected
        if not self._connect_invoked:
            return False

        if self._last_received_time is None:
            # if we have never received a message, it is probably because the client has not created a data channel
            # so we are going to trust aiortc in this case
            return self.pc.connectionState == "connected"
        # Checks if the last received ping was within the last 3 seconds.
        return (time.time() - self._last_received_time) < 3

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

    def ask_to_renegotiate(self):
        if self._renegotiation_in_progress:
            return

        self._renegotiation_in_progress = True
        self.send_app_message(
            {"type": SIGNALLING_TYPE, "message": SignallingMessage.RENEGOTIATE.value}
        )
