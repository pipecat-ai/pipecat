#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""DIDLogic WebSocket call serializer for Pipecat."""

import base64
import json
from typing import Any

from loguru import logger

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.serializers.base_serializer import FrameSerializer


class DidlogicFrameSerializer(FrameSerializer):
    """Serializer for the DIDLogic WebSocket call protocol.

        DIDLogic connects to the bot, which is therefore a WebSocket server. The same
        protocol carries both directions of call:

        - inbound, where somebody dialled a DIDLogic number pointed at this endpoint
        - outbound, placed through the Click2Call API with ``a_type: "stream"``

        Messages are JSON text frames. Audio is PCM16, 24 kHz, mono, base64 in
        ``payload``. The platform sends one 10 ms frame per ``media`` message; audio
        sent back may be any length and is buffered on the platform side, so this
        serializer does not chunk its output.

        The call is answered as soon as the platform has sent ``start``, so audio
        arrives without the bot being asked to announce itself.

        On an **outbound** call, audio flows while the destination is still ringing:
        the bot hears ringback, and anything it says goes into a phone nobody has
        picked up. ``answered`` is what says the person is there. Gate the greeting on
        it — see ``is_outbound`` and the example below.

        Example::

            serializer = DidlogicFrameSerializer()

            transport = FastAPIWebsocketTransport(
                websocket=websocket,
                params=FastAPIWebsocketParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    add_wav_header=False,
                    vad_analyzer=SileroVADAnalyzer(),
                    serializer=serializer,
                ),
            )

        With the development runner, select the ``websocket`` transport and name this
        serializer in the params factory::

            transport_params = {
                "websocket": lambda: FastAPIWebsocketParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    add_wav_header=False,
                    serializer=DidlogicFrameSerializer(),
                ),
            }

    Anything that reads the socket before the transport does — ``parse_telephony_websocket``
        among them — consumes the ``start`` this serializer otherwise learns the call
        from. Pass what it parsed to the constructor; audio, DTMF, ``answered`` and
        ``stop`` are unaffected either way.

        To greet only once the far end is listening::

            class Greeter(FrameProcessor):
                async def process_frame(self, frame, direction):
                    await super().process_frame(frame, direction)
                    if (
                        isinstance(frame, InputTransportMessageFrame)
                        and frame.message.get("event") == "answered"
                    ):
                        await self.push_frame(LLMRunFrame())
                    await self.push_frame(frame, direction)

        On an inbound call the caller is already on the line, so the greeting belongs
        right after the pipeline starts instead.
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for DidlogicFrameSerializer.

        Parameters:
            didlogic_sample_rate: Sample rate on the wire. The platform sends and
                expects 24000 Hz and does not negotiate; ``start`` carries the
                rate in use.
            sample_rate: Optional override for the pipeline input sample rate.
            auto_hang_up: Whether an ``EndFrame`` or ``CancelFrame`` ends the call
                with ``hangup``. Unlike the REST-based providers this needs no
                credentials: it is one more message on the open socket.
        """

        didlogic_sample_rate: int = 24000
        sample_rate: int | None = None
        auto_hang_up: bool = True

    def __init__(
        self,
        params: InputParams | None = None,
        *,
        call_id: str | None = None,
        from_number: str | None = None,
        to_number: str | None = None,
        direction: str | None = None,
    ):
        """Initialize the DidlogicFrameSerializer.

        Args:
            params: Configuration parameters.
            call_id: The call's identifier, for a caller that read ``start``
                before building the serializer. The development runner does,
                because it picks the provider from the handshake. Left unset, all
                four are learned from ``start`` instead.
            from_number: The calling party, on the same terms as ``call_id``.
            to_number: The called party, on the same terms as ``call_id``.
            direction: ``"outbound"`` on an outbound call, same terms again.
        """
        params = params or DidlogicFrameSerializer.InputParams()
        super().__init__(params)
        self._params: DidlogicFrameSerializer.InputParams = params

        self._didlogic_sample_rate = self._params.didlogic_sample_rate
        self._sample_rate = 0

        self._call_id = call_id
        self._from_number = from_number
        self._to_number = to_number
        self._direction = direction
        self._answered = False

        self._input_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )
        self._output_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )

    @property
    def call_id(self) -> str | None:
        """The call's identifier, as given in ``start``.

        On an outbound call this is the ``id`` the Click2Call API returned, so it
        is what correlates this socket with ``GET /api/v1/click2call/:id`` and
        with the ``click2call.*`` webhooks.
        """
        return self._call_id

    @property
    def from_number(self) -> str | None:
        """The calling party: the caller on an inbound call, the caller ID presented to the destination on an outbound one."""
        return self._from_number

    @property
    def to_number(self) -> str | None:
        """The called party: the DIDLogic number dialled, or the outbound destination."""
        return self._to_number

    @property
    def is_outbound(self) -> bool:
        """Whether this is an outbound call, and therefore whether ``answered`` is coming.

        False before ``start`` has arrived.
        """
        return self._direction == "outbound"

    @property
    def answered(self) -> bool:
        """Whether the far end has picked up.

        Always False on an outbound call until ``answered`` arrives. On an inbound
        call the caller is already on the line, so this stays False and means
        nothing — use ``is_outbound`` to tell the two apart.
        """
        return self._answered

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the serializer with pipeline configuration.

        Args:
            setup: Configuration object containing setup parameters.
        """
        self._sample_rate = self._params.sample_rate or setup.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize a Pipecat frame into a DIDLogic message.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            A JSON string, or None when the frame is not one this protocol carries.
        """
        if isinstance(frame, InterruptionFrame):
            # Drop whatever we have sent that has not played yet, so an
            # interrupted reply stops instead of finishing into the conversation.
            return json.dumps({"event": "clear"})

        if isinstance(frame, (EndFrame, CancelFrame)):
            if not self._params.auto_hang_up:
                return None
            return json.dumps({"event": "hangup"})

        if isinstance(frame, AudioRawFrame):
            payload = await self._output_resampler.resample(
                frame.audio, frame.sample_rate, self._didlogic_sample_rate
            )
            if not payload:
                return None
            return json.dumps(
                {"event": "media", "payload": base64.b64encode(payload).decode("utf-8")}
            )

        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize a DIDLogic message into a Pipecat frame.

        Args:
            data: The raw WebSocket message. The protocol is JSON text only;
                bytes are decoded as UTF-8 for transports that hand text frames
                over as bytes.

        Returns:
            A frame for the pipeline, or None when the message needs no frame of
            its own — ``start`` is recorded rather than forwarded.
        """
        message = self._parse(data)
        if message is None:
            return None

        event = message.get("event")

        if event == "media":
            return await self._deserialize_audio(message)

        if event == "start":
            self._handle_start(message)

            return None

        if event == "answered":
            self._answered = True
            logger.debug(f"DIDLogic call {self._call_id} answered by the destination")
            return InputTransportMessageFrame(message=message)

        if event == "dtmf":
            return self._deserialize_dtmf(message)

        if event == "stop":
            logger.debug(f"DIDLogic call {self._call_id} ended")
            return InputTransportMessageFrame(message=message)

        # The protocol is extensible and says to ignore what you do not know.
        logger.debug(f"Ignoring unknown DIDLogic event: {event}")
        return None

    def _parse(self, data: str | bytes) -> dict[str, Any] | None:
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning("Discarding a DIDLogic message that is not UTF-8 text")
                return None
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON message from DIDLogic: {data}")
            return None
        if not isinstance(message, dict):
            logger.warning(f"Discarding a DIDLogic message that is not an object: {data}")
            return None
        return message

    def _handle_start(self, message: dict[str, Any]) -> None:
        self._call_id = message.get("call_id")
        self._from_number = message.get("from")
        self._to_number = message.get("to")
        self._direction = message.get("direction")

        wire_rate = message.get("sample_rate")
        if isinstance(wire_rate, int) and wire_rate != self._didlogic_sample_rate:
            logger.debug(
                f"DIDLogic call {self._call_id} is {wire_rate} Hz, "
                f"not the configured {self._didlogic_sample_rate} Hz; following the call"
            )
            self._didlogic_sample_rate = wire_rate

        logger.debug(
            f"DIDLogic call {self._call_id} started: "
            f"{self._from_number} -> {self._to_number} "
            f"({self._direction or 'inbound'})"
        )

        return None

    async def _deserialize_audio(self, message: dict[str, Any]) -> Frame | None:
        payload = message.get("payload")
        if not payload:
            return None
        try:
            audio = base64.b64decode(payload)
        except (ValueError, TypeError):
            logger.warning("Discarding a DIDLogic media frame with a malformed payload")
            return None

        resampled = await self._input_resampler.resample(
            audio, self._didlogic_sample_rate, self._sample_rate
        )
        if not resampled:
            return None

        return InputAudioRawFrame(
            audio=resampled,
            num_channels=1,
            sample_rate=self._sample_rate,
        )

    def _deserialize_dtmf(self, message: dict[str, Any]) -> Frame | None:
        digit = message.get("digit")
        if digit is None:
            logger.warning(f"DTMF event received but no digit found: {message}")
            return None
        try:
            return InputDTMFFrame(KeypadEntry(str(digit)))
        except ValueError:
            logger.warning(f"Invalid DTMF digit received: {digit}")
            return None
