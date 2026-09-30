"""Asterisk ARI WebSocket serializer for Pipecat.

Handles G.711 mu-law (ulaw) audio at 8kHz sent by Asterisk's
chan_websocket / externalMedia over binary WebSocket frames.

chan_websocket ships in Asterisk 20.16.0, 21.11.0, 22.6.0 and 23.0.0 onwards,
and its control messages arrive as text frames in one of two formats. JSON is
preferred from 20.18.0, 22.8.0 and 23.2.0; plain text remains the default
everywhere and is deprecated. Both are read here, because which one arrives
depends on the operator's chan_websocket.conf or f(<format>) dialstring rather
than on anything this code controls.

https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/
"""

import json
import re
from typing import TYPE_CHECKING

from loguru import logger

from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.enums import EndTaskReason

if TYPE_CHECKING:
    from pipecat.serializers.call_strategies import HangupStrategy, TransferStrategy


# Every documented event name is SCREAMING_SNAKE_CASE. Requiring that shape
# keeps the plain-text branch from turning arbitrary text into a plausible
# looking event, so "unrecognised" in the log means what it says.
_EVENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _parse_control_message(data: str) -> dict | None:
    """Read a chan_websocket control message in either format it can arrive in.

    The plain-text form is the first token as the event name followed by
    space-separated key:value pairs::

        MEDIA_START connection_id:dograh channel:WebSocket/dograh/0x78b3...
            format:ulaw optimal_frame_size:160 ptime:20

    No value contains a space, so splitting on whitespace and then on the first
    colon is the whole grammar. Both forms are normalised to the same shape as
    the JSON one, so callers do not have to care which arrived.

    Returns None only when the message is neither - worth a warning, since it
    means the protocol has moved on.
    """
    text = data.strip()
    if not text:
        return None

    if text.startswith("{"):
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            return None
        return message if isinstance(message, dict) else None

    name, *pairs = text.split()
    if not _EVENT_NAME.match(name):
        return None

    message = {"event": name}
    for pair in pairs:
        key, separator, value = pair.partition(":")
        if separator:
            message[key] = value
    return message


class AsteriskFrameSerializer(FrameSerializer):
    """Serializer for Asterisk ARI WebSocket audio streaming.

    Asterisk's chan_websocket sends raw G.711 mu-law (ulaw) audio at 8kHz
    as binary WebSocket frames. Unlike Twilio, there is no JSON wrapper
    or base64 encoding — audio bytes are sent directly as binary frames.

    On EndFrame/CancelFrame, the serializer will hang up the channel
    via ARI REST API (DELETE /ari/channels/{channel_id}).
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for AsteriskFrameSerializer.

        Parameters:
            asterisk_sample_rate: Sample rate used by Asterisk, defaults to 8000 Hz (ulaw).
            sample_rate: Optional override for pipeline input sample rate.
            auto_hang_up: Whether to automatically terminate channel on EndFrame.
        """

        asterisk_sample_rate: int = 8000
        sample_rate: int | None = None
        auto_hang_up: bool = True

    def __init__(
        self,
        channel_id: str,
        ari_endpoint: str,
        app_name: str,
        app_password: str,
        transfer_strategy: "TransferStrategy | None" = None,
        hangup_strategy: "HangupStrategy | None" = None,
        params: InputParams | None = None,
    ):
        """Initialize the AsteriskFrameSerializer.

        Args:
            channel_id: The Asterisk channel ID.
            ari_endpoint: ARI REST endpoint URL (e.g. http://localhost:8088).
            app_name: ARI application name for authentication.
            app_password: ARI application password for authentication.
            transfer_strategy: Strategy for handling call transfers.
            hangup_strategy: Strategy for handling call hangups.
            params: Configuration parameters.
        """
        params = params or AsteriskFrameSerializer.InputParams()
        super().__init__(params)
        self._params: AsteriskFrameSerializer.InputParams = params

        self._channel_id = channel_id
        self._ari_endpoint = ari_endpoint
        self._app_name = app_name
        self._app_password = app_password
        self._transfer_strategy = transfer_strategy
        self._hangup_strategy = hangup_strategy

        self._asterisk_sample_rate = self._params.asterisk_sample_rate
        self._sample_rate = 0  # Pipeline input rate, set in setup()

        self._input_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )
        self._output_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )
        self._hangup_attempted = False
        self._transfer_attempted = False

    async def setup(self, setup: FrameProcessorSetup):
        """Sets up the serializer with pipeline configuration.

        Args:
            setup: Configuration object containing setup parameters.
        """
        self._sample_rate = self._params.sample_rate or setup.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Asterisk WebSocket format.

        Converts PCM audio to G.711 mu-law and sends as raw binary bytes.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as bytes (ulaw audio) or None if the frame isn't handled.
        """
        if isinstance(frame, (EndFrame, CancelFrame)):
            frame_reason = getattr(frame, "reason", None)
            logger.debug(f"Processing {type(frame).__name__} with reason: {frame_reason}")

            if frame_reason == EndTaskReason.TRANSFER_CALL.value and not self._transfer_attempted:
                self._transfer_attempted = True
                if self._transfer_strategy:
                    context = {
                        "channel_id": self._channel_id,
                        "ari_endpoint": self._ari_endpoint,
                        "app_name": self._app_name,
                        "app_password": self._app_password,
                    }
                    success = await self._transfer_strategy.execute_transfer(context)
                    if not success:
                        logger.error(f"Transfer strategy failed for channel {self._channel_id}")
                else:
                    logger.warning(
                        f"No transfer strategy configured for channel {self._channel_id}"
                    )
                return None
            elif (
                self._params.auto_hang_up
                and not self._hangup_attempted
                and frame_reason != EndTaskReason.TRANSFER_CALL.value
            ):
                self._hangup_attempted = True
                if self._hangup_strategy:
                    context = {
                        "channel_id": self._channel_id,
                        "ari_endpoint": self._ari_endpoint,
                        "app_name": self._app_name,
                        "app_password": self._app_password,
                    }
                    success = await self._hangup_strategy.execute_hangup(context)
                    if not success:
                        logger.error(f"Hangup strategy failed for channel {self._channel_id}")
                else:
                    logger.warning(f"No hangup strategy configured for channel {self._channel_id}")
                return None
        elif isinstance(frame, InterruptionFrame):
            # Asterisk doesn't have a buffer clear command over the audio websocket.
            # Returning None; the transport will stop sending audio.
            return None
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz mu-law for Asterisk
            serialized_data = await pcm_to_ulaw(
                data, frame.sample_rate, self._asterisk_sample_rate, self._output_resampler
            )
            if serialized_data is None or len(serialized_data) == 0:
                return None

            # Asterisk expects raw binary ulaw bytes (no JSON wrapper, no base64)
            return serialized_data

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Asterisk WebSocket data to Pipecat frames.

        Binary messages contain raw G.711 mu-law audio bytes.
        Text messages contain JSON control events (if any).

        Args:
            data: The raw WebSocket data from Asterisk.

        Returns:
            A Pipecat frame corresponding to the data, or None if unhandled.
        """
        if isinstance(data, bytes):
            # Binary message = raw ulaw audio bytes
            deserialized_data = await ulaw_to_pcm(
                data,
                self._asterisk_sample_rate,
                self._sample_rate,
                self._input_resampler,
            )
            if deserialized_data is None or len(deserialized_data) == 0:
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data,
                num_channels=1,  # Asterisk sends mono audio
                sample_rate=self._sample_rate,
            )
            return audio_frame
        else:
            # Text message = control event, JSON or plain text
            message = _parse_control_message(data if isinstance(data, str) else str(data))
            if message is None:
                logger.warning(f"Unrecognised control message from Asterisk: {data}")
                return None

            event = message.get("event") or message.get("type")
            if event == "MEDIA_START":
                self._log_media_start(message)
            else:
                # The payload can carry caller input (DTMF digits), so only the
                # event name is logged above DEBUG.
                logger.info(f"Asterisk WebSocket event: {event}")
                logger.debug(f"Asterisk WebSocket event payload: {message}")
            return None

    def _log_media_start(self, message: dict) -> None:
        """Record what Asterisk declares about the stream it is about to send.

        These are the terms of the audio: they are worth a line of their own
        because this serializer assumes them rather than negotiating them. A
        mismatch here means every sample is being decoded wrong, which is
        otherwise indistinguishable from a bad line.
        """
        declared_format = message.get("format")
        logger.info(
            f"Asterisk MEDIA_START: format={declared_format} "
            f"optimal_frame_size={message.get('optimal_frame_size')} "
            f"ptime={message.get('ptime')} channel_id={message.get('channel_id')}"
        )

        if declared_format and declared_format != "ulaw":
            logger.warning(
                f"Asterisk declared format {declared_format!r}, but this serializer "
                f"decodes ulaw. Audio will be garbled until the dialstring or "
                f"chan_websocket.conf asks for ulaw."
            )
