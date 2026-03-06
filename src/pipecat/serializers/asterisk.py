"""Asterisk ARI WebSocket serializer for Pipecat.

Handles G.711 mu-law (ulaw) audio at 8kHz sent by Asterisk's
chan_websocket / externalMedia over binary WebSocket frames.
"""

import json
from typing import TYPE_CHECKING, Optional

from loguru import logger

from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.enums import EndTaskReason

if TYPE_CHECKING:
    from pipecat.serializers.call_strategies import HangupStrategy, TransferStrategy


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
        sample_rate: Optional[int] = None
        auto_hang_up: bool = True

    def __init__(
        self,
        channel_id: str,
        ari_endpoint: str,
        app_name: str,
        app_password: str,
        transfer_strategy: Optional["TransferStrategy"] = None,
        hangup_strategy: Optional["HangupStrategy"] = None,
        params: Optional[InputParams] = None,
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
        super().__init__(params or AsteriskFrameSerializer.InputParams())

        self._channel_id = channel_id
        self._ari_endpoint = ari_endpoint
        self._app_name = app_name
        self._app_password = app_password
        self._transfer_strategy = transfer_strategy
        self._hangup_strategy = hangup_strategy

        self._asterisk_sample_rate = self._params.asterisk_sample_rate
        self._sample_rate = 0  # Pipeline input rate, set in setup()

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._hangup_attempted = False
        self._transfer_attempted = False

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

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
            # Text message = JSON control event
            try:
                message = json.loads(data)
                event = message.get("type") or message.get("event")
                logger.debug(f"Asterisk WebSocket event: {event} - {message}")
                return None
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON message from Asterisk: {data}")
                return None
