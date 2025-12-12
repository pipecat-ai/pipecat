#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Asterisk WebSocket channel frame serialization interfaces for Pipecat."""

from enum import Enum
import json

from pydantic import BaseModel
from loguru import logger

from pipecat.audio.utils import (
    alaw_to_pcm,
    create_stream_resampler,
    pcm_to_alaw,
    pcm_to_ulaw,
    ulaw_to_pcm,
)
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame, 
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    StartFrame
)
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.serializers.base_serializer import FrameSerializer

class FrameSerializerType(Enum):
    """There only one serialization format in for Asterisk WebSocket channel.

    Parameters:
        MIXED: Binary and text serialization format.
    """
    MIXED = "mixed"

class AsteriskWsFrameSerializer(FrameSerializer):
    """Serializer for Asterisk WebSocket channel frames.
    
    This serializer handles converting between Pipecat frames and Asterisk WebSocket
    channel. It provides raw audio conversion (BINARY), and basic signalling(TEXT): 
        Asterisk to pipecat: 
            - when DTMF detected on Asterisk websocket channel we send InputDTMFFrame to pipecat.
        Pipecat to Asterisk:
            - when an EndFrame or CancelFrame is processed we send HANGUP to Asterisk websocket channel.
            - when an InterruptionFrame is processed we send FLUSH_MEDIA to Asterisk websocket channel.
    """

    class InputParams(BaseModel):
        """Configuration parameters for AsteriskFrameSerializer.

        Parameters:
            asterisk_sample_rate: Sample rate used by Asterisk, defaults to 8000 Hz.
            encoding: Audio encoding (e.g., "ulaw", "alaw").
        """
        asterisk_sample_rate: int = 8000
        encoding: str = "ulaw"

    def __init__(
        self,
        params: AsteriskWsFrameSerializer.InputParams | None = None,
    ):
        """Initialize the AsteriskFrameSerializer.

        Args:
            params: Configuration parameters.
        """
        self._params = params or AsteriskWsFrameSerializer.InputParams()

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()

        self._encoders = {
                "ulaw": pcm_to_ulaw,
                "alaw": pcm_to_alaw,
            }

        self._decoders = {
                "ulaw": ulaw_to_pcm,
                "alaw": alaw_to_pcm,
            }
        self._pipeline_sample_rate = 0  # Pipeline input rate populated during setup
        self._media_buffering_started = False


    @property
    def type(self) -> FrameSerializerType:
        """Get the serialization type supported by this serializer.

        Returns:
            Always returns FrameSerializerType.MIXED.
            It's not used by the serializer websocket transport and defined for compatibility with base class.
        """
        return FrameSerializerType.MIXED

    async def setup(self, frame: StartFrame):
        """Initialize the serializer with startup configuration.
        Defined to set the pipeline input sample rate for resampling.

        Args:
            frame: StartFrame containing initialization parameters.
        """
        self._pipeline_sample_rate = frame.audio_in_sample_rate


    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Convert a frame to its serialized representation.

        Args:
            frame: The frame to serialize.

        Returns:
            Serialized frame data as string, bytes, or None if serialization fails.
        """
        
        if isinstance(frame, AudioRawFrame):
            data = frame.audio

            encoding = self._params.encoding.strip().lower()

            try:
                encoder = self._encoders[encoding]
            except KeyError as e:
                raise ValueError(f"Unsupported encoding: {self._params.encoding}") from e

            serialized_data = await encoder(
                data,
                frame.sample_rate,
                self._params.asterisk_sample_rate,
                self._output_resampler,
            )

            if serialized_data is None or len(serialized_data) == 0:
                return None

            # Asterisk media WebSocket channels can require "START_MEDIA_BUFFERING" to enable audio buffering before sending it to Asterisk core.
            # When media buffering is enabled, we can send raw binary audio in messages of arbitrary sizes, and Asterisk will frame them properly 
            # and it will generate silence to the channel if the buffer is empty and there is no audio to send. 
            # For our use case, it is crucial, as we can send audio frames of arbitrary sizes from Pipecat to Asterisk without worrying about framing.
            # Without media buffering enabled, Asterisk expects audio frames of specific sizes and timings, which complicates the implementation a lot, 
            # and probably not a good idea in general for TCP-based transports such as websockets.
            # Therefore, we always send the "START_MEDIA_BUFFERING" command on before the first audio frame we send to Asterisk.
            if not self._media_buffering_started:
                self._media_buffering_started = True
                return ("START_MEDIA_BUFFERING", serialized_data)

            return serialized_data
        
        elif isinstance(frame, InterruptionFrame):
            return "FLUSH_MEDIA"
        elif isinstance(frame, (EndFrame, CancelFrame)):
            return "HANGUP"

        # Return None for unhandled frames
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Convert serialized data from Asterisk's websocket channel back to a frame object.

        Args:
            data: Serialized frame data as string or bytes.

        Returns:
            Reconstructed Frame object, or None if deserialization fails.
        """

        if isinstance(data, bytes):
            # If data is bytes, it's audio data
            try:
                decoder = self._decoders[self._params.encoding.strip().lower()]
            except KeyError as e:
                raise ValueError(f"Unsupported encoding: {self._params.encoding}") from e
            
            deserialized_data = await decoder(
                data,
                self._params.asterisk_sample_rate,
                self._pipeline_sample_rate,
                self._input_resampler,
            )

            if deserialized_data is None or len(deserialized_data) == 0:
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate
            )

            return audio_frame

        elif isinstance(data, str):
            # if data is str, is't a signalling event, but there are two underlying formats in Asterisk websocket channel's messages.
            # Formats: f(json) and f(plain-text) are defined on Asterisk websocket channel's dialsting parameters.
            # Try to decode as json if fails, try to decode as plain text event
            try:
                message = json.loads(data)
                if message.get("event") == "DTMF_END":
                    digit = message.get("digit")
                    if digit:
                        try:
                            return InputDTMFFrame(KeypadEntry(digit))
                        except ValueError:
                            # Handle case where string doesn't match any enum value
                            logger.warning(f"Invalid DTMF digit received: {digit}")
                            return None
                if message.get("event") == "MEDIA_START":
                    # Media start event, can be ignored or handled as needed
                    # There are a few potentially usefult fields provided by Asterisk in the MEDIA_START event message:
                    #   connection_id: A UUID that will be set on the MEDIA_WEBSOCKET_CONNECTION_ID channel variable.
                    #   channel: The channel name.
                    #   channel_id: The channel's unique id.
                    #   format: The format set on the channel.
                    #   optimal_frame_size: Sending media to Asterisk of this size, or a multiple of this size, ensures the channel driver can properly retime and reframe the media for the best caller experience.
                    #   ptime: The packetization rate in milliseconds.
                    #   channel_variables: An object containing the variables currently set on the channel.
                    logger.info(f"Received MEDIA_START event from Asterisk: {message}")
                    return None
                else:
                    return None

            except json.JSONDecodeError:
                # Not a JSON message, try to decode as plain text event
                if "DTMF_END" in data:
                    # Example plain text DTMF_END message: "DTMF_END 5 duration=500"
                    parts = data.split(" ")
                    if len(parts) >= 2:
                        for part in parts[1:]:
                            if part.startswith("digit:"):
                                _, digit = part.split("digit:")
                                digit = digit.strip()
                                break
                        try:
                            return InputDTMFFrame(KeypadEntry(digit))
                        except ValueError:
                            # Handle case where string doesn't match any enum value
                            logger.warning(f"Invalid DTMF digit received from Asterisk: {digit}")
                            return None
                    else:
                        logger.warning(f"Malformed DTMF_END message from Asterisk: {data}")
                        return None