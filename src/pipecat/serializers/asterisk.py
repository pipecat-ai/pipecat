#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Asterisk WebSocket channel frame serialization interfaces for Pipecat."""

import json
from enum import Enum
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.dtmf.types import KeypadEntry
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
    InputTransportMessageFrame,
    InterruptionFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class FrameSerializerType(Enum):
    """There only one serialization format in for Asterisk WebSocket channel MIXED.

    It's not used in Asterisk WebSocket transport itself, but it's defined here for consistency with other serializers.

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
            encoding: Audio encoding (e.g., "ulaw", "alaw", "slin").
        """

        asterisk_sample_rate: int = 8000
        encoding: str = None  # None means autodetect from Asterisk MEDIA_START event

    def __init__(self, params: Optional[InputParams] = None):
        """Initialize the AsteriskFrameSerializer.

        Args:
            params: Configuration parameters.
        """
        self._asterisk_command_format = None  # Will be set to "json" or "plain-text" after receiving the first MEDIA_START event
        self._input_resampler = create_stream_resampler()
        self._media_buffering_started = False
        self._output_resampler = create_stream_resampler()
        self._params = params or AsteriskWsFrameSerializer.InputParams()
        self._pipeline_sample_rate = 0  # Will be populated during setup

        self._asterisk_event_handlers = {
            "MEDIA_START": self._handle_media_start,
            "MEDIA_XOFF": self._handle_media_xoff,
            "MEDIA_XON": self._handle_media_xon,
            "DTMF_END": self._handle_dtmf_end,
            "QUEUE_DRAINED": self._handle_queue_drained,
        }

        async def raw_media_passthrough(data, in_sr, out_sr, resampler):
            """Pass through raw media without any conversion but resample if needed."""
            return await resampler.resample(data, in_sr, out_sr)

        self._encoders = {
            "ulaw": pcm_to_ulaw,
            "alaw": pcm_to_alaw,
            "slin": raw_media_passthrough,
        }

        self._decoders = {
            "ulaw": ulaw_to_pcm,
            "alaw": alaw_to_pcm,
            "slin": raw_media_passthrough,
        }

    async def setup(self, frame: StartFrame):
        """Initialize the serializer with startup configuration.

        Defined to set the pipeline input sample rate for resampling.

        Args:
            frame: StartFrame containing initialization parameters.
        """
        self._pipeline_sample_rate = frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Convert a frame to its serialized representation sutable for Asterisk WebSocket channel.

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

            return serialized_data

        # Return None for unhandled frames
        return None

    def form_command(self, command: str) -> str:
        """Form a Asterisk WebSocket channel command based on the identified format."""
        if self._asterisk_command_format == "plain-text":
            return command
        else:
            return f'{{"command": "{command}"}}'

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
                audio=deserialized_data, num_channels=1, sample_rate=self._pipeline_sample_rate
            )

            return audio_frame

        elif isinstance(data, str):
            # Identify the format of signalling event from Asterisk websocket channel message based
            # on the first message in the channel: "MEDIA_START", it might be json or plain-text
            if self._asterisk_command_format is None and "MEDIA_START" in data:
                try:
                    message = json.loads(data)
                    if message.get("event") == "MEDIA_START":
                        self._asterisk_command_format = "json"
                except json.JSONDecodeError:
                    self._asterisk_command_format = "plain-text"

            # Parse the message based on the identified format
            event = {}
            if self._asterisk_command_format == "plain-text":
                event_entries = data.split(" ")
                event["event"] = event_entries[0]
                for entry in event_entries[1:]:
                    if ":" in entry:
                        key, value = entry.split(":", 1)
                        event[key] = value
            else:
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning(
                        f'"MEDIA_START" was in json-format, but we failed to parse the following Asterisk websocket message as JSON: {data}'
                    )
                    return None

            handler = self._asterisk_event_handlers.get(event.get("event"))
            if handler:
                return handler(event)
            else:
                # we don't have a handler for this type of event
                return None

    @property
    def type(self) -> FrameSerializerType:
        """Get the serialization type supported by this serializer.

        Returns:
            Always returns FrameSerializerType.MIXED.
            It's not used by the serializer websocket transport and defined for compatibility with base class.
        """
        return FrameSerializerType.MIXED

    # Asterisk event handlers
    def _handle_media_start(self, message: dict):
        """MEDIA_START event handler.

        MEDIA_START event is the first event we receive from Asterisk, we use it in deserialize method to identity which message format is used by Asterisk (json of plain-text).
        But besides that, if the encoding is not defined, we attempt to detect it from the MEDIA_START event.
        There are a few potentially useful parameters provided by Asterisk in the MEDIA_START event message:
          connection_id: A UUID that will be set on the MEDIA_WEBSOCKET_CONNECTION_ID channel variable.
          channel: The channel name on Asterisk.
          channel_id: The channel's unique id on Asterisk.
          format: The audio format set on the channel.
          optimal_frame_size: The optimal frame size from Astersisk's perspective. It's not important as we always use media buffering, Asterisk will reframe and retime the audio as needed.
          ptime: The packetization rate in milliseconds. It's not important as we always use media buffering, Asterisk will reframe and retime the audio as needed.
          channel_variables: An object containing the variables currently set on the channel. This can be very handy for moving data from dialplan variables to the pipeline.
        So we send MEDIA_START event object to the pipeline as InputTransportMessageFrame.

        Args:
            message: The dictionary representing of the MEDIA_START event message from Asterisk.
        """
        logger.info(f"Received MEDIA_START event from Asterisk: {message}")
        logger.info(
            f"Optimal frame size for TTS: {message.get('optimal_frame_size')} bytes, ptime: {message.get('ptime')} ms, you might want to adjust your output trnasport parameters accordingly."
        )
        if self._params.encoding is None:
            logger.info(
                "Encoding is not provided, detecting Asterisk audio encoding from MEDIA_START event..."
            )

            format = message.get("format", "").strip().lower()
            # asterisk slin formats are like "slin12" or "slin16" .. "slin192"
            if format.startswith("slin"):
                _, bitrate = format.split("slin")
                format = "slin"

            if format in self._decoders:
                self._params.encoding = format
                logger.info(f"Detected Asterisk audio encoding: {self._params.encoding}")
                if format.startswith("slin"):
                    if bitrate:
                        self._params.asterisk_sample_rate = int(bitrate) * 1000
                    else:
                        # in asterisk 'slin' assumes 8000 Hz if bitrate if not specified
                        self._params.asterisk_sample_rate = 8000

                    logger.info(
                        f"Detected Asterisk audio format 'slin{bitrate}', setting asterisk_sample_rate to {self._params.asterisk_sample_rate}"
                    )
            else:
                raise ValueError(
                    f"Unsupported or missing audio encoding in Asterisk MEDIA_START event: {format}"
                )
        return InputTransportMessageFrame(message=message)

    def _handle_media_xoff(self, message: dict):
        """MEDIA_XOFF event handler.

        The Asterisk's channel driver will send this event to the app when the frame queue length reaches the high water (XOFF) level.
        The app should then pause sending media. Any media sent after this has a high probability of being dropped.
        Asterisk buffer is ~1000 frames by 20ms (160 bytes for ulaw/alaw at 8000Hz), so it's ~20 seconds of audio or 160KB,
        but the messages is sent when the buffer reaches the high water mark of ~900 frames.

        Args:
            message: The dictionary representing of the MEDIA_XOFF event message from Asterisk.
        """
        logger.error(
            f"Received MEDIA_XOFF event from Asterisk: {message}. Asterisk will drop the following audio frames."
        )
        return None

    def _handle_media_xon(self, message: dict):
        """MEDIA_XON event handler.

        The Asterisk's channel driver will send this event to the app when the frame queue length drops below the low water (XON) level.
        The app can then resume sending media.

        Args:
            message: The dictionary representing of the MEDIA_XON event message from Asterisk.
        """
        logger.info(
            f"Received MEDIA_XON event from Asterisk: {message}. Asterisk audio buffer is ready to receive audio again."
        )
        return None

    def _handle_dtmf_end(self, message: dict) -> Optional[InputDTMFFrame]:
        """DTMF_END event handler.

        Handles DTMF_END events from Asterisk and converts them to InputDTMFFrame.

        Args:
            message: The dictionary representing of the DTMF_END event message from Asterisk.

        Returns:
            An InputDTMFFrame if a valid DTMF digit is found, otherwise None.
        """
        digit = message.get("digit")
        if digit:
            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                # Handle case where string doesn't match any enum value
                logger.warning(f"Invalid DTMF digit received: {digit}")
                return None
        return None

    def _handle_queue_drained(self, message: dict):
        """QUEUE_DRAINED event handler.

        Handles QUEUE_DRAINED events from Asterisk. This event indicates that Asterisk has processed all the queued media.
        We will only receive this event if we requested it by sending "REPORT_QUEUE_DRAINED", and only once per "REPORT_QUEUE_DRAINED".
        Effectively, this means that Asterisk stopped playing audio to the channel.
        It's might be useful when you want to know when Asterisk finished playing all the TTS audio (bot stopped speaking from the remote user's perspective).
        As we use media buffering both in pipecat transport and in Asterisk it's not possible to know exactly
        when Asterisk finished playing all the buffered audio unless we get this event.

        Args:
            message: The dictionary representing of the QUEUE_DRAINED event message from Asterisk.
        """
        logger.info(
            f"Received QUEUE_DRAINED event from Asterisk: {message}. Asterisk has processed all the queued media."
        )
        return InputTransportMessageFrame(message=message)
