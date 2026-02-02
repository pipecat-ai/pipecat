#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Telnyx WebSocket frame serializer for Pipecat."""

import base64
import binascii
import json
import sys
from typing import Optional

import aiohttp
import numpy as np
from loguru import logger

try:
    import ctypes.util
    # Help find libopus on macOS with Homebrew
    if ctypes.util.find_library('opus') is None:
        import os
        for path in ['/opt/homebrew/lib', '/usr/local/lib']:
            lib_path = os.path.join(path, 'libopus.dylib')
            if os.path.exists(lib_path):
                os.environ['DYLD_LIBRARY_PATH'] = path + ':' + os.environ.get('DYLD_LIBRARY_PATH', '')
                break
    import opuslib
    OPUS_AVAILABLE = True
except (ImportError, Exception):
    OPUS_AVAILABLE = False
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
    InterruptionFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class TelnyxFrameSerializer(FrameSerializer):
    """Serializer for Telnyx WebSocket protocol.

    This serializer handles converting between Pipecat frames and Telnyx's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Telnyx call when an EndFrame or CancelFrame is processed, but requires Telnyx
    credentials to be provided.
    """

    class InputParams(BaseModel):
        """Configuration parameters for TelnyxFrameSerializer.

        Parameters:
            telnyx_sample_rate: Sample rate used by Telnyx, defaults to 8000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
            inbound_encoding: Audio encoding for data sent to Telnyx (e.g., "PCMU").
            outbound_encoding: Audio encoding for data received from Telnyx (e.g., "PCMU").
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        telnyx_sample_rate: int = 8000
        sample_rate: Optional[int] = None
        inbound_encoding: str = "PCMU"
        outbound_encoding: str = "PCMU"
        auto_hang_up: bool = True

    def __init__(
        self,
        stream_id: str,
        outbound_encoding: str,
        inbound_encoding: str,
        call_control_id: Optional[str] = None,
        api_key: Optional[str] = None,
        params: Optional[InputParams] = None,
    ):
        """Initialize the TelnyxFrameSerializer.

        Args:
            stream_id: The Stream ID for Telnyx.
            outbound_encoding: The encoding type for outbound audio (e.g., "PCMU").
            inbound_encoding: The encoding type for inbound audio (e.g., "PCMU").
            call_control_id: The Call Control ID for the Telnyx call (optional, but required for auto hang-up).
            api_key: Your Telnyx API key (required for auto hang-up).
            params: Configuration parameters.
        """
        self._stream_id = stream_id
        self._call_control_id = call_control_id
        self._api_key = api_key
        self._params = params or TelnyxFrameSerializer.InputParams()
        self._params.outbound_encoding = outbound_encoding
        self._params.inbound_encoding = inbound_encoding

        self._telnyx_sample_rate = self._params.telnyx_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._hangup_attempted = False

        # OPUS encoder/decoder (lazy init)
        self._opus_encoder = None
        self._opus_decoder = None

        # OPUS frame buffering - must encode in 20ms chunks
        # At 16kHz: 20ms = 320 samples = 640 bytes (16-bit PCM)
        self._opus_frame_samples = self._telnyx_sample_rate // 50  # 20ms
        self._opus_frame_bytes = self._opus_frame_samples * 2  # 16-bit = 2 bytes/sample
        self._opus_encode_buffer = bytearray()

    def _get_opus_encoder(self):
        if self._opus_encoder is None and OPUS_AVAILABLE:
            self._opus_encoder = opuslib.Encoder(
                self._telnyx_sample_rate, 1, opuslib.APPLICATION_VOIP
            )
        return self._opus_encoder

    def _get_opus_decoder(self):
        if self._opus_decoder is None and OPUS_AVAILABLE:
            self._opus_decoder = opuslib.Decoder(self._telnyx_sample_rate, 1)
        return self._opus_decoder

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | list | None:
        """Serializes a Pipecat frame to Telnyx WebSocket format.

        Handles conversion of various frame types to Telnyx WebSocket messages.
        For EndFrames and CancelFrames, initiates call termination if auto_hang_up is enabled.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string, bytes, list of strings (for OPUS), or None.

        Raises:
            ValueError: If an unsupported encoding is specified.
        """
        if (
            self._params.auto_hang_up
            and not self._hangup_attempted
            and isinstance(frame, (EndFrame, CancelFrame))
        ):
            self._hangup_attempted = True
            await self._hang_up_call()
            return None
        elif isinstance(frame, InterruptionFrame):
            answer = {"event": "clear"}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz encoded for Telnyx
            if self._params.inbound_encoding == "PCMU":
                serialized_data = await pcm_to_ulaw(
                    data, frame.sample_rate, self._telnyx_sample_rate, self._output_resampler
                )
            elif self._params.inbound_encoding == "PCMA":
                serialized_data = await pcm_to_alaw(
                    data, frame.sample_rate, self._telnyx_sample_rate, self._output_resampler
                )
            elif self._params.inbound_encoding == "L16":
                # L16 audio to Telnyx - resample then send as little-endian
                resampled_data = await self._output_resampler.resample(
                    data, frame.sample_rate, self._telnyx_sample_rate
                )
                if resampled_data is None or len(resampled_data) == 0:
                    return None
                if len(resampled_data) % 2 != 0:
                    resampled_data = resampled_data[: len(resampled_data) - 1]
                # Telnyx expects little-endian L16
                if sys.byteorder == "little":
                    serialized_data = resampled_data
                else:
                    audio_array = np.frombuffer(resampled_data, dtype=np.int16)
                    serialized_data = audio_array.byteswap().tobytes()
            elif self._params.inbound_encoding == "OPUS":
                if not OPUS_AVAILABLE:
                    raise ValueError("OPUS encoding requires opuslib: pip install opuslib")
                # Resample to target rate first
                resampled_data = await self._output_resampler.resample(
                    data, frame.sample_rate, self._telnyx_sample_rate
                )
                if resampled_data is None or len(resampled_data) == 0:
                    return None

                # Buffer audio and encode in 20ms frames (OPUS requirement)
                # Each OPUS frame must be sent as a separate WebSocket message
                self._opus_encode_buffer.extend(resampled_data)

                # Need at least one full frame
                if len(self._opus_encode_buffer) < self._opus_frame_bytes:
                    return None

                # Encode ALL complete frames and return as a list
                # (Telnyx requires one OPUS packet per WebSocket message)
                encoder = self._get_opus_encoder()
                messages = []
                offset = 0

                while offset + self._opus_frame_bytes <= len(self._opus_encode_buffer):
                    frame_data = bytes(self._opus_encode_buffer[offset : offset + self._opus_frame_bytes])
                    offset += self._opus_frame_bytes

                    opus_packet = encoder.encode(frame_data, self._opus_frame_samples)
                    # Use f-string instead of json.dumps for fixed structure (faster)
                    payload = base64.b64encode(opus_packet).decode("utf-8")
                    messages.append(f'{{"event":"media","media":{{"payload":"{payload}"}}}}')

                # Keep only remaining incomplete frame data
                if offset > 0:
                    self._opus_encode_buffer = self._opus_encode_buffer[offset:]

                return messages if messages else None
            else:
                raise ValueError(f"Unsupported encoding: {self._params.inbound_encoding}")

            if serialized_data is None or len(serialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "media",
                "media": {"payload": payload},
            }

            return json.dumps(answer)

        # Return None for unhandled frames
        return None

    async def _hang_up_call(self):
        """Hang up the Telnyx call using Telnyx's REST API."""
        try:
            call_control_id = self._call_control_id
            api_key = self._api_key

            if not call_control_id or not api_key:
                logger.warning(
                    "Cannot hang up Telnyx call: call_control_id and api_key must be provided"
                )
                return

            # Telnyx API endpoint for hanging up a call
            endpoint = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/hangup"

            # Set headers with API key
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

            # Make the POST request to hang up the call
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers) as response:
                    if response.status == 200:
                        logger.info(f"Successfully terminated Telnyx call {call_control_id}")
                    elif response.status == 422:
                        # Handle the case where the call has already ended
                        # Error code 90018: "Call has already ended"
                        # Source: https://developers.telnyx.com/api/errors/90018
                        try:
                            error_data = await response.json()
                            if any(
                                error.get("code") == "90018"
                                for error in error_data.get("errors", [])
                            ):
                                logger.debug(
                                    f"Telnyx call {call_control_id} was already terminated"
                                )
                                return
                        except:
                            pass  # Fall through to log the raw error

                        # Log other 422 errors
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Telnyx call {call_control_id}: "
                            f"Status {response.status}, Response: {error_text}"
                        )
                    else:
                        # Log other errors
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Telnyx call {call_control_id}: "
                            f"Status {response.status}, Response: {error_text}"
                        )

        except Exception as e:
            logger.error(f"Failed to hang up Telnyx call: {e}")

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Telnyx WebSocket data to Pipecat frames.

        Handles conversion of Telnyx media events to appropriate Pipecat frames,
        including audio data and DTMF keypresses.

        Args:
            data: The raw WebSocket data from Telnyx.

        Returns:
            A Pipecat frame corresponding to the Telnyx event, or None if unhandled.

        Raises:
            ValueError: If an unsupported encoding is specified.
        """
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON message")
            return None

        if not isinstance(message, dict) or "event" not in message:
            return None

        if message["event"] == "media":
            payload_base64 = message["media"]["payload"]
            try:
                payload = base64.b64decode(payload_base64)
            except binascii.Error:
                logger.warning("Failed to decode base64 audio payload")
                return None

            # Input: Convert Telnyx audio to PCM at pipeline input rate
            if self._params.outbound_encoding == "PCMU":
                deserialized_data = await ulaw_to_pcm(
                    payload,
                    self._telnyx_sample_rate,
                    self._sample_rate,
                    self._input_resampler,
                )
            elif self._params.outbound_encoding == "PCMA":
                deserialized_data = await alaw_to_pcm(
                    payload,
                    self._telnyx_sample_rate,
                    self._sample_rate,
                    self._input_resampler,
                )
            elif self._params.outbound_encoding == "L16":
                # L16 audio from Telnyx - little-endian
                if len(payload) % 2 != 0:
                    payload = payload[: len(payload) - 1]
                if len(payload) == 0:
                    return None
                # Telnyx sends little-endian L16
                if sys.byteorder == "little":
                    host_audio = payload
                else:
                    audio_array = np.frombuffer(payload, dtype="<i2")
                    host_audio = audio_array.byteswap().tobytes()
                # Resample if rates differ
                deserialized_data = await self._input_resampler.resample(
                    host_audio,
                    self._telnyx_sample_rate,
                    self._sample_rate,
                )
            elif self._params.outbound_encoding == "OPUS":
                if not OPUS_AVAILABLE:
                    raise ValueError("OPUS decoding requires opuslib: pip install opuslib")
                decoder = self._get_opus_decoder()
                # Decode OPUS to PCM - frame size is samples per channel
                # At 16kHz, 20ms = 320 samples
                frame_size = self._telnyx_sample_rate // 50  # 20ms frame
                pcm_data = decoder.decode(payload, frame_size)
                # Resample if needed
                deserialized_data = await self._input_resampler.resample(
                    pcm_data,
                    self._telnyx_sample_rate,
                    self._sample_rate,
                )
            else:
                raise ValueError(f"Unsupported encoding: {self._params.outbound_encoding}")

            if deserialized_data is None or len(deserialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate
            )
            return audio_frame
        elif message["event"] == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError as e:
                # Handle case where string doesn't match any enum value
                return None
        else:
            return None
