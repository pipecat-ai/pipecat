#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    KeypadEntry,
    StartFrame,
    StartInterruptionFrame,
    TransferCallFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType

try:
    import aiohttp
except ImportError:
    aiohttp = None


class PlivoFrameSerializer(FrameSerializer):
    """Serializer for Plivo Media Streams WebSocket protocol.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Plivo call when an EndFrame or CancelFrame is processed, but requires Plivo
    credentials to be provided.

    Attributes:
        _stream_sid: The Plivo Media Stream SID.
        _call_sid: The associated Plivo Call SID.
        _auth_id: Plivo auth ID for API access.
        _auth_token: Plivo authentication token for API access.
        _params: Configuration parameters.
        _plivo_sample_rate: Sample rate used by Plivo (typically 8kHz).
        _sample_rate: Input sample rate for the pipeline.
        _resampler: Audio resampler for format conversion.
    """

    class InputParams(BaseModel):
        """Configuration parameters for PlivoFrameSerializer.

        Attributes:
            plivo_sample_rate: Sample rate used by Plivo, defaults to 8000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        plivo_sample_rate: int = 8000
        sample_rate: Optional[int] = None
        auto_hang_up: bool = True
        transfer_call_enabled: bool = True
        transfer_call_attempted: bool = False

    def __init__(
        self,
        stream_sid: str,
        call_sid: Optional[str] = None,
        auth_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        params: InputParams = InputParams(),
    ):
        """Initialize the PlivoFrameSerializer.

        Args:
            stream_sid: The Plivo Media Stream SID.
            call_sid: The associated Plivo Call SID (optional, but required for auto hang-up).
            auth_id: Plivo auth ID (required for auto hang-up).
            auth_token: Plivo auth token (required for auto hang-up).
            params: Configuration parameters.
        """
        self._stream_sid = stream_sid
        self._call_sid = call_sid
        self._auth_id = auth_id
        self._auth_token = auth_token
        self._params = params

        self._plivo_sample_rate = self._params.plivo_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        self._resampler = create_default_resampler()
        self._hangup_attempted = False
        self._transfer_call_enabled = self._params.transfer_call_enabled
        self._transfer_call_attempted = self._params.transfer_call_attempted

    @property
    def type(self) -> FrameSerializerType:
        """Gets the serializer type.

        Returns:
            The serializer type, either TEXT or BINARY.
        """
        return FrameSerializerType.TEXT

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Plivo WebSocket format.

        Handles conversion of various frame types to Plivo WebSocket messages.
        For EndFrames, initiates call termination if auto_hang_up is enabled.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.
        """
        logger.trace(f"Plivo serializer processing frame: {type(frame).__name__}")

        # Special debug for TransferCallFrame
        if isinstance(frame, TransferCallFrame):
            logger.error(f"🎯 TransferCallFrame REACHED Plivo serializer! {frame}")

        if (
            self._params.auto_hang_up
            and not self._hangup_attempted
            and isinstance(frame, (EndFrame, CancelFrame))
        ):
            self._hangup_attempted = True
            logger.info(f"Auto hang-up triggered by {type(frame).__name__}")
            await self._hang_up_call()
            return None
        elif (
            self._transfer_call_enabled
            and not self._transfer_call_attempted
            and isinstance(frame, TransferCallFrame)
        ):
            self._transfer_call_attempted = True
            await self._transfer_call(frame)
            return None
        elif isinstance(frame, StartInterruptionFrame):
            answer = {"event": "clearAudio", "streamId": self._stream_sid}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz μ-law for Plivo
            serialized_data = await pcm_to_ulaw(
                data, frame.sample_rate, self._plivo_sample_rate, self._resampler
            )
            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "playAudio",
                "streamId": self._stream_sid,
                "media": {
                    "payload": payload,
                    "sampleRate": self._plivo_sample_rate,
                    "contentType": "audio/x-mulaw",
                },
            }

            return json.dumps(answer)
        elif isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            return json.dumps(frame.message)

        # Return None for unhandled frames
        return None

    async def _hang_up_call(self):
        """Hang up the Plivo call using Plivo's REST API."""
        try:
            if aiohttp is None:
                logger.error("aiohttp is required for call termination but not installed")
                return

            auth_id = self._auth_id
            auth_token = self._auth_token
            call_sid = self._call_sid

            if not call_sid or not auth_id or not auth_token:
                missing = []
                if not call_sid:
                    missing.append("call_sid")
                if not auth_id:
                    missing.append("auth_id")
                if not auth_token:
                    missing.append("auth_token")

                logger.warning(
                    f"Cannot hang up Plivo call: missing required parameters: {', '.join(missing)}"
                )
                return

            endpoint = f"https://api.plivo.com/v1/Account/{auth_id}/Call/{call_sid}"

            auth = aiohttp.BasicAuth(auth_id, auth_token)

            async with aiohttp.ClientSession() as session:
                async with session.delete(endpoint, auth=auth) as response:
                    if (
                        response.status == 204 or response.status == 404
                    ):  # Plivo returns 204 for successful deletion and 404 for already terminated calls
                        logger.info(f"Successfully terminated Plivo call {call_sid}")
                    else:
                        # Get the error details for better debugging
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Plivo call {call_sid}: "
                            f"Status {response.status}, Response: {error_text}"
                        )

        except Exception as e:
            logger.exception(f"Failed to hang up Plivo call: {e}")

    async def _play_music(self, music_url: str):
        url = f"https://api.plivo.com/v1/Account/{self._auth_id}/Call/{self._call_sid}/Play"
        payload = {"urls": music_url, "loop": True}

        auth = aiohttp.BasicAuth(self._auth_id, self._auth_token)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, auth=auth, data=payload) as response:
                response_status = response.status
                logger.debug(f"Plivo API response status: {response_status}")

                if not response.ok:
                    if response_status == 400:
                        response_json = await response.json()
                        logger.warning(
                            f"Failed to play music: Status={response_status}, "
                            f"Reason={response.reason}, Response={response_json}"
                        )
                        return

                logger.info(f"Plivo API Play Music call successful")

    async def _transfer_call(self, frame: TransferCallFrame):
        """Transfer the Plivo call to another number using Plivo's REST API."""
        logger.info(f"Plivo transfer call triggered")
        try:
            if aiohttp is None:
                logger.error("aiohttp is required for call transfer but not installed")
                return

            auth_id = self._auth_id
            auth_token = self._auth_token
            call_sid = self._call_sid

            if not call_sid or not auth_id or not auth_token:
                missing = []
                if not call_sid:
                    missing.append("call_sid")
                if not auth_id:
                    missing.append("auth_id")
                if not auth_token:
                    missing.append("auth_token")

                logger.error(
                    f"Cannot transfer Plivo call: missing required parameters: {', '.join(missing)}"
                )
                return

            if not frame.transfer_call_number:
                logger.error("Cannot transfer call: transfer_call_number is required")
                return

            logger.info(
                f"Initiating Plivo call transfer from {call_sid} to {frame.transfer_call_number}"
            )
            logger.error(call_sid)
            endpoint = f"https://api.plivo.com/v1/Account/{auth_id}/Call/{call_sid}"

            await self._play_music(
                "https://d2ggtb57hogf3o.cloudfront.net/caller-tune/phone_ringtone.wav"
            )

            auth = aiohttp.BasicAuth(auth_id, auth_token)

            transfer_data = {
                "legs": "aleg",
                "aleg_url": "https://7cc9-2401-4900-8830-86ea-2d92-3e2e-a19-7ac7.ngrok-free.app/plivo/transfer",
                "aleg_method": "POST",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{endpoint}", auth=auth, json=transfer_data) as response:
                    if response.status == 202:
                        logger.info(
                            f"Successfully transferred Plivo call {call_sid} to {frame.transfer_call_number}"
                        )
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to transfer Plivo call {call_sid}: Status {response.status}, Response: {error_text}"
                        )

        except Exception as e:
            logger.exception(f"Failed to transfer Plivo call: {e}")

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Plivo WebSocket data to Pipecat frames.

        Handles conversion of Plivo media events to appropriate Pipecat frames.

        Args:
            data: The raw WebSocket data from Plivo.

        Returns:
            A Pipecat frame corresponding to the Plivo event, or None if unhandled.
        """
        try:
            message = json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Plivo WebSocket message: {e}")
            return None

        if message["event"] == "media":
            try:
                payload_base64 = message["media"]["payload"]
                payload = base64.b64decode(payload_base64)

                # Input: Convert Plivo's 8kHz μ-law to PCM at pipeline input rate
                deserialized_data = await ulaw_to_pcm(
                    payload, self._plivo_sample_rate, self._sample_rate, self._resampler
                )
                audio_frame = InputAudioRawFrame(
                    audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate
                )
                return audio_frame
            except (KeyError, base64.binascii.Error) as e:
                logger.error(f"Failed to process Plivo media frame: {e}")
                return None
        elif message["event"] == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                logger.warning(f"Invalid DTMF digit received from Plivo: {digit}")
                return None
        else:
            return None
