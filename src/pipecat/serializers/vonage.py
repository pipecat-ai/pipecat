# SPDX-License-Identifier: BSD-2-Clause
"""Vonage WebSocket serializer (WAV+pydub resample, fixed-size chunking).

Note: DTMF is intentionally not implemented because Vonage Audio Connector
does not expose DTMF events over the WebSocket protocol.
"""

from __future__ import annotations

import io
import json
import wave
from typing import List, Optional, Union

from loguru import logger
from pydantic import BaseModel
from pydub import AudioSegment

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType

# ---- Audio/timing constants --------------------------------------------------

AUDIO_TARGET_RATE_HZ: int = 16_000  # 16 kHz target
AUDIO_CHANNELS_MONO: int = 1  # mono
PCM16_SAMPLE_WIDTH_BYTES: int = 2  # 16-bit PCM
CHUNK_DURATION_MS: int = 20  # telephony frame
SECONDS_PER_MS: float = 1.0 / 1_000.0
CHUNK_PERIOD_SECONDS: float = CHUNK_DURATION_MS * SECONDS_PER_MS
SLEEP_INTERVAL_PER_CHUNK: float = 0.01

BYTES_PER_SAMPLE_MONO: int = AUDIO_CHANNELS_MONO * PCM16_SAMPLE_WIDTH_BYTES
BYTES_PER_CHUNK: int = int(AUDIO_TARGET_RATE_HZ * CHUNK_PERIOD_SECONDS) * BYTES_PER_SAMPLE_MONO


class VonageFrameSerializer(FrameSerializer):
    """Produces 16 kHz mono PCM chunks; resamples using WAV+pydub path."""

    class InputParams(BaseModel):
        """Configuration options for the Vonage frame serializer.

        Controls whether to send a clear-audio event and whether
        to auto-hang-up on End/Cancel frames.

        Hang-up configuration:

        - api_base_url:    Base URL for the OpenTok API.
                           Default: "https://api.opentok.com"
        - project_id:      OpenTok project / API key (used in the URL path).
        - session_id:      OpenTok session ID.
        - connection_id:   Connection ID of the Audio Connector WebSocket connection.
                           May be set at construction time *or later* via
                           VonageFrameSerializer.set_connection_id().
        - jwt:             JWT for OpenTok, used in X-OPENTOK-AUTH header.
        """

        auto_hang_up: bool = True
        send_clear_audio_event: bool = True

        api_base_url: str = "https://api.opentok.com"
        project_id: Optional[str] = None
        session_id: Optional[str] = None
        connection_id: Optional[str] = None
        jwt: Optional[str] = None

    def __init__(self, params: Optional[InputParams] = None) -> None:
        """Initialize the VonageFrameSerializer.

        Args:
            params: Optional configuration parameters for serialization.
        """
        self._params: VonageFrameSerializer.InputParams = (
            params or VonageFrameSerializer.InputParams()
        )
        self._sample_rate_hz: int = AUDIO_TARGET_RATE_HZ
        self._in_resampler = create_stream_resampler()

        # Transport reads this for pacing (one sleep per chunk).
        self.sleep_interval: float = SLEEP_INTERVAL_PER_CHUNK

        # Serializer-side audio format assumptions for pydub path:
        self._channels: int = AUDIO_CHANNELS_MONO
        self._sample_width_bytes: int = PCM16_SAMPLE_WIDTH_BYTES

        # Ensure we only attempt hang-up once
        self._hangup_attempted: bool = False

        # Warn early if auto_hang_up is enabled but core config is incomplete.
        # NOTE: connection_id is intentionally NOT required here, because in
        # the Vonage Audio Connector flow it may only be known after
        # connect_audio_to_websocket() runs. It can be set later with
        # set_connection_id().
        if self._params.auto_hang_up:
            missing = [
                name
                for name, value in (
                    ("project_id", self._params.project_id),
                    ("session_id", self._params.session_id),
                    ("jwt", self._params.jwt),
                )
                if not value
            ]
            if missing:
                logger.warning(
                    "VonageFrameSerializer: auto_hang_up is enabled but the following "
                    f"fields are not configured: {', '.join(missing)}. "
                    "Hang-up requests will be skipped until these are provided."
                )

    # ---- public properties / setters ----------------------------------------

    @property
    def connection_id(self) -> Optional[str]:
        """Current OpenTok connection ID."""
        return self._params.connection_id

    def set_connection_id(self, connection_id: str) -> None:
        """Set or update the OpenTok connection ID.

        This is useful in flows where the Audio Connector connectionId is
        only known after calling /connect in separate component or script.
        """
        self._params.connection_id = connection_id
        logger.debug(
            "VonageFrameSerializer: connection_id updated to %r",
            connection_id,
        )

    @property
    def type(self) -> FrameSerializerType:
        """Return the serializer type (binary frames)."""
        return FrameSerializerType.BINARY

    async def setup(self, frame: StartFrame) -> None:
        """Prepare the serializer for a new session.

        Sets the sample rate and sleep interval for chunk pacing.
        """
        self._sample_rate_hz = AUDIO_TARGET_RATE_HZ
        self.sleep_interval = SLEEP_INTERVAL_PER_CHUNK

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _resample_audio_with_pydub(
        data: bytes,
        src_rate_hz: int,
        num_channels: int,
        sample_width_bytes: int,
        target_rate_hz: int,
    ) -> bytes:
        """Resample via WAV header + pydub.

        NOTE: This assumes `data` contains a WAV header. If your pipeline disables
        WAV headers, switch to a raw-PCM resampler instead.
        """
        with wave.open(io.BytesIO(data), "rb") as wf:
            num_frames = wf.getnframes()
            pcm_data = wf.readframes(num_frames)

        segment = AudioSegment.from_raw(
            io.BytesIO(pcm_data),
            sample_width=sample_width_bytes,
            frame_rate=src_rate_hz,
            channels=num_channels,
        )
        resampled = (
            segment.set_channels(num_channels)
            .set_sample_width(sample_width_bytes)
            .set_frame_rate(target_rate_hz)
        )
        return resampled.raw_data

    @staticmethod
    def _split_into_chunks(audio16: bytes) -> List[bytes]:
        return [audio16[i : i + BYTES_PER_CHUNK] for i in range(0, len(audio16), BYTES_PER_CHUNK)]

    async def _hang_up_call(self) -> None:
        """Hang up the call using OpenTok 'force disconnect' REST API."""
        params = self._params

        missing = [
            name
            for name, value in (
                ("project_id", params.project_id),
                ("session_id", params.session_id),
                ("connection_id", params.connection_id),
                ("jwt", params.jwt),
            )
            if not value
        ]
        if missing:
            logger.warning(
                "VonageFrameSerializer: requested hang-up, but missing required "
                f"OpenTok fields: {', '.join(missing)}. Skipping hang-up."
            )
            return

        base_url = params.api_base_url.rstrip("/")
        endpoint = (
            f"{base_url}/v2/project/{params.project_id}"
            f"/session/{params.session_id}/connection/{params.connection_id}"
        )

        headers = {
            "X-OPENTOK-AUTH": params.jwt,
        }

        logger.info(
            "VonageFrameSerializer: calling force disconnect "
            f"endpoint={endpoint}, jwt_present={bool(headers.get('X-OPENTOK-AUTH'))}, "
            f"connection_id={params.connection_id}"
        )

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.delete(endpoint, headers=headers) as resp:
                    text = await resp.text()
                    if 200 <= resp.status < 300:
                        logger.info(
                            "VonageFrameSerializer: successfully requested force disconnect "
                            f"for connection {params.connection_id} (status={resp.status})."
                        )
                    elif resp.status == 404:
                        logger.debug(
                            "VonageFrameSerializer: connection already disconnected or not found "
                            f"(connection_id={params.connection_id}, status=404)."
                        )
                    else:
                        logger.error(
                            "VonageFrameSerializer: force disconnect request failed "
                            f"(status={resp.status}): {text}"
                        )
        except Exception as exc:
            logger.exception(
                f"VonageFrameSerializer: error while calling OpenTok force disconnect: {exc}"
            )

    # --- API ------------------------------------------------------------------

    async def serialize(self, frame: Frame) -> Optional[Union[str, bytes, list[bytes]]]:
        """Convert a Frame into one or more serialized payloads.

        Args:
            frame: The frame to serialize.

        Returns:
            The serialized data as a string, bytes, or list of bytes.
        """
        # --- Hang-up handling on End/Cancel ----------------------------------
        if isinstance(frame, (EndFrame, CancelFrame)):
            if self._params.auto_hang_up and not self._hangup_attempted:
                self._hangup_attempted = True
                logger.debug("VonageFrameSerializer: End/Cancel observed, triggering hang-up.")
                await self._hang_up_call()
            else:
                logger.debug(
                    "VonageFrameSerializer: End/Cancel observed; "
                    "auto_hang_up disabled or already attempted."
                )
            # No payload needs to be sent to the WebSocket for End/Cancel.
            return None

        # --- Interruption handling ------------------------------------------
        if isinstance(frame, StartInterruptionFrame) and self._params.send_clear_audio_event:
            return json.dumps({"event": "clearAudio"})

        # --- Outbound audio --------------------------------------------------
        if isinstance(frame, OutputAudioRawFrame):
            audio16 = self._resample_audio_with_pydub(
                data=frame.audio,
                src_rate_hz=frame.sample_rate,
                num_channels=self._channels,
                sample_width_bytes=self._sample_width_bytes,
                target_rate_hz=self._sample_rate_hz,
            )
            return self._split_into_chunks(audio16)

        logger.debug(f"VonageFrameSerializer: ignoring frame type {type(frame).__name__}.")
        return None

    async def deserialize(self, data: Union[str, bytes]) -> Optional[Frame]:
        """Convert serialized input data into a Frame.

        Args:
            data: The raw audio or frame payload.

        Returns:
            The corresponding Frame instance, or None if parsing fails.
        """
        # Binary = audio frame from Audio Connector (16-bit PCM, 16 kHz)
        if isinstance(data, (bytes, bytearray)):
            audio = await self._in_resampler.resample(
                bytes(data), self._sample_rate_hz, self._sample_rate_hz
            )
            return InputAudioRawFrame(
                audio=audio,
                num_channels=AUDIO_CHANNELS_MONO,
                sample_rate=self._sample_rate_hz,
            )

        # Text messages (websocket:connected / websocket:media:update / websocket:disconnected)
        logger.info("VonageFrameSerializer: ignoring non-binary inbound data.")
        return None
