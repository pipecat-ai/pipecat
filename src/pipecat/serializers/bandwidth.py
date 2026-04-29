#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bandwidth Programmable Voice WebSocket protocol serializer for Pipecat."""

import base64
import json
from typing import cast

from loguru import logger

from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class BandwidthFrameSerializer(FrameSerializer):
    """Serializer for Bandwidth Programmable Voice WebSocket media streaming.

    This serializer handles converting between Pipecat frames and Bandwidth's
    bidirectional WebSocket media streams protocol, established via the
    ``<StartStream mode="bidirectional">`` BXML verb.

    The Bandwidth wire protocol uses JSON envelopes with an ``eventType``
    field. Inbound media is base64-encoded ``audio/pcmu`` at 8kHz with the
    payload at the top level alongside ``track`` and ``sequenceNumber``.
    Outbound audio is sent as a ``playAudio`` event with a nested ``media``
    object that carries ``contentType`` and ``payload``.

    Bandwidth supports higher-fidelity outbound formats than μ-law — PCM at
    8/16/24 kHz — which can noticeably improve TTS quality. Configure via
    ``outbound_encoding`` and ``outbound_pcm_sample_rate``.

    When ``auto_hang_up`` is enabled (default), the serializer terminates
    the Bandwidth call via the Voice API REST endpoint when an EndFrame or
    CancelFrame is processed; this requires API credentials.

    DTMF is not delivered over the media-stream WebSocket on Bandwidth — it
    is captured via the BXML ``<Gather>`` verb and posted to a separate
    webhook. Handle DTMF in your application's webhook handler, not here.

    Protocol reference:
    https://dev.bandwidth.com/docs/voice/programmable-voice/bxml/startStream/
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for BandwidthFrameSerializer.

        Parameters:
            bandwidth_sample_rate: Sample rate Bandwidth uses on the wire for
                PCMU audio. Always 8000 Hz; exposed for symmetry with other
                serializers.
            sample_rate: Optional override for the pipeline input sample rate.
            outbound_encoding: Audio encoding for media sent back to the
                call — ``"PCMU"`` (μ-law 8kHz, broadest compatibility) or
                ``"PCM"`` (16-bit signed little-endian linear PCM at the rate
                set by ``outbound_pcm_sample_rate``).
            outbound_pcm_sample_rate: Sample rate used when
                ``outbound_encoding == "PCM"``. One of 8000, 16000, or 24000.
                Higher values produce noticeably better TTS quality at the
                cost of bandwidth.
            auto_hang_up: Whether to automatically terminate the call when
                an EndFrame or CancelFrame is processed.
            ignore_rtvi_messages: Inherited from base FrameSerializer.
        """

        bandwidth_sample_rate: int = 8000
        sample_rate: int | None = None
        outbound_encoding: str = "PCMU"
        outbound_pcm_sample_rate: int = 24000
        auto_hang_up: bool = True

    def __init__(
        self,
        stream_id: str,
        call_id: str | None = None,
        account_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        params: InputParams | None = None,
    ):
        """Initialize the BandwidthFrameSerializer.

        Args:
            stream_id: The Bandwidth Stream ID, available in the ``start``
                event metadata.
            call_id: The Bandwidth Call ID (required for auto hang-up).
            account_id: The Bandwidth account ID (required for auto hang-up).
            client_id: OAuth 2.0 Client ID for Bandwidth API authentication
                (required for auto hang-up). Used together with
                ``client_secret`` to obtain a Bearer token from Bandwidth's
                identity provider via the client_credentials grant.
            client_secret: OAuth 2.0 Client Secret (required for auto hang-up).
            params: Configuration parameters.
        """
        params = params or BandwidthFrameSerializer.InputParams()
        super().__init__(params)
        self._params: BandwidthFrameSerializer.InputParams = params

        if self._params.auto_hang_up:
            missing_credentials = []
            if not call_id:
                missing_credentials.append("call_id")
            if not account_id:
                missing_credentials.append("account_id")
            if not client_id:
                missing_credentials.append("client_id")
            if not client_secret:
                missing_credentials.append("client_secret")
            if missing_credentials:
                raise ValueError(
                    "auto_hang_up is enabled but missing required parameters: "
                    f"{', '.join(missing_credentials)}"
                )

        if self._params.outbound_encoding not in ("PCMU", "PCM"):
            raise ValueError(
                f"Unsupported outbound_encoding: {self._params.outbound_encoding}. "
                "Must be 'PCMU' or 'PCM'."
            )

        if (
            self._params.outbound_encoding == "PCM"
            and self._params.outbound_pcm_sample_rate
            not in (
                8000,
                16000,
                24000,
            )
        ):
            raise ValueError(
                "outbound_pcm_sample_rate must be 8000, 16000, or 24000 when "
                f"outbound_encoding is 'PCM'. Got: {self._params.outbound_pcm_sample_rate}"
            )

        self._stream_id = stream_id
        self._call_id = call_id
        self._account_id = account_id
        self._client_id = client_id
        self._client_secret = client_secret

        self._bandwidth_sample_rate = self._params.bandwidth_sample_rate
        self._sample_rate = 0  # Pipeline input rate, set in setup()

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._hangup_attempted = False

    @property
    def stream_id(self) -> str:
        """Get the Bandwidth Stream ID this serializer is bound to."""
        return self._stream_id

    @property
    def call_id(self) -> str | None:
        """Get the Bandwidth Call ID, if provided."""
        return self._call_id

    async def setup(self, frame: StartFrame):
        """Initialize the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize a Pipecat frame into a Bandwidth WebSocket message.

        - EndFrame / CancelFrame: trigger REST hang-up if enabled, return None.
        - InterruptionFrame: emit ``{"eventType": "clear"}`` to flush buffered
          outbound audio so the bot stops talking when the user interrupts.
        - AudioRawFrame: encode according to configured outbound format and
          wrap in a ``playAudio`` event.
        - OutputTransportMessageFrame: pass-through (e.g. RTVI bridging).

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized JSON string, or None if the frame produces no output.
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
            return json.dumps({"eventType": "clear"})
        elif isinstance(frame, AudioRawFrame):
            payload, content_type = await self._encode_outbound_audio(frame)
            if payload is None:
                return None
            return json.dumps(
                {
                    "eventType": "playAudio",
                    "media": {"contentType": content_type, "payload": payload},
                }
            )
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        return None

    async def _encode_outbound_audio(self, frame: AudioRawFrame) -> tuple[str | None, str]:
        """Encode an AudioRawFrame for Bandwidth playback.

        Returns:
            (base64_payload, contentType) for the outbound playAudio event,
            or (None, content_type) if there is no audio to send.
        """
        if self._params.outbound_encoding == "PCMU":
            encoded = await pcm_to_ulaw(
                frame.audio,
                frame.sample_rate,
                self._bandwidth_sample_rate,
                self._output_resampler,
            )
            content_type = "audio/pcmu"
        else:
            target_rate = self._params.outbound_pcm_sample_rate
            if frame.sample_rate != target_rate:
                encoded = await self._output_resampler.resample(
                    frame.audio, frame.sample_rate, target_rate
                )
            else:
                encoded = frame.audio
            content_type = (
                f"audio/pcm;rate={target_rate};channels=1;"
                "bit-depth=16;endian=little;encoding=signed"
            )

        if encoded is None or len(encoded) == 0:
            return None, content_type

        return base64.b64encode(encoded).decode("utf-8"), content_type

    OAUTH_TOKEN_URL = "https://api.bandwidth.com/api/v1/oauth2/token"
    VOICE_API_BASE_URL = "https://voice.bandwidth.com/api/v2"

    async def _hang_up_call(self):
        """Hang up the Bandwidth call via the Voice API REST endpoint.

        Authentication uses the OAuth 2.0 client_credentials grant: we POST
        the client_id/client_secret pair (as Basic Auth) to Bandwidth's IDP
        to obtain a short-lived Bearer token, then send it on the call-update
        request. Legacy API username/password auth is deprecated by
        Bandwidth (sunset 2026-12-02).
        """
        try:
            import aiohttp

            # __init__ guarantees these are non-None whenever auto_hang_up is True,
            # which is the only path that reaches this method.
            account_id = cast(str, self._account_id)
            call_id = cast(str, self._call_id)
            client_id = cast(str, self._client_id)
            client_secret = cast(str, self._client_secret)

            async with aiohttp.ClientSession() as session:
                token_auth = aiohttp.BasicAuth(client_id, client_secret)
                async with session.post(
                    self.OAUTH_TOKEN_URL,
                    auth=token_auth,
                    data={"grant_type": "client_credentials"},
                ) as token_response:
                    if token_response.status != 200:
                        error_text = await token_response.text()
                        logger.error(
                            f"Failed to fetch Bandwidth OAuth token: "
                            f"Status {token_response.status}, Response: {error_text}"
                        )
                        return
                    token_data = await token_response.json()

                access_token = token_data.get("access_token")
                if not access_token:
                    logger.error("Bandwidth OAuth response missing access_token")
                    return

                endpoint = f"{self.VOICE_API_BASE_URL}/accounts/{account_id}/calls/{call_id}"
                headers = {"Authorization": f"Bearer {access_token}"}
                body = {"state": "completed"}

                async with session.post(endpoint, headers=headers, json=body) as response:
                    if 200 <= response.status < 300:
                        logger.info(f"Successfully terminated Bandwidth call {call_id}")
                    elif response.status == 404:
                        # Call already ended.
                        logger.debug(f"Bandwidth call {call_id} was already terminated")
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Bandwidth call {call_id}: "
                            f"Status {response.status}, Response: {error_text}"
                        )
        except Exception as e:
            logger.error(f"Failed to hang up Bandwidth call: {e}")

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize a Bandwidth WebSocket message into a Pipecat frame.

        Bandwidth sends three event types over the media-stream WebSocket:
        ``start``, ``media``, and ``stop``. Only ``media`` events on the
        ``inbound`` track produce frames for the pipeline. ``start`` and
        ``stop`` are handled internally for state tracking.

        Args:
            data: Raw WebSocket data from Bandwidth (JSON text).

        Returns:
            A Pipecat frame corresponding to the event, or None if the
            event is informational only.
        """
        try:
            message = json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Bandwidth WebSocket message: {e}")
            return None

        event_type = message.get("eventType")

        if event_type == "media":
            # Filter to the caller's track. When the BXML uses tracks="both"
            # the WebSocket would otherwise also receive the bot's own
            # outbound audio echoed back, which would cause feedback.
            track = message.get("track", "inbound")
            if track != "inbound":
                return None

            payload_b64 = message.get("payload")
            if not payload_b64:
                return None

            payload = base64.b64decode(payload_b64)

            decoded = await ulaw_to_pcm(
                payload,
                self._bandwidth_sample_rate,
                self._sample_rate,
                self._input_resampler,
            )
            if decoded is None or len(decoded) == 0:
                return None

            return InputAudioRawFrame(audio=decoded, num_channels=1, sample_rate=self._sample_rate)
        elif event_type == "start":
            metadata = message.get("metadata", {})
            logger.debug(
                f"Bandwidth stream started: stream_id={metadata.get('streamId')}, "
                f"call_id={metadata.get('callId')}"
            )
            return None
        elif event_type == "stop":
            logger.debug(f"Bandwidth stream stopped: stream_id={self._stream_id}")
            return None
        else:
            logger.debug(f"Ignoring unknown Bandwidth event type: {event_type}")
            return None
