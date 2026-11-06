#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Twilio Media Streams WebSocket protocol serializer for Pipecat."""

import audioop
import base64
import json
from typing import Any, cast

from loguru import logger
from pydantic import model_validator

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.deprecation import warn_deprecated


def _build_call_resource_url(
    account_sid: str,
    call_sid: str,
    *,
    base_url: str | None = None,
    region: str | None = None,
    edge: str | None = None,
) -> str:
    """Build the REST URL for a Call resource (used by auto hang-up).

    When ``base_url`` is provided it is used verbatim as the API root (scheme +
    host, e.g. ``https://api.twilio.com``), which lets the serializer target a
    Twilio-API-compatible backend or a self-hosted server. Otherwise the host is
    derived from Twilio's ``region``/``edge`` FQDN format, preserving the prior
    default behavior.
    """
    if base_url:
        root = base_url.rstrip("/")
    else:
        region_prefix = f"{region}." if region else ""
        edge_prefix = f"{edge}." if edge else ""
        root = f"https://api.{edge_prefix}{region_prefix}twilio.com"
    return f"{root}/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json"


class TwilioFrameSerializer(FrameSerializer):
    """Serializer for Twilio Media Streams WebSocket protocol.

    This serializer handles converting between Pipecat frames and Twilio's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Twilio call when an EndFrame or CancelFrame is processed, but requires Twilio
    credentials: either auth_token or an API key (api_key_sid + api_key_secret).
    Credentials must belong to the configured region: each Twilio region has its own
    Auth Token and API keys.
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for TwilioFrameSerializer.

        Parameters:
            audio_in_sample_rate: Sample rate of the audio Twilio sends, defaults to 8000 Hz.
            audio_out_sample_rate: Sample rate of the audio sent to Twilio, defaults to 8000 Hz.
            twilio_sample_rate: Sample rate used by Twilio.

                .. deprecated:: 1.16.0
                    Use ``audio_in_sample_rate`` and ``audio_out_sample_rate`` instead.
                    Will be removed in 2.0.0.

            sample_rate: Has no effect: incoming audio is converted to the pipeline's input rate.

                .. deprecated:: 1.16.0
                    Use :attr:`~pipecat.pipeline.worker.PipelineParams.audio_in_sample_rate`
                    instead. Will be removed in 2.0.0.

            auto_hang_up: Whether to automatically terminate call on EndFrame.
            ignore_rtvi_messages: Inherited from base FrameSerializer, defaults to True.
        """

        audio_in_sample_rate: int = 8000
        audio_out_sample_rate: int = 8000
        twilio_sample_rate: int | None = None
        sample_rate: int | None = None
        auto_hang_up: bool = True

        @model_validator(mode="before")
        @classmethod
        def _carry_deprecated_rates(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            if data.get("twilio_sample_rate") is not None:
                warn_deprecated(
                    "`TwilioFrameSerializer.InputParams.twilio_sample_rate` is deprecated since "
                    "1.16.0 and will be removed in 2.0.0. Use `audio_in_sample_rate` and "
                    "`audio_out_sample_rate` instead.",
                    stacklevel=4,
                )
                data.setdefault("audio_in_sample_rate", data["twilio_sample_rate"])
                data.setdefault("audio_out_sample_rate", data["twilio_sample_rate"])
            if data.get("sample_rate") is not None:
                warn_deprecated(
                    "`TwilioFrameSerializer.InputParams.sample_rate` is deprecated since 1.16.0 "
                    "and will be removed in 2.0.0. Use `PipelineParams.audio_in_sample_rate` "
                    "instead.",
                    stacklevel=4,
                )
            return data

    def __init__(
        self,
        stream_sid: str,
        call_sid: str | None = None,
        account_sid: str | None = None,
        auth_token: str | None = None,
        region: str | None = None,
        edge: str | None = None,
        base_url: str | None = None,
        params: InputParams | None = None,
        *,
        api_key_sid: str | None = None,
        api_key_secret: str | None = None,
    ):
        """Initialize the TwilioFrameSerializer.

        Args:
            stream_sid: The Twilio Media Stream SID.
            call_sid: The associated Twilio Call SID (optional, but required for auto hang-up).
            account_sid: Twilio account SID (required for auto hang-up).
            auth_token: Twilio Auth Token for auto hang-up. Mutually exclusive with
                api_key_sid/api_key_secret.
            api_key_sid: Twilio API key SID (starts with "SK"). Alternative to
                auth_token for auto hang-up. Must be provided with api_key_secret.
            api_key_secret: Twilio API key secret. Must be provided with api_key_sid.
            region: Twilio region (e.g., "au1", "ie1"). Must be specified with edge.
            edge: Twilio edge location (e.g., "sydney", "dublin"). Must be specified with region.
            base_url: Optional REST API base URL (scheme + host, e.g.
                ``https://api.twilio.com``) used for auto hang-up. Defaults to the
                host derived from region/edge. Set this to target a Twilio-API-
                compatible backend or a self-hosted server.
            params: Configuration parameters.
        """
        params = params or TwilioFrameSerializer.InputParams()
        super().__init__(params)
        self._params: TwilioFrameSerializer.InputParams = params

        # Validate hangup-related parameters if auto_hang_up is enabled
        if self._params.auto_hang_up:
            # Validate required credentials
            missing_credentials = []
            if not call_sid:
                missing_credentials.append("call_sid")
            if not account_sid:
                missing_credentials.append("account_sid")
            if not auth_token and not api_key_sid and not api_key_secret:
                missing_credentials.append("auth_token or api_key_sid + api_key_secret")

            if missing_credentials:
                raise ValueError(
                    f"auto_hang_up is enabled but missing required parameters: {', '.join(missing_credentials)}"
                )

            # Reject an auth_token alongside any part of an API key. Accepting both
            # would let a leftover auth_token stay deployed without anyone noticing.
            if auth_token and (api_key_sid or api_key_secret):
                raise ValueError(
                    "Provide either auth_token or api_key_sid + api_key_secret, not both."
                )

            # Validate the API key pair is complete. auth_token is ruled out above,
            # so this only sees a lone api_key_sid or api_key_secret.
            if (api_key_sid and not api_key_secret) or (api_key_secret and not api_key_sid):
                missing = "api_key_secret" if api_key_sid else "api_key_sid"
                raise ValueError(
                    "Both api_key_sid and api_key_secret parameters are required if one is set. "
                    f"Missing: {missing}"
                )

            # Validate region and edge are both provided if either is specified.
            # Only applies when deriving Twilio's FQDN host; base_url is used
            # verbatim and ignores region/edge, so the pairing requirement is moot.
            if not base_url and ((region and not edge) or (edge and not region)):
                raise ValueError(
                    "Both edge and region parameters are required if one is set. "
                    f"Twilio's FQDN format requires both: api.{{edge}}.{{region}}.twilio.com. "
                    f"Got: region='{region}', edge='{edge}'"
                )

        self._stream_sid = stream_sid
        self._call_sid = call_sid
        self._account_sid = account_sid
        self._auth_token = auth_token
        self._api_key_sid = api_key_sid
        self._api_key_secret = api_key_secret
        self._region = region
        self._edge = edge
        self._base_url = base_url

        self._hangup_attempted = False

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Twilio WebSocket format.

        Handles conversion of various frame types to Twilio WebSocket messages.
        For EndFrames, initiates call termination if auto_hang_up is enabled.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.
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
            await self._reset_audio_out()
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = await self._resample_audio_out(frame)
            if not data:
                return None

            payload = base64.b64encode(audioop.lin2ulaw(data, 2)).decode("utf-8")
            answer = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            }

            return json.dumps(answer)
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        # Return None for unhandled frames
        return None

    async def _hang_up_call(self):
        """Hang up the Twilio call using Twilio's REST API."""
        try:
            import aiohttp

            # __init__ guarantees account_sid, call_sid and exactly one credential type
            # (auth_token or the API key pair) are set whenever auto_hang_up is True,
            # which is the only path that reaches this method.
            account_sid = cast(str, self._account_sid)
            call_sid = cast(str, self._call_sid)
            # REST endpoint for updating the call. Derived from base_url when set
            # (Twilio-API-compatible / self-hosted), else Twilio's region/edge FQDN.
            endpoint = _build_call_resource_url(
                account_sid,
                call_sid,
                base_url=self._base_url,
                region=self._region,
                edge=self._edge,
            )

            # Basic Auth pair: account_sid + auth_token, or api_key_sid + api_key_secret.
            if self._auth_token:
                login = account_sid
                password = cast(str, self._auth_token)
            else:
                login = cast(str, self._api_key_sid)
                password = cast(str, self._api_key_secret)
            auth = aiohttp.BasicAuth(login, password)

            # Parameters to set the call status to "completed" (hang up)
            params = {"Status": "completed"}

            # Make the POST request to update the call
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, auth=auth, data=params) as response:
                    if response.status == 200:
                        logger.info(f"Successfully terminated Twilio call {call_sid}")
                    elif response.status == 404:
                        # Handle the case where the call has already ended
                        # Error code 20404: "The requested resource was not found"
                        # Source: https://www.twilio.com/docs/errors/20404
                        try:
                            error_data = await response.json()
                            if error_data.get("code") == 20404:
                                logger.debug(f"Twilio call {call_sid} was already terminated")
                                return
                        except Exception:
                            pass  # Fall through to log the raw error

                        # Log other 404 errors
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Twilio call {call_sid}: "
                            f"Status {response.status}, Response: {error_text}"
                        )
                    else:
                        # Log other errors
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Twilio call {call_sid}: "
                            f"Status {response.status}, Response: {error_text}"
                        )

        except Exception as e:
            logger.error(f"Failed to hang up Twilio call: {e}")

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Twilio WebSocket data to Pipecat frames.

        Handles conversion of Twilio media events to appropriate Pipecat frames.

        Args:
            data: The raw WebSocket data from Twilio.

        Returns:
            A Pipecat frame corresponding to the Twilio event, or None if unhandled.
        """
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON message: {data}")
            return None

        if message.get("event") == "media":
            payload_base64 = message.get("media", {}).get("payload")
            if not payload_base64:
                return None
            payload = base64.b64decode(payload_base64)

            audio = await self._resample_audio_in(audioop.ulaw2lin(payload, 2))
            if not audio:
                return None

            return InputAudioRawFrame(audio=audio, num_channels=1, sample_rate=self._sample_rate)
        elif message.get("event") == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                # Handle case where string doesn't match any enum value
                return None
        else:
            return None
