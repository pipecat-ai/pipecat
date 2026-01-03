"""Cloudonix Media Streams WebSocket protocol serializer for Pipecat."""

from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler
from pipecat.serializers.twilio import TwilioFrameSerializer


class CloudonixFrameSerializer(TwilioFrameSerializer):
    """Serializer for Cloudonix Media Streams WebSocket protocol.

    This serializer extends TwilioFrameSerializer for Cloudonix compatibility,
    providing Cloudonix-specific call termination functionality while reusing
    Twilio's audio handling and frame processing capabilities.
    """

    class InputParams(BaseModel):
        """Configuration parameters for CloudonixFrameSerializer.

        Parameters:
            cloudonix_sample_rate: Sample rate same as used by Twilio, defaults to 8000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        cloudonix_sample_rate: int = 8000
        sample_rate: Optional[int] = None
        auto_hang_up: bool = True

    def __init__(
        self,
        stream_sid: str,
        call_sid: Optional[str] = None,
        domain_id: Optional[str] = None,
        bearer_token: Optional[str] = None,
        session_token: Optional[str] = None,
        region: Optional[str] = None,
        edge: Optional[str] = None,
        params: Optional[InputParams] = None,
    ):
        """Initialize the CloudonixFrameSerializer.

        Args:
            stream_sid: The WebSocket Stream SID (Twilio-compatible).
            call_sid: The associated Cloudonix Call SID (optional, but required for auto hang-up).
            domain_id: Cloudonix domain ID (required for auto hang-up).
            bearer_token: Cloudonix bearer token (required for auto hang-up).
            session_token: Cloudonix session token from call initiation (optional, available for hangup).
            region: Optional region parameter (legacy compatibility).
            edge: Optional edge parameter (legacy compatibility).
            params: Configuration parameters.
        """
        self._params = params or CloudonixFrameSerializer.InputParams()

        # Validate hangup-related parameters if auto_hang_up is enabled
        if self._params.auto_hang_up:
            # Validate required credentials
            missing_credentials = []
            if not call_sid:
                missing_credentials.append("call_sid")
            if not domain_id:
                missing_credentials.append("domain_id")
            if not bearer_token:
                missing_credentials.append("bearer_token")

            if missing_credentials:
                raise ValueError(
                    f"auto_hang_up is enabled but missing required parameters: {', '.join(missing_credentials)}"
                )

        self._stream_sid = stream_sid
        self._call_sid = call_sid
        self._domain_id = domain_id
        self._bearer_token = bearer_token
        self._session_token = session_token

        self.cloudonix_sample_rate = self._params.cloudonix_sample_rate
        self._twilio_sample_rate = (
            self._params.cloudonix_sample_rate
        )  # For parent class compatibility
        self._sample_rate = 0  # Pipeline input rate

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._hangup_attempted = False

        logger.info(f"Cloudonix serializer initialized with session_token: {session_token}")
        logger.info(f"Cloudonix serializer params are now {self.__dict__}")

    async def _hang_up_call(self):
        """Terminate the Cloudonix call by issuing a DELETE request to the session endpoint."""
        logger.debug(f"Attempting hangup for call {self._call_sid}")

        # If session_token is not available, fall back to WebSocket close behavior
        if not self._session_token:
            logger.warning(
                f"No session_token available for call {self._call_sid}. "
                f"Relying on WebSocket close for hangup."
            )
            return

        # Validate required parameters for API call
        if not self._domain_id or not self._bearer_token:
            logger.warning(
                f"Missing domain_id or bearer_token for call {self._call_sid}. "
                f"Cannot perform explicit hangup via API."
            )
            return

        try:
            import aiohttp

            # Construct the DELETE session endpoint
            # Using "self" as customer-id as per Cloudonix documentation
            base_url = "https://api.cloudonix.io"
            endpoint = f"{base_url}/customers/self/domains/{self._domain_id}/sessions/{self._session_token}"

            # Prepare headers with Bearer token authentication
            headers = {
                "Authorization": f"Bearer {self._bearer_token}",
                "Content-Type": "application/json",
            }

            logger.info(f"Terminating Cloudonix call {self._call_sid} via DELETE {endpoint}")

            # Make the DELETE request to terminate the session
            async with aiohttp.ClientSession() as session:
                async with session.delete(endpoint, headers=headers) as response:
                    status = response.status
                    response_text = await response.text()

                    if status in (200, 204, 404):
                        # 200/204: Success
                        # 404: Session already terminated (acceptable)
                        logger.info(
                            f"Successfully terminated Cloudonix session {self._session_token} "
                            f"(HTTP {status}), Response: {response_text}"
                        )
                    else:
                        logger.warning(
                            f"Unexpected response terminating Cloudonix session {self._session_token}: "
                            f"HTTP {status}, Response: {response_text}"
                        )

        except Exception as e:
            logger.error(f"Error terminating Cloudonix call {self._call_sid}: {e}", exc_info=True)
