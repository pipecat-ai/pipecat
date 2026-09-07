"""Cloudonix Media Streams WebSocket protocol serializer for Pipecat."""

from typing import TYPE_CHECKING

from loguru import logger

from pipecat.serializers.twilio import TwilioFrameSerializer

if TYPE_CHECKING:
    from pipecat.serializers.call_strategies import HangupStrategy, TransferStrategy


class CloudonixFrameSerializer(TwilioFrameSerializer):
    """Serializer for Cloudonix Media Streams WebSocket protocol.

    This serializer extends TwilioFrameSerializer for Cloudonix compatibility,
    providing Cloudonix-specific call termination functionality while reusing
    Twilio's audio handling and frame processing capabilities.
    """

    def __init__(
        self,
        call_id: str,
        stream_sid: str,
        domain_id: str | None = None,
        bearer_token: str | None = None,
        region: str | None = None,
        edge: str | None = None,
        hangup_strategy: "HangupStrategy | None" = None,
        transfer_strategy: "TransferStrategy | None" = None,
        params: TwilioFrameSerializer.InputParams | None = None,
    ):
        """Initialize the CloudonixFrameSerializer.

        Args:
            call_id: The associated Cloudonix Call ID (required for auto hang-up)
            stream_sid: The associated Cloudonix Stream SID (required for streaming audio)
            domain_id: Cloudonix domain ID (required for auto hang-up).
            bearer_token: Cloudonix bearer token (required for auto hang-up).
            region: Optional region parameter (legacy compatibility).
            edge: Optional edge parameter (legacy compatibility).
            hangup_strategy: Strategy for handling call hangups. The strategy receives
                context with Twilio-compatible keys (call_sid, account_sid, auth_token)
                mapped from Cloudonix values (call_id, domain_id, bearer_token).
            transfer_strategy: Strategy for handling call transfers, invoked on an
                EndFrame carrying the transfer reason. Receives the same
                Twilio-compatible context keys as ``hangup_strategy``.
            params: Configuration parameters.
        """
        self._call_id = call_id

        super().__init__(
            stream_sid=stream_sid,
            call_sid=call_id,
            account_sid=domain_id,
            auth_token=bearer_token,
            region=region,
            edge=edge,
            hangup_strategy=hangup_strategy,
            transfer_strategy=transfer_strategy,
            params=params,
        )

        logger.info(f"Cloudonix serializer initialized with call_id: {self._call_id}")
