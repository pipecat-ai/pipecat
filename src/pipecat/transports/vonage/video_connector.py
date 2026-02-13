#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Vonage Video Connector transport."""

from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    UserImageRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.vonage.client import (
    Session,
    Stream,
    Subscriber,
    VonageClient,
    VonageClientListener,
)

# the following "as" imports help to re-export these types and avoid type checking warnings
# when importing these types from the main transport module
from pipecat.transports.vonage.client import (
    SubscribeSettings as SubscribeSettings,
)
from pipecat.transports.vonage.client import (
    VonageException as VonageException,
)
from pipecat.transports.vonage.client import (
    VonageVideoConnectorTransportParams as VonageVideoConnectorTransportParams,
)


class VonageVideoConnectorInputTransport(BaseInputTransport):
    """Input transport for Vonage, handling audio input from the Vonage session.

    Receives audio from a Vonage Video session and pushes it as input frames.
    """

    _params: VonageVideoConnectorTransportParams

    def __init__(self, client: VonageClient, params: VonageVideoConnectorTransportParams):
        """Initialize the Vonage input transport.

        Args:
            client: The VonageClient instance to use.
            params: Transport parameters for input configuration.
        """
        super().__init__(params)
        self._initialized: bool = False
        self._client: VonageClient = client
        self._listener_id: int = -1
        self._connected: bool = False

    async def start(self, frame: StartFrame) -> None:
        """Start the Vonage input transport.

        Args:
            frame: The StartFrame to initiate the transport.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        if self._params.audio_in_enabled or self._params.video_in_enabled:
            self._listener_id = self._client.add_listener(
                VonageClientListener(
                    on_audio_in=self._audio_in_cb,
                    on_video_in=self._video_in_cb,
                    on_error=self._on_error_cb,
                )
            )
            try:
                await self._client.connect(frame)
                self._connected = True
            except Exception as exc:
                logger.error(f"Error connecting to Vonage session: {exc}")
                await self.push_error("Vonage video connector connection error", fatal=True)
                return

        await self.set_transport_ready(frame)

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self) -> None:
        """Cleanup input transport."""
        await super().cleanup()  # type: ignore
        await self._client.cleanup()

    async def _audio_in_cb(self, _session: Session, audio: InputAudioRawFrame) -> None:
        if self._connected and self._params.audio_in_enabled:
            await self.push_audio_frame(audio)

    async def _video_in_cb(self, _subscriber: Subscriber, video: UserImageRawFrame) -> None:
        if self._connected and self._params.video_in_enabled:
            await self.push_video_frame(video)

    async def _on_error_cb(self, session: Session, description: str, code: int) -> None:
        logger.error(
            f"Vonage input transport error session={session.id} code={code} description={description}"
        )
        if self._connected:
            await self.push_error("Vonage video connector error", fatal=True)

    async def stop(self, frame: EndFrame) -> None:
        """Stop the Vonage input transport.

        Args:
            frame: The EndFrame to stop the transport.
        """
        await super().stop(frame)
        await self._stop_client()

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel the Vonage input transport.

        Args:
            frame: The CancelFrame to cancel the transport.
        """
        await super().cancel(frame)
        await self._stop_client()

    async def _stop_client(self) -> None:
        if self._connected:
            self._client.remove_listener(self._listener_id)
            self._connected = False
            try:
                await self._client.disconnect()
            except Exception:
                pass

    async def subscribe_to_stream(self, stream_id: str, params: SubscribeSettings) -> None:
        """Subscribe to a participant's stream.

        Args:
            stream_id: The ID of the participant to subscribe to.
            params: Subscription parameters for the subscription.
        """
        await self._client.subscribe_to_stream(stream_id, params)


class VonageVideoConnectorOutputTransport(BaseOutputTransport):
    """Output transport for Vonage, handling audio output to the Vonage session.

    Sends audio frames to a Vonage Video session as output.
    """

    _params: VonageVideoConnectorTransportParams

    def __init__(self, client: VonageClient, params: VonageVideoConnectorTransportParams):
        """Initialize the Vonage output transport.

        Args:
            client: The VonageClient instance to use.
            params: Transport parameters for output configuration.
        """
        super().__init__(params)
        self._initialized: bool = False
        self._client = client
        self._connected: bool = False
        self._listener_id: int = -1

    async def start(self, frame: StartFrame) -> None:
        """Start the Vonage output transport.

        Args:
            frame: The StartFrame to initiate the transport.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        if self._params.audio_out_enabled or self._params.video_out_enabled:
            self._listener_id = self._client.add_listener(
                VonageClientListener(on_error=self._on_error_cb)
            )
            try:
                await self._client.connect(frame)
                self._connected = True
            except Exception as exc:
                logger.error(f"Error connecting to Vonage session: {exc}")
                await self.push_error("Vonage video connector connection error", fatal=True)
                return

        await self.set_transport_ready(frame)

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self) -> None:
        """Cleanup output transport."""
        await super().cleanup()  # type: ignore
        await self._client.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process a frame for the Vonage output transport.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # if we get an interruption frame, we need to ensure the buffers inside Vonage Video Connector are cleared
        if (
            self._connected
            and isinstance(frame, InterruptionFrame)
            and self._params.clear_buffers_on_interruption
        ):
            logger.info("Clearing Vonage media buffers due to interruption frame")
            self._client.clear_media_buffers()

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the Vonage session.

        Args:
            frame: The OutputAudioRawFrame to send.
        """
        result = False
        if self._connected and self._params.audio_out_enabled:
            result = await self._client.write_audio(frame)

        return result

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the transport.

        Args:
            frame: The output video frame to write.
        """
        result = False
        if self._connected and self._params.video_out_enabled:
            result = await self._client.write_video(frame)

        return result

    async def stop(self, frame: EndFrame) -> None:
        """Stop the Vonage output transport.

        Args:
            frame: The EndFrame to stop the transport.
        """
        await super().stop(frame)
        await self._stop_client()

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel the Vonage output transport.

        Args:
            frame: The CancelFrame to cancel the transport.
        """
        await super().cancel(frame)
        await self._stop_client()

    async def _stop_client(self) -> None:
        if self._connected:
            self._client.remove_listener(self._listener_id)
            self._connected = False
            try:
                await self._client.disconnect()
            except Exception:
                pass

    async def _on_error_cb(self, session: Session, description: str, code: int) -> None:
        logger.error(
            f"Vonage output transport error session={session.id} code={code} description={description}"
        )
        if self._connected:
            await self.push_error("Vonage video connector error", fatal=True)


class VonageVideoConnectorTransport(BaseTransport):
    """Vonage Video Connector transport implementation for Pipecat.

    Provides input and output audio transport for Vonage Video sessions, supporting event handling
    for session and participant lifecycle.

    Supported features:

    - Audio input and output transport for Vonage Video sessions
    - Event handler registration for session and participant events
    - Publisher and subscriber management
    - Configurable audio and migration parameters
    """

    _params: VonageVideoConnectorTransportParams

    def __init__(
        self,
        application_id: str,
        session_id: str,
        token: str,
        params: VonageVideoConnectorTransportParams,
    ):
        """Initialize the Vonage Video Connector transport.

        Args:
            application_id: The Vonage Video application ID.
            session_id: The session ID to connect to.
            token: The authentication token for the session.
            params: Transport parameters for input/output configuration.
        """
        super().__init__()
        self._params = params

        self._client = VonageClient(application_id, session_id, token, params)

        # Register supported handlers.
        self._register_event_handler("on_joined")
        self._register_event_handler("on_left")
        self._register_event_handler("on_error")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_joined")
        self._register_event_handler("on_participant_left")

        self._client.add_listener(
            VonageClientListener(
                on_connected=self._on_connected,
                on_disconnected=self._on_disconnected,
                on_error=self._on_error,
                on_stream_received=self._on_stream_received,
                on_stream_dropped=self._on_stream_dropped,
                on_subscriber_connected=self._on_subscriber_connected,
                on_subscriber_disconnected=self._on_subscriber_disconnected,
            )
        )

        self._input: Optional[VonageVideoConnectorInputTransport] = None
        self._output: Optional[VonageVideoConnectorOutputTransport] = None
        self._one_stream_received: bool = False

    def input(self) -> FrameProcessor:
        """Get the input transport for Vonage.

        Returns:
            The VonageVideoConnectorInputTransport instance.
        """
        if not self._input:
            self._input = VonageVideoConnectorInputTransport(self._client, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        """Get the output transport for Vonage.

        Returns:
            The VonageVideoConnectorOutputTransport instance.
        """
        if not self._output:
            self._output = VonageVideoConnectorOutputTransport(self._client, self._params)
        return self._output

    async def subscribe_to_stream(self, stream_id: str, params: SubscribeSettings) -> None:
        """Subscribe to a participant's stream.

        Args:
            stream_id: The ID of the participant to subscribe to.
            params: Subscription parameters for the subscription.
        """
        if self._input:
            await self._input.subscribe_to_stream(stream_id, params)

    async def _on_connected(self, session: Session) -> None:
        """Handle session connected event.

        Args:
            session: The connected Session object.
        """
        await self._call_event_handler("on_joined", {"sessionId": session.id})

    async def _on_disconnected(self, session: Session) -> None:
        """Handle session disconnected event.

        Args:
            session: The disconnected Session object.
        """
        await self._call_event_handler("on_left", {"sessionId": session.id})

    async def _on_error(self, _session: Session, description: str, _code: int) -> None:
        """Handle session error event.

        Args:
            _session: The Session object.
            description: Error description.
            _code: Error code.
        """
        await self._call_event_handler("on_error", description)

    async def _on_stream_received(self, session: Session, stream: Stream) -> None:
        """Handle stream received event.

        Args:
            session: The Session object.
            stream: The received Stream object.
        """
        if not self._one_stream_received:
            self._one_stream_received = True
            await self._call_event_handler(
                "on_first_participant_joined",
                {
                    "sessionId": session.id,
                    "streamId": stream.id,
                    "connectionData": stream.connection.data,
                },
            )

        await self._call_event_handler(
            "on_participant_joined",
            {
                "sessionId": session.id,
                "streamId": stream.id,
                "connectionData": stream.connection.data,
            },
        )

    async def _on_stream_dropped(self, session: Session, stream: Stream) -> None:
        """Handle stream dropped event.

        Args:
            session: The Session object.
            stream: The dropped Stream object.
        """
        await self._call_event_handler(
            "on_participant_left",
            {
                "sessionId": session.id,
                "streamId": stream.id,
                "connectionData": stream.connection.data,
            },
        )

    async def _on_subscriber_connected(self, subscriber: Subscriber) -> None:
        """Handle subscriber connected event.

        Args:
            subscriber: The connected Subscriber object.
        """
        await self._call_event_handler(
            "on_client_connected",
            {
                "subscriberId": subscriber.stream.id,
                "streamId": subscriber.stream.id,
                "connectionData": subscriber.stream.connection.data,
            },
        )

    async def _on_subscriber_disconnected(self, subscriber: Subscriber) -> None:
        """Handle subscriber disconnected event.

        Args:
            subscriber: The disconnected Subscriber object.
        """
        await self._call_event_handler(
            "on_client_disconnected",
            {
                "subscriberId": subscriber.stream.id,
                "streamId": subscriber.stream.id,
                "connectionData": subscriber.stream.connection.data,
            },
        )
