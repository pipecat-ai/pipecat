import logging
from functools import wraps
from typing import Any, Callable, Optional, Union

from fastapi import WebSocket
from pipecatcloud.agent import DailySessionArguments, WebsocketSessionArguments
from pydantic import BaseModel, ConfigDict, create_model

from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.transports.network.small_webrtc import (
    SmallWebRTCConnection,
    SmallWebRTCTransport,
)
from pipecat.transports.services.daily import DailyParams, DailyTransport

logger = logging.getLogger(__name__)


def _get_model_fields(model_class):
    """Get all fields from a Pydantic model class."""
    return {
        field_name: (field_info.annotation, field_info.default)
        for field_name, field_info in model_class.model_fields.items()
    }


# Dynamically create PipecatCloudParams by combining fields from all parameter classes
PipecatCloudParams = create_model(
    "PipecatCloudParams",
    __config__=ConfigDict(arbitrary_types_allowed=True),
    __doc__="""Parameters for PipecatCloudTransport.

    This class combines parameters from all transport types. When a specific transport
    is created, it will extract only the parameters relevant to that transport type.
    
    The fields are automatically inherited from:
    - TransportParams
    - FastAPIWebsocketParams
    - DailyParams
    """,
    **{
        **_get_model_fields(TransportParams),
        **_get_model_fields(FastAPIWebsocketParams),
        **_get_model_fields(DailyParams),
    },
)


def _extract_matching_fields(source_obj, target_class):
    """Extract fields from source_obj that match the fields in target_class."""
    target_fields = set(target_class.model_fields.keys())
    matching_fields = {
        field: getattr(source_obj, field) for field in target_fields if hasattr(source_obj, field)
    }
    return target_class(**matching_fields)


# Add the conversion methods to PipecatCloudParams
def to_transport_params(self) -> TransportParams:
    """Convert to TransportParams."""
    return _extract_matching_fields(self, TransportParams)


def to_websocket_params(self) -> FastAPIWebsocketParams:
    """Convert to FastAPIWebsocketParams."""
    return _extract_matching_fields(self, FastAPIWebsocketParams)


def to_daily_params(self) -> DailyParams:
    """Convert to DailyParams."""
    params = _extract_matching_fields(self, DailyParams)
    # Also copy over the base transport params
    transport_params = self.to_transport_params()
    for key, value in transport_params.model_dump().items():
        setattr(params, key, value)
    return params


# Add the conversion methods to the dynamically created class
setattr(PipecatCloudParams, "to_transport_params", to_transport_params)
setattr(PipecatCloudParams, "to_websocket_params", to_websocket_params)
setattr(PipecatCloudParams, "to_daily_params", to_daily_params)


class WebRTCSessionArguments:
    """Arguments for creating a SmallWebRTCTransport session."""

    def __init__(
        self,
        webrtc_connection: SmallWebRTCConnection,
    ):
        self.webrtc_connection = webrtc_connection


SessionArguments = Union[WebsocketSessionArguments, DailySessionArguments, WebRTCSessionArguments]


class PipecatCloudTransport(BaseTransport):
    """A transport that wraps FastAPIWebsocketTransport, SmallWebRTCTransport, and DailyTransport.

    This transport will instantiate one of the three underlying transports based on the
    session arguments provided to the constructor.

    Event handlers:
        @event_handler("on_client_connected"): Called when a client connects. Maps to:
            - FastAPIWebsocketTransport.event_handler("on_client_connected")
            - SmallWebRTCTransport.event_handler("on_client_connected")
            - DailyTransport.event_handler("on_first_participant_joined")

        @event_handler("on_client_disconnected"): Called when a client disconnects. Maps to:
            - FastAPIWebsocketTransport.event_handler("on_client_disconnected")
            - SmallWebRTCTransport.event_handler("on_client_disconnected")
            - DailyTransport.event_handler("on_participant_left")

        Other event handlers are passed through directly to the underlying transport.

    Args:
        session_args: Arguments for creating the session. The type of arguments determines
            which transport will be used.
        params: Configuration parameters for the transport. Parameters will be extracted
            based on the session arguments type.
        input_name: Optional name for the input transport.
        output_name: Optional name for the output transport.
    """

    # Event name mappings for each transport type
    # Only include events that need to be mapped differently
    _EVENT_MAPPINGS = {
        DailyTransport: {
            "on_client_connected": "on_first_participant_joined",
            "on_client_disconnected": "on_participant_left",
        },
    }

    def __init__(
        self,
        session_args: SessionArguments,
        params: Optional[PipecatCloudParams] = None,
        *,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)
        logger.debug(f"SessionArguments: {session_args}")

        params = params or PipecatCloudParams()
        self._pending_handlers = {}

        # Create the appropriate transport based on session arguments type
        if isinstance(session_args, WebsocketSessionArguments):
            logger.info("Using FastAPIWebsocketTransport")
            websocket_params = params.to_websocket_params()
            self._transport = FastAPIWebsocketTransport(
                session_args.websocket,
                websocket_params,
                input_name=input_name,
                output_name=output_name,
            )
        elif isinstance(session_args, DailySessionArguments):
            logger.info("Using DailyTransport")
            daily_params = params.to_daily_params()
            self._transport = DailyTransport(
                session_args.room_url,
                session_args.token,
                session_args.bot_name,
                params=daily_params,
                input_name=input_name,
                output_name=output_name,
            )
        elif isinstance(session_args, WebRTCSessionArguments):
            logger.info("Using SmallWebRTCTransport")
            transport_params = params.to_transport_params()
            self._transport = SmallWebRTCTransport(
                session_args.webrtc_connection,
                transport_params,
                input_name=input_name,
                output_name=output_name,
            )
        else:
            raise ValueError(f"Unsupported session arguments type: {type(session_args)}")

        # Register any handlers that were added before transport creation
        for event_name, handlers in self._pending_handlers.items():
            for handler in handlers:
                self._register_handler(event_name, handler)

    def _register_handler(self, event_name: str, handler: Callable[..., Any]) -> None:
        """Register a handler with the appropriate transport method."""
        transport_type = type(self._transport)

        # If the transport type has mappings and the event needs to be mapped
        if (
            transport_type in self._EVENT_MAPPINGS
            and event_name in self._EVENT_MAPPINGS[transport_type]
        ):
            mapped_event = self._EVENT_MAPPINGS[transport_type][event_name]
        else:
            # Pass through the event name directly if no mapping exists
            mapped_event = event_name

        self._transport.event_handler(mapped_event)(handler)

    def event_handler(self, event_name: str) -> Callable[..., Any]:
        """Register an event handler.

        Args:
            event_name: The name of the event to handle. Common events:
                - "on_client_connected": Called when a client connects
                - "on_client_disconnected": Called when a client disconnects
                Other event names are passed through to the underlying transport.

        Returns:
            A decorator that registers the handler function.
        """

        def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
            if not hasattr(self, "_transport"):
                # Store the handler to be registered when the transport is created
                if event_name not in self._pending_handlers:
                    self._pending_handlers[event_name] = []
                self._pending_handlers[event_name].append(handler)
            else:
                self._register_handler(event_name, handler)
            return handler

        return decorator

    async def start(self, frame):
        """Start the transport."""
        await self._transport.start(frame)

    async def stop(self, frame):
        """Stop the transport."""
        await self._transport.stop(frame)

    async def cancel(self, frame):
        """Cancel the transport."""
        await self._transport.cancel(frame)

    @property
    def input(self):
        """Get the input transport."""
        return self._transport.input

    @property
    def output(self):
        """Get the output transport."""
        return self._transport.output
