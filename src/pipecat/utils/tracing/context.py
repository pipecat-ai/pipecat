from typing import Dict, Optional

from loguru import logger
from opentelemetry import trace

# Global turn context storage
_CURRENT_TURN_ID: Optional[str] = None
_TURN_COUNTER: int = 0
_SERVICE_SPANS: Dict[str, trace.Span] = {}  # turn_id:service_name -> span
_TURN_ACTIVE: bool = False


def get_current_turn_id() -> Optional[str]:
    """Get the ID of the current turn."""
    return _CURRENT_TURN_ID


def get_turn_counter() -> int:
    """Get the current turn count."""
    return _TURN_COUNTER


def increment_turn_counter() -> int:
    """Increment and return the turn counter."""
    global _TURN_COUNTER
    _TURN_COUNTER += 1
    return _TURN_COUNTER


def is_turn_active() -> bool:
    """Check if a turn is currently active."""
    global _TURN_ACTIVE
    return _TURN_ACTIVE


def get_service_span(service_name: str) -> Optional[trace.Span]:
    """Get the service span for the current turn."""
    if not _CURRENT_TURN_ID:
        return None
    key = f"{_CURRENT_TURN_ID}:{service_name}"
    return _SERVICE_SPANS.get(key)


def store_service_span(service_name: str, span: trace.Span) -> None:
    """Store a service span for the current turn."""
    if _CURRENT_TURN_ID:
        key = f"{_CURRENT_TURN_ID}:{service_name}"
        _SERVICE_SPANS[key] = span


def start_turn() -> Optional[str]:
    """Start a new turn context.

    Returns:
        str: The ID of the new turn.
    """
    global _CURRENT_TURN_ID, _TURN_ACTIVE

    # Generate turn ID
    turn_count = get_turn_counter()
    turn_id = f"Turn{turn_count}"

    # Store the current turn ID in the global context
    _CURRENT_TURN_ID = turn_id
    _TURN_ACTIVE = True

    # Get the tracer
    tracer = trace.get_tracer("pipecat")

    # Start a span for the turn
    turn_span = tracer.start_span(turn_id)
    turn_span.set_attribute("turn.id", turn_id)

    # Store the turn span
    store_service_span("turn", turn_span)

    logger.debug(f"Started {turn_id}")
    return turn_id


def end_turn() -> None:
    """End the current turn context."""
    global _CURRENT_TURN_ID, _TURN_ACTIVE

    if not _CURRENT_TURN_ID:
        return

    turn_id = _CURRENT_TURN_ID

    # Clean up all spans for this turn
    keys_to_remove = []
    for key in list(_SERVICE_SPANS.keys()):
        if key.startswith(f"{turn_id}:"):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        span = _SERVICE_SPANS.pop(key)
        if span and span.is_recording():
            span.end()

    # Clear the current turn ID
    _CURRENT_TURN_ID = None
    _TURN_ACTIVE = False

    logger.debug(f"Ended {turn_id}")
