#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame serializer that bridges the eval harness to a bot over RTVI.

The eval harness (:mod:`pipecat.evals.harness`) talks to a bot using the RTVI
protocol over a plain WebSocket (``WebsocketServerTransport``). This serializer
is the only glue needed:

- **Inbound** (harness → bot): JSON RTVI messages are wrapped in an
  :class:`~pipecat.frames.frames.InputTransportMessageFrame` so the bot's
  ``RTVIProcessor`` parses and routes them (``send-text``, ``raw-audio``,
  ``client-ready``, ...). Two control messages are the exception: a
  ``client-message`` with ``t = "eval-reset"`` is short-circuited into an
  :class:`LLMMessagesUpdateFrame` (reseeding the bot's context), and one with
  ``t = "eval-configure"`` into an
  :class:`~pipecat.processors.frameworks.rtvi.frames.RTVIConfigureObserverFrame`
  (raising the function-call report level for the eval). Both keep eval-specific
  behavior out of the bot, and the latter is the trust boundary that lets agents
  keep the secure default report level in production.

- **Outbound** (bot → harness): RTVI server messages (carried as
  ``OutputTransportMessage*Frame`` with the ``rtvi-ai`` label) are emitted as
  their raw JSON. Everything else — notably bot audio — is dropped, since the
  harness only asserts on semantic events.

This lives under ``pipecat.evals`` rather than ``pipecat.serializers`` because
it carries eval-specific behavior (the reset short-circuit, dropping audio) and
is not a general-purpose RTVI transport serializer.
"""

import json
from typing import Any

from loguru import logger

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    Frame,
    InputTransportMessageFrame,
    LLMMessagesUpdateFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frameworks.rtvi.frames import RTVIConfigureObserverFrame
from pipecat.processors.frameworks.rtvi.observer import RTVIFunctionCallReportLevel
from pipecat.serializers.base_serializer import FrameSerializer

# A ``client-message`` with this ``t`` is intercepted by the serializer and
# turned into an ``LLMMessagesUpdateFrame`` instead of being forwarded to the
# RTVIProcessor. Keeps per-eval context seeding out of the bot.
EVAL_RESET_MESSAGE_TYPE = "eval-reset"

# A ``client-message`` with this ``t`` is intercepted and turned into an
# ``RTVIConfigureObserverFrame``. This is the trust boundary for raising the
# function-call report level: only the eval transport understands it, so a
# production RTVI serializer can't be elevated by a remote client.
EVAL_CONFIGURE_MESSAGE_TYPE = "eval-configure"


class RTVIEvalSerializer(FrameSerializer):
    """Bridges JSON RTVI messages and pipeline frames for the eval harness.

    Use as the serializer of a ``WebsocketServerTransport`` when running a bot
    under the eval harness. The bot pipeline must include an ``RTVIProcessor``
    and pass an ``RTVIObserver`` to the task.
    """

    def __init__(self, **kwargs):
        """Initialize the serializer.

        Args:
            **kwargs: Additional arguments passed to ``FrameSerializer``.
        """
        # Do not ignore RTVI messages: the whole point is to put them on the
        # wire so the harness can observe semantic events.
        super().__init__(params=FrameSerializer.InputParams(ignore_rtvi_messages=False), **kwargs)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize an outbound frame to JSON for the harness.

        Only RTVI server messages are forwarded; all other frames (audio,
        control) are dropped.

        Args:
            frame: The frame to serialize.

        Returns:
            JSON text for an RTVI server message, or ``None`` to drop the frame.
        """
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            message = frame.message
            if isinstance(message, dict) and message.get("label") == RTVI.MESSAGE_LABEL:
                return json.dumps(message)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize an inbound JSON RTVI message into a frame.

        Args:
            data: JSON text (or bytes) sent by the harness.

        Returns:
            An ``LLMMessagesUpdateFrame`` for the eval-reset control message, an
            ``InputTransportMessageFrame`` wrapping any other RTVI message, or
            ``None`` if the payload is not a valid RTVI message.
        """
        try:
            message = json.loads(data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"RTVIEvalSerializer: dropping non-JSON message: {e}")
            return None

        if not isinstance(message, dict) or message.get("label") != RTVI.MESSAGE_LABEL:
            logger.warning(f"RTVIEvalSerializer: ignoring non-RTVI message: {message!r}")
            return None

        reset = self._maybe_reset_frame(message)
        if reset is not None:
            return reset

        configure = self._maybe_configure_frame(message)
        if configure is not None:
            return configure

        return InputTransportMessageFrame(message=message)

    def _maybe_reset_frame(self, message: dict) -> Frame | None:
        """Return an ``LLMMessagesUpdateFrame`` for the eval-reset message, else None."""
        if message.get("type") != "client-message":
            return None
        data: Any = message.get("data") or {}
        if not isinstance(data, dict) or data.get("t") != EVAL_RESET_MESSAGE_TYPE:
            return None
        payload = data.get("d") or {}
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        # run_llm=False: seed the context, don't trigger a response.
        return LLMMessagesUpdateFrame(messages=list(messages), run_llm=False)

    def _maybe_configure_frame(self, message: dict) -> Frame | None:
        """Return an ``RTVIConfigureObserverFrame`` for the eval-configure message, else None."""
        if message.get("type") != "client-message":
            return None
        data: Any = message.get("data") or {}
        if not isinstance(data, dict) or data.get("t") != EVAL_CONFIGURE_MESSAGE_TYPE:
            return None
        payload = data.get("d") or {}
        levels = payload.get("function_call_report_level") if isinstance(payload, dict) else None
        report_level = None
        if isinstance(levels, dict):
            # Values arrive as strings ("none"/"name"/"full"); coerce to the enum.
            report_level = {k: RTVIFunctionCallReportLevel(v) for k, v in levels.items()}
        return RTVIConfigureObserverFrame(function_call_report_level=report_level)
