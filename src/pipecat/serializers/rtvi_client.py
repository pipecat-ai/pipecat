#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame serializer for a Pipecat pipeline acting as an RTVI *client*.

This is the client-side mirror of how a bot emits RTVI. It hands the RTVI
*server* messages a bot sends to the pipeline, and serializes a client's
outgoing frames (audio, and RTVI client messages the caller builds) onto the
wire. Pair it with a WebSocket client transport to let a Pipecat pipeline talk
to a bot that runs an RTVI *server* transport (e.g. the eval transport).

It is the inverse of
:class:`~pipecat.processors.frameworks.rtvi.observer.RTVIObserver`, which turns
pipeline frames into server messages on the bot side.

Every message from the bot stays a raw message, an
:class:`~pipecat.frames.frames.InputTransportMessageFrame`. The bot's messages
are about the bot (its speaking, its LLM's text, its function calls, ...) or
about this pipeline as the bot sees it (its ``user-transcription``), and this
pipeline's own frames would pass them off as its own: a
``BotStartedSpeakingFrame`` here means this pipeline's output started playing.
Kept as one kind of frame, they also reach the pipeline in the order the bot
sent them.
"""

import base64
import json

from loguru import logger

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    Frame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class RTVIClientSerializer(FrameSerializer):
    """Bridge pipeline frames and RTVI wire messages from the client side."""

    def __init__(self, **kwargs):
        """Initialize the serializer.

        Args:
            **kwargs: Additional arguments passed to ``FrameSerializer``.
        """
        # We must not ignore RTVI messages: the caller builds RTVI client messages
        # (client-ready, send-text, ...) as transport-message frames for us to emit.
        super().__init__(params=FrameSerializer.InputParams(ignore_rtvi_messages=False), **kwargs)
        self._next_id = 0

    def _message_id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize an outgoing client frame to an RTVI wire message.

        Args:
            frame: The frame to serialize.

        Returns:
            JSON text for an RTVI client message, or ``None`` to drop the frame.
        """
        # An RTVI client message the caller built directly (client-ready, send-text,
        # dtmf, eval-*): emit its envelope as-is.
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            message = frame.message
            if isinstance(message, dict) and message.get("label") == RTVI.MESSAGE_LABEL:
                return json.dumps(message)
            return None
        # The pipeline's audio output (e.g. the user-side TTS) -> raw-audio to the bot.
        if isinstance(frame, OutputAudioRawFrame):
            return json.dumps(
                {
                    "label": RTVI.MESSAGE_LABEL,
                    "type": "raw-audio",
                    "id": self._message_id(),
                    "data": {
                        "base64Audio": base64.b64encode(frame.audio).decode("ascii"),
                        "sampleRate": frame.sample_rate,
                        "numChannels": frame.num_channels,
                    },
                }
            )
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize an incoming RTVI server message, as the raw message it is.

        Args:
            data: JSON text (or bytes) sent by the bot.

        Returns:
            The bot's message as an ``InputTransportMessageFrame``, or ``None`` if
            it is not an RTVI message.
        """
        try:
            message = json.loads(data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"RTVIClientSerializer: dropping non-JSON message: {e}")
            return None

        if not isinstance(message, dict) or message.get("label") != RTVI.MESSAGE_LABEL:
            return None

        return InputTransportMessageFrame(message=message)
