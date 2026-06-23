#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame serializer for a Pipecat pipeline acting as an RTVI *client*.

This is the client-side mirror of how a bot emits RTVI. It deserializes the RTVI
*server* messages a bot sends (``bot-llm-text``, ``user-transcription``, ...) into
pipeline frames, and serializes a client's outgoing frames (audio, and RTVI client
messages the caller builds) onto the wire. Pair it with a WebSocket client
transport to let a Pipecat pipeline talk to a bot that runs an RTVI *server*
transport (e.g. the eval transport).

It is the inverse of
:class:`~pipecat.processors.frameworks.rtvi.observer.RTVIObserver`, which turns
pipeline frames into server messages on the bot side.

.. note::
    This first cut covers the messages a text-mode conversation needs (the LLM
    response lifecycle, transcriptions, speaking/interruption signals). TTS text,
    function-call, metrics, and inbound bot audio are added as the eval-simulation
    work progresses.
"""

import base64
import json

from loguru import logger

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
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
        """Deserialize an incoming RTVI server message into a pipeline frame.

        Args:
            data: JSON text (or bytes) sent by the bot.

        Returns:
            The corresponding frame, or ``None`` if the message is not an RTVI
            message or has no frame mapping yet (handled elsewhere).
        """
        try:
            message = json.loads(data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"RTVIClientSerializer: dropping non-JSON message: {e}")
            return None

        if not isinstance(message, dict) or message.get("label") != RTVI.MESSAGE_LABEL:
            return None

        msg_type = message.get("type")
        payload = message.get("data") or {}

        match msg_type:
            case "bot-llm-started":
                return LLMFullResponseStartFrame()
            case "bot-llm-text":
                return LLMTextFrame(text=payload.get("text", ""))
            case "bot-llm-stopped":
                return LLMFullResponseEndFrame()
            case "bot-started-speaking":
                return BotStartedSpeakingFrame()
            case "bot-stopped-speaking":
                return BotStoppedSpeakingFrame()
            case "bot-interrupted":
                return InterruptionFrame()
            case "user-transcription":
                text = payload.get("text", "")
                user_id = payload.get("user_id", "")
                timestamp = payload.get("timestamp", "")
                if payload.get("final", True):
                    return TranscriptionFrame(text=text, user_id=user_id, timestamp=timestamp)
                return InterimTranscriptionFrame(text=text, user_id=user_id, timestamp=timestamp)
            case _:
                # bot-ready (handshake) and not-yet-mapped messages (tts text,
                # function calls, metrics, bot audio) are handled elsewhere / later.
                return None
