#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""App messages exchanged with a Keyframe avatar in a Daily room.

The bot and the avatar are both participants of the Daily room. The bot sends
its audio and control messages to the avatar as Daily app messages, and the
avatar reports its playback the same way.

From the bot to the avatar::

    {"kfl": 2, "type": "audio", "turn": "t7", "seq": 0, "data": "<base64>", "done": false}
    {"kfl": 2, "type": "interrupt"}
    {"kfl": 2, "type": "emotion_now", "emotion": "happy"}

From the avatar to the bot::

    {"kfl": 2, "type": "hello", "protocol": "2"}
    {"kfl": 2, "type": "playback_started", "turn": "t7"}
    {"kfl": 2, "type": "playback_finished", "turn": "t7", "playback_position": 3.2,
     "interrupted": false}

The audio messages sharing a ``turn`` name are one turn of bot speech, numbered
from ``seq`` 0. The message with ``done`` ends the turn. ``data`` is base64 of a
sequence of records, each ``type (u8) | payload length (u32 LE) | payload``:

- ``0x01`` audio: PCM16LE, 24 kHz, mono.
- ``0x02`` cue: UTF-8 JSON, ``{"emotion": "happy", "at": 57600}``, where ``at``
  is a sample offset from the start of the turn's audio.

``interrupt`` ends every turn sent so far.
"""

import base64
import json
import struct
from typing import Any

PROTOCOL_VERSION = 2

# Sample rate of the audio sent to the avatar: PCM16, mono.
SAMPLE_RATE = 24000

_RECORD_AUDIO = 0x01
_RECORD_CUE = 0x02
_HEADER = struct.Struct("<BI")


def audio_record(pcm: bytes) -> bytes:
    """Encode an audio record.

    Args:
        pcm: PCM16LE, 24 kHz, mono audio.

    Returns:
        The encoded record.
    """
    return _HEADER.pack(_RECORD_AUDIO, len(pcm)) + pcm


def cue_record(emotion: str, at: int) -> bytes:
    """Encode a cue record, which changes the avatar's emotion at a sample of the turn.

    Args:
        emotion: The emotion to change to.
        at: Sample offset from the start of the turn's audio.

    Returns:
        The encoded record.
    """
    body = json.dumps({"emotion": emotion, "at": at}, separators=(",", ":")).encode()
    return _HEADER.pack(_RECORD_CUE, len(body)) + body


def audio_message(turn: str, seq: int, records: bytes, *, done: bool = False) -> dict[str, Any]:
    """Build an audio message.

    Args:
        turn: Name of the turn the message belongs to.
        seq: Position of the message in the turn, starting at 0.
        records: Encoded records. May be empty.
        done: Whether this is the turn's last message.

    Returns:
        The app message payload.
    """
    return {
        "kfl": PROTOCOL_VERSION,
        "type": "audio",
        "turn": turn,
        "seq": seq,
        "data": base64.b64encode(records).decode(),
        "done": done,
    }


def interrupt_message() -> dict[str, Any]:
    """Build an interrupt message, which stops the avatar's playback.

    Returns:
        The app message payload.
    """
    return {"kfl": PROTOCOL_VERSION, "type": "interrupt"}


def emotion_now_message(emotion: str) -> dict[str, Any]:
    """Build a message that changes the avatar's emotion right away.

    Args:
        emotion: The emotion to change to.

    Returns:
        The app message payload.
    """
    return {"kfl": PROTOCOL_VERSION, "type": "emotion_now", "emotion": emotion}


def message_type(message: Any) -> str | None:
    """Get the type of an app message from the avatar.

    Args:
        message: An app message received in the Daily room.

    Returns:
        The message type, such as ``hello`` or ``playback_finished``, or None if
        the message is not one of this protocol.
    """
    if not isinstance(message, dict) or message.get("kfl") != PROTOCOL_VERSION:
        return None
    kind = message.get("type")
    return kind if isinstance(kind, str) else None
