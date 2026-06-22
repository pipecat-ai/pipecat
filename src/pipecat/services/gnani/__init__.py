#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gnani Vachana speech AI service integrations for Pipecat.

STT Services:
- GnaniHttpSTTService: REST-based file transcription (requires VAD)
- GnaniSTTService: WebSocket streaming speech-to-text with VAD

TTS Services:
- GnaniHttpTTSService: REST-based text-to-speech
- GnaniSSETTSService: SSE streaming text-to-speech (lower latency)
- GnaniTTSService: WebSocket streaming text-to-speech with interruption handling

API docs: https://docs.gnani.ai/api/introduction/introduction
"""

from pipecat.services.gnani._common import (
    STT_FORMAT_TRANSCRIBE,
    STT_FORMAT_VERBATIM,
    SUPPORTED_VOICES,
)
from pipecat.services.gnani.stt import (
    GnaniHttpSTTService,
    GnaniHttpSTTSettings,
    GnaniSTTService,
    GnaniSTTSettings,
)
from pipecat.services.gnani.tts import (
    GnaniHttpTTSService,
    GnaniHttpTTSSettings,
    GnaniSSETTSService,
    GnaniSSETTSSettings,
    GnaniTTSService,
    GnaniTTSSettings,
)

__all__ = [
    "GnaniHttpSTTService",
    "GnaniHttpSTTSettings",
    "GnaniSTTService",
    "GnaniSTTSettings",
    "GnaniHttpTTSService",
    "GnaniHttpTTSSettings",
    "GnaniSSETTSService",
    "GnaniSSETTSSettings",
    "GnaniTTSService",
    "GnaniTTSSettings",
    "SUPPORTED_VOICES",
    "STT_FORMAT_VERBATIM",
    "STT_FORMAT_TRANSCRIBE",
]
