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
"""

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
    SUPPORTED_VOICES,
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
]
