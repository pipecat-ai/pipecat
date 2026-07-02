#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gnani Vachana speech AI service integrations for Pipecat.

Re-exports services from the ``pipecat-gnani`` package under the
``pipecat.services.gnani`` namespace.

STT services:

- GnaniHttpSTTService — REST-based file transcription (requires VAD)
- GnaniSTTService — WebSocket streaming speech-to-text with VAD

TTS services:

- GnaniHttpTTSService — REST-based text-to-speech
- GnaniSSETTSService — SSE streaming text-to-speech (lower latency)
- GnaniTTSService — WebSocket streaming synthesis with interruption handling

Voices: Karan (default), Simran, Nara, Riya, Viraj, Raju, Pranav, Kaveri,
Shubhra, Deepak.

API docs: https://docs.gnani.ai/api/introduction/introduction
"""

from pipecat_gnani import (
    STT_FORMAT_TRANSCRIBE,
    STT_FORMAT_VERBATIM,
    SUPPORTED_VOICES,
    GnaniHttpSTTService,
    GnaniHttpSTTSettings,
    GnaniHttpTTSService,
    GnaniHttpTTSSettings,
    GnaniSSETTSService,
    GnaniSSETTSSettings,
    GnaniSTTService,
    GnaniSTTSettings,
    GnaniTTSService,
    GnaniTTSSettings,
)

__all__ = [
    "GnaniHttpSTTService",
    "GnaniHttpSTTSettings",
    "GnaniHttpTTSService",
    "GnaniHttpTTSSettings",
    "GnaniSSETTSService",
    "GnaniSSETTSSettings",
    "GnaniSTTService",
    "GnaniSTTSettings",
    "GnaniTTSService",
    "GnaniTTSSettings",
    "STT_FORMAT_TRANSCRIBE",
    "STT_FORMAT_VERBATIM",
    "SUPPORTED_VOICES",
]
