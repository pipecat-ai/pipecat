#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gnani Vachana speech AI service integrations for Pipecat.

This module re-exports services from the ``pipecat-gnani`` package so that
they are available under the ``pipecat.services.gnani`` namespace.

STT Services:
- GnaniHttpSTTService: REST-based file transcription (requires VAD)
- GnaniSTTService: WebSocket streaming speech-to-text with VAD

TTS Services:
- GnaniHttpTTSService: REST-based text-to-speech
- GnaniSSETTSService: SSE streaming text-to-speech (lower latency)
- GnaniTTSService: WebSocket streaming text-to-speech with interruption handling

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
