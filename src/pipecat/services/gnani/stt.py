#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gnani Vachana Speech-to-Text service implementations.

Re-exports from the ``pipecat-gnani`` package. All implementation logic
lives in ``pipecat_gnani.stt``; this module keeps backward-compatible
imports under ``pipecat.services.gnani.stt``.

Services:
- GnaniHttpSTTService: REST-based file transcription (requires VAD in pipeline)
- GnaniSTTService: WebSocket streaming transcription with real-time VAD

Supported languages: as-IN, bn-IN, en-IN, gu-IN, hi-IN, kn-IN,
ml-IN, mr-IN, or-IN, pa-IN, ta-IN, te-IN.

For API docs see: https://docs.gnani.ai/api/STT/speech-to-text
"""

from pipecat_gnani.stt import (
    GnaniHttpSTTService,
    GnaniHttpSTTSettings,
    GnaniSTTService,
    GnaniSTTSettings,
)

__all__ = [
    "GnaniHttpSTTService",
    "GnaniHttpSTTSettings",
    "GnaniSTTService",
    "GnaniSTTSettings",
]
