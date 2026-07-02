#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gnani Vachana Text-to-Speech service re-exports.

All implementation lives in ``pipecat_gnani.tts``; this module keeps
backward-compatible imports under ``pipecat.services.gnani.tts``.

Services:

- GnaniHttpTTSService — REST-based single-request synthesis
- GnaniSSETTSService — SSE streaming synthesis (lower latency)
- GnaniTTSService — WebSocket streaming synthesis with interruption handling

Voices: Karan (default), Simran, Nara, Riya, Viraj, Raju, Pranav, Kaveri,
Shubhra, Deepak.

API docs: https://docs.gnani.ai/api/TTS/tts-inference
"""

from pipecat_gnani.tts import (
    GnaniHttpTTSService,
    GnaniHttpTTSSettings,
    GnaniSSETTSService,
    GnaniSSETTSSettings,
    GnaniTTSService,
    GnaniTTSSettings,
)

__all__ = [
    "GnaniHttpTTSService",
    "GnaniHttpTTSSettings",
    "GnaniSSETTSService",
    "GnaniSSETTSSettings",
    "GnaniTTSService",
    "GnaniTTSSettings",
]
