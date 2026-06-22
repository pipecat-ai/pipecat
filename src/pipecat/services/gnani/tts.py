#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gnani Vachana text-to-speech service implementations.

Re-exports from the ``pipecat-gnani`` package. All implementation logic
lives in ``pipecat_gnani.tts``; this module keeps backward-compatible
imports under ``pipecat.services.gnani.tts``.

**Voices:** Karan (default), Simran, Nara, Riya, Viraj, Raju

Services:
- GnaniHttpTTSService: REST-based single-request synthesis
- GnaniSSETTSService: SSE streaming synthesis (lower latency than REST)
- GnaniTTSService: WebSocket streaming synthesis with interruption handling

For API docs see: https://docs.gnani.ai/api/TTS/tts-inference
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
