#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Nemo ASR service integration for Pipecat.

This module provides speech-to-text functionality using a custom
Nemo-based ASR server via WebSocket connection.
"""

from pipecat.services.nemo.stt import NemoInputParams, NemoLatencyMode, NemoSTTService

__all__ = ["NemoInputParams", "NemoLatencyMode", "NemoSTTService"]
