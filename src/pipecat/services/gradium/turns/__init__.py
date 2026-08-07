#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gradium turn-based speech-to-text service."""

from pipecat.services.gradium.turns.stt import (
    GradiumTurnsSTTService,
    GradiumTurnsSTTSettings,
)

__all__ = ["GradiumTurnsSTTService", "GradiumTurnsSTTSettings"]
