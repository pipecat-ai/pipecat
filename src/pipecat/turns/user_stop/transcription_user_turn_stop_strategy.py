#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Transcription-based user turn stop strategy (deprecated).

.. deprecated:: 0.0.102
    This module is deprecated. Please use
    ``pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy.SpeechTimeoutUserTurnStopStrategy``
    instead.
"""

import warnings

from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "TranscriptionUserTurnStopStrategy is deprecated. "
        "Please use SpeechTimeoutUserTurnStopStrategy from "
        "pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy instead.",
        DeprecationWarning,
        stacklevel=2,
    )

TranscriptionUserTurnStopStrategy = SpeechTimeoutUserTurnStopStrategy
