#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Import path for :class:`LLMTurnCompletionUserTurnStopStrategy`.

The strategy lives in :mod:`pipecat.turns.user_stop.llm_turn_completion`,
beside the LLM extension it installs.
"""

from pipecat.turns.user_stop.llm_turn_completion.llm_turn_completion_user_turn_stop_strategy import (
    LLMTurnCompletionUserTurnStopStrategy,
)

__all__ = ["LLMTurnCompletionUserTurnStopStrategy"]
