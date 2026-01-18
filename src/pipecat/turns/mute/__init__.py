#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import warnings

from pipecat.turns.user_mute.always_user_mute_strategy import AlwaysUserMuteStrategy
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy
from pipecat.turns.user_mute.first_speech_user_mute_strategy import FirstSpeechUserMuteStrategy
from pipecat.turns.user_mute.function_call_user_mute_strategy import FunctionCallUserMuteStrategy
from pipecat.turns.user_mute.mute_until_first_bot_complete_user_mute_strategy import (
    MuteUntilFirstBotCompleteUserMuteStrategy,
)

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Types in pipecat.turns.mute are deprecated. "
        "Please use the equivalent types from pipecat.turns.user_mute instead.",
        DeprecationWarning,
        stacklevel=2,
    )

__all__ = [
    "AlwaysUserMuteStrategy",
    "BaseUserMuteStrategy",
    "FirstSpeechUserMuteStrategy",
    "FunctionCallUserMuteStrategy",
    "MuteUntilFirstBotCompleteUserMuteStrategy",
]
