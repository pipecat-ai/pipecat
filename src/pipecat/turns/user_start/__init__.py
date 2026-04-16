#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from .base_user_turn_start_strategy import BaseUserTurnStartStrategy, UserTurnStartedParams
from .external_user_turn_start_strategy import ExternalUserTurnStartStrategy
from .min_words_user_turn_start_strategy import MinWordsUserTurnStartStrategy
from .transcription_user_turn_start_strategy import TranscriptionUserTurnStartStrategy
from .vad_user_turn_start_strategy import VADUserTurnStartStrategy
from .wake_phrase_user_turn_start_strategy import WakePhraseUserTurnStartStrategy

_LAZY_IMPORTS = {
    "KrispVivaIPUserTurnStartStrategy": ".krisp_viva_ip_user_turn_start_strategy",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseUserTurnStartStrategy",
    "ExternalUserTurnStartStrategy",
    "KrispVivaIPUserTurnStartStrategy",
    "MinWordsUserTurnStartStrategy",
    "TranscriptionUserTurnStartStrategy",
    "UserTurnStartedParams",
    "VADUserTurnStartStrategy",
    "WakePhraseUserTurnStartStrategy",
]
