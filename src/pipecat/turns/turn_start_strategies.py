#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn start strategy configuration."""

from dataclasses import dataclass
from typing import List, Optional

from pipecat.turns.bot.base_bot_turn_start_strategy import BaseBotTurnStartStrategy
from pipecat.turns.bot.timeout_bot_turn_start_strategy import TimeoutBotTurnStartStrategy
from pipecat.turns.bot.transcription_bot_turn_start_strategy import (
    TranscriptionBotTurnStartStrategy,
)
from pipecat.turns.user.base_user_turn_start_strategy import BaseUserTurnStartStrategy
from pipecat.turns.user.transcription_user_turn_start_strategy import (
    TranscriptionUserTurnStartStrategy,
)
from pipecat.turns.user.vad_user_turn_start_strategy import VADUserTurnStartStrategy


@dataclass
class TurnStartStrategies:
    """Container for user and bot turn start strategies.

    This class groups the configured turn start strategies for both the user
    and the bot.

    If no strategies are specified for the user or the bot, the following
    defaults are used:

        user: [VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy]
         bot: [TranscriptionBotTurnStartStrategy, TimeoutBotTurnStartStrategy]

    Attributes:
        user: A list of user turn start strategies used to detect when the
            user starts speaking.
        bot: A list of bot turn start strategies used to decide when the bot
            should start speaking.

    """

    user: Optional[List[BaseUserTurnStartStrategy]] = None
    bot: Optional[List[BaseBotTurnStartStrategy]] = None

    def __post_init__(self):
        if not self.user:
            self.user = [VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()]
        if not self.bot:
            self.bot = [TranscriptionBotTurnStartStrategy(), TimeoutBotTurnStartStrategy()]
