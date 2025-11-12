#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn start strategy configuration."""

from dataclasses import dataclass
from typing import List

from pipecat.turns.bot.base_bot_turn_start_strategy import BaseBotTurnStartStrategy
from pipecat.turns.user.base_user_turn_start_strategy import BaseUserTurnStartStrategy


@dataclass
class TurnStartStrategies:
    """Container for user and bot turn start strategies.

    This class groups the configured turn start strategies for both the user
    and the bot.

    Attributes:
        user: A list of user turn start strategies used to detect when the
            user starts speaking.
        bot: A list of bot turn start strategies used to decide when the bot
            should start speaking.
    """

    user: List[BaseUserTurnStartStrategy]
    bot: List[BaseBotTurnStartStrategy]
