#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn start strategy configuration."""

from dataclasses import dataclass
from typing import List, Optional

from pipecat.turns.bot import (
    BaseBotTurnStartStrategy,
    ExternalBotTurnStartStrategy,
    TranscriptionBotTurnStartStrategy,
)
from pipecat.turns.user import (
    BaseUserTurnStartStrategy,
    ExternalUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)


@dataclass
class TurnStartStrategies:
    """Container for user and bot turn start strategies.

    This class groups the configured turn start strategies for both the user
    and the bot.

    If no strategies are specified for the user or the bot, the following
    defaults are used:

        user: [VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy]
         bot: [TranscriptionBotTurnStartStrategy]

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
            self.bot = [TranscriptionBotTurnStartStrategy()]


@dataclass
class ExternalTurnStartStrategies(TurnStartStrategies):
    """Default container for external user and bot turn start strategies.

    This class provides a convenience default for configuring external turn
    control. It preconfigures `TurnStartStrategies` with
    `ExternalUserTurnStartStrategy` and `ExternalBotTurnStartStrategy`, allowing
    external processors (such as services) to control when user and bot turns
    start.

    When using this container, the user aggregator does not push
    `UserStartedSpeakingFrame` or `UserStoppedSpeakingFrame` frames, and does
    not generate interruptions. These signals are expected to be provided by an
    external processor.

    """

    def __post_init__(self):
        self.user = [ExternalUserTurnStartStrategy()]
        self.bot = [ExternalBotTurnStartStrategy()]
