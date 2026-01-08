#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn start strategy configuration."""

from dataclasses import dataclass
from typing import List, Optional

from pipecat.turns.user_start import (
    BaseUserTurnStartStrategy,
    ExternalUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    BaseUserTurnStopStrategy,
    ExternalUserTurnStopStrategy,
    TranscriptionUserTurnStopStrategy,
)


@dataclass
class UserTurnStrategies:
    """Container for user turn start and stop strategies.

    If no strategies are specified, the following defaults are used:

        start: [VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy]
         stop: [TranscriptionUserTurnStopStrategy]

    Attributes:
        start: A list of user turn start strategies used to detect when
            the user starts speaking.
        stop: A list of user turn stop strategies used to decide when
            the user stops speaking.

    """

    start: Optional[List[BaseUserTurnStartStrategy]] = None
    stop: Optional[List[BaseUserTurnStopStrategy]] = None

    def __post_init__(self):
        if not self.start:
            self.start = [VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()]
        if not self.stop:
            self.stop = [TranscriptionUserTurnStopStrategy()]


@dataclass
class ExternalUserTurnStrategies(UserTurnStrategies):
    """Default container for external user turn start and stop strategies.

    This class provides a convenience default for configuring external turn
    control. It preconfigures `UserTurnStrategies` with
    `ExternalUserTurnStartStrategy` and `ExternalUserTurnStopStrategy`, allowing
    external processors (such as services) to control when user turn starts and
    stops.

    When using this container, the user aggregator does not push
    `UserStartedSpeakingFrame` or `UserStoppedSpeakingFrame` frames, and does
    not generate interruptions. These signals are expected to be provided by an
    external processor.

    """

    def __post_init__(self):
        self.start = [ExternalUserTurnStartStrategy()]
        self.stop = [ExternalUserTurnStopStrategy()]
