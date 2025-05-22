from enum import Enum

from pydantic import BaseModel


class ActionType(Enum):
    """Enum for the action types."""

    END_CALL = "end_call"
    TRANSFER_CALL = "transfer_call"


class BaseAction(BaseModel):
    """This class is responsible for managing the actions."""

    action_type: ActionType


class EndCallAction(BaseAction):
    """This class is responsible for managing the actions."""

    action_type: ActionType = ActionType.END_CALL
    is_last_turn: bool


class TransferCallAction(BaseAction):
    """This class is responsible for managing the actions."""

    action_type: ActionType = ActionType.TRANSFER_CALL
    reason: str
