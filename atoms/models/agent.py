"""This module contains the CallData class, which is used to store the call data."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator
from utils import get_unallowed_variable_names


class AtomsLLMModels(Enum):
    ELECTRON = "electron"
    GPT_4O = "gpt-4o"


class CallData(BaseModel):
    """Call data model."""

    variables: Optional[Dict[str, Any]] = Field(default=None)

    @field_validator("variables")
    @classmethod
    def validate_required_variables(cls, v):
        if v is not None:
            required_keys = get_unallowed_variable_names()
            missing_keys = [key for key in required_keys if key not in v]
            if missing_keys:
                raise ValueError(f"Missing required keys in variables: {', '.join(missing_keys)}")
        return v
