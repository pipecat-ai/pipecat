#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frames and data models for Inworld AI services."""

from dataclasses import dataclass

from pydantic import AliasChoices, BaseModel, Field

from pipecat.frames.frames import DataFrame


class InworldVoiceProfileLabel(BaseModel):
    """A Voice Profile classification and its confidence score.

    Parameters:
        label: Predicted class name.
        confidence: Prediction confidence from 0.0 to 1.0.
    """

    label: str
    confidence: float


class InworldVoiceProfile(BaseModel):
    """Speaker characteristics returned by Inworld Voice Profile analysis.

    Parameters:
        age: Detected age categories.
        emotion: Detected emotional tones.
        pitch: Detected vocal pitch levels.
        vocal_style: Detected manners of vocal delivery.
        accent: Detected accents as BCP-47 locale codes.
    """

    age: list[InworldVoiceProfileLabel] = Field(default_factory=list)
    emotion: list[InworldVoiceProfileLabel] = Field(default_factory=list)
    pitch: list[InworldVoiceProfileLabel] = Field(default_factory=list)
    vocal_style: list[InworldVoiceProfileLabel] = Field(
        default_factory=list,
        validation_alias=AliasChoices("vocalStyle", "vocal_style"),
    )
    accent: list[InworldVoiceProfileLabel] = Field(default_factory=list)


@dataclass
class InworldVoiceProfileFrame(DataFrame):
    """Frame containing Inworld Voice Profile analysis for a transcription result.

    Parameters:
        user_id: Identifier for the speaker.
        timestamp: When the profile was produced.
        voice_profile: Structured speaker characteristics and confidence scores.
    """

    user_id: str
    timestamp: str
    voice_profile: InworldVoiceProfile
