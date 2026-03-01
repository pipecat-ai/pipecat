#
# Copyright (c) 2024-2026, Onairos contributors
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Onairos personalization and memory services for Pipecat.

This module provides deep personalization capabilities for voice and multimodal
conversational AI agents. Onairos augments your base prompts with rich user context:

- Personality Traits: Numerical scores for user interests and characteristics
- Memory: Textual memories about the user from past interactions
- MBTI Compatibility: Personality type preference scores

Example augmented prompt:
    [Your Base Prompt]

    Personality Traits of User:
    {"Stoic Wisdom Interest": 80, "AI Enthusiasm": 40}

    Memory of User:
    Reads Daily Stoic every morning. Prefers coffee shop meetups.

    MBTI (Personalities User Likes):
    INFJ: 0.627, INTJ: 0.585, ENFJ: 0.580

    Critical Instruction:
    Always check context before asking.

Services:
    OnairosPersonaInjector: Augments prompts with personality traits, memory, MBTI
    OnairosMemoryService: Additional context enhancement
    OnairosContextAggregator: Connection state and onboarding management
    OnairosUserData: Data model for Onairos user context
"""

from pipecat.services.onairos.context import OnairosContextAggregator
from pipecat.services.onairos.memory import OnairosMemoryService
from pipecat.services.onairos.persona import OnairosPersonaInjector, OnairosUserData

__all__ = [
    "OnairosMemoryService",
    "OnairosPersonaInjector",
    "OnairosContextAggregator",
    "OnairosUserData",
]
