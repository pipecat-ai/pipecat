#
# Copyright (c) 2024-2026, Onairos contributors
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Onairos personalization services for Pipecat.

This module provides deep personalization capabilities for voice and multimodal
conversational AI agents. Onairos augments your base prompts with rich user context:

- Personality Traits: Positive traits and areas to improve with scores
- User Summary: Multi-paragraph description of the user
- Archetype: Short personality archetype label
- MBTI Alignment: Personality type preference scores from inference

Example augmented prompt:
    [Your Base Prompt]

    Positive Traits of User:
    Stoic Wisdom Interest: 80, AI Enthusiasm: 40

    Areas to Improve:
    Social Media Engagement: 35

    User Summary:
    You are drawn to deep philosophical thinking...

    Archetype: The Strategic Explorer

    MBTI Alignment (Personalities User Likes):
    INFJ: 0.627, INTJ: 0.585, ENFJ: 0.580

    Critical Instruction:
    Always check context before asking.

Services:
    OnairosPersonaInjector: Augments prompts with personality traits, archetype, MBTI
    OnairosUserData: Data model for Onairos user context
"""

from pipecat.services.onairos.persona import OnairosPersonaInjector, OnairosUserData

__all__ = [
    "OnairosPersonaInjector",
    "OnairosUserData",
]
