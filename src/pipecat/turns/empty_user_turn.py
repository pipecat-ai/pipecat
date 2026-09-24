#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Configuration for recovering from user turns that end without a transcript."""

from dataclasses import dataclass

DEFAULT_EMPTY_USER_TURN_INTERRUPTED_PROMPT = (
    "The user started speaking while you were responding, which cut your response "
    "off, but their speech was not recognized: what they said and meant is unknown. "
    "They may have heard only part of your last response, or none of it. Briefly ask "
    "them to repeat themselves, or briefly repeat the point or question they may have "
    "missed. Do not guess what they said."
)


@dataclass
class EmptyUserTurnConfig:
    """How the user aggregator responds to a user turn with no transcript.

    A user turn can start on voice activity alone and then end with no
    transcript, because the speech could not be recognized or was not speech
    at all. Nothing is written to the context and the LLM does not run, so if
    that turn interrupted the bot, the conversation stalls until the user
    speaks again. With this config, the aggregator appends a developer message
    for such a turn and runs the LLM once.

    Empty turns fall into two cases, each with its own prompt:

    - Interrupted: the turn started while the bot was responding (thinking,
      speaking or running a function call) and interrupted it.
    - Idle: the bot had nothing in progress, so the turn was most likely
      background noise.

    Parameters:
        interrupted_prompt: Developer message for an empty turn that interrupted
            the bot. ``None`` leaves such turns unanswered.
        idle_prompt: Developer message for an empty turn while the bot was idle.
            ``None`` (the default) leaves such turns unanswered.
        max_consecutive_recoveries: How many empty turns in a row are answered.
            Further ones are left unanswered until the user says something that
            is transcribed.
    """

    interrupted_prompt: str | None = DEFAULT_EMPTY_USER_TURN_INTERRUPTED_PROMPT
    idle_prompt: str | None = None
    max_consecutive_recoveries: int = 1
