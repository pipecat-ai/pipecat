#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Process-wide controls for Pipecat's debug logging.

Some debug output grows with the length of a session, such as the LLM context
logged on every inference. The settings here choose how much of it to write.
Each setting can be set in code with :func:`configure_logging` or through an
environment variable; a value set in code takes precedence.

Example::

    from pipecat.utils.log_config import LLMContextLogMode, configure_logging

    configure_logging(llm_context=LLMContextLogMode.OFF)

or, without a code change::

    PIPECAT_LOG_LLM_CONTEXT=off python bot.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

from pipecat.utils.env import InvalidEnvVarValueError

LLM_CONTEXT_LOG_ENV_VAR = "PIPECAT_LOG_LLM_CONTEXT"


class LLMContextLogMode(StrEnum):
    """How much of the context LLM services include in their DEBUG log lines.

    LLM services log a DEBUG line each time they generate a response, and
    realtime services when they set up their server-side conversation. The
    context is built only when DEBUG is enabled.

    Parameters:
        FULL: Every message of the context.
        OFF: None of it; the line is logged without the context.
    """

    FULL = "full"
    OFF = "off"


@dataclass
class _LogConfig:
    llm_context: LLMContextLogMode | None = None


_config = _LogConfig()


def configure_logging(*, llm_context: LLMContextLogMode | str | None = None) -> None:
    """Configure Pipecat's debug logging for the whole process.

    Settings left as ``None`` keep their current value.

    Args:
        llm_context: How much of the context LLM services log.
            Overrides ``PIPECAT_LOG_LLM_CONTEXT``.

    Raises:
        ValueError: If ``llm_context`` is not a valid :class:`LLMContextLogMode`.
    """
    if llm_context is not None:
        _config.llm_context = LLMContextLogMode(llm_context)


def get_llm_context_log_mode() -> LLMContextLogMode:
    """Return how LLM services log the context.

    The value set with :func:`configure_logging` if there is one, otherwise
    ``PIPECAT_LOG_LLM_CONTEXT``, otherwise :attr:`LLMContextLogMode.FULL`. The
    environment variable is read once, on first use.

    Returns:
        The current LLM context log mode.

    Raises:
        InvalidEnvVarValueError: If ``PIPECAT_LOG_LLM_CONTEXT`` is set to an
            unknown mode.
    """
    if _config.llm_context is None:
        _config.llm_context = _llm_context_log_mode_from_env()
    return _config.llm_context


def _llm_context_log_mode_from_env() -> LLMContextLogMode:
    raw = os.getenv(LLM_CONTEXT_LOG_ENV_VAR)
    if raw is None or not raw.strip():
        return LLMContextLogMode.FULL
    try:
        return LLMContextLogMode(raw.strip().lower())
    except ValueError:
        raise InvalidEnvVarValueError(
            name=LLM_CONTEXT_LOG_ENV_VAR,
            value=raw,
            expected=" or ".join(mode.value for mode in LLMContextLogMode),
        ) from None
