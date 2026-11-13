#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Process-wide controls for Pipecat's debug logging.

Some debug output grows with the length of a session, such as the LLM context
logged on every inference. The settings here choose how much of it to write,
and each is read from an environment variable once, on first use.

- ``PIPECAT_LOG_LLM_CONTEXT``: how much of the context LLM services include
  in their DEBUG log lines. ``full`` (the default) logs every message;
  ``off`` logs the line without the context.

Example::

    PIPECAT_LOG_LLM_CONTEXT=off python bot.py
"""

from __future__ import annotations

import os
from enum import StrEnum

from pipecat.utils.env import InvalidEnvVarValueError

_LLM_CONTEXT_LOG_ENV_VAR = "PIPECAT_LOG_LLM_CONTEXT"


class _LLMContextLogMode(StrEnum):
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


_llm_context_log_mode: _LLMContextLogMode | None = None


def _get_llm_context_log_mode() -> _LLMContextLogMode:
    """Return how LLM services log the context.

    The mode named by ``PIPECAT_LOG_LLM_CONTEXT``, or ``full`` when it is unset
    or empty. The environment variable is read once, on first use.

    Returns:
        The current LLM context log mode.

    Raises:
        InvalidEnvVarValueError: If ``PIPECAT_LOG_LLM_CONTEXT`` is set to an
            unknown mode.
    """
    global _llm_context_log_mode
    if _llm_context_log_mode is None:
        _llm_context_log_mode = _llm_context_log_mode_from_env()
    return _llm_context_log_mode


def _llm_context_log_mode_from_env() -> _LLMContextLogMode:
    raw = os.getenv(_LLM_CONTEXT_LOG_ENV_VAR)
    if raw is None or not raw.strip():
        return _LLMContextLogMode.FULL
    try:
        return _LLMContextLogMode(raw.strip().lower())
    except ValueError:
        raise InvalidEnvVarValueError(
            name=_LLM_CONTEXT_LOG_ENV_VAR,
            value=raw,
            expected=" or ".join(mode.value for mode in _LLMContextLogMode),
        ) from None
