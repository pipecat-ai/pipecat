#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the process-wide logging controls in ``pipecat.utils.log_config``."""

import pytest

from pipecat.utils import log_config
from pipecat.utils.env import InvalidEnvVarValueError
from pipecat.utils.log_config import (
    _LLM_CONTEXT_LOG_ENV_VAR,
    _get_llm_context_log_mode,
    _LLMContextLogMode,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(log_config, "_llm_context_log_mode", None)
    monkeypatch.delenv(_LLM_CONTEXT_LOG_ENV_VAR, raising=False)


def test_defaults_to_full():
    assert _get_llm_context_log_mode() == _LLMContextLogMode.FULL


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_env_var_defaults_to_full(monkeypatch, raw):
    monkeypatch.setenv(_LLM_CONTEXT_LOG_ENV_VAR, raw)
    assert _get_llm_context_log_mode() == _LLMContextLogMode.FULL


@pytest.mark.parametrize(
    "raw, expected",
    [
        (" Off ", _LLMContextLogMode.OFF),
        ("FULL", _LLMContextLogMode.FULL),
    ],
)
def test_reads_env_var(monkeypatch, raw, expected):
    monkeypatch.setenv(_LLM_CONTEXT_LOG_ENV_VAR, raw)
    assert _get_llm_context_log_mode() == expected


def test_invalid_env_var_raises(monkeypatch):
    monkeypatch.setenv(_LLM_CONTEXT_LOG_ENV_VAR, "verbose")
    with pytest.raises(InvalidEnvVarValueError) as exc_info:
        _get_llm_context_log_mode()
    assert exc_info.value.name == _LLM_CONTEXT_LOG_ENV_VAR
    assert exc_info.value.value == "verbose"


def test_env_var_read_once(monkeypatch):
    monkeypatch.setenv(_LLM_CONTEXT_LOG_ENV_VAR, "off")
    assert _get_llm_context_log_mode() == _LLMContextLogMode.OFF
    monkeypatch.setenv(_LLM_CONTEXT_LOG_ENV_VAR, "full")
    assert _get_llm_context_log_mode() == _LLMContextLogMode.OFF
