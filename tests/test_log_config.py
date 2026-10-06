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
    LLM_CONTEXT_LOG_ENV_VAR,
    LLMContextLogMode,
    configure_logging,
    get_llm_context_log_mode,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(log_config, "_config", log_config._LogConfig())
    monkeypatch.delenv(LLM_CONTEXT_LOG_ENV_VAR, raising=False)


def test_defaults_to_full():
    assert get_llm_context_log_mode() == LLMContextLogMode.FULL


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_env_var_defaults_to_full(monkeypatch, raw):
    monkeypatch.setenv(LLM_CONTEXT_LOG_ENV_VAR, raw)
    assert get_llm_context_log_mode() == LLMContextLogMode.FULL


@pytest.mark.parametrize(
    "raw, expected",
    [
        (" Off ", LLMContextLogMode.OFF),
        ("FULL", LLMContextLogMode.FULL),
    ],
)
def test_reads_env_var(monkeypatch, raw, expected):
    monkeypatch.setenv(LLM_CONTEXT_LOG_ENV_VAR, raw)
    assert get_llm_context_log_mode() == expected


def test_invalid_env_var_raises(monkeypatch):
    monkeypatch.setenv(LLM_CONTEXT_LOG_ENV_VAR, "verbose")
    with pytest.raises(InvalidEnvVarValueError) as exc_info:
        get_llm_context_log_mode()
    assert exc_info.value.name == LLM_CONTEXT_LOG_ENV_VAR
    assert exc_info.value.value == "verbose"


def test_env_var_read_once(monkeypatch):
    monkeypatch.setenv(LLM_CONTEXT_LOG_ENV_VAR, "off")
    assert get_llm_context_log_mode() == LLMContextLogMode.OFF
    monkeypatch.setenv(LLM_CONTEXT_LOG_ENV_VAR, "full")
    assert get_llm_context_log_mode() == LLMContextLogMode.OFF


def test_configure_overrides_env_var(monkeypatch):
    monkeypatch.setenv(LLM_CONTEXT_LOG_ENV_VAR, "off")
    configure_logging(llm_context=LLMContextLogMode.FULL)
    assert get_llm_context_log_mode() == LLMContextLogMode.FULL


def test_configure_accepts_strings():
    configure_logging(llm_context="off")
    assert get_llm_context_log_mode() == LLMContextLogMode.OFF


def test_configure_none_keeps_current_value():
    configure_logging(llm_context="off")
    configure_logging()
    assert get_llm_context_log_mode() == LLMContextLogMode.OFF


def test_configure_invalid_value_raises():
    with pytest.raises(ValueError):
        configure_logging(llm_context="verbose")
