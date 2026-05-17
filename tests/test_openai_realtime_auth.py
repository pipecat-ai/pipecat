#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAI Realtime authentication headers."""

import base64
import json
from typing import Any

import pytest

from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


def _fake_jwt(payload: dict[str, Any]) -> str:
    def encode(obj: dict[str, Any]) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode(payload)}.signature"


def test_websocket_headers_include_bearer_token():
    service = OpenAIRealtimeLLMService(api_key="test-key")

    assert service._websocket_headers() == {"Authorization": "Bearer test-key"}


def test_websocket_headers_support_codex_auth():
    token = _fake_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"}})
    service = OpenAIRealtimeLLMService(api_key=token, use_codex_auth=True)

    assert service._websocket_headers() == {
        "Authorization": f"Bearer {token}",
        "User-Agent": "codex_cli_rs/0.0.0 (Pipecat)",
        "originator": "codex_cli_rs",
        "ChatGPT-Account-ID": "acct_123",
    }


def test_websocket_headers_load_codex_auth_from_codex_home(monkeypatch, tmp_path):
    token = _fake_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"}})
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(
        json.dumps({"tokens": {"access_token": token}}),
        encoding="utf-8",
    )

    service = OpenAIRealtimeLLMService(use_codex_auth=True)

    assert service._websocket_headers()["Authorization"] == f"Bearer {token}"
    assert service._websocket_headers()["ChatGPT-Account-ID"] == "acct_123"


def test_api_key_required_without_codex_auth():
    with pytest.raises(ValueError, match="api_key is required"):
        OpenAIRealtimeLLMService()
