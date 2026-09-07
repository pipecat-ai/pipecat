#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the pc init --list-options flag."""

import json

from typer.testing import CliRunner

from pipecat.cli.main import app
from pipecat.cli.registry.service_metadata import ServiceRegistry

runner = CliRunner()


class TestListOptions:
    """Tests for --list-options output."""

    def test_exits_successfully(self):
        result = runner.invoke(app, ["init", "--list-options"])
        assert result.exit_code == 0

    def test_returns_valid_json(self):
        result = runner.invoke(app, ["init", "--list-options"])
        data = json.loads(result.stdout)
        assert isinstance(data, dict)

    def test_top_level_keys(self):
        result = runner.invoke(app, ["init", "--list-options"])
        data = json.loads(result.stdout)
        assert set(data.keys()) == {
            "bot_type",
            "transports",
            "stt",
            "llm",
            "tts",
            "realtime",
            "video",
        }

    def test_bot_type_values(self):
        result = runner.invoke(app, ["init", "--list-options"])
        data = json.loads(result.stdout)
        assert data["bot_type"] == ["web", "telephony"]

    def test_transports_grouped_by_bot_type(self):
        result = runner.invoke(app, ["init", "--list-options"])
        data = json.loads(result.stdout)
        assert set(data["transports"].keys()) == {"web", "telephony"}
        assert isinstance(data["transports"]["web"], list)
        assert isinstance(data["transports"]["telephony"], list)

    def test_transports_match_registry(self):
        result = runner.invoke(app, ["init", "--list-options"])
        data = json.loads(result.stdout)
        assert data["transports"]["web"] == [s.value for s in ServiceRegistry.WEBRTC_TRANSPORTS]
        assert data["transports"]["telephony"] == [
            s.value for s in ServiceRegistry.TELEPHONY_TRANSPORTS
        ]

    def test_services_match_registry(self):
        result = runner.invoke(app, ["init", "--list-options"])
        data = json.loads(result.stdout)
        assert data["stt"] == [s.value for s in ServiceRegistry.STT_SERVICES]
        assert data["llm"] == [s.value for s in ServiceRegistry.LLM_SERVICES]
        assert data["tts"] == [s.value for s in ServiceRegistry.TTS_SERVICES]
        assert data["realtime"] == [s.value for s in ServiceRegistry.REALTIME_SERVICES]
        assert data["video"] == [s.value for s in ServiceRegistry.VIDEO_SERVICES]

    def test_all_lists_non_empty(self):
        result = runner.invoke(app, ["init", "--list-options"])
        data = json.loads(result.stdout)
        assert len(data["bot_type"]) > 0
        assert len(data["transports"]["web"]) > 0
        assert len(data["transports"]["telephony"]) > 0
        assert len(data["stt"]) > 0
        assert len(data["llm"]) > 0
        assert len(data["tts"]) > 0
        assert len(data["realtime"]) > 0
        assert len(data["video"]) > 0

    def test_no_project_generated(self):
        """--list-options should exit before any project generation happens."""
        result = runner.invoke(app, ["init", "--list-options", "--name", "should-not-run"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "bot_type" in data
