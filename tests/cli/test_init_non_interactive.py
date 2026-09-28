#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the `pc init` non-interactive command: config-file + flag precedence.

These drive the real Typer command via ``--dry-run``, which resolves the full
config (file values merged with CLI flags) and prints it as JSON without
generating any files — exercising the merge logic in ``resolve_scaffold_config``
end to end.
"""

import json

import pytest
from typer.testing import CliRunner

from pipecat.cli.main import app

runner = CliRunner()


def _write_config(tmp_path, **overrides):
    """Write a JSON config file with sensible cascade defaults plus overrides."""
    data = {
        "project_name": "file-bot",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
    }
    data.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return path


def _dry_run(args):
    """Invoke `init ... --dry-run` and return the parsed resolved config."""
    result = runner.invoke(app, ["init", *args, "--dry-run"])
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


class TestConfigFileResolution:
    """A config file alone fully drives the resolved config."""

    def test_config_file_only(self, tmp_path):
        config_path = _write_config(tmp_path)
        resolved = _dry_run(["--config", str(config_path)])

        assert resolved["project_name"] == "file-bot"
        assert resolved["bot_type"] == "web"
        assert resolved["transports"] == ["daily"]
        assert resolved["mode"] == "cascade"
        assert resolved["stt_service"] == "deepgram_stt"
        assert resolved["llm_service"] == "openai_llm"
        assert resolved["tts_service"] == "cartesia_tts"

    def test_name_alias_in_config_file(self, tmp_path):
        """A config file may use 'name' instead of 'project_name'."""
        config_path = _write_config(tmp_path, project_name=None, name="aliased-bot")
        # Drop the null project_name so only 'name' is present.
        data = json.loads(config_path.read_text())
        del data["project_name"]
        config_path.write_text(json.dumps(data))

        resolved = _dry_run(["--config", str(config_path)])
        assert resolved["project_name"] == "aliased-bot"


class TestFlagPrecedence:
    """CLI flags override values from the config file."""

    def test_flag_overrides_string_value(self, tmp_path):
        config_path = _write_config(tmp_path, tts_service="cartesia_tts")
        resolved = _dry_run(["--config", str(config_path), "--tts", "elevenlabs_tts"])

        # Flag wins over the file value.
        assert resolved["tts_service"] == "elevenlabs_tts"
        # Unspecified fields still come from the file.
        assert resolved["stt_service"] == "deepgram_stt"

    def test_flag_overrides_name(self, tmp_path):
        config_path = _write_config(tmp_path)
        resolved = _dry_run(["--config", str(config_path), "--name", "flag-bot"])
        assert resolved["project_name"] == "flag-bot"

    def test_observability_flag_enables_over_file_default(self, tmp_path):
        config_path = _write_config(tmp_path)  # no observability key -> defaults off
        resolved = _dry_run(["--config", str(config_path), "--observability"])
        assert resolved["enable_observability"] is True

    # Negatable booleans default to False, so an explicit --no-<flag> looks identical
    # to an omitted flag by value alone. These prove the explicit flag still beats a
    # file value that enabled the setting — the general fix, not just for --eval.
    @pytest.mark.parametrize(
        "flag, file_key, output_key",
        [
            ("--no-eval", "enable_eval", "enable_eval"),
            ("--no-recording", "recording", "recording"),
            ("--no-transcription", "transcription", "transcription"),
            ("--no-observability", "enable_observability", "enable_observability"),
        ],
    )
    def test_negated_flag_disables_over_file_value(self, tmp_path, flag, file_key, output_key):
        config_path = _write_config(tmp_path, **{file_key: True})
        resolved = _dry_run(["--config", str(config_path), flag])
        assert resolved[output_key] is False

    def test_no_deploy_to_cloud_flag_disables_over_file_value(self, tmp_path):
        """The True-default flag can be disabled even when the file enables it."""
        config_path = _write_config(tmp_path, deploy_to_cloud=True)
        resolved = _dry_run(["--config", str(config_path), "--no-deploy-to-cloud"])
        assert resolved["deploy_to_cloud"] is False

    def test_enable_krisp_flag_enables_over_file_value(self, tmp_path):
        # Krisp requires deploy_to_cloud (config_validator cross-field rule).
        config_path = _write_config(tmp_path, enable_krisp=False, deploy_to_cloud=True)
        resolved = _dry_run(["--config", str(config_path), "--enable-krisp"])
        assert resolved["enable_krisp"] is True

    def test_boolean_file_value_applies_when_flag_omitted(self, tmp_path):
        """With no flag, the file value still drives the setting — it's the flag that flips it."""
        config_path = _write_config(tmp_path, enable_eval=True)
        resolved = _dry_run(["--config", str(config_path)])
        assert resolved["enable_eval"] is True
