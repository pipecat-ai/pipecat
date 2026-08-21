#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Offline checks for the ``scripts/provider-watch`` tooling.

``inventory.py`` must keep finding every provider service and its default model
as the services tree evolves; ``digest.py`` must render whatever frontmatter the
researcher agents write; ``probe.py`` must refuse to run without credentials
instead of failing later with a provider error. None of these touch the network.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS = REPO_ROOT / "scripts" / "provider-watch"
sys.path.insert(0, str(SCRIPTS))

import digest  # noqa: E402
import inventory  # noqa: E402


@pytest.fixture(scope="module")
def units():
    found = inventory.scan_services()
    inventory.enrich(found)
    return found


class TestInventory:
    def test_covers_the_services_tree(self, units):
        providers = {u.provider for u in units}
        assert len(providers) >= 55
        assert len(units) >= 90
        assert {u.type for u in units} >= {"llm", "stt", "tts", "realtime", "image"}

    def test_unit_ids_are_unique_and_sorted(self, units):
        ids = [u.id for u in units]
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize(
        "unit_id, class_name, expect_model",
        [
            ("openai/llm", "OpenAILLMService", True),
            ("openai/responses-llm", "OpenAIResponsesLLMService", True),
            ("openai/realtime", "OpenAIRealtimeLLMService", True),
            ("cartesia/tts", "CartesiaTTSService", True),
            ("deepgram/stt", "DeepgramSTTService", True),
            ("google/realtime", "GeminiLiveLLMService", True),
            ("groq/llm", "GroqLLMService", True),
            ("azure/tts", "AzureTTSService", False),
        ],
    )
    def test_known_units(self, units, unit_id, class_name, expect_model):
        unit = next(u for u in units if u.id == unit_id)
        assert class_name in [c.name for c in unit.classes]
        assert (unit.default_model is not None) == expect_model

    def test_every_settings_model_literal_is_captured(self, units):
        """A class whose ``__init__`` writes ``model="..."`` into its Settings has a default."""
        literal = re.compile(r"self\.Settings\([^)]*?\bmodel=\"", re.DOTALL)
        for unit in units:
            for cls in unit.classes:
                source = (REPO_ROOT / "src" / Path(*cls.module.split("."))).with_suffix(".py")
                text = source.read_text()
                if f"class {cls.name}(" not in text:
                    continue
                section = text.split(f"class {cls.name}(")[1].split("\nclass ")[0]
                if literal.search(section):
                    assert cls.default_model, cls.name

    def test_thin_wrappers_are_flagged(self, units):
        groq = next(u for u in units if u.id == "groq/llm")
        openai = next(u for u in units if u.id == "openai/llm")
        assert groq.is_thin_wrapper
        assert groq.classes[0].base_url == "https://api.groq.com/openai/v1"
        assert not openai.is_thin_wrapper

    def test_base_classes_and_shims_are_excluded(self, units):
        names = {c.name for u in units for c in u.classes}
        assert "BaseOpenAILLMService" not in names
        assert "AzureBaseTTSService" not in names
        assert "BaseWhisperSTTService" not in names
        assert "BasetenLLMService" in names
        assert not any(u.provider == "grok" for u in units)

    def test_joins(self, units):
        deepgram = next(u for u in units if u.id == "deepgram/stt")
        assert any(e["value"] == "deepgram_stt" for e in deepgram.registry)
        assert "DEEPGRAM_API_KEY" in deepgram.env_vars
        assert any("deepgram" in bot for bot in deepgram.example_bots)
        assert (
            deepgram.docs_url
            == "https://docs.pipecat.ai/api-reference/server/services/stt/deepgram"
        )

    def test_select_and_cli(self, units):
        picked = inventory.select(units, ["openai", "deepgram/stt"], None)
        assert {u.provider for u in picked} == {"openai", "deepgram"}
        assert [u.id for u in picked if u.provider == "deepgram"] == ["deepgram/stt"]
        assert len(inventory.select(units, None, 3)) == 3

        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "inventory.py"), "--json", "--only", "cartesia"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        assert {u["id"] for u in data} == {"cartesia/stt", "cartesia/tts", "cartesia/turns-stt"}


class TestDigest:
    @pytest.fixture
    def reports_dir(self, tmp_path):
        def write(unit, body):
            path = tmp_path / "reports" / unit / "2026-08-20.md"
            path.parent.mkdir(parents=True)
            path.write_text(body)

        write(
            "openai/llm",
            "---\nservice: openai/llm\nstatus: prs-opened\ndefault_model: gpt-4.1\n"
            "summary: gpt-5 is GA and faster\nprs:\n  - url: https://github.com/pipecat-ai/pipecat/pull/1\n"
            "    state: open\n    summary: bump default to gpt-5\n---\n# OpenAI LLM\n",
        )
        write(
            "cartesia/tts",
            "---\nservice: cartesia/tts\nstatus: needs-judgement\ndefault_model: sonic-3.5\n"
            "open_items:\n  - sonic-4 preview needs a voice migration\n---\n",
        )
        write(
            "groq/llm",
            "---\nservice: groq/llm\nstatus: up-to-date\ndefault_model: openai/gpt-oss-120b\n---\n",
        )
        write(
            "fireworks/llm",
            "---\nservice: fireworks/llm\nstatus: prs-withheld\n"
            "default_model: accounts/fireworks/models/firefunction-v2\n"
            "summary: default retired; gpt-oss-120b passes the probe (PR cap reached)\n---\n",
        )
        write("broken/tts", "no frontmatter at all\n")
        return tmp_path

    def test_render_groups_by_status(self, reports_dir):
        reports = digest.load_reports(reports_dir, "2026-08-20")
        text = digest.render(
            reports, date="2026-08-20", highlights="- Big week for LLMs", repo_url="https://x/y"
        )

        assert text.startswith("# Provider watch — 2026-08-20\n\n- Big week for LLMs")
        assert "**5 units researched**" in text
        assert "## PRs opened, to review" in text
        assert text.index("## PRs withheld") < text.index("## Changes to consider")
        assert "fireworks/llm" in text.split("## PRs withheld")[1].split("## Changes")[0]
        assert "https://github.com/pipecat-ai/pipecat/pull/1 — bump default to gpt-5" in text
        assert "## Changes to consider" in text
        assert "  - sonic-4 preview needs a voice migration" in text
        assert "## Errors" in text and "`broken/tts`" not in text and "broken/tts" in text
        assert text.rstrip().endswith(
            "[groq/llm](https://x/y/blob/main/reports/groq/llm/2026-08-20.md)"
        )

    def test_cli_writes_file(self, reports_dir):
        out = reports_dir / "digests" / "2026-08-20.md"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "digest.py"),
                "--reports",
                str(reports_dir),
                "--date",
                "2026-08-20",
                "--out",
                str(out),
                "--repo-url",
                "",
            ],
            check=True,
        )
        assert out.read_text().startswith("# Provider watch — 2026-08-20")
        assert "`openai/llm`" in out.read_text()


class TestProbe:
    def _run(self, *argv, env_overrides=None):
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        env.update(env_overrides or {})
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "probe.py"), *argv],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO_ROOT,
        )

    def test_missing_credentials_exit_2_and_name_only(self):
        result = self._run("--no-dotenv", "run", "--service", "CartesiaTTSService", "--model", "x")
        assert result.returncode == 2, result.stderr
        assert "CARTESIA_API_KEY" in result.stderr
        assert "***" not in result.stderr

    def test_research_only_types_exit_3(self):
        result = self._run("--no-dotenv", "run", "--service", "FalImageGenService", "--model", "x")
        assert result.returncode == 3
        assert "research-only" in result.stderr

    def test_unknown_provider_catalogue_exit_3(self):
        result = self._run("--no-dotenv", "list-models", "--provider", "nosuchprovider")
        assert result.returncode == 3

    def test_setting_values_parse_json_and_scalars(self):
        import probe

        assert probe._kv_pairs(['extra={"reasoning_effort": "low"}', "speed=1.5", "x=none"]) == {
            "extra": {"reasoning_effort": "low"},
            "speed": 1.5,
            "x": None,
        }
        with pytest.raises(SystemExit):
            probe._kv_pairs(["extra={not json"])

    def test_too_many_models_rejected(self):
        result = self._run(
            "run", "--service", "OpenAILLMService", *sum([["--model", m] for m in "abcd"], [])
        )
        assert result.returncode != 0
        assert "at most 3" in result.stderr
