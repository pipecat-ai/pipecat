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
            "---\nservice: openai/llm\ndefault_model: gpt-4.1\nsummary: gpt-5 is GA and faster\n"
            "gaps:\n  - item: gpt-5 supersedes gpt-4.1\n    first_seen: 2026-08-20\n    action: pr\n"
            "prs:\n  - url: https://github.com/pipecat-ai/pipecat/pull/1\n    state: open\n"
            "    summary: bump default to gpt-5\nerror: null\n---\n# OpenAI LLM\n",
        )
        write(
            "cartesia/tts",
            "---\nservice: cartesia/tts\ndefault_model: sonic-3.5\n"
            "gaps:\n  - item: sonic-4 preview needs a voice migration\n    first_seen: 2026-07-30\n"
            "    action: consider\n    note: re-check when GA\nprs: []\nerror: null\n---\n",
        )
        write(
            "groq/llm",
            "---\nservice: groq/llm\ndefault_model: openai/gpt-oss-120b\ngaps: []\nprs: []\nerror: null\n---\n",
        )
        write(
            "fireworks/llm",
            "---\nservice: fireworks/llm\ndefault_model: accounts/fireworks/models/firefunction-v2\n"
            "gaps:\n  - item: firefunction-v2 is retired\n    first_seen: 2026-08-20\n    action: pr\n"
            "prs:\n  - branch: provider-watch/fireworks-llm-default\n    state: branch\n"
            "    summary: Default FireworksLLMService to gpt-oss-120b\nerror: null\n---\n"
            "\n## PRs\n- `provider-watch/fireworks-llm-default` — review: "
            "`git show provider-watch/fireworks-llm-default` — Default FireworksLLMService to gpt-oss-120b\n",
        )
        write("mistral/stt", "---\nservice: mistral/stt\nerror: missing MISTRAL_API_KEY\n---\n")
        write("broken/tts", "no frontmatter at all\n")
        return tmp_path

    def test_render_sections(self, reports_dir):
        reports = digest.load_reports(reports_dir, "2026-08-20")
        text = digest.render(
            reports, date="2026-08-20", highlights="- Big week for LLMs", repo_url="https://x/y"
        )

        assert text.startswith("# Provider watch — 2026-08-20\n\n- Big week for LLMs")
        assert (
            "**6 units researched** — 1 PRs, 1 branches, 1 changes to consider, 2 errors, 1 with nothing new."
            in text
        )
        sections = [line for line in text.splitlines() if line.startswith("## ")]
        assert sections == [
            "## PRs to review",
            "## Branches not opened as PRs (dry run)",
            "## Changes to consider",
            "## Did not complete",
            "## Nothing new",
        ]
        assert "https://github.com/pipecat-ai/pipecat/pull/1 — bump default to gpt-5" in text
        assert "`git show provider-watch/fireworks-llm-default`" in text
        assert (
            "sonic-4 preview needs a voice migration (since 2026-07-30, 3 weeks) — re-check when GA"
            in text
        )
        assert "missing MISTRAL_API_KEY" in text and "report has no frontmatter" in text
        assert (
            "[groq/llm](https://x/y/blob/main/reports/groq/llm/2026-08-20.md)"
            in text.split("## Nothing new")[1]
        )
        assert text.rstrip().endswith("The next run reads these comments.")

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


class TestPublish:
    """publish.py against a fake git/gh so nothing leaves the machine."""

    class FakeShell:
        def __init__(self, open_prs=None, branches=None):
            self.calls = []
            self.open_prs = open_prs or {}
            self.branches = set(branches or [])
            self.pr_counter = 100

        def run(self, *args, cwd=None, check=True):
            self.calls.append(args)
            if args[:3] == ("gh", "pr", "list"):
                head = args[args.index("--head") + 1]
                url = self.open_prs.get(head)
                return json.dumps([{"url": url}] if url else [])
            if args[:3] == ("gh", "pr", "create"):
                self.pr_counter += 1
                return f"https://github.com/pipecat-ai/pipecat/pull/{self.pr_counter}\n"
            if args[:2] == ("git", "log"):
                return (
                    "Default FireworksLLMService to gpt-oss-120b\n"
                    if "--format=%s" in args
                    else "Body.\n"
                )
            if args[:3] == ("gh", "issue", "list"):
                return "[]"
            if args[:3] == ("gh", "issue", "create"):
                return "https://github.com/pipecat-ai/provider-watch/issues/1\n"
            return ""

        def ok(self, *args, cwd=None):
            self.calls.append(args)
            if args[:2] == ("git", "rev-parse"):
                return args[-1] in self.branches
            if args[:2] == ("git", "diff"):
                return False  # something is staged
            return True

    @pytest.fixture
    def reports_dir(self, tmp_path):
        import publish

        def write(unit, branch):
            path = tmp_path / "reports" / unit / "2026-08-20.md"
            path.parent.mkdir(parents=True)
            path.write_text(
                f"---\nservice: {unit}\nprs:\n  - branch: {branch}\n"
                f"    state: branch\n    summary: s\n---\n\n# R\n\n## PRs\n"
                f"- `{branch}` — review: `git show {branch}` — s\n"
            )

        write("fireworks/llm", "provider-watch/fireworks-llm-default")
        write("groq/llm", "provider-watch/groq-llm-example")
        write("ollama/llm", "provider-watch/ollama-llm-default")
        return tmp_path, publish

    def test_opens_adopts_and_caps(self, reports_dir):
        tmp_path, publish = reports_dir
        sh = self.FakeShell(
            open_prs={
                "provider-watch/groq-llm-example": "https://github.com/pipecat-ai/pipecat/pull/7"
            },
            branches=[
                "provider-watch/fireworks-llm-default",
                "provider-watch/groq-llm-example",
                "provider-watch/ollama-llm-default",
            ],
        )
        reports = publish.load_reports(tmp_path, "2026-08-20")
        outcome = publish.publish_prs(
            reports,
            sh=sh,
            repo_root=tmp_path,
            pipecat_repo="pipecat-ai/pipecat",
            reports_repo="pipecat-ai/provider-watch",
            date="2026-08-20",
            cap=1,
        )
        assert outcome.opened == ["https://github.com/pipecat-ai/pipecat/pull/101"]
        assert outcome.adopted == ["https://github.com/pipecat-ai/pipecat/pull/7"]
        assert outcome.capped == ["provider-watch/ollama-llm-default"]

        fireworks = (tmp_path / "reports/fireworks/llm/2026-08-20.md").read_text()
        assert (
            "state: open" in fireworks
            and "url: https://github.com/pipecat-ai/pipecat/pull/101" in fireworks
        )
        assert "- https://github.com/pipecat-ai/pipecat/pull/101 — s" in fireworks
        assert "git diff" not in fireworks
        ollama = (tmp_path / "reports/ollama/llm/2026-08-20.md").read_text()
        assert "capped: true" in ollama and "git show provider-watch/ollama-llm-default" in ollama

        pushes = [c for c in sh.calls if c[:2] == ("git", "push")]
        assert pushes == [("git", "push", "-u", "origin", "provider-watch/fireworks-llm-default")]
        create = next(c for c in sh.calls if c[:3] == ("gh", "pr", "create"))
        assert "--draft" in create and "--label" in create
        assert create[create.index("--title") + 1] == "Default FireworksLLMService to gpt-oss-120b"
        assert "reports/fireworks/llm/2026-08-20.md" in create[create.index("--body") + 1]

    def test_second_pass_is_a_no_op(self, reports_dir):
        tmp_path, publish = reports_dir
        sh = self.FakeShell(
            branches=[
                "provider-watch/fireworks-llm-default",
                "provider-watch/groq-llm-example",
                "provider-watch/ollama-llm-default",
            ]
        )
        kwargs = dict(
            sh=sh,
            repo_root=tmp_path,
            pipecat_repo="p/p",
            reports_repo="p/r",
            date="2026-08-20",
            cap=8,
        )
        publish.publish_prs(publish.load_reports(tmp_path, "2026-08-20"), **kwargs)
        before = len([c for c in sh.calls if c[:3] == ("gh", "pr", "create")])
        publish.publish_prs(publish.load_reports(tmp_path, "2026-08-20"), **kwargs)
        after = len([c for c in sh.calls if c[:3] == ("gh", "pr", "create")])
        assert before == 3 and after == 3

    def test_worth_an_issue(self, reports_dir, tmp_path):
        _, publish = reports_dir
        assert publish.worth_an_issue(publish.load_reports(tmp_path, "2026-08-20"))
        quiet = tmp_path / "quiet" / "reports" / "groq" / "llm"
        quiet.mkdir(parents=True)
        (quiet / "2026-08-20.md").write_text(
            "---\nservice: groq/llm\ngaps: []\nprs: []\nerror: null\n---\n"
        )
        assert not publish.worth_an_issue(publish.load_reports(tmp_path / "quiet", "2026-08-20"))
        (quiet / "2026-08-20.md").write_text("---\nservice: groq/llm\nerror: boom\n---\n")
        assert publish.worth_an_issue(publish.load_reports(tmp_path / "quiet", "2026-08-20"))

    def test_missing_branch_is_skipped(self, reports_dir):
        tmp_path, publish = reports_dir
        sh = self.FakeShell(branches=[])
        outcome = publish.publish_prs(
            publish.load_reports(tmp_path, "2026-08-20"),
            sh=sh,
            repo_root=tmp_path,
            pipecat_repo="p/p",
            reports_repo="p/r",
            date="2026-08-20",
            cap=8,
        )
        assert len(outcome.skipped) == 3 and not outcome.opened


class TestSignals:
    """probe.py signals: SDK derivation from pyproject and spec snapshotting, no network."""

    @pytest.fixture
    def probe(self):
        import probe

        return probe

    def test_sdk_requirements_come_from_pyproject_extras(self, probe, units):
        deepgram = [u for u in units if u.provider == "deepgram"]
        reqs = probe.sdk_requirements("deepgram", deepgram)
        assert any(r.startswith("deepgram-sdk") for r in reqs)
        assert not any(r.startswith("pipecat-ai") for r in reqs)

        google = [u for u in units if u.provider == "google"]
        names = {
            r.split(">")[0].split("<")[0].split("=")[0]
            for r in probe.sdk_requirements("google", google)
        }
        assert {"google-genai", "google-cloud-speech", "google-cloud-texttospeech"} <= names

    def test_thin_wrappers_fall_back_to_openai(self, probe, units):
        groq = [u for u in units if u.provider == "groq"]
        reqs = probe.sdk_requirements("groq", groq)
        assert any(r.startswith("groq") for r in reqs)  # groq has its own extra
        cerebras = [u for u in units if u.provider == "cerebras"]
        reqs = probe.sdk_requirements("cerebras", cerebras)
        assert reqs and all(r.startswith("openai") for r in reqs)

    def test_spec_snapshot_detects_change(self, probe, tmp_path, monkeypatch):
        payloads = iter(
            [b"openapi: 3.0\npaths: {}\n", b"openapi: 3.0\npaths: {}\n", b"openapi: 3.1\n"]
        )
        monkeypatch.setattr(probe, "_http_bytes", lambda url: next(payloads))

        first = probe.spec_snapshot("spec.yml", "https://x/spec.yml", tmp_path)
        assert first["new"] and first["changed"] and (tmp_path / "spec.yml").exists()
        second = probe.spec_snapshot("spec.yml", "https://x/spec.yml", tmp_path)
        assert not second["new"] and not second["changed"] and second["sha256"] == first["sha256"]
        third = probe.spec_snapshot("spec.yml", "https://x/spec.yml", tmp_path)
        assert third["changed"] and third["sha256"] != first["sha256"]
        assert (tmp_path / "spec.yml").read_bytes() == b"openapi: 3.1\n"

    def test_spec_fetch_failure_is_reported_not_raised(self, probe, tmp_path, monkeypatch):
        def boom(url):
            raise OSError("nope")

        monkeypatch.setattr(probe, "_http_bytes", boom)
        result = probe.spec_snapshot("spec.yml", "https://x/spec.yml", tmp_path)
        assert result["error"] == "nope" and not (tmp_path / "spec.yml").exists()

    def test_provider_entry_has_named_specs(self, probe):
        entry = probe.provider_entry("deepgram")
        assert all({"name", "url"} <= set(spec) for spec in entry["specs"])
        assert probe.provider_entry("nosuchprovider") == {}
