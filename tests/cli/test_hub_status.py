"""Tests for the Pipecat Context Hub freshness check.

Every path here must degrade to silence rather than raise: this runs ahead of
every CLI command, so a bad index, a locked database, or an odd project layout
must never break the command the user actually asked for.

Fixtures build a real SQLite database rather than mocking the reader, so the
tests exercise the same contract an installed hub publishes.
"""

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from pipecat.cli.hub_status import (
    freshness_warning,
    project_pipecat_version,
    read_hub_metadata,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Point the check at an empty data dir and away from any active venv."""
    monkeypatch.setenv("PIPECAT_HUB_DATA_DIR", str(tmp_path / "hub"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("PIPECAT_HUB_CHECK", raising=False)
    monkeypatch.delenv("PIPECAT_HUB_STALE_AFTER_DAYS", raising=False)


def _write_index(tmp_path, **metadata) -> None:
    """Create a hub index database carrying *metadata*."""
    data_dir = tmp_path / "hub"
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(data_dir / "metadata.db")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS index_metadata "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.executemany(
        "INSERT OR REPLACE INTO index_metadata VALUES (?, ?, ?)",
        [(k, str(v), "now") for k, v in metadata.items()],
    )
    conn.commit()
    conn.close()


def _ago(days: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def _make_project(tmp_path, version: str, *, editable: bool = False):
    """A project directory with pipecat-ai installed in ./.venv."""
    # Named per version so a test can build more than one project.
    project = tmp_path / f"project-{version}"
    site = project / ".venv" / "lib" / "python3.14" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    (site / f"pipecat_ai-{version}.dist-info").mkdir()
    if editable:
        (site / f"__editable__.pipecat_ai-{version}.pth").touch()
    return project


class TestReadHubMetadata:
    def test_no_index_returns_none(self):
        assert read_hub_metadata() is None

    def test_reads_committed_metadata(self, tmp_path):
        _write_index(tmp_path, last_refresh_at=_ago(1), indexed_framework_version="1.6.0")
        metadata = read_hub_metadata()
        assert metadata is not None
        assert metadata["indexed_framework_version"] == "1.6.0"

    def test_corrupt_database_returns_none(self, tmp_path):
        data_dir = tmp_path / "hub"
        data_dir.mkdir(parents=True)
        (data_dir / "metadata.db").write_text("this is not a database")
        assert read_hub_metadata() is None

    def test_missing_table_returns_none(self, tmp_path):
        """An index from a hub predating the table is unknown, not broken."""
        data_dir = tmp_path / "hub"
        data_dir.mkdir(parents=True)
        sqlite3.connect(data_dir / "metadata.db").close()
        assert read_hub_metadata() is None

    def test_newer_contract_version_is_ignored(self, tmp_path):
        """A hub that changed the contract must not be second-guessed."""
        _write_index(tmp_path, metadata_contract_version=99, last_refresh_at=_ago(1))
        assert read_hub_metadata() is None

    def test_current_contract_version_is_accepted(self, tmp_path):
        _write_index(tmp_path, metadata_contract_version=1, last_refresh_at=_ago(1))
        assert read_hub_metadata() is not None


class TestStaleness:
    def test_silent_without_an_index(self):
        """Most users don't run the hub; the CLI must not nag them."""
        assert freshness_warning() is None

    def test_fresh_index_is_silent(self, tmp_path):
        _write_index(tmp_path, last_refresh_at=_ago(1))
        assert freshness_warning() is None

    def test_stale_index_warns(self, tmp_path):
        _write_index(tmp_path, last_refresh_at=_ago(30))
        warning = freshness_warning()
        assert warning is not None
        assert "30 days old" in warning
        assert "pipecat context-hub refresh" in warning

    def test_threshold_is_configurable(self, tmp_path, monkeypatch):
        _write_index(tmp_path, last_refresh_at=_ago(3))
        assert freshness_warning() is None
        monkeypatch.setenv("PIPECAT_HUB_STALE_AFTER_DAYS", "2")
        assert freshness_warning() is not None

    def test_zero_threshold_disables_staleness(self, tmp_path, monkeypatch):
        _write_index(tmp_path, last_refresh_at=_ago(365))
        monkeypatch.setenv("PIPECAT_HUB_STALE_AFTER_DAYS", "0")
        assert freshness_warning() is None

    def test_index_without_a_completed_refresh(self, tmp_path):
        """Reporting an age would be a lie; 'stale' would be the wrong advice."""
        _write_index(tmp_path, metadata_contract_version=1)
        warning = freshness_warning()
        assert warning is not None
        assert "unbuilt" in warning

    @pytest.mark.parametrize("bad", ["not-a-number", "nan", "inf"])
    def test_bad_threshold_falls_back_to_the_default(self, tmp_path, monkeypatch, bad):
        """inf would make every index look fresh; nan would make every one stale."""
        monkeypatch.setenv("PIPECAT_HUB_STALE_AFTER_DAYS", bad)
        _write_index(tmp_path, last_refresh_at=_ago(30))
        assert freshness_warning() is not None


class TestDisableSwitch:
    @pytest.mark.parametrize("value", ["0", "false", "no", "FALSE"])
    def test_check_can_be_switched_off(self, tmp_path, monkeypatch, value):
        _write_index(tmp_path, last_refresh_at=_ago(365))
        monkeypatch.setenv("PIPECAT_HUB_CHECK", value)
        assert freshness_warning() is None

    def test_other_values_leave_it_on(self, tmp_path, monkeypatch):
        _write_index(tmp_path, last_refresh_at=_ago(365))
        monkeypatch.setenv("PIPECAT_HUB_CHECK", "1")
        assert freshness_warning() is not None


class TestProjectVersion:
    def test_reads_dist_info(self, tmp_path):
        assert project_pipecat_version(_make_project(tmp_path, "1.9.0")) == "1.9.0"

    def test_editable_framework_checkout_is_silent(self, tmp_path):
        """The developer is pipecat; no released version describes their tree."""
        assert project_pipecat_version(_make_project(tmp_path, "1.9.0", editable=True)) is None

    def test_no_venv_returns_none(self, tmp_path):
        assert project_pipecat_version(tmp_path) is None

    def test_active_virtualenv_takes_precedence(self, tmp_path, monkeypatch):
        project = _make_project(tmp_path, "1.9.0")
        monkeypatch.setenv("VIRTUAL_ENV", str(project / ".venv"))
        assert project_pipecat_version(tmp_path) == "1.9.0"


class TestCliIntegration:
    """The notice is emitted by the console-script entry point.

    It lives in `run()` rather than the group callback because click answers a
    bare `pipecat --help` eagerly, without invoking the callback at all.
    """

    def test_warning_goes_to_stderr(self, tmp_path, capsys):
        from pipecat.cli.main import _warn_about_stale_hub_index

        _write_index(tmp_path, last_refresh_at=_ago(42))
        _warn_about_stale_hub_index()
        captured = capsys.readouterr()
        assert "Context Hub" in captured.err
        assert captured.out == ""

    def test_silent_without_an_index(self, capsys):
        from pipecat.cli.main import _warn_about_stale_hub_index

        _warn_about_stale_hub_index()
        assert capsys.readouterr().err == ""

    def test_a_failing_check_never_propagates(self, monkeypatch, capsys):
        """The notice must not be able to break a user's command."""
        import pipecat.cli.hub_status as hub_status
        from pipecat.cli.main import _warn_about_stale_hub_index

        def _boom():
            raise RuntimeError("unexpected")

        monkeypatch.setattr(hub_status, "freshness_warning", _boom)
        _warn_about_stale_hub_index()
        assert capsys.readouterr().err == ""


class TestVersionMismatch:
    def _fresh_index(self, tmp_path, version, commits_ahead=0):
        _write_index(
            tmp_path,
            last_refresh_at=_ago(1),
            indexed_framework_version=version,
            indexed_framework_commits_ahead=commits_ahead,
        )

    def test_project_ahead_by_a_minor_warns(self, tmp_path):
        self._fresh_index(tmp_path, "1.6.0")
        warning = freshness_warning(_make_project(tmp_path, "1.8.0"))
        assert warning is not None
        assert "1.6.0" in warning and "1.8.0" in warning

    def test_matching_minor_is_silent(self, tmp_path):
        self._fresh_index(tmp_path, "1.6.0")
        assert freshness_warning(_make_project(tmp_path, "1.6.3")) is None

    def test_patch_and_dev_segments_are_ignored(self, tmp_path):
        """The overwhelmingly common case; warning here would be pure noise."""
        self._fresh_index(tmp_path, "1.6.0")
        assert freshness_warning(_make_project(tmp_path, "1.6.1.dev55")) is None

    def test_index_ahead_of_project_is_silent(self, tmp_path):
        """A superset index isn't a correctness problem worth interrupting for."""
        self._fresh_index(tmp_path, "1.9.0")
        assert freshness_warning(_make_project(tmp_path, "1.6.0")) is None

    def test_unpinned_index_gets_a_minor_of_slack(self, tmp_path):
        """Tracking the default branch means the recorded tag is a floor.

        An index 55 commits past v1.6.0 may already contain 1.7.0's code, so
        warning at 1.7.0 would fire at every developer on a source checkout.
        """
        self._fresh_index(tmp_path, "1.6.0", commits_ahead=55)
        assert freshness_warning(_make_project(tmp_path, "1.7.0")) is None
        assert freshness_warning(_make_project(tmp_path, "1.8.0")) is not None

    def test_pinned_index_gets_no_slack(self, tmp_path):
        self._fresh_index(tmp_path, "1.6.0", commits_ahead=0)
        assert freshness_warning(_make_project(tmp_path, "1.7.0")) is not None

    def test_major_bump_warns(self, tmp_path):
        self._fresh_index(tmp_path, "1.6.0")
        assert freshness_warning(_make_project(tmp_path, "2.0.0")) is not None

    def test_missing_indexed_version_is_silent(self, tmp_path):
        """Indexes built before the hub recorded provenance."""
        _write_index(tmp_path, last_refresh_at=_ago(1))
        assert freshness_warning(_make_project(tmp_path, "1.9.0")) is None

    def test_unparseable_version_is_silent(self, tmp_path):
        self._fresh_index(tmp_path, "not-a-version")
        assert freshness_warning(_make_project(tmp_path, "1.9.0")) is None

    def test_staleness_takes_priority(self, tmp_path):
        """One warning at a time, and refreshing fixes both."""
        _write_index(tmp_path, last_refresh_at=_ago(30), indexed_framework_version="1.6.0")
        warning = freshness_warning(_make_project(tmp_path, "1.9.0"))
        assert warning is not None
        assert "days old" in warning
