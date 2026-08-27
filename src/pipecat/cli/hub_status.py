#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Freshness check for a locally installed Pipecat Context Hub index.

The hub is a local index of Pipecat docs, examples, and API source that coding
agents query over MCP. When it goes stale, or was built against a different
``pipecat-ai`` release than the project being worked on, an agent cites APIs
that have since changed — and nothing surfaces that until the generated code is
wrong. This module produces the one-line warning the CLI prints in that case.

The hub publishes its ``index_metadata`` table as a documented read contract
precisely so a check like this stays cheap: every in-process hub query opens
ChromaDB, so importing the package or shelling out to its ``status`` command
would cost far more than the command being run. Reading the SQLite table
directly is sub-millisecond and needs nothing outside the standard library.

Every function here answers "unknown" rather than raising. A freshness hint must
never break the command a user actually asked for.
"""

import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

# Highest metadata contract version this reader understands. A higher value in
# the index means a newer hub changed the table's shape or a key's meaning, so
# we stay silent rather than guess.
_SUPPORTED_CONTRACT_VERSION = 1

# Matches the hub's own default and env var, so tuning the threshold moves both
# surfaces together rather than leaving them to disagree.
_STALE_AFTER_ENV = "PIPECAT_HUB_STALE_AFTER_DAYS"
_DEFAULT_STALE_AFTER_DAYS = 7.0

# Set to 0/false/no to silence the check entirely.
_CHECK_ENV = "PIPECAT_HUB_CHECK"
_DISABLED_VALUES = frozenset({"0", "false", "no"})

_DATA_DIR_ENV = "PIPECAT_HUB_DATA_DIR"

# Leading `major.minor` of a PEP 440 version. Only those two components matter
# here, so this deliberately ignores patch, dev, pre, and post segments.
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)")

# `pipecat_ai-<version>.dist-info` in a project's virtualenv.
_DIST_INFO_RE = re.compile(r"^pipecat_ai-(.+?)\.dist-info$")


def _enabled() -> bool:
    """False when the user has switched the check off."""
    return os.environ.get(_CHECK_ENV, "1").strip().lower() not in _DISABLED_VALUES


def _data_dir() -> Path:
    """Where the hub keeps its index.

    Honouring the env var is part of the contract: assuming the default would
    report "no index" at anyone who relocated theirs.
    """
    configured = os.environ.get(_DATA_DIR_ENV, "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".pipecat-context-hub"


def read_hub_metadata() -> dict[str, str] | None:
    """Read the hub's index metadata, or None when it can't be read.

    Opens read-only with no lock wait. The database is WAL, so this neither
    blocks nor is blocked by a concurrent refresh or a running MCP server, and
    sees only committed state.
    """
    db_path = _data_dir() / "metadata.db"
    if not db_path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0)
        try:
            rows = conn.execute("SELECT key, value FROM index_metadata").fetchall()
        finally:
            conn.close()
    except Exception:
        # Unreadable, locked, mid-first-build, on a filesystem without WAL, or
        # written by a hub whose schema predates the table. All mean "unknown".
        return None
    metadata = {str(key): str(value) for key, value in rows}
    try:
        if int(metadata.get("metadata_contract_version", 0)) > _SUPPORTED_CONTRACT_VERSION:
            return None
    except ValueError:
        return None
    return metadata


def _stale_after_days() -> float:
    """Index age, in days, past which the index is called stale."""
    raw = os.environ.get(_STALE_AFTER_ENV, "").strip()
    if not raw:
        return _DEFAULT_STALE_AFTER_DAYS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_STALE_AFTER_DAYS
    # Reject nan/inf: inf would make every index look fresh, nan would make
    # every index look stale.
    if value != value or value in (float("inf"), float("-inf")):
        return _DEFAULT_STALE_AFTER_DAYS
    return value


def _index_age_days(metadata: dict[str, str]) -> float | None:
    """Days since the last completed refresh, or None when unknown."""
    last_refresh = metadata.get("last_refresh_at")
    if not last_refresh:
        return None
    try:
        refreshed = datetime.fromisoformat(last_refresh)
    except ValueError:
        return None
    if refreshed.tzinfo is None:
        refreshed = refreshed.replace(tzinfo=UTC)
    return (datetime.now(UTC) - refreshed).total_seconds() / 86400


def _major_minor(version: str) -> tuple[int, int] | None:
    """Leading ``(major, minor)`` of a version string, or None if unparseable."""
    match = _VERSION_RE.match(version.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def project_pipecat_version(cwd: Path | None = None) -> str | None:
    """The ``pipecat-ai`` version installed in the project's virtualenv.

    Deliberately not ``importlib.metadata.version``: the CLI is normally a global
    ``uv tool``, so that would report the tool's own version rather than the
    project's. Only ``$VIRTUAL_ENV`` and ``./.venv`` are consulted — walking up
    parent directories would find an unrelated venv in a monorepo and warn about
    a project the user isn't working on.

    Returns None when the framework is installed as an editable checkout: the
    developer *is* pipecat, and no released version meaningfully describes it.
    """
    root = cwd or Path.cwd()
    candidates = []
    virtual_env = os.environ.get("VIRTUAL_ENV", "").strip()
    if virtual_env:
        candidates.append(Path(virtual_env))
    candidates.append(root / ".venv")

    for venv in candidates:
        try:
            site_packages = list(venv.glob("lib/python*/site-packages"))
            # Windows venvs put site-packages directly under Lib.
            site_packages += list(venv.glob("Lib/site-packages"))
        except OSError:
            continue
        for site_dir in site_packages:
            try:
                if any(site_dir.glob("__editable__.pipecat_ai-*.pth")):
                    return None
                for entry in site_dir.iterdir():
                    match = _DIST_INFO_RE.match(entry.name)
                    if match:
                        return match.group(1)
            except OSError:
                continue
    return None


def freshness_warning(cwd: Path | None = None) -> str | None:
    """One-line warning about the local hub index, or None when there's nothing to say.

    Silent unless a hub index exists — the CLI must stay quiet for the many users
    who don't run the hub at all.
    """
    if not _enabled():
        return None
    metadata = read_hub_metadata()
    if metadata is None:
        return None

    threshold = _stale_after_days()
    age = _index_age_days(metadata)
    if age is None:
        # An index whose refresh never completed. Reporting an age would be a
        # lie, and calling it stale would be the wrong instruction.
        return (
            "Pipecat Context Hub index looks unbuilt — run `pipecat context-hub refresh` "
            "so coding agents get current Pipecat context."
        )
    if threshold > 0 and age >= threshold:
        days = round(age)
        return (
            f"Pipecat Context Hub index is {days} day{'' if days == 1 else 's'} old — run "
            "`pipecat context-hub refresh` so coding agents don't cite stale APIs."
        )

    return _version_mismatch_warning(metadata, cwd)


def _version_mismatch_warning(metadata: dict[str, str], cwd: Path | None) -> str | None:
    """Warn when the project has moved a release ahead of what the index covers.

    Compares only ``major.minor``; patch and dev/pre-release segments are noise
    here and every one of them would be a false alarm.

    Only fires when the *project* is ahead. An index built from the default
    branch legitimately runs ahead of released users, and a superset index is
    not a correctness problem worth interrupting anyone over.
    """
    indexed = metadata.get("indexed_framework_version")
    project = project_pipecat_version(cwd)
    if not indexed or not project:
        return None

    indexed_mm = _major_minor(indexed)
    project_mm = _major_minor(project)
    if indexed_mm is None or project_mm is None:
        return None

    # An unpinned refresh tracks the default branch, so the recorded tag is a
    # floor: the index already contains commits published after it. Allow a
    # minor of slack in that case, or every developer on a source checkout gets
    # warned about an index that is in fact newer than its own tag.
    slack = 0
    try:
        if int(metadata.get("indexed_framework_commits_ahead", "0")) > 0:
            slack = 1
    except ValueError:
        slack = 1

    indexed_major, indexed_minor = indexed_mm
    project_major, project_minor = project_mm
    allowed = (indexed_major, indexed_minor + slack)
    if (project_major, project_minor) <= allowed:
        return None

    return (
        f"Pipecat Context Hub index was built for pipecat-ai {indexed}, but this "
        f"project uses {project} — run `pipecat context-hub refresh` so coding agents "
        "match your version."
    )
