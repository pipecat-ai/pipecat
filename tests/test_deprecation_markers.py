#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audit of deprecation conventions across ``src/pipecat``.

The ``.. deprecated::`` docstring directive is the single source of truth for
deprecations — the registry generator (``scripts/deprecations/generate.py``)
parses it into ``deprecations.json``. This audit and that generator share one
parser (``scripts/deprecations/scan.py``), so the enforced grammar and the
generated registry cannot drift.

The parsing and validation rules live in ``scan.py``; the tests below are thin
assertions over its validators, plus runtime checks that the converted shims
still emit ``DeprecationWarning`` without warning at import time.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

# The shared parser lives under scripts/ (build tooling, not shipped runtime
# code). Put it on the path so the audit and the generator validate identically.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from deprecations import generate as dgen  # noqa: E402
from deprecations import generate_removals as drem  # noqa: E402
from deprecations import scan as dscan  # noqa: E402

from pipecat.frames.frames import (  # noqa: E402
    CancelTaskFrame,
    EndTaskFrame,
    InterruptionTaskFrame,
    StopTaskFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.worker import PipelineTask, PipelineTaskParams  # noqa: E402
from pipecat.processors.filters.identity_filter import IdentityFilter  # noqa: E402

SRC_ROOT = Path(__file__).parent.parent / "src" / "pipecat"
_SCAN = dscan.scan_source(SRC_ROOT)


# --- Directive enforcement (the registry source of truth) --------------------


def test_directives_parse():
    """Every ``.. deprecated::`` directive yields a version and a target/no-replacement."""
    assert _SCAN.directives, "expected .. deprecated:: directives in src/pipecat"
    bad = dscan.check_directives_parse(_SCAN)
    assert not bad, (
        "These `.. deprecated::` directives don't parse — give them a version after `::` "
        "and a body that names a replacement (`Use :class:`X` instead.`) or says "
        "`No replacement.`:\n" + "\n".join(f"  {b}" for b in bad)
    )


def test_directives_state_removal_version():
    """Every directive states a concrete removal version ("removed in X.Y.Z").

    Lets the registry record ``removed_in`` for parameter/module/behavior
    deprecations, which have no ``@deprecated`` message to carry it.
    """
    bad = dscan.check_directive_removal_versions(_SCAN)
    assert not bad, (
        "These `.. deprecated::` directives don't state a removal version — add "
        "`Will be removed in X.Y.Z.` (a concrete semantic version):\n"
        + "\n".join(f"  {b}" for b in bad)
    )


# --- @deprecated decorator message consistency -------------------------------


def test_deprecated_messages_follow_template():
    """Every @deprecated call site uses a literal message matching the template."""
    assert any(s.has_decorator for s in _SCAN.symbols), "expected @deprecated call sites"
    bad = dscan.check_decorator_messages(_SCAN)
    assert not bad, (
        "These @deprecated messages don't follow the canonical template from "
        "pipecat.utils.deprecation:\n" + "\n".join(f"  {b}" for b in bad)
    )


def test_deprecated_subject_names_decorated_symbol():
    """The `Subject` in the message is the symbol the decorator is applied to."""
    bad = dscan.check_decorator_subjects(_SCAN)
    assert not bad, "These @deprecated subjects don't match the decorated symbol:\n" + "\n".join(
        f"  {b}" for b in bad
    )


def test_deprecated_version_matches_docstring_directive():
    """The `since` version in the message agrees with the docstring directive."""
    bad = dscan.check_decorator_versions(_SCAN)
    assert not bad, (
        "These @deprecated message versions disagree with their docstring directive:\n"
        + "\n".join(f"  {b}" for b in bad)
    )


def test_deprecated_replacement_targets_exist():
    """Backticked class/function replacement targets in messages name real symbols."""
    bad = dscan.check_decorator_replacements_exist(_SCAN)
    assert not bad, (
        "These @deprecated replacement targets aren't defined in src/pipecat (typo?):\n"
        + "\n".join(f"  {b}" for b in bad)
    )


# --- Generated registry ------------------------------------------------------


def test_deprecations_registry_is_up_to_date():
    """The committed registry matches a fresh build from the source.

    Regenerate with ``uv run python scripts/deprecations/generate.py`` when this
    fails (the same check CI runs as a drift guard).
    """
    committed = json.loads(dgen.REGISTRY_PATH.read_text(encoding="utf-8"))
    fresh = dgen.build_registry(SRC_ROOT)
    assert committed == fresh, (
        f"{dgen.REGISTRY_PATH.name} is stale — run "
        "`uv run python scripts/deprecations/generate.py` and commit the result."
    )


# --- Runtime behavior --------------------------------------------------------


def test_pipeline_task_warns():
    with pytest.warns(DeprecationWarning, match="`PipelineTask` is deprecated"):
        PipelineTask(Pipeline([IdentityFilter()]))


@pytest.mark.asyncio
async def test_pipeline_task_params_warns():
    with pytest.warns(DeprecationWarning, match="`PipelineTaskParams` is deprecated"):
        PipelineTaskParams(loop=asyncio.get_running_loop())


@pytest.mark.asyncio
async def test_pipeline_runner_warns():
    with pytest.warns(DeprecationWarning, match="`PipelineRunner` is deprecated"):
        PipelineRunner()


@pytest.mark.parametrize(
    "frame_cls",
    [EndTaskFrame, StopTaskFrame, CancelTaskFrame, InterruptionTaskFrame],
)
def test_task_frame_aliases_warn(frame_cls):
    with pytest.warns(DeprecationWarning, match=f"`{frame_cls.__name__}` is deprecated"):
        frame_cls()


def test_no_deprecation_warnings_at_import_time():
    """Importing the modules with deprecated shims must not emit a pipecat DeprecationWarning.

    Guards the suppression of @deprecated subclassing warnings for the task frame
    aliases defined inside pipecat.frames.frames. Done in a subprocess for a
    fresh import, and filtered to pipecat's own deprecations (the canonical
    "... is deprecated since X.Y.Z ..." message) so unrelated stdlib warnings —
    e.g. ``audioop`` on Python 3.12 — don't trip it.
    """
    script = (
        "import sys, warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import pipecat.frames.frames\n"
        "    import pipecat.pipeline.worker\n"
        "    import pipecat.pipeline.runner\n"
        "bad = [str(w.message) for w in caught\n"
        "       if issubclass(w.category, DeprecationWarning)\n"
        "       and ' is deprecated since ' in str(w.message)]\n"
        "if bad:\n"
        "    sys.stderr.write('pipecat import-time deprecation warnings:\\n' + '\\n'.join(bad))\n"
        "    sys.exit(1)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --- Removal history (removals.json) ------------------------------------------
#
# Removals are detected at release-prep by diffing the previous release tag's
# registry against the working tree (generate_removals.py). The full
# rebuild-from-tags drift check is deferred until removals actually exist
# (~1.6.0/2.0.0); for now we keep a schema-sanity backstop plus unit tests over
# the pure diff logic.

_REMOVAL_FIELDS = {
    "subject",
    "module",
    "kind",
    "deprecated_in",
    "removed_in",
    "announced_removed_in",
    "relation",
    "replacement",
    "message",
}


def _dep_registry(*records):
    """A minimal ``deprecations.json``-shaped document for the given records."""
    return {"schema_version": 1, "deprecations": list(records)}


def _dep_record(subject, **overrides):
    """A deprecation record with sensible defaults, overridable per field."""
    rec = {
        "subject": subject,
        "module": "pipecat.x",
        "kind": "class",
        "deprecated_in": "1.3.0",
        "removed_in": "2.0.0",
        "relation": "use_existing",
        "replacement": "Y",
        "message": f"`{subject}` is deprecated since 1.3.0 and will be removed in 2.0.0. Use `Y` instead.",
        "location": "pipecat/x.py:1",
    }
    rec.update(overrides)
    return rec


def test_removals_registry_schema_is_valid():
    """The committed removals.json is well-formed (schema-sanity backstop)."""
    doc = json.loads(drem.REMOVALS_PATH.read_text(encoding="utf-8"))
    assert doc.get("schema_version") == drem.SCHEMA_VERSION
    assert isinstance(doc.get("removals"), list)
    seen = set()
    for rec in doc["removals"]:
        assert set(rec) == _REMOVAL_FIELDS, f"unexpected fields on {rec.get('subject')!r}"
        assert "location" not in rec  # dropped — points at source that's gone
        assert rec["subject"] and rec["subject"] not in seen, f"duplicate {rec['subject']!r}"
        seen.add(rec["subject"])
        assert drem._VERSION_RE.match(rec["removed_in"]), rec["removed_in"]


def test_compute_removals_detects_disappeared_symbol():
    prev = _dep_registry(_dep_record("Gone"), _dep_record("Kept"))
    current = _dep_registry(_dep_record("Kept"))
    removals = drem.compute_removals(prev, current, "2.0.0", [])
    assert [r["subject"] for r in removals] == ["Gone"]
    gone = removals[0]
    assert set(gone) == _REMOVAL_FIELDS  # exactly the removal schema, no location
    assert gone["removed_in"] == "2.0.0"
    assert gone["replacement"] == "Y"


def test_compute_removals_no_change_when_nothing_removed():
    prev = _dep_registry(_dep_record("A"))
    current = _dep_registry(_dep_record("A"))
    assert drem.compute_removals(prev, current, "2.0.0", []) == []


def test_compute_removals_is_idempotent():
    """A subject already recorded as removed is not appended again."""
    prev = _dep_registry(_dep_record("Gone"))
    current = _dep_registry()
    first = drem.compute_removals(prev, current, "2.0.0", [])
    second = drem.compute_removals(prev, current, "2.0.0", first)
    assert first == second


def test_compute_removals_bootstrap_has_no_previous():
    """The first registry-bearing release has no previous registry → empty."""
    current = _dep_registry(_dep_record("A"))
    assert drem.compute_removals(None, current, "1.4.0", []) == []


def test_compute_removals_records_actual_vs_announced_version():
    """removed_in is the real disappearance; announced_removed_in is the promise."""
    prev = _dep_registry(_dep_record("Slipped", removed_in="2.0.0"))
    current = _dep_registry()
    removals = drem.compute_removals(prev, current, "2.1.0", [])  # slipped past 2.0.0
    assert removals[0]["removed_in"] == "2.1.0"
    assert removals[0]["announced_removed_in"] == "2.0.0"
