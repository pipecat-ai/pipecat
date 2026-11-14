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

import inspect
import json
import subprocess
import sys
import warnings
from dataclasses import dataclass
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
    EagerEndOfTurnCancelFrame,
    EndTaskFrame,
    InterruptionTaskFrame,
    StartFrame,
    StopTaskFrame,
    UserTurnResponseCancelFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.worker import PipelineTask, PipelineTaskParams  # noqa: E402
from pipecat.processors.filters.identity_filter import IdentityFilter  # noqa: E402
from pipecat.utils.asyncio.task_manager import TaskManager  # noqa: E402
from pipecat.utils.deprecation import renamed_init_field, warn_deprecated_read  # noqa: E402

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


def test_no_replacement_directive_extracts_no_replacement():
    """A directive leading with "No replacement." records none — even with backticks.

    The first-reference rule treats the first backtick/role token as the
    replacement, so a no-replacement body may freely backtick contextual symbols
    (the deprecated thing itself, related types) without one being mistaken for a
    replacement, as long as it leads with the explicit marker.
    """
    body = (
        "No replacement. ``FlowResult`` is no longer referenced by any handler; "
        "the upstream contract is ``Any``. Will be removed in 2.0.0."
    )
    assert dscan.first_reference(body) is None
    assert dscan.relation_for(body, dscan.first_reference(body)) == "none"

    # An incidental relation verb in later prose doesn't override the marker.
    moved = "No replacement. The old behavior moved to a different layer entirely."
    assert dscan.relation_for(moved, dscan.first_reference(moved)) == "none"

    # A real replacement is still extracted as before.
    use = "Use :class:`Foo` instead. Will be removed in 2.0.0."
    assert dscan.first_reference(use) == "Foo"
    assert dscan.relation_for(use, "Foo") == "use_existing"


def test_directives_are_attributed_to_the_entry_they_sit_under():
    """A directive under an event bullet or an Args entry deprecates that entry alone."""
    docstring = """A processor.

    Events:

    - on_ready: Triggered when ready.
    - on_old_event: Triggered by the old thing.

      .. deprecated:: 1.8.0
          Use ``on_ready`` instead. Will be removed in 2.0.0.

    Args:
        old_param: An old parameter.

            .. deprecated:: 1.8.0
                Use ``new_param`` instead. Will be removed in 2.0.0.
    """
    entries = [(param, kind) for _, _, param, kind in dscan.iter_directives(docstring)]

    assert entries == [("on_old_event", "event"), ("old_param", "parameter")]


def test_top_level_directive_deprecates_the_symbol():
    """A directive outside any entry deprecates the symbol the docstring belongs to."""
    docstring = """A processor.

    .. deprecated:: 1.8.0
        Use ``NewProcessor`` instead. Will be removed in 2.0.0.

    Events:

    - on_ready: Triggered when ready.
    """
    entries = [(param, kind) for _, _, param, kind in dscan.iter_directives(docstring)]

    assert entries == [(None, "parameter")]


# --- @deprecated decorator message consistency -------------------------------


def test_deprecated_messages_follow_template():
    """Every @deprecated call site uses a literal message matching the template."""
    assert any(s.has_decorator for s in _SCAN.symbols), "expected @deprecated call sites"
    bad = dscan.check_decorator_messages(_SCAN)
    assert not bad, (
        "These @deprecated messages don't follow the canonical template from "
        "pipecat.utils.deprecation:\n" + "\n".join(f"  {b}" for b in bad)
    )


def test_warn_messages_follow_template():
    """Every hand-written deprecation warning uses a literal message matching the template."""
    assert _SCAN.warn_sites, "expected warnings.warn(..., DeprecationWarning) call sites"
    bad = dscan.check_warn_messages(_SCAN)
    assert not bad, (
        "These deprecation warnings don't follow the canonical template from "
        "pipecat.utils.deprecation:\n" + "\n".join(f"  {b}" for b in bad)
    )


def test_deprecation_warnings_use_the_helper():
    """Every hand-written deprecation warning is raised through ``warn_deprecated()``."""
    bad = dscan.check_warn_helper(_SCAN)
    assert not bad, (
        "These deprecation warnings call warnings.warn() directly; use "
        "pipecat.utils.deprecation.warn_deprecated():\n" + "\n".join(f"  {b}" for b in bad)
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
        PipelineTaskParams(task_manager=TaskManager())


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


def test_eager_end_of_turn_cancel_frame_warns():
    with pytest.warns(DeprecationWarning, match="`EagerEndOfTurnCancelFrame` is deprecated"):
        frame = EagerEndOfTurnCancelFrame()
    assert isinstance(frame, UserTurnResponseCancelFrame)


def test_deferring_a_stop_strategy_warns_once():
    from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy, deferred

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deferred(SpeechTimeoutUserTurnStopStrategy())
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert messages == [
        "`deferred` is deprecated since 1.15.0 and will be removed in 2.0.0. No replacement."
    ]


def test_external_user_turn_completion_stop_strategy_warns():
    from pipecat.turns.user_stop import ExternalUserTurnCompletionStopStrategy

    with pytest.warns(
        DeprecationWarning, match="`ExternalUserTurnCompletionStopStrategy` is deprecated"
    ):
        ExternalUserTurnCompletionStopStrategy()


def test_no_deprecation_warnings_at_import_time():
    """Importing the modules with deprecated shims must not emit a pipecat DeprecationWarning.

    Guards against @deprecated subclassing warnings from the task frame aliases
    in pipecat.frames.frames, the deprecated stop strategies in
    pipecat.turns.user_stop, and the deprecated TTS base classes in
    pipecat.services.tts_service. Done in a subprocess for a
    fresh import, and filtered to pipecat's own deprecations (the canonical
    "... is deprecated since X.Y.Z ..." message) so unrelated stdlib and
    third-party warnings don't trip it.
    """
    script = (
        "import sys, warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import pipecat.frames.frames\n"
        "    import pipecat.pipeline.worker\n"
        "    import pipecat.pipeline.runner\n"
        "    import pipecat.turns.user_stop\n"
        "    import pipecat.services.tts_service\n"
        "bad =[str(w.message) for w in caught\n"
        "       if issubclass(w.category, DeprecationWarning)\n"
        "       and ' is deprecated since ' in str(w.message)]\n"
        "if bad:\n"
        "    sys.stderr.write('pipecat import-time deprecation warnings:\\n' + '\\n'.join(bad))\n"
        "    sys.exit(1)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --- Intercepted field reads (warn_deprecated) --------------------------------
#
# A field whose reads are intercepted by ``__getattribute__`` warns once per call
# site. The shim passes ``stacklevel=2`` to name the reading line, and the helper
# keys its record of warned sites on that same line, so the tests below pin both
# the count and the reported location: a call layer added between the shim and
# the helper would otherwise silently collapse every reader into one entry.


def _read_enable_metrics(frame):
    return frame.enable_metrics


def test_intercepted_read_warns_once_per_call_site():
    frame = StartFrame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(10):
            _read_enable_metrics(frame)
    assert len(caught) == 1


def test_intercepted_read_warns_for_every_call_site():
    frame = StartFrame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = frame.enable_metrics
        _ = frame.enable_metrics  # Same field, second call site.
    assert len(caught) == 2


def test_intercepted_read_reports_the_reading_line():
    frame = StartFrame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = frame.enable_metrics
        lineno = inspect.currentframe().f_lineno - 1
    assert caught[0].filename == __file__
    assert caught[0].lineno == lineno


class _LegacyShim:
    old = 1

    def __getattribute__(self, name):
        if name == "old":
            warn_deprecated_read(
                "`_LegacyShim.old` is deprecated since 1.0.0 and will be removed in 2.0.0. "
                "No replacement."
            )
        return object.__getattribute__(self, name)


def test_deprecated_warn_deprecated_read_still_names_the_reading_line():
    """``warn_deprecated_read()`` warns that it is deprecated, and still reports the read."""
    shim = _LegacyShim()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = shim.old
        lineno = inspect.currentframe().f_lineno - 1
    messages = {str(w.message).split(" is deprecated")[0]: w for w in caught}
    assert set(messages) == {"`warn_deprecated_read`", "`_LegacyShim.old`"}
    assert messages["`_LegacyShim.old`"].filename == __file__
    assert messages["`_LegacyShim.old`"].lineno == lineno


def test_intercepted_read_warns_through_an_ignore_filter():
    """The warning reaches a reader who has filtered ``DeprecationWarning`` out.

    Many deprecations are detected inside Pipecat, where Python's default
    filters would hide them, so ``warn_deprecated()`` raises them under its own
    ``always`` filter. Run in a subprocess to get an interpreter whose filters
    ignore the category.
    """
    script = "from pipecat.frames.frames import StartFrame\n_ = StartFrame().enable_metrics\n"
    result = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "`StartFrame.enable_metrics` is deprecated" in result.stderr


# --- Renamed dataclass fields (renamed_init_field) ----------------------------
#
# A class's constructor is wrapped the first time the class is built, so each
# test builds a hierarchy of its own that no other test has built before.


@renamed_init_field(
    "old",
    "new",
    "`_RenamedBase.old` is deprecated since 1.0.0 and will be removed in 2.0.0. Use `new` instead.",
)
@dataclass
class _RenamedBase:
    new: int


@dataclass
class _RenamedChild(_RenamedBase):
    extra: int = 0


class _PassesOldNameToSuper(_RenamedChild):
    def __init__(self):
        super().__init__(old=1)


def test_renamed_field_reaches_a_constructor_that_passes_it_to_super():
    with pytest.warns(DeprecationWarning, match="`_RenamedBase.old`"):
        frame = _PassesOldNameToSuper()
    assert frame.new == 1


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


def test_compute_removals_ignores_subject_recorded_by_member():
    """A subject now recorded per member (``X`` → ``X.member``) wasn't removed."""
    prev = _dep_registry(_dep_record("Proc"))
    current = _dep_registry(_dep_record("Proc.on_event", kind="event"))
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


# --- Relocations (moves/removals that were never deprecated) -------------------
#
# compute_removals only sees subjects deprecated in the previous release, so an
# alpha feature moved outright is invisible to it. relocations.json is the
# hand-maintained input for those; the generator merges it into removals.json.


def _relocation(subject, **overrides):
    rec = {
        "subject": subject,
        "module": "pipecat.old",
        "kind": "class",
        "deprecated_in": None,
        "removed_in": "2.0.0",
        "announced_removed_in": None,
        "relation": "move",
        "replacement": "pipecat.new.Thing",
        "message": f"`{subject}` moved to `pipecat.new` in 2.0.0.",
    }
    rec.update(overrides)
    return rec


def test_relocations_registry_schema_is_valid():
    """The committed relocations.json passes the same validation the generator applies."""
    assert drem.load_relocations()


def test_every_relocation_is_recorded_in_removals():
    """removals.json is generated from relocations.json; a stale copy would drop a record."""
    committed = {
        r["subject"]: r
        for r in json.loads(drem.REMOVALS_PATH.read_text(encoding="utf-8"))["removals"]
    }
    for rec in drem.load_relocations():
        assert committed.get(rec["subject"]) == rec, (
            f"{rec['subject']!r} missing or different in removals.json; "
            "run scripts/deprecations/generate_removals.py"
        )


def test_compute_removals_merges_relocations():
    removals = drem.compute_removals(None, _dep_registry(), "2.0.0", [], [_relocation("Moved")])
    assert [r["subject"] for r in removals] == ["Moved"]
    assert removals[0]["deprecated_in"] is None
    assert removals[0]["replacement"] == "pipecat.new.Thing"


def test_compute_removals_relocations_are_idempotent_and_do_not_override():
    reloc = [_relocation("Moved")]
    first = drem.compute_removals(None, _dep_registry(), "2.0.0", [], reloc)
    assert drem.compute_removals(None, _dep_registry(), "2.0.0", first, reloc) == first
    # A subject already recorded (e.g. detected as a deprecated-then-removed symbol)
    # keeps its record; the relocation does not overwrite it.
    existing = [_relocation("Moved", message="detected", deprecated_in="1.0.0")]
    merged = drem.compute_removals(None, _dep_registry(), "2.0.0", existing, reloc)
    assert [r["message"] for r in merged] == ["detected"]


def test_compute_removals_edited_relocation_replaces_recorded_draft():
    first = drem.compute_removals(None, _dep_registry(), "2.0.0", [], [_relocation("Moved")])
    fixed = _relocation("Moved", replacement="pipecat.better.Thing", message="fixed by reviewer")
    again = drem.compute_removals(None, _dep_registry(), "2.0.0", first, [fixed])
    assert again == [{k: fixed[k] for k in _REMOVAL_FIELDS}]


def test_compute_removals_deleted_or_renamed_relocation_is_dropped():
    first = drem.compute_removals(
        None, _dep_registry(), "2.0.0", [], [_relocation("Moved"), _relocation("Gone")]
    )
    kept = drem.compute_removals(None, _dep_registry(), "2.0.0", first, [_relocation("Moved")])
    assert [r["subject"] for r in kept] == ["Moved"]
    renamed = drem.compute_removals(None, _dep_registry(), "2.0.0", first, [_relocation("Renamed")])
    assert [r["subject"] for r in renamed] == ["Renamed"]


def test_compute_removals_relocation_edit_keeps_detected_records():
    detected = _relocation("Detected", deprecated_in="1.0.0", announced_removed_in="2.0.0")
    first = drem.compute_removals(
        None, _dep_registry(), "2.0.0", [detected], [_relocation("Moved")]
    )
    again = drem.compute_removals(None, _dep_registry(), "2.0.0", first, [])
    assert [r["subject"] for r in again] == ["Detected"]


def test_compute_removals_relocations_drop_unknown_fields():
    rec = _relocation("Moved")
    rec["location"] = "pipecat/old.py:1"
    removals = drem.compute_removals(None, _dep_registry(), "2.0.0", [], [rec])
    assert set(removals[0]) == _REMOVAL_FIELDS


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("message"),  # missing field
        lambda r: r.update(removed_in="2.0"),  # not X.Y.Z
        lambda r: r.update(replacement=""),  # nothing to point at
        lambda r: r.update(extra="x"),  # unexpected field
    ],
)
def test_load_relocations_rejects_malformed_record(tmp_path, mutate):
    rec = _relocation("Moved")
    mutate(rec)
    path = tmp_path / "relocations.json"
    path.write_text(json.dumps({"relocations": [rec]}), encoding="utf-8")
    with pytest.raises(ValueError):
        drem.load_relocations(path)


def test_load_relocations_rejects_duplicate_subject(tmp_path):
    path = tmp_path / "relocations.json"
    path.write_text(
        json.dumps({"relocations": [_relocation("Moved"), _relocation("Moved")]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        drem.load_relocations(path)


def test_module_name_skips_private_modules():
    assert drem._module_name("src/pipecat/processors/tone/reader/model.py") == (
        "pipecat.processors.tone.reader.model"
    )
    assert drem._module_name("src/pipecat/processors/tone/__init__.py") == (
        "pipecat.processors.tone"
    )
    assert drem._module_name("src/pipecat/processors/tone/_models.py") is None
    assert drem._module_name("src/pipecat/_private/thing.py") is None


def test_vanished_module_without_record_is_flagged():
    prev = {"pipecat.a", "pipecat.old", "pipecat.old.sub"}
    assert drem.unrecorded_vanished_modules(prev, {"pipecat.a"}, []) == [
        "pipecat.old",
        "pipecat.old.sub",
    ]


def test_vanished_module_is_accounted_for_by_a_package_record_or_module_field():
    prev = {"pipecat.a", "pipecat.old", "pipecat.old.sub", "pipecat.gone"}
    current = {"pipecat.a"}
    records = [
        _relocation("pipecat.old", module="pipecat.old", kind="module"),  # package + children
        _relocation("Gone", module="pipecat.gone"),  # named by `module`
    ]
    assert drem.unrecorded_vanished_modules(prev, current, records) == []


def test_modules_still_present_are_not_flagged():
    modules = {"pipecat.a", "pipecat.b"}
    assert drem.unrecorded_vanished_modules(modules, set(modules), []) == []


# --- Drafting relocations automatically ---------------------------------------


def test_public_classes_and_exported_names():
    src = "from .a import Foo as Bar, Baz\n__all__ = ['Qux']\nclass Public: ...\nclass _Private: ...\n"
    assert drem._public_classes(src) == ["Public"]
    assert drem._exported_names(src) == {"Bar", "Baz", "Qux"}
    assert drem._public_classes("def broken(") == []
    assert drem._exported_names("def broken(") == set()


def test_public_home_walks_up_to_the_shortest_reexporting_package():
    reexports = {"pkg.a": {"Thing"}, "pkg.a.b": {"Thing"}, "pkg": set()}
    assert drem._public_home("pkg.a.b.impl", "Thing", reexports) == "pkg.a"
    assert drem._public_home("pkg.a.b.impl", "Other", reexports) == "pkg.a.b.impl"


def test_draft_relocations_tracks_classes_by_name():
    drafts = drem.draft_relocations(
        vanished=["pkg.old", "pkg.old.impl"],
        old_classes={"pkg.old": [], "pkg.old.impl": ["Moved", "Deleted", "Ambiguous"]},
        homes={"Moved": ["pkg.new.impl"], "Ambiguous": ["pkg.x", "pkg.y"]},
        reexports={"pkg.new": {"Moved"}},
        version="2.0.0",
        old_reexports={"pkg.old": {"Moved"}},
    )
    by = {d["subject"]: d for d in drafts}
    assert set(by) == {"pkg.old", "Moved", "Deleted"}  # ambiguous is left for a human
    assert by["Moved"]["module"] == "pkg.old"  # the public path users imported from
    assert by["Moved"]["replacement"] == "pkg.new.Moved"  # re-exported home, not pkg.new.impl
    assert (by["Deleted"]["relation"], by["Deleted"]["replacement"]) == ("none", "")
    assert by["pkg.old"]["kind"] == "module" and by["pkg.old"]["replacement"] == "pkg.new"
    assert all(d["removed_in"] == "2.0.0" and d["deprecated_in"] is None for d in drafts)


def test_draft_relocations_one_module_record_per_package_root():
    drafts = drem.draft_relocations(["pkg.old", "pkg.old.a", "pkg.old.b"], {}, {}, {}, "2.0.0")
    assert [d["subject"] for d in drafts] == ["pkg.old"]
    assert drafts[0]["relation"] == "none"


def test_drafted_records_pass_validation_and_round_trip(tmp_path):
    drafts = drem.draft_relocations(
        ["pkg.old"], {"pkg.old": ["Moved", "Gone"]}, {"Moved": ["pkg.new"]}, {}, "2.0.0"
    )
    path = tmp_path / "relocations.json"
    path.write_text(json.dumps({"_comment": "keep me", "relocations": []}), encoding="utf-8")
    drem.write_relocations(drafts, path)
    assert json.loads(path.read_text(encoding="utf-8"))["_comment"] == "keep me"
    assert (
        drem.load_relocations(path) == drafts
    )  # a deletion (relation none) may have no replacement
    assert drem.compute_removals(None, _dep_registry(), "2.0.0", [], drafts)
