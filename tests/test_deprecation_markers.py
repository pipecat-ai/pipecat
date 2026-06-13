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
