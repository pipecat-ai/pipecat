#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audit of deprecation conventions across ``src/pipecat``.

The ``.. deprecated::`` docstring directive is the single source of truth for
deprecations — downstream tooling parses it into a registry — so this audit
enforces:

- **Every directive parses** (``test_directives_parse``): a version after
  ``::`` plus a body that names a replacement target (a Sphinx role or a
  backticked name) or states "No replacement.". The replacement is the *first*
  reference in the body (the one in the leading relation clause, e.g. after
  "Use"); any later references are contextual prose, so a registry parser
  anchors on the first target rather than collecting every reference.
- **Every runtime deprecation is documented** (``test_deprecation_warns_have_directive``):
  a ``warnings.warn(DeprecationWarning)`` has a directive in its module, its
  function, or an enclosing class. A small allowlist covers shared warning
  helpers and cross-symbol cases the static check can't associate.

Deprecated classes/functions additionally carry the PEP 702 ``@deprecated``
decorator (for runtime warning + IDE/type-checker detection); its message is
kept consistent with the directive (``test_deprecated_*``). Parameters, module
moves, and behavior changes — which the decorator can't mark — use a plain
``warnings.warn(..., DeprecationWarning)`` and rely on the directive alone.
"""

import ast
import asyncio
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from pipecat.frames.frames import (
    CancelTaskFrame,
    EndTaskFrame,
    InterruptionTaskFrame,
    StopTaskFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineTask, PipelineTaskParams
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.utils.deprecation import DEPRECATION_MESSAGE_RE

SRC_ROOT = Path(__file__).parent.parent / "src" / "pipecat"

_BACKTICK_TOKEN_RE = re.compile(r"`([^`]+)`")

# A directive body names a replacement via a Sphinx cross-reference role or a
# backticked name, or explicitly states there is none.
_DIRECTIVE_TARGET_RE = re.compile(
    r":(?:class|meth|func|attr|mod|obj|data|exc):`~?[^`]+`|``?[^`]+``?"
)
# Phrases that genuinely signal "no replacement". Deliberately excludes
# "removed" — it appears in the removal-timeline sentence of nearly every
# directive ("will be removed in 2.0.0"), so matching it would let a directive
# whose replacement is bare, unparseable prose pass as if it had none.
_NO_REPLACEMENT_RE = re.compile(
    r"no replacement|no longer|always|discontinued|unmaintained", re.IGNORECASE
)

# ``warnings.warn(DeprecationWarning)`` call sites with no associable directive
# in their module / function / enclosing class. These are structural cases the
# static check can't follow, not undocumented deprecations:
#   - shared helpers that emit warnings on behalf of their callers
#   - the actual directive lives on a parameter/field in a *different* symbol
#   - behavior/value-change deprecations with no single owning symbol
#   - one non-deprecation that (mis)uses DeprecationWarning
# Keyed by "<path relative to src>::<enclosing qualified name>". Should shrink.
_WARN_WITHOUT_DIRECTIVE = {
    # Not a deprecation — DeprecationWarning used for a "couldn't inject
    # built-in tools" runtime notice; left as-is pending a category review.
    "pipecat/adapters/base_llm_adapter.py::BaseLLMAdapter.from_standard_tools",
    # Shared helpers that warn on behalf of the deprecated param/arg, which
    # carries its own directive at the call site.
    "pipecat/services/ai_service.py::AIService._warn_init_param_moved_to_settings",
    "pipecat/services/speechmatics/stt.py::SpeechmaticsSTTService._check_deprecated_args._deprecation_warning",
    # Directive lives on the deprecated parameter/field, in a different symbol
    # than where the warning fires.
    "pipecat/services/mem0/memory.py::Mem0MemoryService.__init__",
    "pipecat/transports/base_output.py::BaseOutputTransport.__init__",
    "pipecat/utils/startup.py::run_setup_hook",
    # Behavior / value-change deprecations with no single owning symbol to
    # attach a directive to (dict-form settings frames, None-coercion, a
    # deprecated value passed in a config list).
    "pipecat/frames/frames.py::TTSSpeakFrame.__post_init__",
    "pipecat/observers/loggers/metrics_log_observer.py::MetricsLogObserver.__init__",
    "pipecat/services/google/stt.py::GoogleSTTService._update_settings",
    "pipecat/services/llm_service.py::LLMService.process_frame",
    "pipecat/services/stt_service.py::STTService.process_frame",
    "pipecat/services/tts_service.py::TTSService.process_frame",
}


def _iter_directives(docstring: str):
    """Yield ``(version, body)`` for each ``.. deprecated::`` block in a docstring."""
    if not docstring:
        return
    lines = docstring.splitlines()
    i = 0
    while i < len(lines):
        match = re.match(r"(\s*)\.\. deprecated::\s*(.*)$", lines[i])
        if not match:
            i += 1
            continue
        base_indent = len(match.group(1))
        version = match.group(2).strip() or None
        body: list[str] = []
        j = i + 1
        while j < len(lines):
            if lines[j].strip() == "":
                j += 1
                if (
                    j < len(lines)
                    and lines[j].strip()
                    and (len(lines[j]) - len(lines[j].lstrip())) > base_indent
                ):
                    continue
                break
            if (len(lines[j]) - len(lines[j].lstrip())) <= base_indent:
                break
            body.append(lines[j].strip())
            j += 1
        yield version, " ".join(body).strip()
        i = j


def _directive_parses(version: str | None, body: str) -> bool:
    """Whether a directive yields the fields the registry needs."""
    if not version or not re.fullmatch(r"v?\d+\.\d+\.\d+", version):
        return False
    return bool(_DIRECTIVE_TARGET_RE.search(body) or _NO_REPLACEMENT_RE.search(body))


@dataclass
class _Symbol:
    relpath: str
    qualname: str
    lineno: int
    name: str
    docstring: str
    has_decorator: bool
    decorator_message: str | None  # @deprecated literal arg, if any (None if absent/non-literal)

    @property
    def location(self) -> str:
        return f"{self.relpath}:{self.lineno} ({self.qualname})"


@dataclass
class _Directive:
    relpath: str
    owner: str  # qualname of the symbol, or "<module>"
    version: str | None
    body: str

    @property
    def location(self) -> str:
        return f"{self.relpath} ({self.owner})"


@dataclass
class _WarnSite:
    relpath: str
    enclosing: str  # qualname of the symbol the warn lives in ("" if module level)

    @property
    def key(self) -> str:
        return (
            f"{self.relpath}::{self.enclosing}" if self.enclosing else f"{self.relpath}::<module>"
        )


def _deprecated_message(node) -> tuple[bool, str | None]:
    for d in getattr(node, "decorator_list", []):
        if not isinstance(d, ast.Call):
            continue
        func = d.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "deprecated":
            continue
        if d.args and isinstance(d.args[0], ast.Constant) and isinstance(d.args[0].value, str):
            return True, d.args[0].value
        return True, None
    return False, None


def _is_deprecation_warn(call) -> bool:
    func = call.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    if name != "warn":
        return False
    category = call.args[1] if len(call.args) >= 2 else None
    for kw in call.keywords:
        if kw.arg == "category":
            category = kw.value
    return (
        getattr(category, "id", None) or getattr(category, "attr", None)
    ) == "DeprecationWarning"


@dataclass
class _Scan:
    symbols: list[_Symbol] = field(default_factory=list)
    definitions: set[str] = field(default_factory=set)
    directives: list[_Directive] = field(default_factory=list)
    warn_sites: list[_WarnSite] = field(default_factory=list)
    # qualnames (and "<module>") that have at least one .. deprecated:: directive
    documented: set[str] = field(default_factory=set)


def _scan_source() -> _Scan:
    scan = _Scan()
    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        relpath = str(py_file.relative_to(SRC_ROOT.parent))
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

        for version, body in _iter_directives(ast.get_docstring(tree) or ""):
            scan.directives.append(_Directive(relpath, "<module>", version, body))
            scan.documented.add(f"{relpath}::<module>")

        def visit(node, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            scan.definitions.add(target.id)
                elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    scan.definitions.add(child.target.id)
                if isinstance(child, ast.Call) and _is_deprecation_warn(child):
                    scan.warn_sites.append(_WarnSite(relpath, prefix))
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = f"{prefix}.{child.name}" if prefix else child.name
                    scan.definitions.add(child.name)
                    has_dec, message = _deprecated_message(child)
                    docstring = ast.get_docstring(child) or ""
                    scan.symbols.append(
                        _Symbol(
                            relpath=relpath,
                            qualname=qualname,
                            lineno=child.lineno,
                            name=child.name,
                            docstring=docstring,
                            has_decorator=has_dec,
                            decorator_message=message,
                        )
                    )
                    for version, body in _iter_directives(docstring):
                        scan.directives.append(_Directive(relpath, qualname, version, body))
                        scan.documented.add(f"{relpath}::{qualname}")
                    visit(child, qualname)
                else:
                    visit(child, prefix)

        visit(tree, "")
    return scan


_SCAN = _scan_source()
_DECORATED = [s for s in _SCAN.symbols if s.has_decorator]


def _parsed_decorated():
    for sym in _DECORATED:
        if sym.decorator_message is None:
            continue
        match = DEPRECATION_MESSAGE_RE.match(sym.decorator_message)
        if match:
            yield sym, match.groupdict()


# --- Directive enforcement (the registry source of truth) --------------------


def test_directives_parse():
    """Every ``.. deprecated::`` directive yields a version and a target/no-replacement."""
    assert _SCAN.directives, "expected .. deprecated:: directives in src/pipecat"
    bad = [d for d in _SCAN.directives if not _directive_parses(d.version, d.body)]
    assert not bad, (
        "These `.. deprecated::` directives don't parse — give them a version after `::` "
        "and a body that names a replacement (`Use :class:`X` instead.`) or says "
        "`No replacement.`:\n"
        + "\n".join(f"  {d.location}: v={d.version!r} {d.body[:90]!r}" for d in bad)
    )


def test_deprecation_warns_have_directive():
    """Each warnings.warn(DeprecationWarning) is documented by a directive.

    The directive may live on the warning's module, its enclosing function, or
    any ancestor class. Structural exceptions are tracked in
    ``_WARN_WITHOUT_DIRECTIVE`` (which should only shrink).
    """
    undocumented = []
    for site in _SCAN.warn_sites:
        if site.key in _WARN_WITHOUT_DIRECTIVE:
            continue
        if not site.enclosing:
            covered = f"{site.relpath}::<module>" in _SCAN.documented
        else:
            parts = site.enclosing.split(".")
            covered = (
                any(
                    f"{site.relpath}::{'.'.join(parts[:k])}" in _SCAN.documented
                    for k in range(len(parts), 0, -1)
                )
                or f"{site.relpath}::<module>" in _SCAN.documented
            )
        if not covered:
            undocumented.append(site)
    assert not undocumented, (
        "These warnings.warn(DeprecationWarning) sites have no `.. deprecated::` directive "
        "in their module, function, or enclosing class — add one, or (for a structural case) "
        "add the key to _WARN_WITHOUT_DIRECTIVE:\n" + "\n".join(f"  {s.key}" for s in undocumented)
    )


def test_warn_without_directive_allowlist_is_current():
    """Every _WARN_WITHOUT_DIRECTIVE entry still names a real warn site."""
    live = {s.key for s in _SCAN.warn_sites}
    stale = [key for key in _WARN_WITHOUT_DIRECTIVE if key not in live]
    assert not stale, "_WARN_WITHOUT_DIRECTIVE has stale entries — delete them:\n" + "\n".join(
        f"  {key}" for key in sorted(stale)
    )


# --- @deprecated decorator message consistency -------------------------------


def test_deprecated_messages_follow_template():
    """Every @deprecated call site uses a literal message matching the template."""
    assert _DECORATED, "expected at least one @deprecated call site in src/pipecat"
    for sym in _DECORATED:
        assert sym.decorator_message is not None, (
            f"{sym.location}: @deprecated argument must be a string literal "
            "(type checkers cannot display a computed message)"
        )
        assert DEPRECATION_MESSAGE_RE.match(sym.decorator_message), (
            f"{sym.location}: message does not follow the canonical template from "
            f"pipecat.utils.deprecation: {sym.decorator_message!r}"
        )


def test_deprecated_subject_names_decorated_symbol():
    """The `Subject` in the message is the symbol the decorator is applied to."""
    for sym, fields in _parsed_decorated():
        subject = fields["subject"]
        assert subject == sym.name or subject.endswith(f".{sym.name}"), (
            f"{sym.location}: message subject `{subject}` does not name the "
            f"decorated symbol {sym.name} (copy-pasted message?)"
        )


def test_deprecated_version_matches_docstring_directive():
    """The `since` version in the message agrees with the docstring directive."""
    for sym, fields in _parsed_decorated():
        directive = f".. deprecated:: {fields['version']}"
        assert directive in sym.docstring, (
            f"{sym.location}: docstring must contain '{directive}' matching the "
            f"message version (found docstring: {sym.docstring[:120]!r})"
        )


def test_deprecated_replacement_targets_exist():
    """Backticked replacement targets in @deprecated messages name real symbols."""
    for sym, fields in _parsed_decorated():
        replacement = fields["replacement"]
        if replacement is None:  # "No replacement."
            continue
        tokens = _BACKTICK_TOKEN_RE.findall(replacement)
        assert tokens, (
            f"{sym.location}: replacement clause has no backticked target: {replacement!r}"
        )
        for token in tokens:
            name = token.removesuffix("()").split(".")[-1]
            is_function = token.endswith("()")
            is_class = name[:1].isupper() and name.isidentifier()
            if not (is_function or is_class):
                continue
            assert name in _SCAN.definitions, (
                f"{sym.location}: replacement target `{token}` does not name a "
                "class or function defined in src/pipecat (typo?)"
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
    """Importing the modules with deprecated shims must not warn.

    Guards the suppression of @deprecated subclassing warnings for the
    task frame aliases defined inside pipecat.frames.frames.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            "import pipecat.frames.frames, pipecat.pipeline.worker, pipecat.pipeline.runner",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
