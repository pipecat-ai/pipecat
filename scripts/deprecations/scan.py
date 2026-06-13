#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared parser for pipecat deprecation markers.

Single source of truth for reading ``.. deprecated::`` directives and PEP 702
``@deprecated`` decorators out of the pipecat source tree. Both the audit
(``tests/test_deprecation_markers.py``) and the registry generator
(``scripts/deprecations/generate.py``) import this module, so the enforced
grammar and the generated registry can never drift.

The module is pure tooling: it depends only on the standard library plus
``DEPRECATION_MESSAGE_RE`` from ``pipecat.utils.deprecation``, and reads source
via ``ast`` — it is never imported at runtime by the framework itself.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from pipecat.utils.deprecation import DEPRECATION_MESSAGE_RE

# Files excluded from the scan: the convention's home module, whose docstring
# carries illustrative ``.. deprecated::`` examples that are not real
# deprecations.
DEFAULT_EXCLUDE = frozenset({"pipecat/utils/deprecation.py"})

# A backtick-wrapped token (single or double backticks are matched the same).
BACKTICK_TOKEN_RE = re.compile(r"`([^`]+)`")
# A directive body names a replacement via a Sphinx cross-reference role or a
# backticked name.
DIRECTIVE_TARGET_RE = re.compile(
    r":(?:class|meth|func|attr|mod|obj|data|exc):`~?[^`]+`|``?[^`]+``?"
)
# Phrases that genuinely signal "no replacement". Deliberately excludes
# "removed" — it appears in the removal-timeline sentence of nearly every
# directive ("will be removed in 2.0.0"), so matching it would let a directive
# whose replacement is bare, unparseable prose pass as if it had none.
NO_REPLACEMENT_RE = re.compile(
    r"no replacement|no longer|always|discontinued|unmaintained", re.IGNORECASE
)
# The removal version stated in a directive body ("... will be removed in 2.0.0.").
REMOVAL_RE = re.compile(r"removed in (\d+\.\d+\.\d+)", re.IGNORECASE)
_DIRECTIVE_VERSION_RE = re.compile(r"v?\d+\.\d+\.\d+")

# ``warnings.warn(DeprecationWarning)`` call sites with no associable directive
# in their module / function / enclosing class. These are structural cases the
# static check can't follow, not undocumented deprecations:
#   - shared helpers that emit warnings on behalf of their callers
#   - the actual directive lives on a parameter/field in a *different* symbol
#   - behavior/value-change deprecations with no single owning symbol
#   - one non-deprecation that (mis)uses DeprecationWarning
# Keyed by "<path relative to src>::<enclosing qualified name>". Should shrink.
WARN_WITHOUT_DIRECTIVE = frozenset(
    {
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
)


@dataclass
class Symbol:
    """A class/function/method/property definition in the source tree."""

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
class Directive:
    """A single ``.. deprecated::`` block found in a docstring."""

    relpath: str
    owner: str  # qualname of the symbol the docstring belongs to, or "<module>"
    version: str | None
    body: str

    @property
    def location(self) -> str:
        return f"{self.relpath} ({self.owner})"


@dataclass
class WarnSite:
    """A ``warnings.warn(..., DeprecationWarning)`` call site."""

    relpath: str
    enclosing: str  # qualname of the symbol the warn lives in ("" if module level)

    @property
    def key(self) -> str:
        return (
            f"{self.relpath}::{self.enclosing}" if self.enclosing else f"{self.relpath}::<module>"
        )


@dataclass
class Scan:
    """Everything extracted from one pass over the source tree."""

    symbols: list[Symbol] = field(default_factory=list)
    definitions: set[str] = field(default_factory=set)
    directives: list[Directive] = field(default_factory=list)
    warn_sites: list[WarnSite] = field(default_factory=list)
    # qualnames (and "<module>") that have at least one .. deprecated:: directive
    documented: set[str] = field(default_factory=set)


def iter_directives(docstring: str):
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


def directive_parses(version: str | None, body: str) -> bool:
    """Whether a directive states a version and a replacement (or no-replacement)."""
    if not version or not _DIRECTIVE_VERSION_RE.fullmatch(version):
        return False
    return bool(DIRECTIVE_TARGET_RE.search(body) or NO_REPLACEMENT_RE.search(body))


def directive_removal(body: str) -> str | None:
    """The removal version stated in a directive body, or None if absent."""
    match = REMOVAL_RE.search(body)
    return match.group(1) if match else None


def deprecated_message(node) -> tuple[bool, str | None]:
    """(has @deprecated, literal message or None) for a class/function node."""
    for decorator in getattr(node, "decorator_list", []):
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "deprecated":
            continue
        if (
            decorator.args
            and isinstance(decorator.args[0], ast.Constant)
            and isinstance(decorator.args[0].value, str)
        ):
            return True, decorator.args[0].value
        return True, None
    return False, None


def is_deprecation_warn(call) -> bool:
    """Whether a Call is ``warnings.warn(..., DeprecationWarning)``."""
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


def scan_source(src_root: Path, *, exclude: frozenset[str] = DEFAULT_EXCLUDE) -> Scan:
    """Walk ``src_root`` and collect deprecation markers.

    Args:
        src_root: The ``.../src/pipecat`` directory.
        exclude: Paths (relative to ``src_root``'s parent) to skip.

    Returns:
        A :class:`Scan` with every symbol, directive, and warn site.
    """
    scan = Scan()
    for py_file in sorted(src_root.rglob("*.py")):
        relpath = str(py_file.relative_to(src_root.parent))
        if relpath in exclude:
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

        for version, body in iter_directives(ast.get_docstring(tree) or ""):
            scan.directives.append(Directive(relpath, "<module>", version, body))
            scan.documented.add(f"{relpath}::<module>")

        def visit(node, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            scan.definitions.add(target.id)
                elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    scan.definitions.add(child.target.id)
                if isinstance(child, ast.Call) and is_deprecation_warn(child):
                    scan.warn_sites.append(WarnSite(relpath, prefix))
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = f"{prefix}.{child.name}" if prefix else child.name
                    scan.definitions.add(child.name)
                    has_dec, message = deprecated_message(child)
                    docstring = ast.get_docstring(child) or ""
                    scan.symbols.append(
                        Symbol(
                            relpath=relpath,
                            qualname=qualname,
                            lineno=child.lineno,
                            name=child.name,
                            docstring=docstring,
                            has_decorator=has_dec,
                            decorator_message=message,
                        )
                    )
                    for version, body in iter_directives(docstring):
                        scan.directives.append(Directive(relpath, qualname, version, body))
                        scan.documented.add(f"{relpath}::{qualname}")
                    visit(child, qualname)
                else:
                    visit(child, prefix)

        visit(tree, "")
    return scan


def parsed_decorated(scan: Scan):
    """Yield ``(symbol, parsed_message_fields)`` for each well-formed @deprecated symbol."""
    for sym in scan.symbols:
        if not sym.has_decorator or sym.decorator_message is None:
            continue
        match = DEPRECATION_MESSAGE_RE.match(sym.decorator_message)
        if match:
            yield sym, match.groupdict()


# --- Validators: each returns a list of human-readable violation strings ----


def check_directives_parse(scan: Scan) -> list[str]:
    """Every directive states a version and a replacement target (or no-replacement)."""
    return [
        f"{d.location}: v={d.version!r} {d.body[:90]!r}"
        for d in scan.directives
        if not directive_parses(d.version, d.body)
    ]


def check_directive_removal_versions(scan: Scan) -> list[str]:
    """Every directive states a concrete removal version ("removed in X.Y.Z")."""
    return [
        f"{d.location}: {d.body[:90]!r}"
        for d in scan.directives
        if directive_removal(d.body) is None
    ]


def check_decorator_messages(scan: Scan) -> list[str]:
    """Every @deprecated argument is a string literal matching the canonical template."""
    violations = []
    for sym in scan.symbols:
        if not sym.has_decorator:
            continue
        if sym.decorator_message is None:
            violations.append(f"{sym.location}: @deprecated argument is not a string literal")
        elif not DEPRECATION_MESSAGE_RE.match(sym.decorator_message):
            violations.append(f"{sym.location}: message off-template: {sym.decorator_message!r}")
    return violations


def check_decorator_subjects(scan: Scan) -> list[str]:
    """The message `Subject` names the decorated symbol."""
    violations = []
    for sym, fields in parsed_decorated(scan):
        subject = fields["subject"]
        if not (subject == sym.name or subject.endswith(f".{sym.name}")):
            violations.append(f"{sym.location}: subject `{subject}` != decorated `{sym.name}`")
    return violations


def check_decorator_versions(scan: Scan) -> list[str]:
    """The message `since` version agrees with the docstring directive."""
    violations = []
    for sym, fields in parsed_decorated(scan):
        if f".. deprecated:: {fields['version']}" not in sym.docstring:
            violations.append(
                f"{sym.location}: message version {fields['version']} not in directive"
            )
    return violations


def check_decorator_replacements_exist(scan: Scan) -> list[str]:
    """Class/function replacement targets in messages name real defined symbols."""
    violations = []
    for sym, fields in parsed_decorated(scan):
        replacement = fields["replacement"]
        if replacement is None:
            continue
        tokens = BACKTICK_TOKEN_RE.findall(replacement)
        if not tokens:
            violations.append(
                f"{sym.location}: replacement has no backticked target: {replacement!r}"
            )
            continue
        for token in tokens:
            name = token.removesuffix("()").split(".")[-1]
            is_function = token.endswith("()")
            is_class = name[:1].isupper() and name.isidentifier()
            if (is_function or is_class) and name not in scan.definitions:
                violations.append(f"{sym.location}: replacement `{token}` not defined in src")
    return violations


def check_warn_coverage(
    scan: Scan, allowlist: frozenset[str] = WARN_WITHOUT_DIRECTIVE
) -> list[str]:
    """Every warnings.warn(DeprecationWarning) is documented by a directive."""
    violations = []
    for site in scan.warn_sites:
        if site.key in allowlist:
            continue
        if not site.enclosing:
            covered = f"{site.relpath}::<module>" in scan.documented
        else:
            parts = site.enclosing.split(".")
            covered = (
                any(
                    f"{site.relpath}::{'.'.join(parts[:k])}" in scan.documented
                    for k in range(len(parts), 0, -1)
                )
                or f"{site.relpath}::<module>" in scan.documented
            )
        if not covered:
            violations.append(site.key)
    return violations


def check_warn_allowlist_current(
    scan: Scan, allowlist: frozenset[str] = WARN_WITHOUT_DIRECTIVE
) -> list[str]:
    """Every allowlist entry still names a real warn site."""
    live = {s.key for s in scan.warn_sites}
    return sorted(key for key in allowlist if key not in live)
