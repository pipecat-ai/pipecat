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
grammar, the validation, and the generated registry can never drift.

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
# The first reference in a directive body (role or backtick) — the replacement,
# per the "first reference is the replacement" rule. Capture groups: role
# target, double-backtick token, single-backtick token.
FIRST_REFERENCE_RE = re.compile(
    r":(?:class|meth|func|attr|mod|obj|data|exc):`~?([^`]+)`|``([^`]+)``|`([^`]+)`"
)
# Phrases that genuinely signal "no replacement". Deliberately excludes
# "removed" — it appears in the removal-timeline sentence of nearly every
# directive ("will be removed in 2.0.0"), so matching it would let a directive
# whose replacement is bare, unparseable prose pass as if it had none.
NO_REPLACEMENT_RE = re.compile(
    r"no replacement|no longer|always|discontinued|unmaintained", re.IGNORECASE
)
# An explicit, leading "No replacement" — the convention's authoritative signal
# that nothing replaces the deprecated thing (from CONTRIBUTING.md: "lead with
# `No replacement.`"). No trailing period required, so "No replacement — ..." or
# "No replacement; ..." also match.
NO_REPLACEMENT_LEAD_RE = re.compile(r"^\s*no replacement\b", re.IGNORECASE)
# The removal version stated in a directive body ("... will be removed in 2.0.0.").
REMOVAL_RE = re.compile(r"removed in (\d+\.\d+\.\d+)", re.IGNORECASE)
_DIRECTIVE_VERSION_RE = re.compile(r"v?\d+\.\d+\.\d+")
# Section headers that end the parameter list while scanning a docstring.
_SECTION_RE = re.compile(
    r"^(Parameters|Args|Arguments|Attributes|Returns|Raises|Yields|Example)s?:"
)
# A parameter/field entry within an Args/Parameters section ("voice: The voice ...").
_PARAM_ENTRY_RE = re.compile(r"^([A-Za-z_]\w*):\s")


@dataclass
class Symbol:
    """A class/function/method/property definition in the source tree."""

    relpath: str
    qualname: str
    lineno: int
    name: str
    kind: str  # class | function | method | property
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
    param: str | None  # parameter/field name when the directive is nested under one
    version: str | None
    body: str

    @property
    def location(self) -> str:
        where = f"{self.owner}.{self.param}" if self.param else self.owner
        return f"{self.relpath} ({where})"


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


def iter_directives(docstring: str):
    """Yield ``(version, body, param)`` for each ``.. deprecated::`` block.

    ``param`` is the name of the ``Args:``/``Parameters:`` entry the directive
    is nested under (a parameter/field deprecation), or ``None`` when the
    directive sits at the docstring's top level (the symbol itself is deprecated).
    """
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
        param = _enclosing_param(lines, i, base_indent)
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
        yield version, " ".join(body).strip(), param
        i = j


def _enclosing_param(lines: list[str], directive_index: int, directive_indent: int) -> str | None:
    """The parameter entry a directive is nested under, scanning upward, or None."""
    for k in range(directive_index - 1, -1, -1):
        stripped = lines[k].strip()
        if not stripped:
            continue
        indent = len(lines[k]) - len(lines[k].lstrip())
        if indent >= directive_indent:
            continue
        # A less-indented line: either the param entry that owns the directive,
        # or a section header / prose that means the directive is top-level.
        if _SECTION_RE.match(stripped):
            return None
        entry = _PARAM_ENTRY_RE.match(stripped)
        return entry.group(1) if entry else None
    return None


def directive_parses(version: str | None, body: str) -> bool:
    """Whether a directive states a version and a replacement (or no-replacement)."""
    if not version or not _DIRECTIVE_VERSION_RE.fullmatch(version):
        return False
    return bool(DIRECTIVE_TARGET_RE.search(body) or NO_REPLACEMENT_RE.search(body))


def directive_removal(body: str) -> str | None:
    """The removal version stated in a directive body, or None if absent."""
    match = REMOVAL_RE.search(body)
    return match.group(1) if match else None


def first_reference(body: str) -> str | None:
    """The first role/backtick reference in a directive body — the replacement.

    Returns ``None`` when the body leads with an explicit "No replacement.", so
    contextual role/backtick references after it aren't mistaken for a replacement.
    """
    if NO_REPLACEMENT_LEAD_RE.match(body):
        return None
    match = FIRST_REFERENCE_RE.search(body)
    if not match:
        return None
    return next(group for group in match.groups() if group)


_FIRST_REF_IS_MODULE_RE = re.compile(r"^[^`]*:mod:`")


def relation_for(body: str, replacement: str | None) -> str:
    """Classify the migration relation from the directive's leading verb/target."""
    if NO_REPLACEMENT_LEAD_RE.match(body):
        # An explicit leading "No replacement." overrides any relation verb that
        # appears later in contextual prose (e.g. "... merged into X years ago").
        return "none"
    low = body.lower()
    if re.search(r"\brenamed to\b", low):
        return "rename"
    if re.search(r"\bmerged into\b", low):
        return "merged"
    if re.search(r"\bmoved to\b", low) or _FIRST_REF_IS_MODULE_RE.match(body):
        # A directive whose first reference is a :mod: target is a module move
        # (e.g. ``Use :mod:`pipecat.services.xai.llm` instead``).
        return "move"
    if replacement is None:
        return "none"
    return "use_existing"


def primary_replacement(text: str | None) -> str | None:
    """The first backticked target in a replacement clause, or None."""
    if not text:
        return None
    tokens = BACKTICK_TOKEN_RE.findall(text)
    return tokens[0] if tokens else None


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


def _symbol_kind(node, parent_is_class: bool) -> str:
    if isinstance(node, ast.ClassDef):
        return "class"
    decorators = {
        (d.id if isinstance(d, ast.Name) else getattr(d, "attr", None)) for d in node.decorator_list
    }
    if "property" in decorators:
        return "property"
    return "method" if parent_is_class else "function"


def module_path(relpath: str) -> str:
    """Dotted module path for a source file ("pipecat/services/grok/llm.py")."""
    parts = relpath[:-3].split("/") if relpath.endswith(".py") else relpath.split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def scan_source(src_root: Path, *, exclude: frozenset[str] = DEFAULT_EXCLUDE) -> Scan:
    """Walk ``src_root`` and collect deprecation markers."""
    scan = Scan()
    for py_file in sorted(src_root.rglob("*.py")):
        relpath = str(py_file.relative_to(src_root.parent))
        if relpath in exclude:
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

        for version, body, param in iter_directives(ast.get_docstring(tree) or ""):
            scan.directives.append(Directive(relpath, "<module>", param, version, body))

        def visit(node, prefix: str, parent_is_class: bool) -> None:
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
                            kind=_symbol_kind(child, parent_is_class),
                            docstring=docstring,
                            has_decorator=has_dec,
                            decorator_message=message,
                        )
                    )
                    for version, body, param in iter_directives(docstring):
                        scan.directives.append(Directive(relpath, qualname, param, version, body))
                    visit(child, qualname, isinstance(child, ast.ClassDef))
                else:
                    visit(child, prefix, parent_is_class)

        visit(tree, "", False)
    return scan


def parsed_decorated(scan: Scan):
    """Yield ``(symbol, parsed_message_fields)`` for each well-formed @deprecated symbol."""
    for sym in scan.symbols:
        if not sym.has_decorator or sym.decorator_message is None:
            continue
        match = DEPRECATION_MESSAGE_RE.match(sym.decorator_message)
        if match:
            yield sym, match.groupdict()


# --- Registry records --------------------------------------------------------


def build_records(scan: Scan) -> list[dict]:
    """Build deprecation registry records from the scan.

    One record per deprecated thing — a symbol (from its ``@deprecated``
    message, the structured source), or a parameter/module deprecation (from
    its ``.. deprecated::`` directive). Behavior deprecations that have only a
    bare ``warnings.warn`` and no directive are not captured (see
    :func:`undocumented_warn_sites`).
    """
    decorated = {(s.relpath, s.qualname): s for s in scan.symbols if s.has_decorator}
    symbol_directive = {(d.relpath, d.owner): d for d in scan.directives if d.param is None}
    symbol_kind = {(s.relpath, s.qualname): s.kind for s in scan.symbols}

    records: list[dict] = []

    # 1. @deprecated symbols — fields come from the structured decorator message;
    #    the relation verb (rename/move/merged) comes from the docstring directive.
    for (relpath, qualname), sym in decorated.items():
        match = DEPRECATION_MESSAGE_RE.match(sym.decorator_message or "")
        if not match:
            continue  # off-template; the audit fails the build before we get here
        fields = match.groupdict()
        directive = symbol_directive.get((relpath, qualname))
        body = directive.body if directive else ""
        replacement = primary_replacement(fields["replacement"])
        records.append(
            {
                "subject": fields["subject"],
                "module": module_path(relpath),
                "kind": sym.kind,
                "deprecated_in": fields["version"],
                "removed_in": fields["removal"],
                "relation": relation_for(body, replacement),
                "replacement": replacement,
                "message": sym.decorator_message,
                # File only — no line number, which would churn the registry on
                # any edit that shifts the symbol. The symbol is locatable by name.
                "location": relpath,
            }
        )

    # 2. Directive-only deprecations: parameters, and module-move shims.
    for directive in scan.directives:
        is_symbol_level_of_decorated = (
            directive.param is None and (directive.relpath, directive.owner) in decorated
        )
        if is_symbol_level_of_decorated:
            continue  # already captured from the decorator above
        if directive.owner == "<module>" and directive.param is None:
            subject = module_path(directive.relpath)
            kind = "module"
        elif directive.param is not None:
            owner = directive.owner.removesuffix(".__init__")
            subject = f"{owner}.{directive.param}" if owner else directive.param
            kind = "parameter"
        else:
            # A directive on a non-decorated symbol (a method/function deprecated
            # via directive + manual warn rather than @deprecated).
            subject = directive.owner
            kind = symbol_kind.get((directive.relpath, directive.owner), "symbol")
        replacement = first_reference(directive.body)
        records.append(
            {
                "subject": subject,
                "module": module_path(directive.relpath),
                "kind": kind,
                "deprecated_in": directive.version,
                "removed_in": directive_removal(directive.body),
                "relation": relation_for(directive.body, replacement),
                "replacement": replacement,
                "message": directive.body,
                # File only — consistent with the decorator branch above; the
                # directive's owner is already captured in ``subject``.
                "location": directive.relpath,
            }
        )

    records.sort(key=lambda r: (r["module"], r["subject"], r["kind"]))
    return records


def undocumented_warn_sites(scan: Scan) -> list[str]:
    """warnings.warn(DeprecationWarning) sites with no directive in module/function/class.

    These are not captured in the registry — shared warning helpers, behavior
    deprecations with no owning symbol, or cross-symbol param warns whose
    directive lives elsewhere. Surfaced (non-gating) so a maintainer can see
    what the registry does not cover.
    """
    documented = {f"{d.relpath}::{d.owner}" for d in scan.directives if d.owner != "<module>"} | {
        f"{d.relpath}::<module>" for d in scan.directives if d.owner == "<module>"
    }
    out = []
    for site in scan.warn_sites:
        if not site.enclosing:
            covered = f"{site.relpath}::<module>" in documented
        else:
            parts = site.enclosing.split(".")
            covered = (
                any(
                    f"{site.relpath}::{'.'.join(parts[:k])}" in documented
                    for k in range(len(parts), 0, -1)
                )
                or f"{site.relpath}::<module>" in documented
            )
        if not covered:
            out.append(site.key)
    return sorted(out)


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


def all_violations(scan: Scan) -> list[str]:
    """Every validation failure, for the generator to refuse on."""
    out = []
    for check in (
        check_directives_parse,
        check_directive_removal_versions,
        check_decorator_messages,
        check_decorator_subjects,
        check_decorator_versions,
        check_decorator_replacements_exist,
    ):
        out.extend(f"{check.__name__}: {v}" for v in check(scan))
    return out
