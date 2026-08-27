#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Enumerate Pipecat's provider services as research units for ``/provider-watch``.

A *unit* is one provider × one service type (``openai/llm``, ``cartesia/tts``,
``google/realtime``) and bundles its variant classes (WebSocket + HTTP, SageMaker
deployments, ...). For each unit this reports the concrete classes, the default
model each declares in its ``default_settings = self.Settings(model=...)`` block,
the ``Settings`` fields, whether it is a thin wrapper over another service, and
pointers into the CLI registry, the release-eval bots, ``env.example`` and the
docs site.

The scan is pure ``ast`` over ``src/pipecat/services`` so it runs before
``uv sync`` and in offline tests; the registry/manifest/docs joins are
best-effort and degrade to empty fields. Run::

    uv run python scripts/provider-watch/inventory.py --md
    uv run python scripts/provider-watch/inventory.py --json --only openai,deepgram
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICES_DIR = REPO_ROOT / "src" / "pipecat" / "services"
MANIFEST = REPO_ROOT / "scripts" / "release-evals" / "manifest.yaml"
ENV_EXAMPLE = REPO_ROOT / "env.example"
README = REPO_ROOT / "README.md"

# Provider directories that only re-export another provider's classes.
SHIM_PROVIDERS = {"grok"}

# Subpackages that are deployment variants of their parent unit rather than a
# distinct product line.
FOLDED_SUBPACKAGES = {"sagemaker"}

# Subpackages whose classes are realtime (speech-to-speech) services.
REALTIME_SUBPACKAGES = {"realtime", "gemini_live", "nova_sonic"}

MODULE_TYPES = {
    "llm": "llm",
    "stt": "stt",
    "tts": "tts",
    "image": "image",
    "vision": "vision",
    "video": "video",
    "memory": "memory",
    "search": "tool",
    "agent_core": "other",
}

# Base classes whose subclasses inherit the whole implementation and only
# override endpoint and default model.
THIN_WRAPPER_BASES = {
    "OpenAILLMService",
    "OpenAIRealtimeLLMService",
    "BaseWhisperSTTService",
    "GoogleLLMService",
    "GeminiLiveLLMService",
}

# Concrete service classes; abstract bases are named ``Base...`` / ``...Base...``.
CONCRETE_CLASS = re.compile(r"(Service|Search|Processor)(MLX|REST)?$")
BASE_CLASS = re.compile(r"(^|[a-z])Base[A-Z]|^_")

# Name fragments marking a deployment/transport variant rather than the primary class.
VARIANT_MARKERS = ("SageMaker", "Http", "HTTP", "REST", "Realtime", "Segmented", "NonJson", "MLX")

# docs.pipecat.ai path segment per unit type.
DOCS_SEGMENTS = {
    "llm": "llm",
    "stt": "stt",
    "tts": "tts",
    "realtime": "s2s",
    "image": "image-generation",
    "vision": "vision",
    "video": "video",
    "memory": "memory",
}


@dataclass
class ServiceClass:
    """One concrete service class."""

    name: str
    module: str
    bases: list[str]
    default_model: str | None
    default_model_expr: str | None
    settings_class: str | None
    settings_fields: list[str]
    base_url: str | None
    is_thin_wrapper: bool


@dataclass
class Unit:
    """One provider × service type research unit."""

    id: str
    provider: str
    type: str
    variant: str | None
    classes: list[ServiceClass]
    default_model: str | None
    is_thin_wrapper: bool
    registry: list[dict] = field(default_factory=list)
    env_vars: list[str] = field(default_factory=list)
    example_bots: list[str] = field(default_factory=list)
    docs_url: str | None = None
    source_files: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- AST


def _name(node: ast.expr) -> str:
    return ast.unparse(node)


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` assignments."""
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    out[target.id] = node.value.value
    return out


def _settings_call_model(init: ast.FunctionDef, constants: dict[str, str]):
    """Find ``model=`` in the ``self.Settings(...)`` / ``XSettings(...)`` call of ``__init__``."""
    for node in ast.walk(init):
        if not isinstance(node, ast.Call):
            continue
        func = _name(node.func)
        if not (func == "self.Settings" or func.endswith("Settings")):
            continue
        for kw in node.keywords:
            if kw.arg != "model":
                continue
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
                return (value if isinstance(value, str) else None), None
            expr = _name(kw.value)
            if expr in constants:
                return constants[expr], expr
            return None, expr
    return None, None


def _init_default(init: ast.FunctionDef, arg_name: str) -> str | None:
    """String default of a keyword argument in ``__init__``."""
    args = init.args
    for names, defaults in (
        (args.kwonlyargs, args.kw_defaults),
        (args.args[-len(args.defaults) :] if args.defaults else [], args.defaults),
    ):
        for arg, default in zip(names, defaults):
            if arg.arg == arg_name and isinstance(default, ast.Constant):
                if isinstance(default.value, str):
                    return default.value
    return None


def _init_of(cls: ast.ClassDef) -> ast.FunctionDef | None:
    return next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None
    )


def _dataclass_fields(cls: ast.ClassDef) -> list[str]:
    return [
        node.target.id
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]


def _scan_module(path: Path) -> list[ServiceClass]:
    tree = ast.parse(path.read_text(), filename=str(path))
    constants = _module_constants(tree)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    module = ".".join(path.relative_to(REPO_ROOT / "src").with_suffix("").parts)

    # Defaults declared by private/base classes in the same module are inherited
    # by the concrete classes that leave ``model`` unset (e.g. the OpenAI
    # Responses services share one ``_Base...`` constructor).
    inherited: dict[str, tuple[str | None, str | None]] = {}
    for cls in classes.values():
        init = _init_of(cls)
        if init is not None:
            inherited[cls.name] = _settings_call_model(init, constants)

    found: list[ServiceClass] = []
    for cls in classes.values():
        if not CONCRETE_CLASS.search(cls.name) or BASE_CLASS.search(cls.name):
            continue
        bases = [_name(b) for b in cls.bases]
        init = _init_of(cls)
        settings_class = None
        for node in cls.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "Settings" for t in node.targets
            ):
                settings_class = _name(node.value)
        settings_fields: list[str] = []
        if settings_class and settings_class in classes:
            settings_fields = _dataclass_fields(classes[settings_class])

        default_model = default_expr = base_url = None
        if init is not None:
            default_model, default_expr = _settings_call_model(init, constants)
            base_url = _init_default(init, "base_url")
        if default_model is None and default_expr is None:
            for base in bases:
                if base in inherited and inherited[base] != (None, None):
                    default_model, default_expr = inherited[base]
                    break

        found.append(
            ServiceClass(
                name=cls.name,
                module=module,
                bases=bases,
                default_model=default_model,
                default_model_expr=default_expr,
                settings_class=settings_class,
                settings_fields=settings_fields,
                base_url=base_url,
                is_thin_wrapper=any(b.split(".")[-1] in THIN_WRAPPER_BASES for b in bases),
            )
        )
    return found


def _classify(rel_parts: tuple[str, ...], cls: ServiceClass) -> tuple[str, str | None] | None:
    """Return ``(type, variant)`` for a module path like ``("openai", "responses", "llm")``."""
    stem = rel_parts[-1]
    if stem not in MODULE_TYPES:
        return None
    subpackages = [p for p in rel_parts[1:-1] if p not in FOLDED_SUBPACKAGES]
    unit_type = MODULE_TYPES[stem]
    if unit_type == "llm" and (
        any(p in REALTIME_SUBPACKAGES for p in subpackages) or "Realtime" in cls.name
    ):
        unit_type = "realtime"
        subpackages = [p for p in subpackages if p not in REALTIME_SUBPACKAGES]
    variant = "-".join(subpackages) or None
    return unit_type, variant


def scan_services() -> list[Unit]:
    """Walk the services tree and group classes into units."""
    units: dict[str, Unit] = {}
    for path in sorted(SERVICES_DIR.rglob("*.py")):
        rel = path.relative_to(SERVICES_DIR)
        if len(rel.parts) < 2 or rel.parts[0] in SHIM_PROVIDERS or "__pycache__" in rel.parts:
            continue
        provider = rel.parts[0]
        rel_parts = rel.with_suffix("").parts
        for cls in _scan_module(path):
            classified = _classify(rel_parts, cls)
            if classified is None:
                continue
            unit_type, variant = classified
            suffix = f"{variant}-{unit_type}" if variant else unit_type
            unit_id = f"{provider}/{suffix}"
            unit = units.setdefault(
                unit_id,
                Unit(
                    id=unit_id,
                    provider=provider,
                    type=unit_type,
                    variant=variant,
                    classes=[],
                    default_model=None,
                    is_thin_wrapper=False,
                ),
            )
            unit.classes.append(cls)
            unit.source_files.append(str(path.relative_to(REPO_ROOT)))

    for unit in units.values():
        unit.source_files = sorted(set(unit.source_files))
        unit.classes.sort(key=lambda c: (any(m in c.name for m in VARIANT_MARKERS), c.name))
        models = [c.default_model for c in unit.classes if c.default_model]
        unit.default_model = models[0] if models else None
        unit.is_thin_wrapper = all(c.is_thin_wrapper for c in unit.classes)
    return [units[k] for k in sorted(units)]


# ------------------------------------------------------------------------- joins


def _registry_by_class() -> dict[str, dict]:
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from pipecat.cli.registry.service_metadata import ServiceRegistry
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for group in (
        ServiceRegistry.STT_SERVICES,
        ServiceRegistry.LLM_SERVICES,
        ServiceRegistry.TTS_SERVICES,
        ServiceRegistry.REALTIME_SERVICES,
        ServiceRegistry.VIDEO_SERVICES,
    ):
        for definition in group:
            entry = {
                "value": definition.value,
                "class_names": list(definition.class_name or []),
                "label": definition.label,
                "package": definition.package,
                "env_prefix": definition.env_prefix,
                "include_params": definition.include_params or [],
                "settings_params": definition.settings_params or [],
                "param_defaults": definition.param_defaults or {},
            }
            for class_name in definition.class_name or []:
                out.setdefault(class_name, entry)
    return out


def _manifest_bots() -> list[str]:
    if not MANIFEST.exists():
        return []
    return sorted(set(re.findall(r"^\s*-\s*bot:\s*(\S+)", MANIFEST.read_text(), re.MULTILINE)))


def _env_example_vars() -> list[str]:
    if not ENV_EXAMPLE.exists():
        return []
    return re.findall(r"^([A-Z][A-Z0-9_]*)=", ENV_EXAMPLE.read_text(), re.MULTILINE)


def _docs_links() -> dict[tuple[str, str], str]:
    """``(segment, slug) -> url`` for every services link in the README table."""
    if not README.exists():
        return {}
    pattern = r"https://docs\.pipecat\.ai/api-reference/server/services/([a-z0-9-]+)/([a-z0-9-]+)"
    return {
        (seg, slug): f"https://docs.pipecat.ai/api-reference/server/services/{seg}/{slug}"
        for seg, slug in re.findall(pattern, README.read_text())
    }


def enrich(units: list[Unit]) -> None:
    """Attach registry, env, example-bot and docs pointers to each unit in place."""
    registry = _registry_by_class()
    bots = _manifest_bots()
    env_vars = _env_example_vars()
    docs = _docs_links()

    for unit in units:
        seen: set[str] = set()
        for cls in unit.classes:
            entry = registry.get(cls.name)
            if entry and entry["value"] not in seen:
                seen.add(entry["value"])
                unit.registry.append(entry)

        prefixes = {unit.provider.upper().replace("-", "_")}
        prefixes.update(e["env_prefix"] for e in unit.registry if e.get("env_prefix"))
        unit.env_vars = sorted(
            v for v in env_vars if any(v == p or v.startswith(p + "_") for p in prefixes)
        )

        slug = unit.provider.replace("_", "-")
        unit.example_bots = [b for b in bots if slug in Path(b).name]

        segment = DOCS_SEGMENTS.get(unit.type)
        if segment:
            candidates = [slug, unit.provider]
            if unit.variant:
                candidates.insert(0, f"{slug}-{unit.variant}")
            for candidate in candidates:
                if (segment, candidate) in docs:
                    unit.docs_url = docs[(segment, candidate)]
                    break


# ------------------------------------------------------------------------ output


def select(units: list[Unit], only: list[str] | None, limit: int | None) -> list[Unit]:
    """Filter by provider or unit-id prefix, then cap the count (deterministic order)."""
    if only:
        wanted = [o.strip() for o in only if o.strip()]
        units = [
            u
            for u in units
            if any(
                u.id == w or u.id.startswith(w.rstrip("/") + "/") or u.provider == w for w in wanted
            )
        ]
    if limit is not None:
        units = units[:limit]
    return units


def to_markdown(units: list[Unit]) -> str:
    lines = [
        "| Unit | Classes | Default model | Thin wrapper | Registry | Example bots |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for u in units:
        lines.append(
            "| {id} | {classes} | {model} | {thin} | {registry} | {bots} |".format(
                id=u.id,
                classes=", ".join(c.name for c in u.classes),
                model=u.default_model
                or next((c.default_model_expr for c in u.classes if c.default_model_expr), None)
                or "—",
                thin="yes" if u.is_thin_wrapper else "",
                registry=", ".join(e["value"] for e in u.registry) or "—",
                bots=len(u.example_bots),
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="emit JSON (default)")
    parser.add_argument("--md", action="store_true", help="emit a Markdown table")
    parser.add_argument(
        "--only", help="comma-separated providers or unit ids (e.g. openai,deepgram/stt)"
    )
    parser.add_argument("--limit", type=int, help="keep only the first N units")
    args = parser.parse_args(argv)

    units = scan_services()
    enrich(units)
    units = select(units, args.only.split(",") if args.only else None, args.limit)

    if args.md:
        print(to_markdown(units))
    else:
        print(json.dumps([asdict(u) for u in units], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
