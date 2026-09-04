#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""YAML loading for scenario files.

Both kinds of scenario file, scripted and simulated, are read the same way: a
``SafeLoader`` that resolves only plain-decimal integers (so a DTMF ``012``
keeps its digits) and an ``!include <path>`` tag that splices in another YAML
file relative to the including one, so scenarios can share ``user:`` and
``judge:`` blocks.
"""

import re
from pathlib import Path
from typing import Any

import yaml


class _ScenarioLoader(yaml.SafeLoader):
    """SafeLoader that resolves only plain-decimal numeric scalars as ints.

    PyYAML's SafeLoader follows YAML 1.1, which reinterprets unquoted numeric
    scalars as octal (``010`` -> 8), hex (``0x10`` -> 16), or binary before
    application code sees them. For DTMF that silently rewrites the digit
    sequence the user typed (``dtmf: 012`` would load as ``10``). Dropping those
    resolvers and keeping only plain decimal means ``dtmf: 123`` still loads as
    an int (so the unquoted-digits convenience works), while leading-zero, hex,
    and binary tokens stay strings and reach DTMF validation with their digits
    intact. No scenario field wants an octal/hex literal, so this is safe
    document-wide.
    """


# Strip the inherited int resolvers (which match octal/hex/binary/sexagesimal)
# and register a decimal-only replacement. Underscores stay allowed to match
# YAML's grouping syntax (e.g. ``1_000``); a leading zero (``012``) no longer
# matches, so such tokens load as strings.
_ScenarioLoader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:int"]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
yaml.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^[-+]?(?:0|[1-9][0-9_]*)$"),
    list("-+0123456789"),
    Loader=_ScenarioLoader,
)


def _load_mapping(path: Path) -> dict:
    """Load a scenario file's top-level mapping, resolving ``!include`` tags.

    Includes resolve relative to the file's directory. The constructor is
    registered on a private loader subclass (not the global SafeLoader) so it
    has no global side effects.

    Raises:
        ValueError: If the top level is not a mapping.
    """

    class _Loader(_ScenarioLoader):
        pass

    _add_include_constructor(_Loader, path.parent)
    with path.open() as f:
        data = yaml.load(f, _Loader)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def _add_include_constructor(loader_class: type[yaml.SafeLoader], base_dir: Path) -> None:
    """Register an ``!include <relative-path>`` constructor on ``loader_class``.

    Included files load with the same loader class, so nested includes work and
    scalars get the same resolver treatment as the top-level document. Paths
    resolve against ``base_dir`` (the scenario file's directory).
    """

    def _include(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
        if not isinstance(node, yaml.ScalarNode):
            raise yaml.constructor.ConstructorError(
                None, None, "!include expects a file path", node.start_mark
            )
        include_path = base_dir / str(loader.construct_scalar(node))
        with include_path.open() as f:
            return yaml.load(f, loader_class)

    loader_class.add_constructor("!include", _include)
