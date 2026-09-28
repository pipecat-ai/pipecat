#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""YAML loading helpers shared across Pipecat's YAML-based configuration.

The main helper is :func:`include_loader`, which builds a loader class that
understands an ``!include <relative-path>`` tag so a document can pull in
sibling files::

    judge: !include judge_audio.yaml
    task_messages: !include prompts/greeting.yaml
"""

from pathlib import Path
from typing import Any

import yaml


def add_include_constructor(loader_class: type[yaml.SafeLoader], base_dir: Path) -> None:
    """Register an ``!include <relative-path>`` constructor on ``loader_class``.

    Included files load with the same loader class, so nested includes work and
    scalars get the same resolver treatment as the top-level document. Paths
    resolve against ``base_dir``.

    This mutates ``loader_class``. Register on a private subclass rather than
    on ``yaml.SafeLoader`` itself so the constructor has no global side
    effects; :func:`include_loader` does that for you.

    Args:
        loader_class: The loader class to register the constructor on.
        base_dir: Directory that ``!include`` paths resolve against.
    """

    def _include(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
        if not isinstance(node, yaml.ScalarNode):
            raise yaml.constructor.ConstructorError(
                None, None, "!include expects a file path", node.start_mark
            )
        include_path = base_dir / str(loader.construct_scalar(node))
        with include_path.open(encoding="utf-8") as f:
            return yaml.load(f, loader_class)

    loader_class.add_constructor("!include", _include)


def include_loader(
    base_dir: Path, base: type[yaml.SafeLoader] = yaml.SafeLoader
) -> type[yaml.SafeLoader]:
    """Build a loader class that resolves ``!include`` relative to ``base_dir``.

    Returns a fresh subclass of ``base`` with the include constructor
    registered, so ``base`` itself is left untouched.

    Args:
        base_dir: Directory that ``!include`` paths resolve against, typically
            the directory of the document being loaded.
        base: Loader class to subclass. Defaults to ``yaml.SafeLoader``; pass a
            custom SafeLoader subclass to keep its resolvers and constructors.

    Returns:
        A loader class to pass as the ``Loader`` argument of ``yaml.load``.

    Example::

        with path.open() as f:
            data = yaml.load(f, include_loader(path.parent))
    """

    class _IncludeLoader(base):  # type: ignore[valid-type,misc]
        pass

    add_include_constructor(_IncludeLoader, base_dir)
    return _IncludeLoader
