#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""YAML loading helpers shared across Pipecat's YAML-based configuration.

The main helper is :func:`include_loader`, which builds a loader class that
understands an ``!include <relative-path>`` tag so a document can pull in
other files::

    judge: !include judge_audio.yaml
    task_messages: !include prompts/greeting.yaml

A path resolves relative to the file that names it, so an included file's own
includes resolve against its directory. Paths are not confined to that
directory; ``..`` and absolute paths reach anywhere the process can read. Use
these loaders only on trusted documents.
"""

from pathlib import Path
from typing import Any

import yaml


def add_include_constructor(loader_class: type[yaml.SafeLoader], base_dir: Path) -> None:
    """Register an ``!include <relative-path>`` constructor on ``loader_class``.

    Paths resolve against ``base_dir`` and may reach outside it. An included
    file loads with a subclass of the loading class whose own includes resolve
    against that file's directory, so a fragment's includes work wherever the
    fragment is included from, and scalars and constructors get the same
    treatment as in the top-level document.

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
            return yaml.load(f, include_loader(include_path.parent, base=type(loader)))

    loader_class.add_constructor("!include", _include)


def include_loader(
    base_dir: Path, base: type[yaml.SafeLoader] = yaml.SafeLoader
) -> type[yaml.SafeLoader]:
    """Build a loader class that resolves ``!include`` relative to ``base_dir``.

    Returns a fresh subclass of ``base`` with the include constructor
    registered, so ``base`` itself is left untouched.

    Args:
        base_dir: Directory that the document's ``!include`` paths resolve
            against, typically the directory of the document being loaded. An
            included file's own includes resolve against its directory.
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
