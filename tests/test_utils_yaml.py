#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import re
import tempfile
import unittest
from pathlib import Path

import yaml

from pipecat.utils.yaml import add_include_constructor, include_loader


def _tmpdir() -> Path:
    return Path(tempfile.mkdtemp())


class TestIncludeLoader(unittest.TestCase):
    def test_include_resolves_relative_to_base_dir(self):
        d = _tmpdir()
        (d / "fragment.yaml").write_text("a: 1\nb: two\n", encoding="utf-8")

        data = yaml.load("top: !include fragment.yaml\n", include_loader(d))
        self.assertEqual(data, {"top": {"a": 1, "b": "two"}})

    def test_nested_include_uses_same_loader(self):
        d = _tmpdir()
        (d / "sub").mkdir()
        (d / "leaf.yaml").write_text("leaf: true\n", encoding="utf-8")
        (d / "sub" / "middle.yaml").write_text("inner: !include leaf.yaml\n", encoding="utf-8")

        # Nested includes resolve against the same base_dir as the top-level
        # document, not against the including file.
        data = yaml.load("outer: !include sub/middle.yaml\n", include_loader(d))
        self.assertEqual(data, {"outer": {"inner": {"leaf": True}}})

    def test_include_scalar_document(self):
        d = _tmpdir()
        (d / "prompt.yaml").write_text("Say hello, then ask for the order.\n", encoding="utf-8")

        data = yaml.load("content: !include prompt.yaml\n", include_loader(d))
        self.assertEqual(data, {"content": "Say hello, then ask for the order."})

    def test_missing_file_raises(self):
        d = _tmpdir()
        with self.assertRaises(FileNotFoundError):
            yaml.load("x: !include nope.yaml\n", include_loader(d))

    def test_non_scalar_node_raises(self):
        d = _tmpdir()
        with self.assertRaises(yaml.constructor.ConstructorError) as cm:
            yaml.load("x: !include [a, b]\n", include_loader(d))
        self.assertIn("expects a file path", str(cm.exception))

    def test_base_loader_is_not_mutated(self):
        d = _tmpdir()
        include_loader(d)
        with self.assertRaises(yaml.constructor.ConstructorError):
            yaml.load("x: !include fragment.yaml\n", yaml.SafeLoader)

    def test_custom_base_loader_keeps_its_resolvers(self):
        # A subclass that keeps leading-zero tokens as strings, like the evals
        # scenario loader does for DTMF digits.
        class _DecimalOnly(yaml.SafeLoader):
            pass

        _DecimalOnly.yaml_implicit_resolvers = {
            ch: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:int"]
            for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
        }
        yaml.add_implicit_resolver(
            "tag:yaml.org,2002:int",
            re.compile(r"^[-+]?(?:0|[1-9][0-9_]*)$"),
            list("-+0123456789"),
            Loader=_DecimalOnly,
        )

        d = _tmpdir()
        (d / "digits.yaml").write_text("dtmf: 012\n", encoding="utf-8")

        data = yaml.load("t: !include digits.yaml\n", include_loader(d, base=_DecimalOnly))
        self.assertEqual(data, {"t": {"dtmf": "012"}})

    def test_add_include_constructor_on_explicit_subclass(self):
        d = _tmpdir()
        (d / "fragment.yaml").write_text("k: v\n", encoding="utf-8")

        class _Loader(yaml.SafeLoader):
            pass

        add_include_constructor(_Loader, d)
        self.assertEqual(yaml.load("x: !include fragment.yaml\n", _Loader), {"x": {"k": "v"}})


if __name__ == "__main__":
    unittest.main()
