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

    def test_nested_include_resolves_against_including_file(self):
        d = _tmpdir()
        (d / "sub").mkdir()
        (d / "leaf.yaml").write_text("leaf: top\n", encoding="utf-8")
        (d / "sub" / "leaf.yaml").write_text("leaf: sub\n", encoding="utf-8")
        (d / "sub" / "middle.yaml").write_text("inner: !include leaf.yaml\n", encoding="utf-8")

        data = yaml.load("outer: !include sub/middle.yaml\n", include_loader(d))
        self.assertEqual(data, {"outer": {"inner": {"leaf": "sub"}}})

    def test_nested_includes_chain_across_directories(self):
        d = _tmpdir()
        (d / "scenarios").mkdir()
        (d / "shared" / "judges").mkdir(parents=True)
        (d / "shared" / "rubric.yaml").write_text("criterion: polite\n", encoding="utf-8")
        (d / "shared" / "judges" / "text.yaml").write_text(
            "rubric: !include ../rubric.yaml\n", encoding="utf-8"
        )
        (d / "shared" / "judge.yaml").write_text(
            "text: !include judges/text.yaml\n", encoding="utf-8"
        )

        data = yaml.load("judge: !include ../shared/judge.yaml\n", include_loader(d / "scenarios"))
        self.assertEqual(data, {"judge": {"text": {"rubric": {"criterion": "polite"}}}})

    def test_fragment_resolves_the_same_from_any_includer(self):
        d = _tmpdir()
        (d / "a" / "deep").mkdir(parents=True)
        (d / "shared").mkdir()
        (d / "shared" / "leaf.yaml").write_text("leaf: shared\n", encoding="utf-8")
        (d / "shared" / "fragment.yaml").write_text("x: !include leaf.yaml\n", encoding="utf-8")

        top = yaml.load("f: !include shared/fragment.yaml\n", include_loader(d))
        deep = yaml.load(
            "f: !include ../../shared/fragment.yaml\n", include_loader(d / "a" / "deep")
        )
        self.assertEqual(top, {"f": {"x": {"leaf": "shared"}}})
        self.assertEqual(deep, top)

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
        d = _tmpdir()
        (d / "digits.yaml").write_text("dtmf: 012\n", encoding="utf-8")

        data = yaml.load("t: !include digits.yaml\n", include_loader(d, base=_decimal_only()))
        self.assertEqual(data, {"t": {"dtmf": "012"}})

    def test_nested_include_keeps_custom_base_resolvers(self):
        d = _tmpdir()
        (d / "sub").mkdir()
        (d / "sub" / "digits.yaml").write_text("dtmf: 012\n", encoding="utf-8")
        (d / "sub" / "middle.yaml").write_text("inner: !include digits.yaml\n", encoding="utf-8")

        base = _decimal_only()
        data = yaml.load("outer: !include sub/middle.yaml\n", include_loader(d, base=base))
        self.assertEqual(data, {"outer": {"inner": {"dtmf": "012"}}})
        self.assertNotIn("!include", base.yaml_constructors)

    def test_nested_include_keeps_constructors_of_a_loader_subclass(self):
        d = _tmpdir()
        (d / "sub").mkdir()
        (d / "sub" / "leaf.yaml").write_text("v: !upper deep\n", encoding="utf-8")
        (d / "sub" / "middle.yaml").write_text(
            "v: !upper mid\ninner: !include leaf.yaml\n", encoding="utf-8"
        )

        class _Upper(include_loader(d)):
            pass

        _Upper.add_constructor("!upper", lambda loader, node: node.value.upper())

        data = yaml.load("outer: !include sub/middle.yaml\n", _Upper)
        self.assertEqual(data, {"outer": {"v": "MID", "inner": {"v": "DEEP"}}})

    def test_add_include_constructor_on_explicit_subclass(self):
        d = _tmpdir()
        (d / "sub").mkdir()
        (d / "fragment.yaml").write_text("k: v\n", encoding="utf-8")
        (d / "sub" / "leaf.yaml").write_text("leaf: sub\n", encoding="utf-8")
        (d / "sub" / "middle.yaml").write_text("inner: !include leaf.yaml\n", encoding="utf-8")

        class _Loader(yaml.SafeLoader):
            pass

        add_include_constructor(_Loader, d)
        self.assertEqual(yaml.load("x: !include fragment.yaml\n", _Loader), {"x": {"k": "v"}})
        self.assertEqual(
            yaml.load("x: !include sub/middle.yaml\n", _Loader),
            {"x": {"inner": {"leaf": "sub"}}},
        )
        # A nested load leaves the caller's loader resolving against its own base_dir.
        self.assertEqual(yaml.load("x: !include fragment.yaml\n", _Loader), {"x": {"k": "v"}})


def _decimal_only() -> type[yaml.SafeLoader]:
    """A loader that keeps leading-zero tokens as strings, like the evals scenario loader."""

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
    return _DecimalOnly


if __name__ == "__main__":
    unittest.main()
