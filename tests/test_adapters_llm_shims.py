#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The deprecated ``pipecat.adapters`` paths still export what moved to ``pipecat.adapters.llm``."""

import importlib
import pkgutil
import sys
import unittest
import warnings

import pipecat.adapters.llm

NEW_PREFIX = "pipecat.adapters.llm."
OLD_PREFIX = "pipecat.adapters."


def _moved_modules() -> list[str]:
    return sorted(
        info.name
        for info in pkgutil.walk_packages(pipecat.adapters.llm.__path__, NEW_PREFIX)
        if not info.ispkg
    )


class TestAdaptersLLMShims(unittest.TestCase):
    def test_every_moved_module_has_a_shim(self):
        self.assertTrue(_moved_modules())
        for new_name in _moved_modules():
            old_name = OLD_PREFIX + new_name.removeprefix(NEW_PREFIX)
            with self.subTest(module=old_name):
                new = importlib.import_module(new_name)
                sys.modules.pop(old_name, None)
                with warnings.catch_warnings(record=True) as caught:
                    old = importlib.import_module(old_name)

                messages = [
                    str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
                ]
                self.assertIn(
                    f"`{old_name}` is deprecated since 1.14.0 and will be removed in 2.0.0. "
                    f"Use `{new_name}` instead.",
                    messages,
                )

                public = [name for name in vars(new) if not name.startswith("_")]
                for name in public:
                    self.assertIs(getattr(old, name), getattr(new, name), name)


if __name__ == "__main__":
    unittest.main()
