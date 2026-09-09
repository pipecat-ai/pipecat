#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""``pipecat eval`` loads the nearest ``.env`` for the harness's own services."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipecat.cli.commands.eval import _eval_callback


class TestEvalLoadsDotenv(unittest.TestCase):
    """The root conftest stubs ``load_dotenv`` for the whole session, so the
    check is on the call: the file found walking up from the working directory,
    loaded without overriding the shell."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        self.base = Path(self._tmp.name).resolve()
        (self.base / ".env").write_text("OPENAI_API_KEY=from-file\n")
        (self.base / "bots").mkdir()
        os.chdir(self.base / "bots")

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def test_the_nearest_env_file_is_loaded_without_overriding_the_shell(self):
        calls: list[tuple[str, dict]] = []
        with mock.patch(
            "pipecat.cli.commands.eval.load_dotenv",
            side_effect=lambda path, **kw: calls.append((path, kw)) or True,
        ):
            _eval_callback()
        self.assertEqual(len(calls), 1)
        path, kwargs = calls[0]
        self.assertEqual(Path(path).resolve(), self.base / ".env")
        self.assertFalse(kwargs.get("override", False))
