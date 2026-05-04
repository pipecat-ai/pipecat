#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from pipecat.runner.run import _resolve_download_path


class TestRunnerDownloads(unittest.TestCase):
    def test_resolve_download_path_allows_files_inside_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloads = Path(tmpdir) / "downloads"
            nested = downloads / "nested"
            nested.mkdir(parents=True)
            file_path = nested / "recording.txt"
            file_path.write_text("session transcript")

            resolved = _resolve_download_path(str(downloads), "nested/recording.txt")

            self.assertEqual(resolved, file_path.resolve())

    def test_resolve_download_path_blocks_parent_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            downloads = root / "downloads"
            downloads.mkdir()
            (root / "secret.txt").write_text("secret")

            with self.assertRaises(HTTPException) as context:
                _resolve_download_path(str(downloads), "../secret.txt")

            self.assertEqual(context.exception.status_code, 403)

    def test_resolve_download_path_blocks_decoded_encoded_slashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            downloads = root / "media"
            downloads.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret.txt").write_text("secret")

            with self.assertRaises(HTTPException) as context:
                _resolve_download_path(str(downloads), "../outside/secret.txt")

            self.assertEqual(context.exception.status_code, 403)

    def test_resolve_download_path_blocks_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloads = Path(tmpdir) / "downloads"
            downloads.mkdir()

            with self.assertRaises(HTTPException) as context:
                _resolve_download_path(str(downloads), "/etc/passwd")

            self.assertEqual(context.exception.status_code, 403)

    @unittest.skipUnless(hasattr(os, "symlink"), "os.symlink is not available")
    def test_resolve_download_path_blocks_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            downloads = root / "downloads"
            outside = root / "outside"
            downloads.mkdir()
            outside.mkdir()
            (outside / "secret.txt").write_text("secret")

            try:
                os.symlink(outside, downloads / "linked")
            except OSError as e:
                self.skipTest(f"Unable to create symlink: {e}")

            with self.assertRaises(HTTPException) as context:
                _resolve_download_path(str(downloads), "linked/secret.txt")

            self.assertEqual(context.exception.status_code, 403)
