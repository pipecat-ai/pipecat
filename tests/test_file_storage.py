#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import tempfile
import unittest
from pathlib import Path

from pipecat.utils.file_storage import LocalFileStorage


class TestLocalFileStorage(unittest.IsolatedAsyncioTestCase):
    async def test_save_then_load_roundtrips_contents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalFileStorage(tmpdir)
            file_id = await storage.save("report.pdf", b"contents")

            self.assertEqual(await storage.load(file_id), b"contents")

    async def test_save_does_not_leak_filename_into_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalFileStorage(tmpdir)
            file_id = await storage.save("../../etc/passwd", b"x")

            self.assertRegex(file_id, r"^pipecat:[0-9a-f]{32}$")

    async def test_load_missing_id_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalFileStorage(tmpdir)

            with self.assertRaises(FileNotFoundError):
                await storage.load("a" * 32)

    async def test_load_rejects_path_traversal_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            uploads = root / "uploads"
            uploads.mkdir()
            (root / "secret.txt").write_text("secret")
            storage = LocalFileStorage(str(uploads))

            with self.assertRaises(FileNotFoundError):
                await storage.load("../secret.txt")

    async def test_load_rejects_absolute_path_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalFileStorage(tmpdir)

            with self.assertRaises(FileNotFoundError):
                await storage.load("/etc/passwd")

    async def test_delete_removes_saved_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalFileStorage(tmpdir)
            file_id = await storage.save("report.pdf", b"contents")

            await storage.delete(file_id)

            with self.assertRaises(FileNotFoundError):
                await storage.load(file_id)

    async def test_delete_missing_id_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalFileStorage(tmpdir)

            await storage.delete("a" * 32)

    async def test_save_trims_oldest_file_once_over_max(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalFileStorage(tmpdir, max_files=2)
            first = await storage.save("a.txt", b"1")
            await storage.save("b.txt", b"2")
            await storage.save("c.txt", b"3")

            with self.assertRaises(FileNotFoundError):
                await storage.load(first)
            self.assertEqual(len(list(Path(tmpdir).iterdir())), 2)

    async def test_max_files_zero_disables_trimming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalFileStorage(tmpdir, max_files=0)
            for i in range(5):
                await storage.save(f"{i}.txt", str(i).encode())

            self.assertEqual(len(list(Path(tmpdir).iterdir())), 5)
