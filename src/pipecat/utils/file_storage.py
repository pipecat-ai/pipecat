#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pluggable storage for files uploaded by clients and referenced by ID later.

:class:`FileStorage` is the shared interface between the development runner's
``POST /files`` upload endpoint and :class:`~pipecat.processors.frameworks.rtvi.processor.RTVIProcessor`,
which resolves file ID references sent by RTVI clients. Passing the same storage
instance to both means they agree on where uploads live without each needing its
own copy of the configuration, and lets a deployment swap in a backend other than
local disk (e.g. S3) by implementing this interface. IDs are opaque: a caller
must pass back exactly what :meth:`FileStorage.save` returned, and each
implementation is free to choose its own ID format.
"""

import asyncio
import re
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import aiofiles
from loguru import logger


class FileStorage(ABC):
    """Storage backend for files uploaded by clients and referenced by ID later."""

    @abstractmethod
    async def save(self, filename: str, contents: bytes) -> str:
        """Store a file and return the ID later passed to :meth:`load` or :meth:`delete`.

        Args:
            filename: The original filename, for backends that want to keep it
                (e.g. for content-type sniffing). Storage implementations should
                not use it to derive the returned ID.
            contents: The raw file contents.

        Returns:
            An opaque ID identifying the stored file. Callers must pass it back
            unmodified to :meth:`load` or :meth:`delete`.
        """
        raise NotImplementedError

    @abstractmethod
    async def load(self, file_id: str) -> bytes:
        """Return the stored contents for `file_id`.

        Args:
            file_id: An ID previously returned by :meth:`save`.

        Raises:
            FileNotFoundError: If `file_id` is invalid or no longer stored.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, file_id: str) -> None:
        """Remove the stored file for `file_id`, if present.

        Args:
            file_id: An ID previously returned by :meth:`save`.
        """
        raise NotImplementedError


class LocalFileStorage(FileStorage):
    """Default storage backend: files on local disk, keyed by a random hex ID.

    Suitable for local development and single-process deployments. IDs are
    prefixed (e.g. ``pipecat:<hex>``) so they're recognizable in logs and client
    payloads; the hex suffix can't be used for path traversal and doesn't leak
    the original filename.
    """

    _ID_PREFIX = "pipecat:"
    _ID_SUFFIX_PATTERN = re.compile(r"[0-9a-f]{32}")

    def __init__(self, folder: str, max_files: int = 10):
        """Initialize local file storage.

        Args:
            folder: Directory to store files in. Created on first save if missing.
            max_files: Maximum number of files to retain; oldest files (by mtime)
                are deleted once this is exceeded. Set to 0 to disable trimming.
        """
        self._folder = Path(folder)
        self._max_files = max_files

    async def save(self, filename: str, contents: bytes) -> str:
        """Write `contents` to a new randomly-named file and return its ID."""
        self._folder.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex
        async with aiofiles.open(self._folder / suffix, "wb") as f:
            await f.write(contents)
        self._trim()
        return f"{self._ID_PREFIX}{suffix}"

    async def load(self, file_id: str) -> bytes:
        """Read and return the contents stored under `file_id`."""
        suffix = self._resolve_suffix(file_id)
        async with aiofiles.open(self._folder / suffix, "rb") as f:
            return await f.read()

    async def delete(self, file_id: str) -> None:
        """Delete the file stored under `file_id`, if it exists."""
        try:
            suffix = self._resolve_suffix(file_id)
        except FileNotFoundError:
            return
        try:
            await asyncio.to_thread((self._folder / suffix).unlink)
        except OSError as e:
            logger.warning(f"Failed to remove uploaded file {file_id}: {e}")

    def _resolve_suffix(self, file_id: str) -> str:
        """Validate `file_id` and return the on-disk filename it maps to."""
        suffix = file_id.removeprefix(self._ID_PREFIX)
        if not self._ID_SUFFIX_PATTERN.fullmatch(suffix):
            raise FileNotFoundError(file_id)
        return suffix

    def _trim(self):
        """Keep only the most recent `max_files` files; delete oldest by mtime."""
        if self._max_files <= 0:
            return
        try:
            files = [p for p in self._folder.iterdir() if p.is_file()]
            if len(files) <= self._max_files:
                return
            by_mtime = sorted(files, key=lambda p: p.stat().st_mtime)
            for p in by_mtime[: len(files) - self._max_files]:
                try:
                    p.unlink()
                    logger.debug(f"Trimmed upload {p.name} from {self._folder}")
                except OSError as e:
                    logger.warning(f"Failed to trim upload {p}: {e}")
        except OSError as e:
            logger.warning(f"Failed to list uploads folder {self._folder}: {e}")
