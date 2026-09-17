#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pluggable storage for files uploaded by clients and referenced by URL later.

:class:`FileStorage` is the shared interface between the development runner's
``POST /files`` upload endpoint, which saves uploads and hands the resulting
URL back to the client, and the LLM service's
:class:`~pipecat.utils.file_resolver.FileResolver`, which loads the bytes back
when the client references that URL in a send-file message. Passing the same
storage instance to both means they agree on where uploads live without each
needing its own copy of the configuration, and lets a deployment swap in a
backend other than local disk by implementing this interface.

The URL is the contract: :meth:`FileStorage.save` returns one, callers pass it
back unmodified, and each implementation mints whatever form it can later
resolve — ``pipecat:<id>`` for :class:`LocalFileStorage`, or e.g. ``gs://`` /
``s3://`` for a custom cloud backend (which the LLM provider may then even
fetch itself, without :meth:`FileStorage.load` being involved at all).
"""

import asyncio
import re
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import aiofiles
from loguru import logger


class FileStorage(ABC):
    """Storage backend for files uploaded by clients and referenced by URL later."""

    # Whether a file should be deleted once its bytes have been consumed
    # (loaded and inlined into the LLM context by a FileResolver, which then
    # has no further use for the stored copy). Safe only when every URL this
    # backend resolves is an upload it owns; leave False for backends whose
    # URLs may point at shared data (e.g. arbitrary objects in a cloud
    # bucket), where "read into a conversation" must not mean "destroy".
    delete_after_load: bool = False

    @abstractmethod
    async def save(self, filename: str, contents: bytes) -> str:
        """Store a file and return the URL later passed to :meth:`load` or :meth:`delete`.

        Args:
            filename: The original filename, for backends that want to keep it
                (e.g. for content-type sniffing). Storage implementations should
                not use it to derive the returned URL.
            contents: The raw file contents.

        Returns:
            A URL identifying the stored file, in whatever form this backend
            can later resolve. Callers must pass it back unmodified to
            :meth:`load` or :meth:`delete`.
        """
        raise NotImplementedError

    @abstractmethod
    async def load(self, file_url: str) -> bytes:
        """Return the stored contents for `file_url`.

        Args:
            file_url: A URL previously returned by :meth:`save`.

        Raises:
            FileNotFoundError: If `file_url` is invalid or no longer stored.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, file_url: str) -> None:
        """Remove the stored file for `file_url`, if present.

        Args:
            file_url: A URL previously returned by :meth:`save`.
        """
        raise NotImplementedError


class LocalFileStorage(FileStorage):
    """Default storage backend: files on local disk, addressed as ``pipecat:<hex>``.

    Suitable for local development and single-process deployments. The
    ``pipecat:`` scheme makes the URLs recognizable in logs and client
    payloads; the random hex suffix can't be used for path traversal and
    doesn't leak the original filename.
    """

    # Every pipecat: URL is an upload this backend owns, so the file is
    # removed once its bytes have been consumed into the LLM context.
    delete_after_load = True

    _URL_PREFIX = "pipecat:"
    _URL_SUFFIX_PATTERN = re.compile(r"[0-9a-f]{32}")

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
        """Write `contents` to a new randomly-named file and return its ``pipecat:`` URL."""
        self._folder.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex
        async with aiofiles.open(self._folder / suffix, "wb") as f:
            await f.write(contents)
        await asyncio.to_thread(self._trim)
        return f"{self._URL_PREFIX}{suffix}"

    async def load(self, file_url: str) -> bytes:
        """Read and return the contents stored under `file_url`."""
        suffix = self._resolve_suffix(file_url)
        async with aiofiles.open(self._folder / suffix, "rb") as f:
            return await f.read()

    async def delete(self, file_url: str) -> None:
        """Delete the file stored under `file_url`, if it exists."""
        try:
            suffix = self._resolve_suffix(file_url)
        except FileNotFoundError:
            return
        try:
            # missing_ok: a consumed upload (delete_after_load) may be deleted
            # again by session-end cleanup.
            await asyncio.to_thread((self._folder / suffix).unlink, missing_ok=True)
        except OSError as e:
            logger.warning(f"Failed to remove uploaded file {file_url}: {e}")

    def _resolve_suffix(self, file_url: str) -> str:
        """Validate `file_url` and return the on-disk filename it maps to."""
        suffix = file_url.removeprefix(self._URL_PREFIX)
        if not self._URL_SUFFIX_PATTERN.fullmatch(suffix):
            raise FileNotFoundError(file_url)
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
