#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

from pipecat.utils.tracing.langfuse import LangfuseRecordingUploader

TRACE_ID = int("ab" * 16, 16)


class _FakeTurnContext:
    def __init__(self, trace_id: int, span_id: int):
        self.trace_id = trace_id
        self.span_id = span_id


class _FakeTraceObserver:
    """Stands in for TurnTraceObserver: turn contexts for known turns, nothing else."""

    def __init__(self, turns: dict[int, int]):
        self._turns = {n: _FakeTurnContext(TRACE_ID, span_id) for n, span_id in turns.items()}

    def get_turn_context(self, turn_number: int):
        return self._turns.get(turn_number)

    def get_current_turn_context(self):
        return None


class _FakeLangfuse:
    """A local Langfuse media API: records requests, serves presigned upload URLs."""

    def __init__(self):
        self.posts = []
        self.puts = []
        self.patches = []
        self.dedup = False
        self._counter = 0

        app = web.Application()
        app.router.add_post("/api/public/media", self._post_media)
        app.router.add_put("/storage/{media_id}", self._put_storage)
        app.router.add_patch("/api/public/media/{media_id}", self._patch_media)
        self.server = TestServer(app)

    async def __aenter__(self):
        await self.server.start_server()
        return self

    async def __aexit__(self, *exc):
        await self.server.close()

    @property
    def host(self) -> str:
        return str(self.server.make_url("")).rstrip("/")

    async def _post_media(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.posts.append({"auth": request.headers.get("Authorization"), **body})
        self._counter += 1
        media_id = f"media-{self._counter}"
        upload_url = None if self.dedup else str(self.server.make_url(f"/storage/{media_id}"))
        return web.json_response({"mediaId": media_id, "uploadUrl": upload_url})

    async def _put_storage(self, request: web.Request) -> web.Response:
        self.puts.append(
            {
                "media_id": request.match_info["media_id"],
                "auth": request.headers.get("Authorization"),
                "checksum": request.headers.get("x-amz-checksum-sha256"),
                "content_type": request.headers.get("Content-Type"),
                "size": len(await request.read()),
            }
        )
        return web.Response()

    async def _patch_media(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.patches.append({"media_id": request.match_info["media_id"], **body})
        return web.Response()


def _make_uploader(host: str) -> LangfuseRecordingUploader:
    uploader = LangfuseRecordingUploader(host=host, public_key="pk-test", secret_key="sk-test")
    # Preload collected audio, as if attach()/stop_and_collect() already ran.
    uploader._pcm = bytearray(b"\x01\x00" * 3200)  # 0.1s stereo at 16kHz
    uploader._sample_rate = 16000
    uploader._num_channels = 2
    uploader._turn_clips = {
        1: {"input": b"\x02\x00" * 800, "output": b"\x03\x00" * 800},
        2: {"input": b"\x04\x00" * 800},
    }
    uploader._turn_sample_rate = 16000
    return uploader


class TestLangfuseRecordingUploader(unittest.IsolatedAsyncioTestCase):
    async def test_uploads_whole_call_and_turn_clips(self):
        async with _FakeLangfuse() as langfuse:
            uploader = _make_uploader(langfuse.host)
            observer = _FakeTraceObserver({1: 0x1111, 2: 0x2222})
            worker = SimpleNamespace(turn_trace_observer=observer)

            await uploader.upload(worker)

        # Whole call + turn 1 input/output + turn 2 input.
        self.assertEqual(len(langfuse.posts), 4)
        self.assertEqual(len(langfuse.puts), 4)
        self.assertEqual(len(langfuse.patches), 4)

        trace_hex = format(TRACE_ID, "032x")
        for post in langfuse.posts:
            self.assertEqual(post["traceId"], trace_hex)
            self.assertEqual(post["contentType"], "audio/wav")
            self.assertTrue(post["auth"].startswith("Basic "))

        whole_call = [p for p in langfuse.posts if "observationId" not in p]
        self.assertEqual(len(whole_call), 1)
        self.assertEqual(whole_call[0]["field"], "output")

        turn_posts = [p for p in langfuse.posts if "observationId" in p]
        self.assertEqual(
            sorted((p["observationId"], p["field"]) for p in turn_posts),
            [
                (format(0x1111, "016x"), "input"),
                (format(0x1111, "016x"), "output"),
                (format(0x2222, "016x"), "input"),
            ],
        )

        # The presigned PUT carries the checksum but no Langfuse auth.
        for put in langfuse.puts:
            self.assertIsNone(put["auth"])
            self.assertTrue(put["checksum"])
            self.assertEqual(put["content_type"], "audio/wav")

        for patch_body in langfuse.patches:
            self.assertEqual(patch_body["uploadHttpStatus"], 200)
            self.assertIn("uploadedAt", patch_body)

    async def test_dedup_skips_put_and_patch(self):
        async with _FakeLangfuse() as langfuse:
            langfuse.dedup = True
            uploader = _make_uploader(langfuse.host)
            observer = _FakeTraceObserver({1: 0x1111, 2: 0x2222})
            worker = SimpleNamespace(turn_trace_observer=observer)

            await uploader.upload(worker)

        # A null uploadUrl means the content is already stored and linked.
        self.assertEqual(len(langfuse.posts), 4)
        self.assertEqual(len(langfuse.puts), 0)
        self.assertEqual(len(langfuse.patches), 0)

    async def test_turn_without_span_is_skipped(self):
        async with _FakeLangfuse() as langfuse:
            uploader = _make_uploader(langfuse.host)
            # Only turn 1 has a span; turn 2's clip has nowhere to link.
            observer = _FakeTraceObserver({1: 0x1111})
            worker = SimpleNamespace(turn_trace_observer=observer)

            await uploader.upload(worker)

        self.assertEqual(len(langfuse.posts), 3)  # whole call + turn 1 input/output

    async def test_no_trace_id_skips_upload(self):
        async with _FakeLangfuse() as langfuse:
            uploader = _make_uploader(langfuse.host)
            worker = SimpleNamespace(turn_trace_observer=_FakeTraceObserver({}))

            await uploader.upload(worker)

        self.assertEqual(len(langfuse.posts), 0)

    async def test_upload_never_raises(self):
        # Point at a closed port; upload() must swallow the connection error.
        uploader = _make_uploader("http://127.0.0.1:1")
        worker = SimpleNamespace(turn_trace_observer=_FakeTraceObserver({1: 0x1111}))
        await uploader.upload(worker)


class TestFromEnv(unittest.TestCase):
    def test_returns_none_without_keys(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(LangfuseRecordingUploader.from_env())

    def test_builds_uploader_with_keys(self):
        env = {
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
            "LANGFUSE_HOST": "https://cloud.langfuse.com/",
        }
        with patch.dict("os.environ", env, clear=True):
            uploader = LangfuseRecordingUploader.from_env()
        self.assertIsNotNone(uploader)
        self.assertEqual(uploader._host, "https://cloud.langfuse.com")


if __name__ == "__main__":
    unittest.main()
