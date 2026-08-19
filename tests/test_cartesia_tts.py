#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import contextlib
import json
import unittest

from websockets.protocol import State

from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    AggregationType,
    TTSTextFrame,
)
from pipecat.services.cartesia.tts import _IN_FLIGHT_MAX_AGE_S, CartesiaTTSService
from pipecat.services.settings import TTSSettings
from pipecat.transcriptions.language import Language
from pipecat.utils.context.aggregated_frame_sequencer import AggregatedFrameSequencer
from pipecat.utils.context.word_completion_tracker import WordCompletionTracker
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text


def _service(language: str) -> CartesiaTTSService:
    service = CartesiaTTSService.__new__(CartesiaTTSService)
    service._settings = TTSSettings(language=language)
    return service


def _process_word_timestamps(
    words: list[str], starts: list[float], language: str
) -> list[tuple[str, float]]:
    return _service(language)._normalize_word_timestamps(words, starts)


def _concatenate_processed_timestamps(
    timestamp_groups: list[tuple[list[str], list[float]]], language: str
) -> str:
    service = _service(language)
    text_parts = []
    for words, starts in timestamp_groups:
        processed_timestamps = service._normalize_word_timestamps(words, starts)
        includes_inter_frame_spaces = service._word_timestamps_include_inter_frame_spaces()
        text_parts.extend(
            TextPartForConcatenation(
                word,
                includes_inter_part_spaces=includes_inter_frame_spaces,
            )
            for word, _timestamp in processed_timestamps
        )
    return concatenate_aggregated_text(text_parts)


def test_cartesia_chinese_word_timestamps_join_without_spaces():
    assert _process_word_timestamps(
        words=["你", "好", "。"],
        starts=[0.0, 0.1, 0.2],
        language="zh",
    ) == [("你好。", 0.0)]


def test_cartesia_japanese_word_timestamps_join_without_spaces():
    assert _process_word_timestamps(
        words=["こ", "ん", "に", "ち", "は", "。"],
        starts=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        language="ja",
    ) == [("こんにちは。", 0.0)]


def test_cartesia_korean_word_timestamps_preserve_words_and_timestamps():
    assert _process_word_timestamps(
        words=["안녕하세요", "반갑습니다"],
        starts=[0.0, 0.2],
        language="ko",
    ) == [("안녕하세요", 0.0), ("반갑습니다", 0.2)]


def test_cartesia_korean_word_timestamps_do_not_join_latin_and_hangul():
    assert _process_word_timestamps(
        words=["AI", "어시스턴트입니다."],
        starts=[3.7026982, 4.1999383],
        language="ko",
    ) == [("AI", 3.7026982), ("어시스턴트입니다.", 4.1999383)]


def test_cartesia_japanese_timestamp_groups_reassemble_without_spaces():
    assert (
        _concatenate_processed_timestamps(
            [
                (["こ", "ん", "に", "ち", "は", "、", "私"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
                (["は", "あ", "な", "た", "の"], [1.0, 1.1, 1.2, 1.3, 1.4]),
            ],
            language="ja",
        )
        == "こんにちは、私はあなたの"
    )


def test_cartesia_chinese_timestamp_groups_reassemble_without_spaces():
    assert (
        _concatenate_processed_timestamps(
            [
                (["你", "好", "，", "我", "是"], [0.1, 0.2, 0.3, 0.4, 0.5]),
                (["你", "的", "智", "能"], [1.0, 1.1, 1.2, 1.3]),
            ],
            language="zh",
        )
        == "你好，我是你的智能"
    )


def test_cartesia_korean_timestamp_groups_reassemble_with_spaces():
    assert (
        _concatenate_processed_timestamps(
            [
                (["저는"], [1.6]),
                (["여러분의"], [1.8]),
                (["AI", "어시스턴트입니다."], [3.7, 4.2]),
            ],
            language="ko",
        )
        == "저는 여러분의 AI 어시스턴트입니다."
    )


def test_cartesia_spell_tag_keeps_its_word_attached_to_following_punctuation():
    assert _process_word_timestamps(
        words=["<spell>1234</spell>."],
        starts=[0.0],
        language="en",
    ) == [("1234.", 0.0)]


def test_cartesia_tag_between_two_words_keeps_them_separated():
    assert _process_word_timestamps(
        words=["to<spell>1234</spell>"],
        starts=[0.0],
        language="en",
    ) == [("to 1234", 0.0)]


def test_cartesia_tag_only_token_is_dropped():
    assert (
        _process_word_timestamps(
            words=['<break time="80ms"/>'],
            starts=[0.0],
            language="en",
        )
        == []
    )


def test_cartesia_spell_token_matches_the_text_sent_for_synthesis():
    """Every normalized token has to be recognised by the word tracker.

    A token the tracker cannot place force-completes the slot, which emits all the
    text left unspoken — synthesis tags included — as one TTSTextFrame, ending the
    turn's word-level tracking.
    """
    text = "Hello, I love to <spell>1234</spell>."
    tracker = WordCompletionTracker(text, llm_text=text, user_facing_text=text)

    for word, _ in _process_word_timestamps(
        words=["Hello,", "I", "love", "to", "<spell>1234</spell>."],
        starts=[0.0, 0.1, 0.2, 0.3, 0.4],
        language="en",
    ):
        assert tracker.word_belongs_here(word), f"{word!r} was not recognised"
        tracker.add_word_and_check_complete(word)

    assert tracker.is_complete
    assert tracker.get_accumulated_user_facing_text() == text


class TestCartesiaUpdateSettingsFinalizesOldContext(unittest.IsolatedAsyncioTestCase):
    """A mid-reply voice/model/language change re-mints the turn context. The old
    context's still-pending sentence must be finalized first, or the already-heard
    prefix's word-timestamps land on no slot and drop out of the transcript.
    """

    async def _service_with_pending_prefix(self, old_ctx: str):
        service = CartesiaTTSService.__new__(CartesiaTTSService)
        service._name = "CartesiaTTSService#0"
        service._settings = CartesiaTTSService.Settings(
            model="sonic-3.5",
            voice="voiceA",
            language=Language.EN,
            generation_config=None,
            pronunciation_dict_id=None,
        )
        # Applying a settings delta reports the service usable again.
        service._is_usable = True
        # Real streaming sequencer with a mid-sentence prefix pending on the turn ctx.
        seq = AggregatedFrameSequencer(name=service._name, streaming=True)
        service._aggregated_frame_sequencer = seq
        service._turn_context_id = old_ctx
        for token in ("Hi", " there"):
            frame = AggregatedTextFrame(token, AggregationType.SENTENCE, raw_text=token)
            await seq.register_spoken(frame, old_ctx, token, append_to_context=True)
        assert seq._slots == []  # nothing promoted — sentence has no boundary yet

        # Stub I/O so _update_settings exercises the finalize/flush/re-mint logic
        # without a websocket. Capture frames the finalize pushes.
        pushed: list = []

        async def fake_push(frames, context_id):
            pushed.extend(frames)

        async def fake_flush(context_id=None):
            service._flushed = context_id

        service._flushed = None
        service._push_sequencer_frames = fake_push
        service.flush_audio = fake_flush
        service.audio_context_available = lambda context_id: True
        service.create_context_id = lambda: "ctx-new"
        return service, seq, pushed

    async def test_voice_change_finalizes_and_rescues_prefix(self):
        old_ctx = "ctx-old"
        service, seq, pushed = await self._service_with_pending_prefix(old_ctx)

        await service._update_settings(CartesiaTTSService.Settings(voice="voiceB"))

        # The old context's pending sentence was force-promoted into a real slot.
        self.assertEqual([s.frame.text for s in seq._slots], ["Hi there"])
        self.assertEqual(seq._slots[0].context_id, old_ctx)
        # The finalize pushed the promoted sentence anchor downstream.
        self.assertTrue(
            any(isinstance(f, AggregatedTextFrame) and f.text == "Hi there" for f in pushed)
        )
        # The context was flushed and the turn context re-minted afterwards.
        self.assertEqual(service._flushed, old_ctx)
        self.assertEqual(service._turn_context_id, "ctx-new")

        # A word-timestamp for the flushed prefix (arriving on the OLD context during
        # playout) still finds the promoted slot and emits a progress frame.
        result = seq.process_word("Hi", pts=10, context_id=old_ctx)
        self.assertTrue(any(isinstance(f, TTSTextFrame) and f.text == "Hi" for f in result))
        progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0].accumulated_text, "Hi")

    async def test_non_remint_change_does_not_finalize(self):
        # A change that does not re-mint the context (e.g. pronunciation_dict_id)
        # must NOT finalize — that would prematurely promote and mis-segment the
        # ongoing reply.
        old_ctx = "ctx-old"
        service, seq, pushed = await self._service_with_pending_prefix(old_ctx)

        await service._update_settings(CartesiaTTSService.Settings(pronunciation_dict_id="dict-1"))

        self.assertEqual(seq._slots, [])  # still pending, not promoted
        self.assertEqual(pushed, [])
        self.assertIsNone(service._flushed)
        self.assertEqual(service._turn_context_id, old_ctx)


class _FakeWebSocket:
    """Minimal stand-in for the Cartesia websocket that records sends."""

    def __init__(self):
        self.state = State.OPEN
        self.sent: list[dict] = []

    async def send(self, data: str):
        self.sent.append(json.loads(data))

    async def close(self):
        self.state = State.CLOSED

    async def ping(self):
        pass


def _make_ws_service() -> CartesiaTTSService:
    return CartesiaTTSService(
        api_key="test-key",
        settings=CartesiaTTSService.Settings(voice="test-voice"),
    )


async def _drain_run_tts(service: CartesiaTTSService, text: str, context_id: str):
    async for _ in service.run_tts(text, context_id):
        pass


class TestCartesiaReconnectReplaysLostUtterance(unittest.IsolatedAsyncioTestCase):
    """A websocket drop AFTER a successful send but BEFORE any audio comes back
    must not lose the utterance. The reconnect has to re-send the transcript of
    every context that received zero audio, so the bot still speaks the reply
    instead of staying silent until the next user turn.
    """

    def _service_with_socket(self) -> tuple[CartesiaTTSService, _FakeWebSocket]:
        service = _make_ws_service()
        ws = _FakeWebSocket()
        service._websocket = ws
        return service, ws

    async def _reconnect(self, service: CartesiaTTSService) -> _FakeWebSocket:
        """Run the real reconnect path with the network stubbed out."""
        new_ws = _FakeWebSocket()

        async def fake_connect(uri, **kwargs):
            return new_ws

        service._websocket_connect = fake_connect
        await service._reconnect_websocket(1)
        return new_ws

    async def test_reconnect_replays_transcript_that_received_no_audio(self):
        service, _ = self._service_with_socket()
        await _drain_run_tts(service, "The capital of Japan is Tokyo.", "ctx-1")
        await service.flush_audio("ctx-1")

        new_ws = await self._reconnect(service)

        transcripts = [(m["transcript"], m["continue"]) for m in new_ws.sent]
        self.assertEqual(
            transcripts,
            [("The capital of Japan is Tokyo.", True), ("", False)],
        )
        self.assertTrue(all(m["context_id"] == "ctx-1" for m in new_ws.sent))

    async def test_reconnect_does_not_replay_context_that_received_audio(self):
        service, _ = self._service_with_socket()
        await _drain_run_tts(service, "The capital of Japan is Tokyo.", "ctx-1")

        # An audio chunk arrives for the context: the user has started hearing
        # it, so a replay would repeat words.
        entry = service._in_flight_contexts["ctx-1"]
        entry.received_audio = True
        entry.transcripts.clear()

        new_ws = await self._reconnect(service)

        self.assertEqual(new_ws.sent, [])

    async def test_interrupted_context_is_not_replayed(self):
        service, _ = self._service_with_socket()
        await _drain_run_tts(service, "The capital of Japan is Tokyo.", "ctx-1")

        await service.on_audio_context_interrupted("ctx-1")

        new_ws = await self._reconnect(service)
        self.assertEqual(new_ws.sent, [])

    async def test_context_finished_by_server_done_is_not_replayed(self):
        service = _make_ws_service()
        ws = _StreamingFakeWebSocket()
        service._websocket = ws
        await service.create_audio_context("ctx-1")
        await _drain_run_tts(service, "The capital of Japan is Tokyo.", "ctx-1")

        # The server finishes the context with its "done" message: the
        # utterance was fully delivered, so there is nothing to replay.
        await ws.messages.put(json.dumps({"type": "done", "context_id": "ctx-1"}))
        receive_task = asyncio.create_task(service._process_messages())
        try:
            for _ in range(10):
                await asyncio.sleep(0)
        finally:
            receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receive_task

        self.assertNotIn("ctx-1", service._in_flight_contexts)
        new_ws = await self._reconnect(service)
        self.assertEqual(new_ws.sent, [])

    async def test_reconnect_keeps_the_starved_audio_context_alive(self):
        """The reconnect's own disconnect must not tear down the audio context
        that is still owed audio: removing it is what turned a transient socket
        drop into a silently lost utterance (its queue gets the end-of-context
        sentinel and force_complete pushes the never-spoken text downstream).
        """
        service, _ = self._service_with_socket()
        await service.create_audio_context("ctx-1")
        service._playing_context_id = "ctx-1"
        await _drain_run_tts(service, "The capital of Japan is Tokyo.", "ctx-1")

        # Reconnect-style disconnect (not an intentional teardown).
        await service._disconnect_websocket()

        self.assertTrue(service.audio_context_available("ctx-1"))
        self.assertEqual(service._playing_context_id, "ctx-1")
        # No end-of-context sentinel was queued.
        queued = []
        queue = service._audio_contexts["ctx-1"]
        while not queue.empty():
            queued.append(queue.get_nowait())
        self.assertNotIn(None, queued)

    async def test_intentional_disconnect_still_removes_the_audio_context(self):
        service, _ = self._service_with_socket()
        await service.create_audio_context("ctx-1")
        service._playing_context_id = "ctx-1"
        await _drain_run_tts(service, "The capital of Japan is Tokyo.", "ctx-1")

        # Intentional teardown (stop/cancel/cleanup) sets _disconnecting.
        service._disconnecting = True
        await service._disconnect_websocket()

        self.assertIsNone(service._playing_context_id)
        self.assertEqual(service._in_flight_contexts, {})
        # The context queue got the end-of-context sentinel.
        queue = service._audio_contexts["ctx-1"]
        queued = []
        while not queue.empty():
            queued.append(queue.get_nowait())
        self.assertIn(None, queued)


class _StreamingFakeWebSocket(_FakeWebSocket):
    """Fake websocket whose messages can be consumed by a real receive loop.

    ``deliver_during_send`` simulates the race the receive task creates: the
    server's first message arrives (and is fully processed) while ``send`` is
    still being awaited by run_tts.
    """

    def __init__(self):
        super().__init__()
        self.messages: asyncio.Queue = asyncio.Queue()
        self.deliver_during_send: str | None = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.messages.get()

    async def send(self, data: str):
        await super().send(data)
        if self.deliver_during_send is not None:
            await self.messages.put(self.deliver_during_send)
            self.deliver_during_send = None
            # Yield until the concurrent receive task has processed the
            # message, so it lands before this send returns.
            for _ in range(10):
                await asyncio.sleep(0)


class TestCartesiaInFlightTrackingRaces(unittest.IsolatedAsyncioTestCase):
    """Lifetime edges of the retained-transcript tracking.

    Two hazards from review: (1) audio arriving concurrently with the send must
    not be missed, or a reconnect replays speech the user already heard; and
    (2) a dead socket can go undetected past the audio-context idle timeout,
    so the retained transcript must survive the timeout-driven completion or
    there is nothing left to replay when the reconnect finally happens.
    """

    def _service_with_socket(self) -> tuple[CartesiaTTSService, _FakeWebSocket]:
        service = _make_ws_service()
        ws = _FakeWebSocket()
        service._websocket = ws
        return service, ws

    async def _reconnect(self, service: CartesiaTTSService) -> _FakeWebSocket:
        new_ws = _FakeWebSocket()

        async def fake_connect(uri, **kwargs):
            return new_ws

        service._websocket_connect = fake_connect
        await service._reconnect_websocket(1)
        return new_ws

    async def test_chunk_arriving_during_send_marks_context_as_heard(self):
        """The receive loop runs in its own task, so the first audio chunk can
        arrive while run_tts is still awaiting the send. The context must
        count as having produced audio; treating it as zero-audio would make
        a later reconnect replay speech the user already heard.
        """
        service = _make_ws_service()
        ws = _StreamingFakeWebSocket()
        service._websocket = ws
        await service.create_audio_context("ctx-1")

        receive_task = asyncio.create_task(service._process_messages())
        try:
            ws.deliver_during_send = json.dumps(
                {
                    "type": "chunk",
                    "context_id": "ctx-1",
                    "data": base64.b64encode(b"\x00\x00").decode(),
                }
            )
            await _drain_run_tts(service, "The capital of Japan is Tokyo.", "ctx-1")
        finally:
            receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receive_task

        entry = service._in_flight_contexts["ctx-1"]
        self.assertTrue(entry.received_audio)
        self.assertEqual(entry.transcripts, [])

        new_ws = await self._reconnect(service)
        self.assertEqual(new_ws.sent, [])

    async def test_failed_send_does_not_retain_the_transcript(self):
        """A transcript that never made it onto the wire must not be replayed."""

        class _FailingSendWebSocket(_FakeWebSocket):
            async def send(self, data: str):
                raise ConnectionError("boom")

        service = _make_ws_service()
        service._websocket = _FailingSendWebSocket()

        async def noop():
            pass

        # run_tts recovers from a failed send with a disconnect/connect cycle;
        # stub those so the test stays off the network.
        service._disconnect = noop
        service._connect = noop

        await _drain_run_tts(service, "The capital of Japan is Tokyo.", "ctx-1")

        self.assertNotIn("ctx-1", service._in_flight_contexts)

    async def test_slowly_detected_drop_survives_context_timeout_and_replays(self):
        """A dead socket can go undetected well past stop_frame_timeout_s. The
        base class then completes and deletes the starved audio context (its
        frames are exactly what stopped flowing) BEFORE any reconnect happens.
        The retained transcript must survive that completion, and the replay
        must recreate the audio context so the late audio still has somewhere
        to land.
        """
        service, _ = self._service_with_socket()
        await service.create_audio_context("ctx-1")
        await _drain_run_tts(service, "The capital of Japan is Tokyo.", "ctx-1")
        await service.flush_audio("ctx-1")

        # Idle-timeout completion, as _audio_context_task_handler does it: the
        # context is deleted, then the completion hook fires.
        del service._audio_contexts["ctx-1"]
        await service.on_audio_context_completed("ctx-1")

        new_ws = await self._reconnect(service)

        transcripts = [(m["transcript"], m["continue"]) for m in new_ws.sent]
        self.assertEqual(
            transcripts,
            [("The capital of Japan is Tokyo.", True), ("", False)],
        )
        # The replay recreated the audio context for the incoming audio.
        self.assertTrue(service.audio_context_available("ctx-1"))

    async def test_timeout_completion_still_drops_contexts_that_produced_audio(self):
        """Only zero-audio entries survive a timeout completion. A context that
        already produced audio is dropped there: it is never replayed, so
        keeping it would only accumulate state.
        """
        service, _ = self._service_with_socket()
        await _drain_run_tts(service, "The capital of Japan is Tokyo.", "ctx-1")
        entry = service._in_flight_contexts["ctx-1"]
        entry.received_audio = True
        entry.transcripts.clear()

        await service.on_audio_context_completed("ctx-1")

        self.assertNotIn("ctx-1", service._in_flight_contexts)

    async def test_entries_older_than_the_age_bound_are_not_replayed(self):
        """Retention is bounded: entries past the age bound are pruned instead
        of replayed, so they cannot accumulate for the life of the process.
        """
        service, _ = self._service_with_socket()
        await _drain_run_tts(service, "The capital of Japan is Tokyo.", "ctx-1")
        service._in_flight_contexts["ctx-1"].created_at -= _IN_FLIGHT_MAX_AGE_S + 1.0

        new_ws = await self._reconnect(service)

        self.assertEqual(new_ws.sent, [])
        self.assertEqual(service._in_flight_contexts, {})
