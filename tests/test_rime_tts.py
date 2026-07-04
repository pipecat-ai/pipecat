#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.services.rime.tts import RimeTTSService


def _service() -> RimeTTSService:
    service = RimeTTSService.__new__(RimeTTSService)
    service._audio_remainder = b""
    service._audio_remainder_context_id = None
    return service


def test_even_chunks_pass_through_unchanged():
    service = _service()
    assert service._sample_aligned_audio("ctx", b"\x01\x02\x03\x04") == b"\x01\x02\x03\x04"
    assert service._sample_aligned_audio("ctx", b"\x05\x06") == b"\x05\x06"
    assert service._audio_remainder == b""


def test_odd_chunk_holds_back_dangling_byte():
    service = _service()
    assert service._sample_aligned_audio("ctx", b"\x01\x02\x03") == b"\x01\x02"
    assert service._audio_remainder == b"\x03"
    # The held-back byte completes the first sample of the next chunk.
    assert service._sample_aligned_audio("ctx", b"\x04\x05\x06") == b"\x03\x04\x05\x06"
    assert service._audio_remainder == b""


def test_byte_stream_preserved_across_odd_boundaries():
    # Chunk sizes observed from Rime's ws3 endpoint: consecutive odd-length
    # chunks that restore alignment overall.
    sizes = [1024, 4070, 20, 4063, 4068, 509, 1024, 1856]
    stream = bytes(i % 251 for i in range(sum(sizes)))
    chunks, pos = [], 0
    for size in sizes:
        chunks.append(stream[pos : pos + size])
        pos += size

    service = _service()
    out = b"".join(service._sample_aligned_audio("ctx", chunk) for chunk in chunks)
    assert out == stream
    assert all(len(service._sample_aligned_audio("ctx2", chunk)) % 2 == 0 for chunk in chunks)


def test_single_byte_chunk_returns_empty():
    service = _service()
    assert service._sample_aligned_audio("ctx", b"\x01") == b""
    assert service._audio_remainder == b"\x01"


def test_context_switch_drops_stale_remainder():
    service = _service()
    service._sample_aligned_audio("old", b"\x01\x02\x03")
    assert service._audio_remainder == b"\x03"
    # A new context must not inherit the old context's dangling byte.
    assert service._sample_aligned_audio("new", b"\x0a\x0b") == b"\x0a\x0b"
    assert service._audio_remainder == b""
