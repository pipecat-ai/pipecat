#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the MoQ (Media over QUIC) transport.

Three areas covered:

1. **``_downmix_s16_to_mono``** — the workaround for ``@moq/publish``'s
   browser-side encoder publishing stereo even when the source mic
   reports mono. Tests cover the no-op paths (already-mono, malformed
   length) and the arithmetic (averaging, saturation guard).

2. **Cert-hash helpers in ``runner/run.py``** — ``_cert_hash_from_pem``
   (client-mode PEM read) and ``_hex_to_b64`` (serve-mode fingerprint
   conversion). We hit a real ``certHash=None`` bug here once; locking
   the round-trip in stops a regression.

3. **``MOQTransportClient.__init__`` characterization** — the publish
   broadcast and transcript track must be created synchronously,
   because :class:`MOQOutputTransport.start` opens the audio track
   immediately without waiting for ``_run()``'s async bring-up. If a
   future refactor moves either into ``_run()``, the bot will lose its
   first few hundred ms of audio (this was a real bug PR #4557's
   self-review fixed).
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

# The ``moq`` extra is optional; skip the whole module when ``moq-rs``
# isn't installed, matching the default CI unit test environment which
# doesn't pull optional extras.
pytest.importorskip("moq")

from pipecat.transports.moq.transport import (  # noqa: E402
    MOQParams,
    MOQTransport,
    _downmix_s16_to_mono,
)

# ----------------------------------------------------------------------
# _downmix_s16_to_mono
# ----------------------------------------------------------------------


def _pack(samples):
    """Pack a list of signed-16 ints into bytes (little-endian S16)."""
    import array

    return array.array("h", samples).tobytes()


def _unpack(buf):
    """Unpack S16 bytes back into a list of ints."""
    import array

    arr = array.array("h")
    arr.frombytes(buf)
    return list(arr)


class TestDownmixS16ToMono(unittest.TestCase):
    """Cover the channel-downmix helper used in ``_forward_peer_audio``.

    The browser side of ``@moq/publish`` 0.2.9 publishes stereo Opus
    even when the source ``MediaStreamTrack`` reports mono, because
    ``MediaStreamAudioSourceNode.channelCount`` defaults to 2 when
    ``track.getSettings()`` omits the ``channelCount`` field (observed
    on macOS). The bot's ``moq-rs`` Opus decoder won't downmix on its
    own, so we decode at the source channel count and average in
    Python before pushing audio downstream.
    """

    def test_mono_passthrough_returns_input_unchanged(self):
        """No-op when channels<=1 (callers shouldn't call us, but be safe)."""
        pcm = _pack([100, 200, 300, 400])
        self.assertIs(_downmix_s16_to_mono(pcm, 1), pcm)
        self.assertIs(_downmix_s16_to_mono(pcm, 0), pcm)

    def test_odd_length_returns_input_unchanged(self):
        """Malformed buffer (not evenly divisible by channel count) is a
        no-op rather than a partial-frame decode. Keeps push_received_audio
        from crashing on a single byte split across an Opus frame
        boundary."""
        # 5 samples, 2 channels: not evenly divisible.
        pcm = _pack([1, 2, 3, 4, 5])
        self.assertIs(_downmix_s16_to_mono(pcm, 2), pcm)

    def test_stereo_equal_channels_preserves_volume(self):
        """When L=R (mono-delivered-as-fake-stereo, the common case
        through ``@moq/publish``), the average equals each channel —
        no volume loss."""
        pcm = _pack([1000, 1000, 2000, 2000, 3000, 3000])
        mono = _unpack(_downmix_s16_to_mono(pcm, 2))
        self.assertEqual(mono, [1000, 2000, 3000])

    def test_stereo_different_channels_averages(self):
        """Genuine stereo input is averaged per frame."""
        # L=[1000, 2000, 3000, 4000], R=[3000, 4000, 5000, 6000]
        # → mono=[2000, 3000, 4000, 5000]
        pcm = _pack([1000, 3000, 2000, 4000, 3000, 5000, 4000, 6000])
        mono = _unpack(_downmix_s16_to_mono(pcm, 2))
        self.assertEqual(mono, [2000, 3000, 4000, 5000])

    def test_three_channels_averages(self):
        """Defensive: the catalog could in principle advertise N>2
        (multi-channel mic, surround). The helper averages across all
        channels rather than only L+R."""
        # 2 frames of 3 channels each.
        # frame 0: [3, 6, 9] → 6
        # frame 1: [10, 20, 30] → 20
        pcm = _pack([3, 6, 9, 10, 20, 30])
        mono = _unpack(_downmix_s16_to_mono(pcm, 3))
        self.assertEqual(mono, [6, 20])

    def test_does_not_overflow_int16_sum(self):
        """``acc = sum(samples)`` uses Python ints (unbounded) so adding
        two max-positive S16 values can't overflow before the divide.
        Without this, a naive C-style implementation would wrap to
        negative on the addition step."""
        # Both channels at +32767. Sum would overflow int16 (=> 65534),
        # but Python's int addition is fine, then //2 = 32767.
        pcm = _pack([32767, 32767])
        mono = _unpack(_downmix_s16_to_mono(pcm, 2))
        self.assertEqual(mono, [32767])

    def test_clips_to_int16_range(self):
        """Saturation guard: even if the average somehow lands outside
        the S16 range (rounding edge cases on negative-asymmetric inputs),
        the output stays in [-32768, 32767]."""
        # The averaged result should always fit, but the guard is
        # belt-and-suspenders. Pick values that exercise the lower bound.
        pcm = _pack([-32768, -32768])
        mono = _unpack(_downmix_s16_to_mono(pcm, 2))
        self.assertEqual(mono, [-32768])


# ----------------------------------------------------------------------
# Cert hash helpers (runner/run.py)
# ----------------------------------------------------------------------


def _self_signed_pem(tmp_path):
    """Mint a self-signed cert into ``tmp_path`` and return (pem_path,
    expected_b64_sha256)."""
    import base64
    import hashlib

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    # Use the same algorithm (ECDSA P-256) as the dev script + moq-rs
    # in-process mint, so the helper sees a realistic input.
    key = ec.generate_private_key(ec.SECP256R1())
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(__import__("datetime").datetime.now(__import__("datetime").timezone.utc))
        .not_valid_after(
            __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            + __import__("datetime").timedelta(days=1)
        )
        .sign(key, hashes.SHA256())
    )

    pem_path = tmp_path / "test-cert.pem"
    pem_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    der = cert.public_bytes(serialization.Encoding.DER)
    expected = base64.b64encode(hashlib.sha256(der).digest()).decode()
    return str(pem_path), expected


# The runner module pulls in FastAPI/uvicorn (the `runner` extra). Skip
# the cert-hash helper tests when that's not installed; the helpers are
# defined in run.py, so import = require runner extra.
fastapi = pytest.importorskip("fastapi")
from pipecat.runner.moq import (  # noqa: E402
    _build_moq_client_config,
    _cert_hash_from_pem,
    _hex_to_b64,
)


class TestCertHashHelpers(unittest.TestCase):
    """``/start`` must hand the browser a base64 SHA-256 of the cert that
    the bot is presenting, so WebTransport's ``serverCertificateHashes``
    pin matches at handshake. We hit a real ``certHash=None`` bug once
    where the fallback path silently returned ``None``; lock in the
    round-trip."""

    def test_cert_hash_from_pem_matches_openssl(self):
        """``_cert_hash_from_pem`` should produce the same digest as
        ``openssl x509 -outform der | openssl dgst -sha256 | base64``,
        which is what the old dev script (and the WebTransport spec)
        defines."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            pem_path, expected = _self_signed_pem(Path(td))
            got = _cert_hash_from_pem(pem_path)
            self.assertEqual(got, expected)

    def test_cert_hash_from_pem_missing_file_returns_none(self):
        """Bad path → ``None`` rather than crash. The /start handler
        falls back to ``certHash: null`` in the JSON, which the browser
        treats as ``cert=none`` (CA-signed)."""
        self.assertIsNone(_cert_hash_from_pem("/nonexistent/cert.pem"))

    def test_cert_hash_from_pem_malformed_pem_returns_none(self):
        """A real file but not a PEM-encoded cert → ``None``."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pem", mode="w", delete=False) as f:
            f.write("not a real cert\n")
            path = f.name
        try:
            self.assertIsNone(_cert_hash_from_pem(path))
        finally:
            import os

            os.unlink(path)

    def test_hex_to_b64_round_trip(self):
        """The bot's serve-mode ``cert_fingerprints()`` returns hex; the
        browser wants base64. Verify the encoding is a straight
        bytes-equal conversion."""
        import base64

        # Arbitrary 32-byte digest, hex-encoded.
        digest = bytes(range(32))
        hex_str = digest.hex()
        expected = base64.b64encode(digest).decode()
        self.assertEqual(_hex_to_b64(hex_str), expected)

    def test_hex_to_b64_malformed_returns_none(self):
        """Garbage hex → ``None``. /start handles the None by sending
        ``certHash: null``."""
        self.assertIsNone(_hex_to_b64("not-hex"))
        self.assertIsNone(_hex_to_b64("a"))  # odd length

    def test_build_moq_client_config_serve_uses_runner_fingerprint(self):
        """In serve mode the bot's in-process cert fingerprint takes
        precedence over any ``--moq-cert`` path. Verifies the priority
        order in ``_build_moq_client_config`` so a future reshuffle
        doesn't silently regress."""
        args = MagicMock()
        args.moq_host = "localhost"
        args.moq_port = 4080
        args.moq_path = "/"
        args.moq_serve = True
        args.moq_tls_cert = None  # serve-mode: no PEM on disk
        args.moq_client_id = "client0"
        args.moq_bot_id = "bot0"

        digest = bytes(range(32))
        cfg = _build_moq_client_config(args, namespace="pipecat", cert_fingerprints=[digest.hex()])

        import base64

        self.assertEqual(cfg["certHash"], base64.b64encode(digest).decode())
        self.assertEqual(cfg["relayUrl"], "https://localhost:4080/")
        self.assertTrue(cfg["serve"])
        # Track names are NOT pinned — the bot publishes a catalog and
        # the browser reads track names from it at runtime.
        self.assertNotIn("publishTrack", cfg)
        self.assertNotIn("subscribeTrack", cfg)
        self.assertEqual(cfg["transcriptTrack"], "transcript.json.z")

    def test_build_moq_client_config_client_mode_falls_back_to_pem(self):
        """In client mode (no serve, ``--moq-cert /path``), the helper
        reads the PEM and computes the hash. Locks in the fallback
        ordering."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            pem_path, expected = _self_signed_pem(Path(td))

            args = MagicMock()
            args.moq_host = "relay.example.com"
            args.moq_port = 4080
            args.moq_path = "/moq"
            args.moq_serve = False
            args.moq_tls_cert = pem_path
            args.moq_client_id = "client0"
            args.moq_bot_id = "bot0"

            cfg = _build_moq_client_config(args, namespace="pipecat", cert_fingerprints=[])
            self.assertEqual(cfg["certHash"], expected)
            self.assertFalse(cfg["serve"])

    def test_build_moq_client_config_no_cert_returns_null_hash(self):
        """CA-signed deployment: no ``--moq-cert``, no serve-mode
        fingerprint → ``certHash: None`` (becomes ``null`` in JSON,
        which the browser interprets as "no pinning, trust normally")."""
        args = MagicMock()
        args.moq_host = "moq.example.com"
        args.moq_port = 4080
        args.moq_path = "/moq"
        args.moq_serve = False
        args.moq_tls_cert = None
        args.moq_client_id = "client0"
        args.moq_bot_id = "bot0"

        cfg = _build_moq_client_config(args, namespace="pipecat", cert_fingerprints=None)
        self.assertIsNone(cfg["certHash"])


# ----------------------------------------------------------------------
# MOQTransport.__init__ characterization
# ----------------------------------------------------------------------


class TestMOQTransportInit(unittest.TestCase):
    """Lock in the synchronous-construction contract:

    The publish broadcast and transcript track MUST be created in
    ``MOQTransportClient.__init__``, NOT in ``_run()``'s async bring-up.

    Why: :class:`MOQOutputTransport.start` runs as part of pipecat's
    StartFrame propagation. It calls ``open_audio_track`` immediately,
    which needs ``self._publish_broadcast`` to exist. If broadcast
    creation were deferred to ``_run()`` (the connection task), the
    output transport could try to publish audio before the broadcast
    producer existed → silent drops, hundreds of ms of bot speech lost
    at startup.

    PR #4557's self-review commit explicitly fixed this regression.
    These assertions stop a future refactor from re-introducing it.
    """

    def _make_transport(self):
        """Construct a MOQTransport with the moq library's BroadcastProducer
        mocked so we don't need a real QUIC stack just to check that the
        producer methods got called."""
        params = MOQParams(audio_in_enabled=True, audio_out_enabled=True)

        # Patch ``moq.BroadcastProducer`` so we can observe what __init__
        # calls on it without standing up an actual broadcast.
        with patch("pipecat.transports.moq.transport.moq") as moq_mock:
            broadcast = MagicMock(name="broadcast")
            track = MagicMock(name="transcript_stream")
            broadcast.publish_json_stream.return_value = track
            moq_mock.BroadcastProducer.return_value = broadcast

            transport = MOQTransport(params=params, host="localhost", port=4080)
            return transport, broadcast, track, moq_mock

    def test_publish_broadcast_created_synchronously(self):
        """The bot's broadcast producer exists immediately after
        ``__init__`` — not lazily inside ``_run()``."""
        transport, broadcast, _track, _moq = self._make_transport()
        self.assertIsNotNone(transport._client._publish_broadcast)
        self.assertIs(transport._client._publish_broadcast, broadcast)

    def test_transcript_track_created_synchronously(self):
        """Same constraint for the transcript JSON stream: ``send_message``
        on the output transport appends RTVI messages into it, and that can
        happen before ``_run()`` finishes dialing. Compression is on (the
        ``.z`` suffix)."""
        transport, broadcast, track, _moq = self._make_transport()
        self.assertIs(transport._client._transcript_out, track)
        broadcast.publish_json_stream.assert_called_once_with("transcript.json.z", compression=True)

    def test_audio_track_is_lazy(self):
        """The audio track, by contrast, is intentionally lazy. We don't
        know the pipeline's output sample rate until StartFrame arrives,
        which fires :class:`MOQOutputTransport.start` → ``open_audio_track``.
        If __init__ were to eagerly open the track here, we'd commit to
        the wrong sample rate."""
        transport, broadcast, _track, _moq = self._make_transport()
        self.assertIsNone(transport._client._audio_out)
        broadcast.publish_audio.assert_not_called()

    def test_broadcast_paths_built_from_params(self):
        """``<namespace>/<participant_id>`` and ``<namespace>/<peer_id>``
        are computed from MOQParams once at __init__ — the bot doesn't
        re-resolve them per connection. A future "rooms" refactor that
        wants per-connection namespacing would need to either re-thread
        these or rebuild the transport per connection."""
        params = MOQParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            namespace="myroom",
            participant_id="alice",
            peer_id="bob",
        )
        with patch("pipecat.transports.moq.transport.moq") as moq_mock:
            moq_mock.BroadcastProducer.return_value = MagicMock()
            transport = MOQTransport(params=params, host="localhost", port=4080)

        self.assertEqual(transport._client._broadcast_path, "myroom/alice")
        self.assertEqual(transport._client._peer_broadcast_path, "myroom/bob")

    def test_cert_fingerprints_initially_empty(self):
        """Serve-mode cert fingerprints get populated by ``_run()`` once
        the moq.Server has bound. Before that, the runner reads ``[]``
        — which ``_build_moq_client_config`` falls through to the
        ``--moq-cert`` path. Verifies the published initial state."""
        transport, _broadcast, _track, _moq = self._make_transport()
        self.assertEqual(transport.cert_fingerprints, [])


if __name__ == "__main__":
    unittest.main()
