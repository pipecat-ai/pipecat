#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the MoQ (Media over QUIC) transport.

Four areas covered:

1. **``_downmix_s16_to_mono``** — the workaround for ``@moq/publish``'s
   browser-side encoder publishing stereo even when the source mic
   reports mono. Tests cover the no-op paths (already-mono, malformed
   length) and the arithmetic (averaging, saturation guard).

2. **Cert-hash helpers in ``runner/run.py``** — ``_cert_hash_from_pem``
   (client-mode PEM read) and ``_hex_to_b64`` (serve-mode fingerprint
   conversion). We hit a real ``certHash=None`` bug here once; locking
   the round-trip in stops a regression.

3. **Mode + namespace resolution in ``runner/moq.py``** — which of serve
   and client mode a given set of flags selects, and the namespace each
   gets. Client mode meets the browser on a shared relay, so the
   namespace carries isolation duty that serve mode's private socket
   handles for free.

4. **``MOQTransportClient.__init__`` characterization** — the publish
   broadcast and transcript track must be created synchronously,
   because :class:`MOQOutputTransport.start` opens the audio track
   immediately without waiting for ``_run()``'s async bring-up. If a
   future refactor moves either into ``_run()``, the bot will lose its
   first few hundred ms of audio (this was a real bug PR #4557's
   self-review fixed).
"""

import argparse
import unittest
from unittest.mock import MagicMock, patch

import pytest

# The ``moq`` extra is optional; skip the whole module when ``moq-rs``
# isn't installed, matching the default CI unit test environment which
# doesn't pull optional extras.
pytest.importorskip("moq")

import moq  # noqa: E402

from pipecat.transports.moq.transport import (  # noqa: E402
    MOQParams,
    MOQTransport,
    _downmix_s16_to_mono,
    _is_normal_close,
    _is_peer_gone,
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
class TestIsPeerGone(unittest.TestCase):
    """``_is_peer_gone`` decides whether a per-peer subscription error is
    the normal end of a call (peer closed its session, relay tore down
    its broadcast) or a real failure that must propagate."""

    def test_remote_error_code_is_peer_gone(self):
        """The peer hanging up surfaces as ``remote error: code=N`` on the
        audio/transcript subscription being consumed."""
        self.assertTrue(_is_peer_gone(moq.Error.Audio("moq: remote error: code=4")))

    def test_normal_close_is_peer_gone(self):
        """Session-level normal close counts as the peer leaving too."""
        self.assertTrue(_is_peer_gone(moq.Error.Protocol("webtransport error: closed")))

    def test_dropped_producer_is_peer_gone(self):
        """A peer that vanishes mid-call drops its producer without finishing.

        moq-rs 0.4 raises that locally with the reason as the message tail
        rather than as a reset code, and ``Dropped`` is the one normal-close
        reason with no typed binding. Both shapes observed against a real
        stack: bare from an in-process track, prefixed through the audio path.
        """
        self.assertTrue(_is_peer_gone(moq.Error.JsonTrack("dropped")))
        self.assertTrue(_is_peer_gone(moq.Error.Audio("moq: dropped")))

    def test_shutdown_variants_are_peer_gone(self):
        """``Cancelled``/``Closed`` are typed, so they need no message match."""
        self.assertTrue(_is_peer_gone(moq.Error.Cancelled("cancelled")))
        self.assertTrue(_is_peer_gone(moq.Error.Closed("closed")))

    def test_other_moq_errors_propagate(self):
        self.assertFalse(_is_peer_gone(moq.Error.Mux("json: cancelled")))

    def test_non_moq_errors_propagate(self):
        self.assertFalse(_is_peer_gone(RuntimeError("remote error: code=4")))


fastapi = pytest.importorskip("fastapi")
from pipecat.runner.moq import (  # noqa: E402
    _build_moq_client_config,
    _cert_hash_from_pem,
    _hex_to_b64,
    _new_session_namespace,
    _validate_moq_args,
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
        args.moq_client_id = "request"
        args.moq_bot_id = "response"

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
            args.moq_client_id = "request"
            args.moq_bot_id = "response"

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
        args.moq_client_id = "request"
        args.moq_bot_id = "response"

        cfg = _build_moq_client_config(args, namespace="pipecat", cert_fingerprints=None)
        self.assertIsNone(cfg["certHash"])


# ----------------------------------------------------------------------
# Mode + namespace resolution
# ----------------------------------------------------------------------


def _moq_args(**overrides) -> argparse.Namespace:
    """Build an args namespace the way the parser leaves it before validation."""
    defaults = dict(
        moq_serve=None,
        moq_connect=None,
        moq_bind=None,
        moq_namespace=None,
        moq_tls_cert=None,
        moq_tls_key=None,
        moq_tls_generate=None,
        moq_tls_insecure=False,
        moq_bot_id="response",
        moq_client_id="request",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestMoqModeResolution(unittest.TestCase):
    """Naming a relay with ``--moq-connect`` is the only thing that selects
    client mode; there's no default relay. Server mode stays the default for
    a bare ``-t moq`` so local dev keeps working offline."""

    def test_no_flags_defaults_to_serve(self):
        args = _moq_args()
        self.assertTrue(_validate_moq_args(args))
        self.assertTrue(args.moq_serve)

    def test_connect_url_selects_client_mode(self):
        """Passing a relay is what opts into client mode."""
        args = _moq_args(moq_connect="https://relay.example.com:4443/moq")
        self.assertTrue(_validate_moq_args(args))
        self.assertFalse(args.moq_serve)
        self.assertEqual(args.moq_host, "relay.example.com")
        self.assertEqual(args.moq_port, 4443)
        self.assertEqual(args.moq_path, "/moq")

    def test_connect_url_may_omit_the_port(self):
        """A relay on standard HTTPS needn't spell out :443."""
        args = _moq_args(moq_connect="https://cdn.moq.dev/anon")
        self.assertTrue(_validate_moq_args(args))
        self.assertFalse(args.moq_serve)
        self.assertEqual(args.moq_host, "cdn.moq.dev")
        self.assertEqual(args.moq_port, 443)
        self.assertEqual(args.moq_path, "/anon")

    def test_explicit_serve_wins_over_connect(self):
        """--moq-serve is explicit, so it isn't overridden by --moq-connect."""
        args = _moq_args(moq_serve=True, moq_connect="https://relay.example.com:4443/moq")
        self.assertTrue(_validate_moq_args(args))
        self.assertTrue(args.moq_serve)


class TestMoqNamespaceResolution(unittest.TestCase):
    """In client mode the namespace is the only thing separating one
    session from another on a shared relay — and on an anonymous relay,
    the only thing gating access. So it must be per-session and
    unguessable there, while serve mode can keep a stable, readable name."""

    def test_serve_mode_gets_the_fixed_default(self):
        args = _moq_args(moq_serve=True)
        self.assertTrue(_validate_moq_args(args))
        self.assertEqual(args.moq_namespace, "pipecat")

    def test_client_mode_left_unresolved_for_per_session_minting(self):
        """Left as None so each /start mints its own; a fixed default here
        would silently put every session on the same public path."""
        args = _moq_args(moq_connect="https://cdn.moq.dev/anon")
        self.assertTrue(_validate_moq_args(args))
        self.assertIsNone(args.moq_namespace)

    def test_explicit_namespace_survives_both_modes(self):
        for extra in ({"moq_serve": True}, {"moq_connect": "https://cdn.moq.dev/anon"}):
            args = _moq_args(moq_namespace="my-room", **extra)
            self.assertTrue(_validate_moq_args(args))
            self.assertEqual(args.moq_namespace, "my-room")

    def test_minted_namespaces_are_unique_and_unguessable(self):
        minted = {_new_session_namespace() for _ in range(100)}
        self.assertEqual(len(minted), 100)
        # 8 bytes of entropy, rendered hex, on a readable prefix.
        for ns in minted:
            self.assertTrue(ns.startswith("pipecat-"))
            self.assertEqual(len(ns.removeprefix("pipecat-")), 16)

    def test_minted_namespace_is_a_single_path_segment(self):
        """The namespace is joined into ``<namespace>/<id>``; a stray
        separator would silently reshape the broadcast path."""
        for _ in range(20):
            self.assertNotIn("/", _new_session_namespace())


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
        """Construct a MOQTransport with the moq library's origin mocked so we
        don't need a real QUIC stack just to check that the producer methods
        got called."""
        params = MOQParams(audio_in_enabled=True, audio_out_enabled=True)

        # A broadcast is created ON an origin, so patch the origin and observe
        # what __init__ asks it for without standing up an actual broadcast.
        with patch("pipecat.transports.moq.transport.moq") as moq_mock:
            broadcast = MagicMock(name="broadcast")
            track = MagicMock(name="transcript_stream")
            broadcast.publish_json_stream.return_value = track
            origin = MagicMock(name="publish_origin")
            origin.create_broadcast.return_value = broadcast
            moq_mock.OriginProducer.return_value = origin

            transport = MOQTransport(params=params, host="localhost", port=4080)
            return transport, broadcast, track, moq_mock

    def test_publish_broadcast_created_synchronously(self):
        """The bot's broadcast producer exists immediately after
        ``__init__`` — not lazily inside ``_run()``."""
        transport, broadcast, _track, _moq = self._make_transport()
        self.assertIsNotNone(transport._client._publish_broadcast)
        self.assertIs(transport._client._publish_broadcast, broadcast)
        # Created at its final path: the origin carries the broadcast into the
        # session, so there is no later attach step to forget.
        transport._client._publish_origin.create_broadcast.assert_called_once_with(
            transport._client._broadcast_path
        )

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

    def _paths_for(self, **kwargs):
        params = MOQParams(audio_in_enabled=True, audio_out_enabled=True, **kwargs)
        with patch("pipecat.transports.moq.transport.moq") as moq_mock:
            moq_mock.BroadcastProducer.return_value = MagicMock()
            transport = MOQTransport(params=params, host="localhost", port=4080)
        return transport._client._broadcast_path, transport._client._peer_broadcast_path

    def _bind_for(self, **kwargs):
        params = MOQParams(audio_in_enabled=True, audio_out_enabled=True, **kwargs)
        with patch("pipecat.transports.moq.transport.moq") as moq_mock:
            moq_mock.BroadcastProducer.return_value = MagicMock()
            transport = MOQTransport(params=params, host="localhost", port=4080)
        return transport._client._bind

    def test_serve_mode_defaults_the_bind_to_the_port(self):
        """Serve mode needs a concrete listen address; unset, it falls
        back to the constructor's port."""
        self.assertEqual(self._bind_for(serve=True), "[::]:4080")

    def test_serve_mode_honors_an_explicit_bind(self):
        self.assertEqual(self._bind_for(serve=True, bind="[::]:9000"), "[::]:9000")

    def test_client_mode_binds_ephemeral_by_default(self):
        """None means moq.Client picks an ephemeral source port — the
        port default is serve-only and must not leak into client mode."""
        self.assertIsNone(self._bind_for(serve=False))

    def test_client_mode_honors_an_explicit_bind(self):
        """A chosen, non-ephemeral source port is valid when dialing a
        relay — it isn't ignored."""
        self.assertEqual(self._bind_for(serve=False, bind="[::]:9000"), "[::]:9000")

    def _client_kwargs(self, **params):
        """The kwargs ``_make_transport`` hands ``moq.Client`` in client mode."""
        p = MOQParams(audio_in_enabled=True, audio_out_enabled=True, serve=False, **params)
        with patch("pipecat.transports.moq.transport.moq") as moq_mock:
            moq_mock.BroadcastProducer.return_value = MagicMock()
            transport = MOQTransport(params=p, host="localhost", port=4080)
            client = transport._client
            client._make_transport(MagicMock(), MagicMock())
            return moq_mock.Client.call_args.kwargs

    def test_client_cert_is_presented_when_both_halves_are_set(self):
        """A relay that authenticates its peers with mTLS needs the client
        cert; without it the dial is anonymous and the relay tiers it as an
        ordinary connection."""
        kwargs = self._client_kwargs(client_tls_cert="/c.pem", client_tls_key="/k.pem")
        self.assertEqual(kwargs["tls_cert"], "/c.pem")
        self.assertEqual(kwargs["tls_key"], "/k.pem")

    def test_no_client_cert_by_default(self):
        kwargs = self._client_kwargs()
        self.assertNotIn("tls_cert", kwargs)
        self.assertNotIn("tls_key", kwargs)

    def test_half_a_client_cert_is_ignored(self):
        """A cert without its key can't be loaded, so passing one alone would
        fail the dial rather than degrade to anonymous."""
        self.assertNotIn("tls_cert", self._client_kwargs(client_tls_cert="/c.pem"))
        self.assertNotIn("tls_key", self._client_kwargs(client_tls_key="/k.pem"))

    def test_custom_roots_and_pins_are_passed_through(self):
        """Both are alternatives to switching ``verify_ssl`` off: a private CA
        and a self-signed relay can each be verified rather than trusted
        blindly."""
        kwargs = self._client_kwargs(
            client_tls_roots=["/ca.pem"], client_tls_fingerprints=["ab:cd"]
        )
        self.assertEqual(kwargs["tls_roots"], ["/ca.pem"])
        self.assertEqual(kwargs["tls_fingerprints"], ["ab:cd"])

    def test_no_roots_or_pins_by_default(self):
        kwargs = self._client_kwargs()
        self.assertNotIn("tls_roots", kwargs)
        self.assertNotIn("tls_fingerprints", kwargs)

    def test_deprecated_serve_bind_still_sets_the_bind(self):
        """Pydantic drops unknown fields, so without the alias a bot that
        pinned the pre-1.8.0 ``serve_bind`` would silently listen on the
        default address instead."""
        with self.assertWarns(DeprecationWarning):
            bind = self._bind_for(serve=True, serve_bind="[::]:9000")
        self.assertEqual(bind, "[::]:9000")

    def test_bind_wins_over_deprecated_serve_bind(self):
        with self.assertWarns(DeprecationWarning):
            bind = self._bind_for(serve=True, bind="[::]:1", serve_bind="[::]:2")
        self.assertEqual(bind, "[::]:1")

    def test_explicit_paths_override_the_namespace_layer(self):
        """``response_path``/``request_path`` win over ``<namespace>/<id>``.

        The namespace model needs both peers to agree on a namespace up
        front. That works when one side hands the other a config blob, but
        not when the paths are assigned externally — e.g. a relay that routes
        on a path prefix and derives the bot's path from the peer's, so
        there's no namespace to agree on.
        """
        publish, subscribe = self._paths_for(
            namespace="ignored",
            participant_id="ignored",
            peer_id="ignored",
            response_path="room1/agent.hang",
            request_path="room1.hang",
        )
        self.assertEqual(publish, "room1/agent.hang")
        self.assertEqual(subscribe, "room1.hang")

    def test_paths_override_independently(self):
        """Either path may be overridden alone; the other still derives.

        Nothing requires both to come from the same place, and silently
        ignoring one because the other was set would be a nasty surprise.
        """
        publish, subscribe = self._paths_for(
            namespace="myroom",
            participant_id="alice",
            peer_id="bob",
            response_path="somewhere/else",
        )
        self.assertEqual(publish, "somewhere/else")
        self.assertEqual(subscribe, "myroom/bob")

        publish, subscribe = self._paths_for(
            namespace="myroom",
            participant_id="alice",
            peer_id="bob",
            request_path="somewhere/else",
        )
        self.assertEqual(publish, "myroom/alice")
        self.assertEqual(subscribe, "somewhere/else")

    def test_paths_default_to_the_namespace_layer(self):
        """Unset (the default), the namespace model is unchanged."""
        publish, subscribe = self._paths_for()
        self.assertEqual(publish, "pipecat/response")
        self.assertEqual(subscribe, "pipecat/request")

    def test_cert_fingerprints_initially_empty(self):
        """Serve-mode cert fingerprints get populated by ``_run()`` once
        the moq.Server has bound. Before that, the runner reads ``[]``
        — which ``_build_moq_client_config`` falls through to the
        ``--moq-cert`` path. Verifies the published initial state."""
        transport, _broadcast, _track, _moq = self._make_transport()
        self.assertEqual(transport.cert_fingerprints, [])


# ----------------------------------------------------------------------
# _is_normal_close
# ----------------------------------------------------------------------


class TestIsNormalClose(unittest.TestCase):
    """Cover which MoQ errors count as a hangup rather than a failure.

    A disconnect surfaces at two levels: the session reports a
    WebTransport close, and every in-flight track subscription is reset
    with a numeric remote code. Misclassifying either turns an ordinary
    hangup into an ERROR log, a traceback, and an ``on_error`` callback
    that application code may act on.

    The codes come from moq-net's ``Error::to_code``. Application codes
    are offset by 64 there, so ``code=240`` (``App(176)``) shares a
    prefix with the ``Dropped`` code and must not be matched by it.
    """

    def _audio_error(self, message):
        import moq

        return moq.Error.Audio(message)

    def test_session_close_is_normal(self):
        self.assertTrue(_is_normal_close(self._audio_error("webtransport error: closed")))

    def test_peer_dropped_producer_is_normal(self):
        """A browser leaving mid-call drops its mic producer without finishing."""
        self.assertTrue(_is_normal_close(self._audio_error("moq: remote error: code=24")))

    def test_cancel_and_closed_are_normal(self):
        for code in (0, 25):
            with self.subTest(code=code):
                self.assertTrue(
                    _is_normal_close(self._audio_error(f"moq: remote error: code={code}"))
                )

    def test_real_failures_are_not_normal(self):
        # Decode, Lagged, and an application code that starts with "24".
        for code in (5, 26, 240):
            with self.subTest(code=code):
                self.assertFalse(
                    _is_normal_close(self._audio_error(f"moq: remote error: code={code}"))
                )

    def test_non_moq_exception_is_not_normal(self):
        self.assertFalse(_is_normal_close(RuntimeError("moq: remote error: code=24")))


if __name__ == "__main__":
    unittest.main()
