#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the MoQ (Media over QUIC) transport.

Areas covered:

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

5. **Transcript record metadata** — every published record carries
   ``seq`` and ``epoch``, and a subscriber drops the replay it gets on
   every (re)subscribe while passing records without them through.

6. **``MOQRunnerArguments.relay_url``** — reaches the transport unchanged,
   query string included, and host/port still compose a URL.

7. **Client-mode reconnect** — the session loop redials a dropped
   session within ``connection_timeout``, never retries a refused dial,
   reports the peer gone exactly once, and pushes errors with the
   category and permanence the pipeline acts on.
"""

import argparse
import asyncio
import itertools
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The ``moq`` extra is optional; skip the whole module when ``moq-rs``
# isn't installed, matching the default CI unit test environment which
# doesn't pull optional extras.
pytest.importorskip("moq")

import moq  # noqa: E402

import pipecat.transports.moq.transport as moq_transport  # noqa: E402
from pipecat.runner.types import MOQRunnerArguments  # noqa: E402
from pipecat.runner.utils import create_transport  # noqa: E402
from pipecat.transports.moq.transport import (  # noqa: E402
    TRANSCRIPT_EPOCH_FIELD,
    TRANSCRIPT_SEQ_FIELD,
    MOQCallbacks,
    MOQParams,
    MOQTransport,
    MOQTransportClient,
    _downmix_s16_to_mono,
    _is_normal_close,
    _is_peer_gone,
)
from pipecat.utils.asyncio.task_manager import TaskManager  # noqa: E402
from pipecat.utils.errors import ErrorCategory  # noqa: E402

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


# ----------------------------------------------------------------------
# Transcript record metadata and replay dedupe
# ----------------------------------------------------------------------


def _fake_origin(**_kwargs):
    """A stand-in ``moq.OriginProducer`` whose broadcast hands out mock producers."""
    origin = MagicMock(name="origin")
    broadcast = MagicMock(name="broadcast")
    broadcast.publish_json_stream.return_value = MagicMock(name="transcript_stream")
    broadcast.publish_audio.return_value = MagicMock(name="audio_track")
    origin.create_broadcast.return_value = broadcast
    return origin


def _client_with_fake_moq(params: MOQParams | None = None, url: str = "https://relay/moq"):
    """Build a ``MOQTransportClient`` whose moq origins and producers are mocks.

    Returns the client and the transcript stream mock its records land on.
    """
    params = params or MOQParams(audio_in_enabled=True, audio_out_enabled=True)
    callbacks = MOQCallbacks(
        on_connected=AsyncMock(),
        on_disconnected=AsyncMock(),
        on_client_connected=AsyncMock(),
        on_client_disconnected=AsyncMock(),
        on_track_subscribed=AsyncMock(),
        on_error=AsyncMock(),
        on_audio_received=AsyncMock(),
        on_message_received=AsyncMock(),
    )
    with patch.object(moq_transport.moq, "OriginProducer") as origin_cls:
        stream = MagicMock(name="transcript_stream")
        broadcast = MagicMock(name="broadcast")
        broadcast.publish_json_stream.return_value = stream
        origin_cls.return_value.create_broadcast.return_value = broadcast
        client = MOQTransportClient(params=params, url=url, bind=None, callbacks=callbacks)
    return client, stream


class TestTranscriptRecords(unittest.TestCase):
    """Every published record carries ``seq`` and ``epoch``; a subscriber
    uses them to drop the replay it gets on every (re)subscribe."""

    def test_records_are_numbered_from_zero_per_instance(self):
        client, stream = _client_with_fake_moq()
        client.publish_transcript({"label": "rtvi-ai", "type": "a"})
        client.publish_transcript({"label": "rtvi-ai", "type": "b"})
        first, second = (call.args[0] for call in stream.append.call_args_list)
        self.assertEqual(first[TRANSCRIPT_SEQ_FIELD], 0)
        self.assertEqual(second[TRANSCRIPT_SEQ_FIELD], 1)
        self.assertEqual(first[TRANSCRIPT_EPOCH_FIELD], second[TRANSCRIPT_EPOCH_FIELD])
        self.assertEqual(first["type"], "a")

    def test_epoch_differs_between_instances(self):
        """A restarted bot is a new peer, so its count must not collide with
        the watermark a subscriber kept from the previous one."""
        a, stream_a = _client_with_fake_moq()
        b, stream_b = _client_with_fake_moq()
        a.publish_transcript({"type": "x"})
        b.publish_transcript({"type": "x"})
        self.assertNotEqual(
            stream_a.append.call_args.args[0][TRANSCRIPT_EPOCH_FIELD],
            stream_b.append.call_args.args[0][TRANSCRIPT_EPOCH_FIELD],
        )

    def test_the_message_itself_is_not_modified(self):
        client, _stream = _client_with_fake_moq()
        message = {"label": "rtvi-ai", "type": "a"}
        client.publish_transcript(message)
        self.assertEqual(message, {"label": "rtvi-ai", "type": "a"})

    def test_new_records_are_delivered_with_the_fields_stripped(self):
        client, _stream = _client_with_fake_moq()
        record = {"label": "rtvi-ai", "type": "a", "seq": 0, "epoch": "e1"}
        self.assertEqual(client._accept_peer_record(record), {"label": "rtvi-ai", "type": "a"})

    def test_replayed_records_are_dropped(self):
        client, _stream = _client_with_fake_moq()
        for seq in (0, 1, 2):
            self.assertIsNotNone(
                client._accept_peer_record({"type": "a", "seq": seq, "epoch": "e"})
            )
        # The whole log comes back on a resubscribe.
        for seq in (0, 1, 2):
            self.assertIsNone(client._accept_peer_record({"type": "a", "seq": seq, "epoch": "e"}))
        self.assertIsNotNone(client._accept_peer_record({"type": "a", "seq": 3, "epoch": "e"}))

    def test_a_new_epoch_starts_the_count_over(self):
        client, _stream = _client_with_fake_moq()
        client._accept_peer_record({"type": "a", "seq": 5, "epoch": "old"})
        self.assertIsNotNone(client._accept_peer_record({"type": "a", "seq": 0, "epoch": "new"}))
        self.assertIsNone(client._accept_peer_record({"type": "a", "seq": 0, "epoch": "new"}))

    def test_records_without_a_sequence_pass_through_unchanged(self):
        """A peer that predates the fields cannot be deduplicated, but it
        must keep working."""
        client, _stream = _client_with_fake_moq()
        record = {"label": "rtvi-ai", "type": "client-ready"}
        self.assertIs(client._accept_peer_record(record), record)
        self.assertIs(client._accept_peer_record(record), record)

    def test_a_boolean_sequence_is_not_a_sequence(self):
        client, _stream = _client_with_fake_moq()
        record = {"type": "a", "seq": True}
        self.assertIs(client._accept_peer_record(record), record)


# ----------------------------------------------------------------------
# Runner arguments: relay_url
# ----------------------------------------------------------------------


class TestRunnerRelayUrl(unittest.IsolatedAsyncioTestCase):
    """``MOQRunnerArguments.relay_url`` reaches the transport unchanged."""

    async def _transport_for(self, args: MOQRunnerArguments) -> MOQTransport:
        with patch("pipecat.transports.moq.transport.moq") as moq_mock:
            moq_mock.BroadcastProducer.return_value = MagicMock()
            transport = await create_transport(
                args, {"moq": lambda: MOQParams(audio_in_enabled=True)}
            )
        return transport

    async def test_relay_url_is_dialed_as_given_query_string_included(self):
        url = "https://relay.example.com/?jwt=eyJhbGciOi.eyJyb290Ijo.sig"
        transport = await self._transport_for(MOQRunnerArguments(relay_url=url, namespace="ns"))
        self.assertEqual(transport._client._url, url)
        self.assertEqual(transport._params.relay_url, url)
        self.assertEqual(transport._params.namespace, "ns")

    async def test_host_and_port_still_compose_the_url(self):
        transport = await self._transport_for(MOQRunnerArguments("relay.example.com", 4443))
        self.assertEqual(transport._client._url, "https://relay.example.com:4443/moq")

    async def test_relay_url_wins_over_host_and_port_with_a_warning(self):
        url = "https://relay.example.com/?jwt=t"
        with patch("pipecat.runner.types.logger") as log:
            args = MOQRunnerArguments("other.example.com", 4443, relay_url=url)
        log.warning.assert_called_once()
        transport = await self._transport_for(args)
        self.assertEqual(transport._client._url, url)

    def test_relay_url_alone_logs_no_warning(self):
        with patch("pipecat.runner.types.logger") as log:
            MOQRunnerArguments(relay_url="https://relay.example.com/")
        log.warning.assert_not_called()

    def test_client_mode_needs_a_dial_target(self):
        with self.assertRaises(ValueError):
            MOQRunnerArguments()
        with self.assertRaises(ValueError):
            MOQRunnerArguments(host="relay.example.com")

    def test_serve_mode_needs_no_dial_target(self):
        self.assertTrue(MOQRunnerArguments(serve=True).serve)


# ----------------------------------------------------------------------
# Client-mode reconnect
# ----------------------------------------------------------------------


class _FakeSession:
    """A dialed session that closes when the test says so, with or without an error."""

    def __init__(self, closed: bool = False, error: Exception | None = None):
        self._closed = asyncio.Event()
        self._error = error
        if closed:
            self._closed.set()

    async def closed(self):
        await self._closed.wait()
        if self._error is not None:
            raise self._error

    def drop(self, error: Exception | None = None):
        self._error = error
        self._closed.set()


# What moq-ffi raises from ``Session.closed()`` when the relay refused the token.
_UNAUTHORIZED_CLOSE = "transport: webtransport error: closed: code=6 reason=unauthorized"


class _FakeClient:
    """Stands in for ``moq.Client``: each dial takes the next scripted outcome.

    An outcome is a :class:`_FakeSession` (the dial succeeds and yields it)
    or an exception (the dial raises it). Every dial's URL is recorded.
    """

    def __init__(self, script: list, dials: list, url: str, **kwargs):
        self._script = script
        self._dials = dials
        self._url = url
        self.session = None

    async def __aenter__(self):
        self._dials.append(self._url)
        outcome = self._script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        self.session = outcome
        return self

    async def __aexit__(self, *exc):
        return None


class TestClientReconnect(unittest.IsolatedAsyncioTestCase):
    """The client-mode session loop: redial on a dropped session, stop on a
    refused dial, give up at the end of the window.

    ``_consume_peer`` is replaced by a stand-in that reports the peer
    available the way the real one does and then either waits (the peer
    stays) or returns (the peer left); the real one is exercised against
    a relay, not here.
    """

    URL = "https://relay.example.com/?jwt=tok.en"

    async def asyncSetUp(self):
        self.dials: list[str] = []
        self.script: list = []
        patches = [
            patch.object(moq_transport.moq, "OriginProducer", side_effect=_fake_origin),
            patch.object(moq_transport, "_RECONNECT_BACKOFF_INITIAL_S", 0.001),
            patch.object(moq_transport, "_RECONNECT_BACKOFF_MAX_S", 0.002),
            patch.object(moq_transport, "_SESSION_CLOSE_GRACE_S", 0.01),
            patch.object(moq_transport, "_SESSION_STALL_POLL_S", 0.005),
            patch.object(moq_transport, "_SESSION_STALL_S", 0.02),
            patch.object(
                moq_transport.moq,
                "Client",
                side_effect=lambda url, **kw: _FakeClient(self.script, self.dials, url, **kw),
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _make_client(self, **params) -> MOQTransportClient:
        client, _stream = _client_with_fake_moq(
            MOQParams(audio_in_enabled=True, audio_out_enabled=True, **params), url=self.URL
        )
        client._task_manager = TaskManager()
        return client

    @staticmethod
    def _peer_stays(client: MOQTransportClient):
        """A stand-in ``_consume_peer``: the peer is seen and never leaves."""

        async def consume(_origin):
            await client._on_peer_available()
            await client._on_peer_data()
            await asyncio.Event().wait()

        client._consume_peer = consume  # type: ignore[method-assign]

    @staticmethod
    def _peer_leaves_after(client: MOQTransportClient, sessions_seen: int):
        """A stand-in ``_consume_peer`` whose peer says goodbye and leaves during session N."""
        seen = 0

        async def consume(_origin):
            nonlocal seen
            seen += 1
            await client._on_peer_available()
            await client._on_peer_data()
            if seen >= sessions_seen:
                client._peer_goodbye = True
                return True
            await asyncio.Event().wait()

        client._consume_peer = consume  # type: ignore[method-assign]

    @staticmethod
    def _peer_tracks_end_without_goodbye(client: MOQTransportClient, comes_back: bool):
        """A stand-in ``_consume_peer``: in the first session the peer is seen and
        its tracks keep ending without the marker, so the session is dropped.
        On the next session the peer either comes back and then says goodbye,
        or never appears."""
        seen = 0

        async def consume(_origin):
            nonlocal seen
            seen += 1
            if seen == 1:
                await client._on_peer_available()
                await client._on_peer_data()
                return False
            if comes_back:
                await client._on_peer_available()
                await client._on_peer_data()
                client._peer_goodbye = True
            return True

        client._consume_peer = consume  # type: ignore[method-assign]

    @staticmethod
    def _peer_never_returns(client: MOQTransportClient):
        """A stand-in ``_consume_peer``: the peer is seen in the first session only."""
        seen = 0

        async def consume(_origin):
            nonlocal seen
            seen += 1
            if seen == 1:
                await client._on_peer_available()
                await client._on_peer_data()
            await asyncio.Event().wait()

        client._consume_peer = consume  # type: ignore[method-assign]

    async def _wait_for_dials(self, n: int):
        for _ in range(200):
            if len(self.dials) >= n:
                return
            await asyncio.sleep(0.005)
        self.fail(f"expected {n} dials, saw {len(self.dials)}")

    async def test_a_dropped_session_is_redialed_and_the_peer_survives_it(self):
        first, second = _FakeSession(), _FakeSession()
        self.script[:] = [first, second]
        client = self._make_client()
        self._peer_leaves_after(client, sessions_seen=2)

        run = asyncio.create_task(client._run())
        await self._wait_for_dials(1)
        first.drop()
        await asyncio.wait_for(run, timeout=2)

        cb = client._callbacks
        # Each relay session is a connect and a disconnect of its own.
        self.assertEqual(cb.on_connected.await_count, 2)
        self.assertEqual(cb.on_disconnected.await_count, 2)
        # The peer joined once; its return after the redial is a
        # reconnect, not a second client. It was reported gone exactly
        # once, when it left the second session, not when the first
        # session dropped.
        cb.on_client_connected.assert_awaited_once()
        cb.on_client_disconnected.assert_awaited_once()
        cb.on_error.assert_not_awaited()
        self.assertEqual(self.dials, [self.URL, self.URL])

    async def test_a_session_whose_inbound_traffic_stalls_is_redialed(self):
        """A dead network path freezes the inbound byte counter long before
        QUIC's idle timeout reports the session closed; the stall alone
        forces the redial."""
        first, second = _FakeSession(), _FakeSession()
        first.stats = lambda: SimpleNamespace(bytes_received=1000)  # type: ignore[attr-defined]
        self.script[:] = [first, second]
        client = self._make_client()
        self._peer_leaves_after(client, sessions_seen=2)

        await asyncio.wait_for(client._run(), timeout=2)

        self.assertEqual(self.dials, [self.URL] * 2)
        self.assertEqual(client._callbacks.on_connected.await_count, 2)

    async def test_a_session_with_flowing_traffic_is_not_redialed(self):
        """Keepalive/ACK traffic keeps the counter moving on a healthy
        session, so the watchdog stays quiet."""
        counter = itertools.count(1)
        session = _FakeSession()
        session.stats = lambda: SimpleNamespace(bytes_received=next(counter))  # type: ignore[attr-defined]
        self.script[:] = [session]
        client = self._make_client()
        self._peer_leaves_after(client, sessions_seen=1)

        await asyncio.wait_for(client._run(), timeout=2)

        self.assertEqual(self.dials, [self.URL])
        client._callbacks.on_connected.assert_awaited_once()

    async def test_a_session_reporting_no_counters_is_not_redialed(self):
        """The WebSocket fallback reports no counters; the watchdog stays
        dormant and ``session.closed()`` remains the only drop signal."""
        session = _FakeSession()
        session.stats = lambda: SimpleNamespace(bytes_received=None)  # type: ignore[attr-defined]
        self.script[:] = [session]
        client = self._make_client()
        self._peer_leaves_after(client, sessions_seen=1)

        await asyncio.wait_for(client._run(), timeout=2)

        self.assertEqual(self.dials, [self.URL])
        client._callbacks.on_connected.assert_awaited_once()

    async def test_the_url_is_dialed_unchanged_on_every_attempt(self):
        """The relay token rides in the query string, so the redial must send
        the URL byte for byte."""
        self.script[:] = [ConnectionError("refused"), ConnectionError("refused"), _FakeSession()]
        client = self._make_client()
        self._peer_leaves_after(client, sessions_seen=1)
        await asyncio.wait_for(client._run(), timeout=2)
        self.assertEqual(self.dials, [self.URL] * 3)

    async def test_a_refused_first_dial_is_not_retried(self):
        self.script[:] = [moq.Error.Forbidden("403 denied")]
        client = self._make_client()
        self._peer_stays(client)
        await asyncio.wait_for(client._run(), timeout=2)

        cb = client._callbacks
        self.assertEqual(self.dials, [self.URL])
        cb.on_connected.assert_not_awaited()
        cb.on_client_disconnected.assert_not_awaited()
        cb.on_error.assert_awaited_once()
        _message, exc, category, permanent = cb.on_error.await_args.args
        self.assertIsInstance(exc, moq.Error.Forbidden)
        self.assertIs(category, ErrorCategory.AUTHORIZATION)
        self.assertFalse(permanent)
        cb.on_disconnected.assert_awaited_once()

    async def test_a_refused_redial_ends_the_call(self):
        """The token expired mid-session: the relay closed the session and
        refuses the redial. The peer it had is reported gone, then the
        refusal."""
        first = _FakeSession()
        self.script[:] = [first, moq.Error.Unauthorized("401 expired")]
        client = self._make_client()
        self._peer_stays(client)

        run = asyncio.create_task(client._run())
        await self._wait_for_dials(1)
        first.drop()
        await asyncio.wait_for(run, timeout=2)

        cb = client._callbacks
        cb.on_connected.assert_awaited_once()
        cb.on_disconnected.assert_awaited_once()
        cb.on_client_disconnected.assert_awaited_once()
        _message, exc, category, _permanent = cb.on_error.await_args.args
        self.assertIsInstance(exc, moq.Error.Unauthorized)
        self.assertIs(category, ErrorCategory.AUTHENTICATION)

    async def test_an_unauthorized_session_close_is_not_retried(self):
        """The relay accepts the connection and then closes the session as
        unauthorized, so the refusal is read off the close, not the dial."""
        first = _FakeSession()
        self.script[:] = [first, _FakeSession(), _FakeSession()]
        client = self._make_client()
        self._peer_stays(client)

        run = asyncio.create_task(client._run())
        await self._wait_for_dials(1)
        first.drop(moq.Error.Protocol(_UNAUTHORIZED_CLOSE))
        await asyncio.wait_for(run, timeout=2)

        cb = client._callbacks
        self.assertEqual(self.dials, [self.URL])
        cb.on_disconnected.assert_awaited_once()
        cb.on_client_disconnected.assert_awaited_once()
        _message, exc, category, _permanent = cb.on_error.await_args.args
        self.assertIsInstance(exc, moq.Error.Protocol)
        self.assertIs(category, ErrorCategory.AUTHENTICATION)

    async def test_sessions_that_close_before_the_peer_is_back_keep_the_window_running(self):
        """A relay that accepts every dial and closes the session at once
        must not restart the window on each dial, or the loop never ends."""
        first = _FakeSession()
        self.script[:] = [first] + [_FakeSession(closed=True) for _ in range(1000)]
        client = self._make_client(connection_timeout=0.05)
        self._peer_never_returns(client)

        run = asyncio.create_task(client._run())
        await self._wait_for_dials(1)
        first.drop()
        await asyncio.wait_for(run, timeout=2)

        cb = client._callbacks
        self.assertGreater(len(self.dials), 2)
        self.assertLess(len(self.dials), 1000)
        cb.on_client_disconnected.assert_awaited_once()
        _message, _exc, category, permanent = cb.on_error.await_args.args
        self.assertIs(category, ErrorCategory.CONNECTIVITY)
        self.assertTrue(permanent)

    async def test_the_window_bounds_the_redials(self):
        first = _FakeSession()
        self.script[:] = [first] + [ConnectionError("refused")] * 1000
        client = self._make_client(connection_timeout=0.05)
        self._peer_stays(client)

        run = asyncio.create_task(client._run())
        await self._wait_for_dials(1)
        first.drop()
        await asyncio.wait_for(run, timeout=2)

        cb = client._callbacks
        self.assertGreater(len(self.dials), 1)
        self.assertLess(len(self.dials), 1000)
        cb.on_client_disconnected.assert_awaited_once()
        _message, exc, category, permanent = cb.on_error.await_args.args
        self.assertIsInstance(exc, ConnectionError)
        self.assertIsInstance(exc.__cause__, ConnectionError)
        self.assertIs(category, ErrorCategory.CONNECTIVITY)
        self.assertTrue(permanent)
        cb.on_disconnected.assert_awaited_once()

    async def test_a_redial_that_succeeds_does_not_restart_the_count(self):
        """Only the peer coming back stops the count, so a relay that is
        back while the peer is not leaves what was left of it."""
        first, second = _FakeSession(), _FakeSession()
        self.script[:] = [first, second]
        client = self._make_client(connection_timeout=30)
        self._peer_never_returns(client)

        run = asyncio.create_task(client._run())
        await self._wait_for_dials(1)
        first.drop()
        await self._wait_for_dials(2)
        started = client._peer_missing_since
        self.assertIsNotNone(started)
        await asyncio.sleep(0.02)
        self.assertEqual(client._peer_missing_since, started)
        self.assertLess(client._peer_wait_remaining(), 30)

        run.cancel()
        await asyncio.gather(run, return_exceptions=True)

    async def test_the_peer_coming_back_stops_the_count(self):
        first, second = _FakeSession(), _FakeSession()
        self.script[:] = [first, second]
        client = self._make_client()
        self._peer_stays(client)

        run = asyncio.create_task(client._run())
        await self._wait_for_dials(1)
        first.drop()
        await self._wait_for_dials(2)
        for _ in range(200):
            if client._peer_missing_since is None:
                break
            await asyncio.sleep(0.005)
        self.assertIsNone(client._peer_missing_since)
        self.assertEqual(client._redial_attempt, 0)

        run.cancel()
        await asyncio.gather(run, return_exceptions=True)

    async def test_a_redial_presents_a_fresh_publisher_and_replays_the_log(self):
        """A relay that still holds a dead route for the old publisher never
        serves a re-announce under that identity, so every redial publishes
        from a new origin, with the log replayed and the audio track reopened."""
        first, second = _FakeSession(), _FakeSession()
        self.script[:] = [first, second]
        client = self._make_client()
        self._peer_leaves_after(client, sessions_seen=2)
        client.open_audio_track(24000)
        client.publish_transcript({"type": "a"})
        client.publish_transcript({"type": "b"})
        old_origin, old_stream = client._publish_origin, client._transcript_out
        old_audio = client._audio_out

        run = asyncio.create_task(client._run())
        await self._wait_for_dials(1)
        self.assertIs(client._publish_origin, old_origin)
        first.drop()
        await asyncio.wait_for(run, timeout=2)

        self.assertIsNot(client._publish_origin, old_origin)
        old_stream.finish.assert_called_once()
        old_audio.finish.assert_called()
        new_stream = client._transcript_out
        self.assertIsNot(new_stream, old_stream)
        self.assertEqual([call.args[0]["seq"] for call in new_stream.append.call_args_list], [0, 1])
        self.assertEqual(new_stream.append.call_args_list[1].args[0]["type"], "b")
        self.assertIsNot(client._audio_out, old_audio)
        client._publish_broadcast.publish_audio.assert_called_once()

    async def test_peer_tracks_ending_without_goodbye_redial_and_the_peer_comes_back(self):
        """A relay between the peers failing ends the peer's tracks the way a
        hangup does; without the marker the transport redials and the peer
        reappears on the new session."""
        self.script[:] = [_FakeSession(), _FakeSession()]
        client = self._make_client()
        self._peer_tracks_end_without_goodbye(client, comes_back=True)
        await asyncio.wait_for(client._run(), timeout=2)

        cb = client._callbacks
        self.assertEqual(self.dials, [self.URL, self.URL])
        self.assertEqual(cb.on_connected.await_count, 2)
        cb.on_client_connected.assert_awaited_once()
        cb.on_client_disconnected.assert_awaited_once()
        cb.on_error.assert_not_awaited()

    async def test_peer_tracks_ending_without_goodbye_and_no_return_report_the_peer_gone(self):
        """A hard hangup also ends the tracks without the marker; the peer
        then fails to appear on the redialed session and is reported gone."""
        self.script[:] = [_FakeSession(), _FakeSession()]
        client = self._make_client()
        self._peer_tracks_end_without_goodbye(client, comes_back=False)
        await asyncio.wait_for(client._run(), timeout=2)

        cb = client._callbacks
        self.assertEqual(self.dials, [self.URL, self.URL])
        self.assertEqual(cb.on_connected.await_count, 2)
        cb.on_client_disconnected.assert_awaited_once()
        cb.on_error.assert_not_awaited()
        self.assertEqual(cb.on_disconnected.await_count, 2)

    async def test_a_peer_that_leaves_ends_the_loop_without_redialing(self):
        self.script[:] = [_FakeSession()]
        client = self._make_client()
        self._peer_leaves_after(client, sessions_seen=1)
        await asyncio.wait_for(client._run(), timeout=2)

        cb = client._callbacks
        self.assertEqual(self.dials, [self.URL])
        cb.on_connected.assert_awaited_once()
        cb.on_client_disconnected.assert_awaited_once()
        cb.on_error.assert_not_awaited()


class _FakeJsonStream:
    """Stands in for a ``JsonStreamConsumer``: yields scripted records, then ends."""

    def __init__(self, records):
        self._records = list(records)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._records:
            raise StopAsyncIteration
        return self._records.pop(0)

    def cancel(self):
        pass


class _FakeAnnounced:
    """Stands in for ``AnnouncedBroadcast``: resolves to a broadcast token at
    once, or never when ``broadcast`` is ``None``."""

    def __init__(self, broadcast):
        self._broadcast = broadcast

    async def available(self):
        if self._broadcast is None:
            await asyncio.Event().wait()
        return self._broadcast

    def cancel(self):
        pass


class TestConsumePeer(unittest.IsolatedAsyncioTestCase):
    """``_consume_peer`` in client mode: one more subscription on the same
    session when the peer's tracks end without its marker, and a
    subscription that serves nothing is given up on."""

    async def asyncSetUp(self):
        p = patch.object(moq_transport, "_PEER_DATA_GRACE_S", 0.05)
        p.start()
        self.addCleanup(p.stop)

    def _client(self, **params) -> MOQTransportClient:
        client, _stream = _client_with_fake_moq(
            MOQParams(audio_in_enabled=True, audio_out_enabled=True, **params)
        )
        client._task_manager = TaskManager()
        return client

    @staticmethod
    def _origin(*broadcasts):
        """An origin whose announcements resolve to each broadcast in turn;
        ``None`` never resolves. The last one repeats."""
        broadcasts = list(broadcasts) or ["peer-broadcast"]
        origin = MagicMock(name="subscribe_origin")

        def announced(_path):
            broadcast = broadcasts.pop(0) if len(broadcasts) > 1 else broadcasts[0]
            return _FakeAnnounced(broadcast)

        origin.consume.return_value.announced_broadcast.side_effect = announced
        return origin

    @staticmethod
    def _tracks(client: MOQTransportClient, outcomes: list[str]):
        """Script ``_forward_peer_tracks``: ``data`` delivers then ends,
        ``silent`` delivers nothing and never ends, ``goodbye`` delivers the
        marker then ends."""
        calls: list[str] = []

        async def forward(_broadcast):
            outcome = outcomes[len(calls)]
            calls.append(outcome)
            if outcome == "silent":
                # Like a real pump: it ends only when its moq consumer is
                # cancelled, which is what the watchdog does.
                ended = asyncio.Event()
                consumer = MagicMock(name="silent_consumer")
                consumer.cancel.side_effect = ended.set
                client._active_consumers.append(consumer)
                await ended.wait()
                return
            await client._on_peer_data()
            if outcome == "goodbye":
                client._peer_goodbye = True

        client._forward_peer_tracks = forward  # type: ignore[method-assign]
        return calls

    async def test_tracks_ending_twice_without_goodbye_drop_the_session(self):
        client = self._client()
        calls = self._tracks(client, ["data", "data"])
        gone = await asyncio.wait_for(client._consume_peer(self._origin()), timeout=2)
        self.assertEqual(calls, ["data", "data"])
        self.assertFalse(gone)
        client._callbacks.on_client_connected.assert_awaited_once()

    async def test_a_peer_that_redialed_carries_on_after_one_more_subscription(self):
        """The peer's tracks end once (its redial), then it says goodbye at
        the end of the call: one retry, then the peer is gone."""
        client = self._client()
        calls = self._tracks(client, ["data", "goodbye"])
        gone = await asyncio.wait_for(client._consume_peer(self._origin()), timeout=2)
        self.assertEqual(calls, ["data", "goodbye"])
        self.assertTrue(gone)

    async def test_goodbye_ends_without_another_subscription(self):
        client = self._client()
        calls = self._tracks(client, ["goodbye", "data"])
        gone = await asyncio.wait_for(client._consume_peer(self._origin()), timeout=2)
        self.assertEqual(calls, ["goodbye"])
        self.assertTrue(gone)

    async def test_a_peer_that_does_not_reappear_is_gone(self):
        """After a hangup without the marker the broadcast never comes
        back; waiting ``connection_timeout`` on this session settles it
        without a redial."""
        client = self._client(connection_timeout=0.05)
        calls = self._tracks(client, ["data"])
        gone = await asyncio.wait_for(
            client._consume_peer(self._origin("peer-broadcast", None)), timeout=2
        )
        self.assertEqual(calls, ["data"])
        self.assertTrue(gone)

    async def test_a_silent_subscription_is_given_up_on(self):
        """A relay keeps announcing a path whose route died and serves
        nothing on it; waiting on it forever would hide the outage."""
        client = self._client()
        calls = self._tracks(client, ["data", "silent"])
        gone = await asyncio.wait_for(client._consume_peer(self._origin()), timeout=2)
        self.assertEqual(calls, ["data", "silent"])
        self.assertFalse(gone)
        self.assertFalse(client._peer_data_seen)

    async def test_the_subscription_a_peer_joins_on_has_no_watchdog(self):
        """A client may hold its first message until it is ready to hear
        the bot, so silence from a peer that has just joined is not a
        failure; from one seen before, on a retry or a redialed session,
        it is."""
        flags: list[bool] = []
        client = self._client()
        real = client._forward_peer

        async def spy(broadcast, watchdog):
            flags.append(watchdog)
            return await real(broadcast, watchdog)

        client._forward_peer = spy  # type: ignore[method-assign]
        self._tracks(client, ["data", "goodbye"])
        await asyncio.wait_for(client._consume_peer(self._origin()), timeout=2)
        self.assertEqual(flags, [False, True])

        # A redialed session, for a peer seen before the outage.
        flags.clear()
        client = self._client()
        client._peer_connected = True
        client._peer_missing_since = time.monotonic()
        real = client._forward_peer
        client._forward_peer = spy  # type: ignore[method-assign]
        self._tracks(client, ["goodbye"])
        await asyncio.wait_for(client._consume_peer(self._origin()), timeout=2)
        self.assertEqual(flags, [True])

    async def test_a_peer_joining_after_failed_dials_has_no_watchdog(self):
        """Dials that failed before the peer was ever seen leave the count
        running; the peer's first subscription is still a join."""
        flags: list[bool] = []
        client = self._client()
        client._peer_missing_since = time.monotonic()
        client._redial_attempt = 2
        real = client._forward_peer

        async def spy(broadcast, watchdog):
            flags.append(watchdog)
            return await real(broadcast, watchdog)

        client._forward_peer = spy  # type: ignore[method-assign]
        self._tracks(client, ["goodbye"])
        await asyncio.wait_for(client._consume_peer(self._origin()), timeout=2)
        self.assertEqual(flags, [False])

    async def test_a_join_stops_the_count_and_an_announcement_in_an_outage_does_not(self):
        client = self._client()
        client._peer_missing_since = time.monotonic()
        await client._on_peer_available()
        self.assertIsNone(client._peer_missing_since)

        client._peer_missing_since = started = time.monotonic()
        await client._on_peer_available()
        self.assertEqual(client._peer_missing_since, started)
        await client._on_peer_data()
        self.assertIsNone(client._peer_missing_since)

    async def test_the_wait_for_the_peer_gets_what_is_left_of_the_count(self):
        """A redialed session does not give a missing peer a fresh
        ``connection_timeout``."""
        client = self._client(connection_timeout=30)
        client._peer_connected = True
        client._peer_missing_since = time.monotonic() - 29.95
        gone = await asyncio.wait_for(client._consume_peer(self._origin(None)), timeout=2)
        self.assertTrue(gone)

    async def test_serve_mode_takes_the_tracks_ending_at_face_value(self):
        client = self._client(serve=True)
        calls = self._tracks(client, ["data", "data"])
        gone = await asyncio.wait_for(client._consume_peer(self._origin()), timeout=2)
        self.assertEqual(calls, ["data"])
        self.assertTrue(gone)


class TestPeerGoodbye(unittest.IsolatedAsyncioTestCase):
    """The peer's session-ending marker is read off its transcript stream and
    never reaches the pipeline."""

    async def test_marker_sets_goodbye_and_is_not_forwarded(self):
        client, _stream = _client_with_fake_moq()
        peer_broadcast = MagicMock()
        peer_broadcast.subscribe_json_stream = AsyncMock(
            return_value=_FakeJsonStream(
                [
                    {"label": "rtvi-ai", "type": "client-ready", "seq": 0, "epoch": "e"},
                    {"label": "moq-transport", "type": "session-ending", "seq": 1, "epoch": "e"},
                ]
            )
        )

        await client._forward_peer_transcript(peer_broadcast)

        client._callbacks.on_message_received.assert_awaited_once_with(
            {"label": "rtvi-ai", "type": "client-ready"}
        )
        self.assertTrue(client._peer_goodbye)

    async def test_tracks_ending_without_the_marker_leave_goodbye_unset(self):
        client, _stream = _client_with_fake_moq()
        peer_broadcast = MagicMock()
        peer_broadcast.subscribe_json_stream = AsyncMock(
            return_value=_FakeJsonStream([{"label": "rtvi-ai", "type": "client-ready"}])
        )

        await client._forward_peer_transcript(peer_broadcast)

        self.assertFalse(client._peer_goodbye)


class TestIsUnauthorized(unittest.TestCase):
    """A refused token shows up either as an HTTP status on the dial or as
    the session closing with moq-net's ``Unauthorized`` code."""

    def test_http_refusals_count(self):
        self.assertTrue(moq_transport._is_unauthorized(moq.Error.Unauthorized("401")))
        self.assertTrue(moq_transport._is_unauthorized(moq.Error.Forbidden("403")))

    def test_unauthorized_session_close_counts(self):
        self.assertTrue(moq_transport._is_unauthorized(moq.Error.Protocol(_UNAUTHORIZED_CLOSE)))

    def test_other_closes_do_not(self):
        for message in (
            "transport: webtransport error: closed: code=4 reason=transport",
            "transport: webtransport error: closed: code=0 reason=remote err",
            "webtransport error: closed",
        ):
            with self.subTest(message=message):
                self.assertFalse(moq_transport._is_unauthorized(moq.Error.Protocol(message)))

    def test_non_moq_errors_do_not(self):
        self.assertFalse(moq_transport._is_unauthorized(ConnectionError(_UNAUTHORIZED_CLOSE)))


# ----------------------------------------------------------------------
# Errors reach the pipeline
# ----------------------------------------------------------------------


class TestErrorsReachThePipeline(unittest.IsolatedAsyncioTestCase):
    """A transport error fires ``on_error`` and pushes an ``ErrorFrame`` from
    the input transport, with the category and permanence the session loop
    worked out."""

    def _transport(self) -> MOQTransport:
        with patch("pipecat.transports.moq.transport.moq") as moq_mock:
            moq_mock.BroadcastProducer.return_value = MagicMock()
            transport = MOQTransport(
                params=MOQParams(audio_in_enabled=True), host="localhost", port=4080
            )
        transport.input()
        transport._input.push_error = AsyncMock()  # type: ignore[method-assign]
        return transport

    async def test_error_is_pushed_from_the_input_transport(self):
        transport = self._transport()
        seen = []
        fired = asyncio.Event()

        @transport.event_handler("on_error")
        async def on_error(_transport, message, exception):
            seen.append((message, exception))
            fired.set()

        exc = moq.Error.Unauthorized("401")
        await transport._on_error("401", exc, ErrorCategory.AUTHENTICATION, False)

        # Event handlers run in a background task.
        await asyncio.wait_for(fired.wait(), timeout=1)
        self.assertEqual(seen, [("401", exc)])
        transport._input.push_error.assert_awaited_once_with(
            "401", exc, category=ErrorCategory.AUTHENTICATION, force_treat_as_permanent=False
        )

    async def test_permanence_is_forwarded(self):
        transport = self._transport()
        exc = ConnectionError("gave up")
        await transport._on_error("gave up", exc, ErrorCategory.CONNECTIVITY, True)
        transport._input.push_error.assert_awaited_once_with(
            "gave up", exc, category=ErrorCategory.CONNECTIVITY, force_treat_as_permanent=True
        )


if __name__ == "__main__":
    unittest.main()
