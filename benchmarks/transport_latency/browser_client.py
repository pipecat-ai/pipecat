"""Browser measuring client: Playwright driver for the web/ bench page.

Two measurement methods, one page (``web/index.html``):

- Method B (in-page, the per-transport comparison number): the page generates
  the probe on its AudioContext clock, publishes it through the pipecat client
  SDK, and records the echoed track with a worklet stamping 20 ms chunks on
  the same clock. This module pulls the capture out of the page and feeds it
  through the shared ``analyze_capture`` — the browser is measured with the
  exact pairing logic used for the headless client.

- Method A (device loopback, the glass-to-glass number): Chrome captures a
  real input device and plays bot audio to a real output device (two BlackHole
  virtual devices); ``AudioDeviceConnector`` plays/records the probe through
  them with ``sounddevice``, so ``run_trial`` measures the full user-felt path
  — OS audio I/O, browser capture/playout, SDK, transport — on Python's
  monotonic clock.

The page is served from ``web/`` by a local static server; it talks to the
bot's HTTP endpoints cross-origin (the dev runner allows all origins).
Build the page first: ``cd web && npm install && npm run build``.
"""

import asyncio
import base64
import collections
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from client_core import TrialResult, analyze_capture, run_trial
from probe import SAMPLE_RATE, gen_probe

WEB_DIR = Path(__file__).parent / "web"
PAGE_PORT = 7871
CHUNK_SAMPLES = SAMPLE_RATE * 20 // 1000


class PageServer:
    """Static server for the built bench page (web/), on a background thread."""

    def __init__(self, port: int = PAGE_PORT) -> None:
        if not (WEB_DIR / "dist" / "bench.js").exists():
            raise RuntimeError(
                f"bench page not built — run: cd {WEB_DIR} && npm install && npm run build"
            )

        class _QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

        handler = partial(_QuietHandler, directory=str(WEB_DIR))
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self.url = f"http://127.0.0.1:{port}"

    def __enter__(self) -> "PageServer":
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()


@dataclass
class BrowserPageConfig:
    """Query parameters handed to the bench page."""

    transport: str  # "smallwebrtc" | "moq"
    method: str  # "a" | "b"
    server: str = "http://localhost:7860"
    jitter_ms: int = 60
    turn_url: str | None = None
    turn_username: str | None = None
    turn_credential: str | None = None
    force_relay: bool = False
    mic_label: str | None = None
    spk_label: str | None = None
    cert_hash_hex: str | None = None  # self-signed MoQ relay fingerprint (hex sha-256)

    def query(self) -> str:
        q = {"transport": self.transport, "method": self.method, "server": self.server}
        q["jitter_ms"] = str(self.jitter_ms)
        if self.cert_hash_hex:
            q["cert_hash_hex"] = self.cert_hash_hex
        if self.turn_url:
            q["turn_url"] = self.turn_url
            q["turn_username"] = self.turn_username or ""
            q["turn_credential"] = self.turn_credential or ""
        if self.force_relay:
            q["force_relay"] = "1"
        if self.mic_label:
            q["mic_label"] = self.mic_label
        if self.spk_label:
            q["spk_label"] = self.spk_label
        return urlencode(q)


@dataclass
class BrowserSession:
    """A connected bench page, ready to run a trial or act as a connector."""

    page: object
    diagnostics: dict = field(default_factory=dict)


CHROME_ARGS = [
    "--autoplay-policy=no-user-gesture-required",
    "--use-fake-ui-for-media-stream",
    # The page is never focused under Playwright; keep timers and rendering
    # at full rate so the audio graph and trial timing never throttle.
    "--disable-renderer-backgrounding",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
]


class BrowserDriver:
    """Owns Playwright + Chrome and one bench page per trial."""

    def __init__(self, headed: bool = False, console_log: Path | None = None) -> None:
        self._headed = headed
        self._console_log = console_log
        self._pw = None
        self._browser = None

    async def __aenter__(self) -> "BrowserDriver":
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            channel="chrome", headless=not self._headed, args=CHROME_ARGS
        )
        # Headless Chrome silently auto-denies the local-network-access
        # permission, which stalls ICE gathering toward loopback TURN/relay
        # targets forever (no candidates, no errors). Grant it up front.
        self._context = await self._browser.new_context()
        await self._context.grant_permissions(["local-network-access"])
        return self

    async def __aexit__(self, *exc) -> None:
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    @property
    def chrome_version(self) -> str | None:
        return self._browser.version if self._browser else None

    async def open(self, base_url: str, config: BrowserPageConfig) -> BrowserSession:
        page = await self._context.new_page()
        if self._console_log:
            log = open(self._console_log, "a")
            page.on("console", lambda m: (log.write(f"{m.type}: {m.text}\n"), log.flush()))
        await page.goto(f"{base_url}/index.html?{config.query()}")
        await page.wait_for_function("window.bench && window.bench.ready")
        await asyncio.wait_for(page.evaluate("bench.setup()"), timeout=40.0)
        return BrowserSession(page=page)

    async def close_page(self, session: BrowserSession) -> None:
        try:
            await asyncio.wait_for(session.page.evaluate("bench.teardown()"), timeout=10.0)
        finally:
            await session.page.close()


def _assert_ice_topology(webrtc: dict | None, force_relay: bool) -> None:
    """Browser twin of the headless client's selected-pair assertion.

    With ``iceTransportPolicy: relay`` Chrome only gathers relay candidates,
    so no direct pair can form regardless of what the stats report — but its
    selected local candidate sometimes shows as ``prflx`` (discovered during
    checks arriving via the TURN allocation; coturn's byte accounting
    confirms the media traverses the relay). Accept both labels.
    """
    if webrtc is None or webrtc.get("icePair") is None:
        raise RuntimeError("no selected ICE pair reported by the page")
    pair = webrtc["icePair"]
    local, remote = pair.get("local"), pair.get("remote")
    if force_relay and local not in ("relay", "prflx"):
        raise RuntimeError(f"TURN scenario got ICE pair {(local, remote)}, expected local relay")
    if not force_relay and (local, remote) != ("host", "host"):
        raise RuntimeError(f"direct scenario got ICE pair {(local, remote)}, expected host/host")


async def run_page_trial(
    driver: BrowserDriver,
    page_url: str,
    config: BrowserPageConfig,
    duration_s: float = 60.0,
    warmup_s: float = 5.0,
    period_ms: int = 250,
    gap_every_s: float | None = None,
    gap_s: float = 2.0,
) -> tuple[TrialResult, dict]:
    """Method B: run one in-page trial; returns (result, diagnostics)."""
    assert config.method == "b"
    session = await driver.open(page_url, config)
    page = session.page
    try:
        run_info = await asyncio.wait_for(
            page.evaluate(
                "opts => bench.run(opts.duration, "
                "{periodMs: opts.periodMs, gapEveryS: opts.gap, gapS: opts.gapS})",
                {
                    "duration": duration_s,
                    "periodMs": period_ms,
                    "gap": gap_every_s,
                    "gapS": gap_s,
                },
            ),
            timeout=duration_s + 60.0,
        )
        res = await asyncio.wait_for(page.evaluate("bench.result()"), timeout=60.0)
    finally:
        await driver.close_page(session)

    # The page's probe must be the same probe (guards the JS port drifting).
    _, marker_positions = gen_probe(
        duration_s, period_ms=period_ms, gap_every_s=gap_every_s, gap_s=gap_s
    )
    if list(marker_positions) != list(run_info["markerPositions"]):
        raise RuntimeError("page probe marker positions differ from probe.py")

    pcm = np.frombuffer(base64.b64decode(res["recvPcmB64"]), dtype=np.int16)
    stamps = list(res["recvStamps"])
    if len(pcm) != CHUNK_SAMPLES * len(stamps):
        raise RuntimeError(f"pcm/stamp mismatch: {len(pcm)} samples, {len(stamps)} stamps")
    send_handoffs = [res["sendT0"] + i * res["chunkS"] for i in range(run_info["nChunks"])]

    analysis = analyze_capture(
        pcm,
        stamps,
        [CHUNK_SAMPLES] * len(stamps),
        marker_positions,
        send_handoffs,
        warmup_s=warmup_s,
        period_ms=period_ms,
    )

    if config.transport == "smallwebrtc":
        _assert_ice_topology(res.get("webrtc"), config.force_relay)

    join_start = res["joinStartT"]
    track_start = res.get("trackStartT")
    result = TrialResult(
        rtts_ms=analysis.rtts_ms,
        drops=analysis.drops,
        ambiguous=analysis.ambiguous,
        markers_sent=analysis.markers_sent,
        spurt_rtts_ms=analysis.spurt_rtts_ms,
        steady_rtts_ms=analysis.steady_rtts_ms,
        series=analysis.series,
        join_ms=(track_start - join_start) * 1000.0 if track_start is not None else None,
        first_echo_ms=(
            (analysis.first_onset_stamp - join_start) * 1000.0
            if analysis.first_onset_stamp is not None
            else None
        ),
        recv_wav=pcm,
        recv_chunk_stats={"n_chunks": len(stamps), "median_chunk_ms": 20.0},
    )
    diagnostics = {
        "webrtc": res.get("webrtc"),
        "moq_config": res.get("moqConfig"),
        "user_agent": res.get("userAgent"),
        "chrome_version": driver.chrome_version,
    }
    return result, diagnostics


# ---------------------------------------------------------------------------
# Method A: OS device loopback through two BlackHole virtual devices.


def _find_device(sd, name: str, *, want_input: bool) -> int:
    matches = [
        i
        for i, d in enumerate(sd.query_devices())
        if name.lower() in d["name"].lower()
        and (d["max_input_channels"] if want_input else d["max_output_channels"]) > 0
    ]
    if not matches:
        kind = "input" if want_input else "output"
        names = [d["name"] for d in sd.query_devices()]
        raise RuntimeError(f"no {kind} device matching {name!r}; devices: {names}")
    return matches[0]


class AudioDeviceConnector:
    """Plays the probe into one audio device and records another.

    ``send_chunk`` queues PCM for a callback-driven output stream (underflow
    plays silence), so handoff stamps stay non-blocking like the network
    connectors; received chunks arrive from the input stream's callback. With
    Chrome capturing the output device as its mic and sinking bot audio to the
    input device, ``run_trial`` measures the true mic-to-speaker round trip.
    """

    def __init__(self, mic_device: str, spk_device: str) -> None:
        self._mic_name = mic_device  # we play the probe INTO this (Chrome's mic)
        self._spk_name = spk_device  # we record FROM this (Chrome's speaker)
        self._pending: collections.deque[np.ndarray] = collections.deque()
        self._leftover: np.ndarray | None = None

    async def start(self) -> None:
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        self._q: asyncio.Queue[bytes] = asyncio.Queue()
        pending = self._pending

        def out_cb(outdata, frames, t, status):
            filled = 0
            if self._leftover is not None:
                n = min(frames, len(self._leftover))
                outdata[:n, 0] = self._leftover[:n]
                self._leftover = self._leftover[n:] if n < len(self._leftover) else None
                filled = n
            while filled < frames and pending:
                chunk = pending.popleft()
                n = min(frames - filled, len(chunk))
                outdata[filled : filled + n, 0] = chunk[:n]
                if n < len(chunk):
                    self._leftover = chunk[n:]
                filled += n
            if filled < frames:
                outdata[filled:, 0] = 0

        def in_cb(indata, frames, t, status):
            loop.call_soon_threadsafe(self._q.put_nowait, bytes(indata))

        self._out = sd.OutputStream(
            device=_find_device(sd, self._mic_name, want_input=False),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=out_cb,
        )
        self._in = sd.InputStream(
            device=_find_device(sd, self._spk_name, want_input=True),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=in_cb,
        )
        self._in.start()
        self._out.start()

    async def send_chunk(self, pcm: bytes) -> None:
        self._pending.append(np.frombuffer(pcm, dtype=np.int16))

    async def recv_chunks(self) -> AsyncIterator[bytes]:
        while True:
            yield await self._q.get()

    async def stop(self) -> None:
        for stream in (self._out, self._in):
            stream.stop()
            stream.close()


async def run_device_trial(
    driver: BrowserDriver,
    page_url: str,
    config: BrowserPageConfig,
    duration_s: float = 60.0,
    warmup_s: float = 5.0,
    period_ms: int = 250,
    gap_every_s: float | None = None,
    gap_s: float = 2.0,
) -> tuple[TrialResult, dict]:
    """Method A: browser connects devices; Python measures through them."""
    assert config.method == "a" and config.mic_label and config.spk_label
    session = await driver.open(page_url, config)
    try:
        connector = AudioDeviceConnector(config.mic_label, config.spk_label)
        result = await run_trial(
            connector,
            duration_s=duration_s,
            warmup_s=warmup_s,
            period_ms=period_ms,
            gap_every_s=gap_every_s,
            gap_s=gap_s,
        )
        res = await asyncio.wait_for(session.page.evaluate("bench.result()"), timeout=30.0)
        if config.transport == "smallwebrtc":
            _assert_ice_topology(res.get("webrtc"), config.force_relay)
        diagnostics = {
            "webrtc": res.get("webrtc"),
            "moq_config": res.get("moqConfig"),
            "user_agent": res.get("userAgent"),
            "chrome_version": driver.chrome_version,
        }
    finally:
        await driver.close_page(session)
    return result, diagnostics
