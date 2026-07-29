"""CLI + scenario runner for the transport latency benchmark.

Spawns the echo bot as a subprocess, runs the measuring client against it,
and writes per-trial JSON plus ``results/summary.md`` and chart PNGs.

Scenarios are named topologies (see ``scenarios.py``); run them with
``--scenario <slug>`` (repeatable) or ``--scenario all-local``.

Relay containers are never auto-started; the runner checks they're up and
prints the command to run them (foreground, own terminal) if not:
  - moq-relay-local:   pipecat-moq-relay/moq-relay-dev.sh relay    (moq-relay, :4443)
  - webrtc-turn-local: benchmarks/transport_latency/coturn-local/run.sh   (coturn, :3478)

Deployed-tier scenarios need external endpoints:
  - moq-relay-deployed:   --relay-url https://<your-relay>/<path>
  - webrtc-turn-deployed: --turn-url/--turn-username/--turn-credential

Examples:
    uv run python benchmarks/transport_latency/transport_latency.py --scenario all-local
    uv run python benchmarks/transport_latency/transport_latency.py \
        --scenario moq-relay-deployed --relay-url https://relay.example.com/anon
    uv run python benchmarks/transport_latency/transport_latency.py --floors
"""

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from client_core import run_trial
from scenarios import ALL_LOCAL, SCENARIOS, Scenario
from stats import capture_environment, render_summary, summarize

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
RESULTS = HERE / "results"
MOQ_RELAY_URL = "https://localhost:4443/anon"
MOQ_RELAY_HTTP = "http://localhost:4443/certificate.sha256"


def _port_open_udp_hint_tcp(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


async def _wait_for_port_free(port: int, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _port_open_udp_hint_tcp(port):
            return
        await asyncio.sleep(0.25)
    raise RuntimeError(f"port {port} still in use — a previous bot did not shut down")


def _stop_bot(bot: subprocess.Popen) -> None:
    """SIGINT the bot's whole process group; escalate to SIGKILL on timeout."""
    try:
        os.killpg(bot.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        bot.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(bot.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        bot.wait(timeout=5)


async def _wait_for_bot(base_url: str, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    async with aiohttp.ClientSession() as http:
        while time.monotonic() < deadline:
            try:
                async with http.get(base_url, timeout=aiohttp.ClientTimeout(total=2)):
                    return
            except Exception:
                await asyncio.sleep(0.5)
    raise RuntimeError(f"bot did not come up at {base_url}")


class TurnCredentials:
    """One TURN server: URL(s) + shared username/credential."""

    def __init__(self, urls: list[str], username: str, credential: str) -> None:
        self.urls = urls
        self.username = username
        self.credential = credential

    @property
    def udp_url(self) -> str:
        """The UDP turn: URL (matches local coturn's transport)."""
        for url in self.urls:
            if url.startswith("turn:") and ("transport=udp" in url or "?" not in url):
                return url
        raise RuntimeError(f"no UDP turn: URL among {self.urls}")


_turn_cache: TurnCredentials | None = None


async def _turn_credentials(args: argparse.Namespace) -> TurnCredentials:
    """Resolve TURN credentials for webrtc-turn-deployed (cached per run)."""
    global _turn_cache
    if _turn_cache is not None:
        return _turn_cache
    if not args.turn_url:
        raise RuntimeError(
            "webrtc-turn-deployed needs --turn-url/--turn-username/--turn-credential"
        )
    if not (args.turn_username and args.turn_credential):
        raise RuntimeError("--turn-url needs --turn-username and --turn-credential")
    _turn_cache = TurnCredentials([args.turn_url], args.turn_username, args.turn_credential)
    return _turn_cache


def _require_local_moq_relay() -> str:
    """Check the local moq-relay is up; returns its cert fingerprint."""
    try:
        return urllib.request.urlopen(MOQ_RELAY_HTTP, timeout=3).read().decode().strip()
    except Exception as e:
        raise RuntimeError(
            "moq relay not reachable on :4443 — start it in another terminal: "
            "pipecat-moq-relay/moq-relay-dev.sh relay"
        ) from e


def _require_local_coturn() -> None:
    if not _port_open_udp_hint_tcp(3478):
        raise RuntimeError(
            "coturn not reachable on :3478 — start it in another terminal: "
            "benchmarks/transport_latency/coturn-local/run.sh"
        )


async def _bot_command(scenario: Scenario, args: argparse.Namespace) -> list[str]:
    cmd = ["uv", "run", "python", str(HERE / "echo_bot.py")]
    slug = scenario.slug
    if slug in ("webrtc-local", "webrtc-turn-local"):
        cmd += ["-t", "webrtc"]
        if slug == "webrtc-turn-local":
            _require_local_coturn()
    elif slug == "webrtc-turn-deployed":
        # A public TURN can't reach the bot's private host candidates, so the
        # bot needs the TURN server too (srflx/relay candidates).
        turn = await _turn_credentials(args)
        cmd += ["-t", "webrtc"]
        cmd += ["--webrtc-ice-url", turn.udp_url]
        cmd += ["--webrtc-ice-username", turn.username]
        cmd += ["--webrtc-ice-credential", turn.credential]
    elif slug == "moq-serve":
        cmd += ["-t", "moq"]
    elif slug == "moq-relay-local":
        _require_local_moq_relay()
        cmd += ["-t", "moq", "--moq-connect", MOQ_RELAY_URL, "--moq-tls-insecure"]
    elif slug == "moq-relay-deployed":
        if not args.relay_url:
            raise RuntimeError("moq-relay-deployed needs --relay-url https://<relay>/<path>")
        cmd += ["-t", "moq", "--moq-connect", args.relay_url]
    else:
        raise RuntimeError(f"no local bot command for scenario {slug}")
    return cmd


async def _make_connector(scenario: Scenario, args: argparse.Namespace):
    if scenario.transport == "moq":
        from moq_client import MoqConnector

        # The local relay's cert is self-signed (dev fallback when mkcert is
        # absent); dial insecure like the bot does (--moq-tls-insecure).
        tls_verify = False if scenario.slug == "moq-relay-local" else None
        return MoqConnector(jitter_ms=args.jitter_ms, tls_verify=tls_verify)
    if scenario.transport == "webrtc":
        from aiortc import RTCIceServer
        from webrtc_client import WebRTCConnector

        if scenario.slug == "webrtc-local":
            return WebRTCConnector(turn=None)
        if scenario.slug == "webrtc-turn-local":
            turn = RTCIceServer(
                urls="turn:127.0.0.1:3478", username="pipecat", credential="pipecat"
            )
        else:
            creds = await _turn_credentials(args)
            turn = RTCIceServer(
                urls=creds.udp_url, username=creds.username, credential=creds.credential
            )
        return WebRTCConnector(turn=turn)
    raise RuntimeError(f"no connector for transport {scenario.transport}")


def _floor_connector(transport: str, jitter_ms: int):
    if transport == "moq":
        from moq_client import MoqLocalConnector

        return MoqLocalConnector(jitter_ms=jitter_ms)
    from webrtc_client import LoopbackConnector

    return LoopbackConnector()


async def run_floor(transport: str, jitter_ms: int, duration: float = 15.0) -> dict:
    result = await run_trial(_floor_connector(transport, jitter_ms), duration_s=duration)
    stats = summarize(result.rtts_ms)
    out = {"kind": "floor", "transport": transport, "stats": stats, "jitter_ms": jitter_ms}
    (RESULTS / f"floor-{transport}.json").write_text(json.dumps(out, indent=2))
    print(f"floor {transport}: {stats}")
    return out


async def run_scenario(scenario: Scenario, args: argparse.Namespace, env_info: dict) -> None:
    bot_command = await _bot_command(scenario, args)
    for trial in range(1, args.trials + 1):
        bot_stats_path = RESULTS / f"bot-{scenario.slug}-{trial}.json"
        env = os.environ | {
            "BENCH_BOT_STATS": str(bot_stats_path),
            "BENCH_JITTER_MS": str(args.jitter_ms),
        }
        await _wait_for_port_free(7860)
        # New session so teardown can signal the whole group — `uv run` wraps
        # the python bot, and signalling only the wrapper leaks the bot.
        bot_log = open(RESULTS / f"bot-{scenario.slug}-{trial}.log", "wb")
        bot = subprocess.Popen(
            bot_command,
            cwd=REPO_ROOT,
            env=env,
            stdout=bot_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        bot_log.close()
        try:
            await _wait_for_bot("http://localhost:7860/")
            await asyncio.sleep(2.0)
            connector = await _make_connector(scenario, args)
            result = await run_trial(connector, duration_s=args.duration)
        finally:
            _stop_bot(bot)

        bot_stats = None
        if bot_stats_path.exists():
            bot_stats = json.loads(bot_stats_path.read_text())

        if args.save_wav and result.recv_wav is not None:
            import wave

            for name, data in (("sent", result.sent_wav), ("recv", result.recv_wav)):
                with wave.open(str(RESULTS / f"{scenario.slug}-{trial}-{name}.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(48000)
                    w.writeframes(np.asarray(data, dtype=np.int16).tobytes())

        out = {
            "scenario": scenario.slug,
            "transport": scenario.transport,
            "group": scenario.group,
            "trial": trial,
            "drops": result.drops,
            "ambiguous": result.ambiguous,
            "markers_sent": result.markers_sent,
            "jitter_ms_config": args.jitter_ms,
            "stats": summarize(result.rtts_ms),
            "bot_stats": bot_stats,
            "ice_pair": getattr(connector, "selected_ice_pair", None),
            "moq_relay_url": getattr(connector, "moq_config", {}).get("relayUrl"),
            "environment": env_info,
        }
        (RESULTS / f"{scenario.slug}-{trial}.json").write_text(json.dumps(out, indent=2))
        print(f"{scenario.slug} trial {trial}: {out['stats']} drops={result.drops}")
        await asyncio.sleep(1.0)


def _render_outputs() -> None:
    (RESULTS / "summary.md").write_text(render_summary(RESULTS))
    print(f"\nsummary -> {RESULTS / 'summary.md'}")
    try:
        from charts import render_charts
    except ImportError as e:
        print(f"charts skipped ({e}) — install with: uv sync --group bench")
        return
    for png in render_charts(RESULTS):
        print(f"chart   -> {png}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[*SCENARIOS, "all-local"],
        help="Scenario slug (repeatable). 'all-local' = "
        + ", ".join(ALL_LOCAL)
        + ". See scenarios.py.",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--jitter-ms", type=int, default=60)
    parser.add_argument("--save-wav", action="store_true")
    parser.add_argument("--relay-url", help="Deployed moq-relay URL (moq-relay-deployed scenario)")
    parser.add_argument("--turn-url", help="TURN URL for webrtc-turn-deployed")
    parser.add_argument("--turn-username")
    parser.add_argument("--turn-credential")
    parser.add_argument(
        "--floors", action="store_true", help="run only the client-stack floor trials"
    )
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)

    if args.floors:
        for t in ("moq", "webrtc"):
            await run_floor(t, args.jitter_ms)
        _render_outputs()
        return

    if not args.scenario:
        parser.error("--scenario is required (or use --floors)")
    slugs: list[str] = []
    for s in args.scenario:
        for slug in ALL_LOCAL if s == "all-local" else [s]:
            if slug not in slugs:
                slugs.append(slug)

    env_info = capture_environment(REPO_ROOT)
    for slug in slugs:
        scenario = SCENARIOS[slug]
        if scenario.floor_key:
            floor_file = RESULTS / f"floor-{scenario.floor_key}.json"
            if not floor_file.exists():
                await run_floor(scenario.floor_key, args.jitter_ms)
        await run_scenario(scenario, args, env_info)

    _render_outputs()


if __name__ == "__main__":
    asyncio.run(main())
