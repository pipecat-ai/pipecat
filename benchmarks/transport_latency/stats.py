"""Statistics, environment capture, and report rendering for the benchmark."""

import json
import platform
import subprocess
from datetime import UTC, datetime, timezone
from pathlib import Path

import numpy as np
from scenarios import scenario_order


def summarize(rtts_ms: list[float]) -> dict:
    if not rtts_ms:
        return {"n": 0}
    arr = np.array(rtts_ms)
    # RFC 3550-style smoothed jitter over consecutive-marker RTT deltas.
    jitter = 0.0
    for d in np.abs(np.diff(arr)):
        jitter += (d - jitter) / 16.0
    return {
        "n": int(len(arr)),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "std_ms": round(float(arr.std()), 2),
        "rfc3550_jitter_ms": round(float(jitter), 2),
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
    }


def _cmd(args: list[str]) -> str | None:
    try:
        return (
            subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip() or None
        )
    except Exception:
        return None


def capture_environment(repo_root: Path) -> dict:
    import importlib.metadata as md

    def ver(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "pipecat_commit": _cmd(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"]),
        "pipecat_branch": _cmd(["git", "-C", str(repo_root), "branch", "--show-current"]),
        "versions": {
            p: ver(p) for p in ("pipecat-ai", "moq-rs", "moq-ffi", "aiortc", "av", "numpy")
        },
        "moq_relay_image": _cmd(
            ["docker", "inspect", "pipecat-moq-relay-dev", "--format", "{{.Config.Image}}"]
        ),
        "coturn_image": _cmd(
            ["docker", "inspect", "pipecat-coturn-dev", "--format", "{{.Config.Image}}"]
        ),
        "platform": platform.platform(),
        "cpu": _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
    }


def render_summary(results_dir: Path) -> str:
    """Render summary.md from every trial JSON in the results directory."""
    rows = []
    floors: dict[str, dict] = {}
    for f in sorted(results_dir.glob("*.json")):
        data = json.loads(f.read_text())
        if data.get("kind") == "floor":
            floors[data["transport"]] = data["stats"]
        elif "stats" in data:  # skip bot-*.json observer sidecars
            rows.append(data)
    rows.sort(key=lambda r: (scenario_order(r.get("scenario", "")), r.get("trial", 0)))

    lines = [
        "# Transport latency results",
        "",
        "RTT = client->bot->client audio round trip (chirp markers, one",
        "monotonic clock). `excess` = p50 minus the same client stack's",
        "in-process floor — the comparable bot-path number. Scenarios are",
        "topology-symmetric and never blended.",
        "",
        "| scenario | transport | trial | n | drops | p50 ms | p95 ms | p99 ms | jitter ms | excess p50 ms | bot-internal p50 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["stats"]
        floor = floors.get(r["transport"], {}).get("p50_ms")
        excess = round(s["p50_ms"] - floor, 2) if floor is not None and s.get("p50_ms") else None
        bot = (r.get("bot_stats") or {}).get("internal_ms", {}).get("p50")
        lines.append(
            f"| {r['scenario']} | {r['transport']} | {r['trial']} | {s.get('n', 0)} "
            f"| {r.get('drops', 0)} | {s.get('p50_ms', '-')} | {s.get('p95_ms', '-')} "
            f"| {s.get('p99_ms', '-')} | {s.get('rfc3550_jitter_ms', '-')} "
            f"| {excess if excess is not None else '-'} "
            f"| {round(bot, 2) if bot is not None else '-'} |"
        )

    lines += ["", "## Client-stack floors (in-process, no bot, no network)", ""]
    lines += ["| transport | n | p50 ms | p95 ms |", "|---|---|---|---|"]
    for t, s in sorted(floors.items()):
        lines.append(f"| {t} | {s.get('n', 0)} | {s.get('p50_ms', '-')} | {s.get('p95_ms', '-')} |")

    if rows:
        env = rows[-1].get("environment", {})
        lines += [
            "",
            "## Environment",
            "",
            "```json",
            json.dumps(env, indent=2),
            "```",
        ]
    return "\n".join(lines) + "\n"
