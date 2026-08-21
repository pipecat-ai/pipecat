#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Exercise a Pipecat service class against a provider with a given model.

``run`` constructs the service exactly as a user would — credentials from the
environment, ``settings=Cls.Settings(model=...)`` — and pushes one real turn
through a three-processor pipeline, reporting whether the expected output frame
arrived and how long it took:

- LLM: a user message → first ``LLMTextFrame``
- TTS: a ``TTSSpeakFrame`` → first ``TTSAudioRawFrame``
- STT: a bundled 16 kHz speech clip → first ``TranscriptionFrame``
- realtime (speech-to-speech): connect on ``StartFrame`` and stay error-free

Latency comes from the service's own metrics when it emits them (``ttfb_ms``;
for LLMs also ``ttfat_ms``, time to the first *answer* token, and
``thinking_ms`` spent on reasoning), falling back to wall-clock from the
request to the first output frame. Compare LLM candidates on ``ttfat_ms``.

``list-models`` queries the provider's model catalogue where one exists
(OpenAI-compatible ``/models``, Anthropic, Google, Deepgram, ElevenLabs).

``signals`` gathers the cheap change signals a researcher compares against the
previous report: the latest PyPI version of each SDK the provider's extra
depends on, and a content hash of each published API spec listed for the
provider in ``providers.yaml`` (or passed with ``--spec``). Specs are
snapshotted under the reports checkout so ``git diff`` shows what changed.

Credentials come from the repo's ``.env`` loaded *without* override, so exported
variables win and CI runs without a ``.env`` at all (``--no-dotenv`` ignores it). Every value
of a secret-looking environment variable is redacted from output. Only the
variable *names* of missing credentials are printed. Run::

    uv run python scripts/provider-watch/probe.py run --service CartesiaTTSService --model sonic-3.5
    uv run python scripts/provider-watch/probe.py run --service OpenAILLMService --model gpt-4.1 --model gpt-5 --json
    uv run python scripts/provider-watch/probe.py list-models --provider groq
    uv run python scripts/provider-watch/probe.py signals --provider deepgram

Exit codes: 0 all probes passed · 1 a probe failed · 2 missing credentials ·
3 unsupported (no probe for this service type / no model catalogue for this provider).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request
import wave
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import inventory  # noqa: E402

DEFAULT_WAV = HERE / "assets" / "speech-16k.wav"
PROVIDERS_YAML = inventory.REPO_ROOT / ".claude" / "skills" / "provider-watch" / "providers.yaml"
PYPROJECT = inventory.REPO_ROOT / "pyproject.toml"
DEFAULT_SPECS_DIR = inventory.REPO_ROOT / "_reports" / "specs"
DEFAULT_TEXT = "In one short sentence, what is the capital of France?"
DEFAULT_TTS_TEXT = "Hello from Pipecat. This is a short synthesis check."
MAX_MODELS_PER_RUN = 3
SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH|PRIVATE", re.IGNORECASE)

# Constructor parameter → environment-variable suffix, matching the CLI
# scaffolder's convention (``scripts/cli/configs/config_generator.py``).
PARAM_TO_ENV_SUFFIX = {
    "api_key": "API_KEY",
    "credentials": "APPLICATION_CREDENTIALS",
    "credentials_path": "TEST_CREDENTIALS",
    "region": "REGION",
    "region_name": "REGION",
    "aws_region": "REGION",
    "voice_id": "VOICE_ID",
    "voice": "VOICE_ID",
    "replica_id": "REPLICA_ID",
    "face_id": "FACE_ID",
    "model": "MODEL",
    "base_url": "BASE_URL",
    "endpoint": "ENDPOINT",
}

EXIT_OK, EXIT_FAILED, EXIT_MISSING_ENV, EXIT_UNSUPPORTED = 0, 1, 2, 3


@dataclass
class Timings:
    """Latency figures gathered while a probe runs, in seconds."""

    wall_clock_ttfb: float | None = None
    ttfb: float | None = None
    ttfat: float | None = None
    thinking: float | None = None


@dataclass
class ProbeResult:
    """Outcome of one service × model probe."""

    service: str
    model: str | None
    ok: bool
    ttfb_ms: int | None
    ttfat_ms: int | None
    thinking_ms: int | None
    frames: dict[str, int]
    error: str | None
    first_text: str | None = None
    note: str | None = None


# ---------------------------------------------------------------------- helpers


def _redactor() -> Callable[[str | None], str | None]:
    secrets = sorted(
        {v for k, v in os.environ.items() if SECRET_ENV.search(k) and len(v) >= 8},
        key=len,
        reverse=True,
    )

    def redact(text: str | None) -> str | None:
        if text is None:
            return None
        for value in secrets:
            text = text.replace(value, "***")
        return text

    return redact


def _parse_literal(value: str) -> Any:
    """``key=value`` values: JSON for dicts/lists, then bool/None/int/float, else the string."""
    if value[:1] in "{[":
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise SystemExit(f"expected JSON for {value!r}: {e}") from None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _kv_pairs(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {item!r}")
        out[key] = _parse_literal(value)
    return out


def _find_unit(service: str) -> tuple[inventory.Unit, inventory.ServiceClass]:
    units = inventory.scan_services()
    inventory.enrich(units)
    for unit in units:
        for cls in unit.classes:
            if cls.name == service:
                return unit, cls
    raise SystemExit(f"unknown service class {service!r}; see inventory.py --md")


def _registry_entry(unit: inventory.Unit, service: str) -> dict | None:
    for entry in unit.registry:
        if service in entry.get("class_names", []):
            return entry
    return unit.registry[0] if unit.registry else None


def _env_prefix(unit: inventory.Unit, entry: dict | None) -> str:
    if entry and entry.get("env_prefix"):
        return entry["env_prefix"]
    return unit.provider.upper().replace("-", "_")


def build_kwargs(
    cls: type,
    unit: inventory.Unit,
    entry: dict | None,
    *,
    model: str | None,
    settings_overrides: dict[str, Any],
    kwarg_overrides: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Constructor kwargs from the environment plus overrides; also the missing env names."""
    params = inspect.signature(cls.__init__).parameters
    prefix = _env_prefix(unit, entry)
    include = list((entry or {}).get("include_params") or [])
    if "api_key" in params and "api_key" not in include:
        include.insert(0, "api_key")

    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for name in include:
        if name not in params or name in kwarg_overrides:
            continue
        env_name = f"{prefix}_{PARAM_TO_ENV_SUFFIX.get(name, name.upper())}"
        value = os.environ.get(env_name)
        if value:
            kwargs[name] = value
        elif params[name].default is inspect.Parameter.empty:
            missing.append(env_name)
    kwargs.update(kwarg_overrides)

    settings_cls = getattr(cls, "Settings", None)
    if settings_cls is not None and "settings" in params:
        fields = {
            f for f in getattr(settings_cls, "__dataclass_fields__", {}) if not f.startswith("_")
        }
        settings_kwargs: dict[str, Any] = {}
        for key, value in ((entry or {}).get("param_defaults") or {}).items():
            if key in fields and key != "model":
                settings_kwargs[key] = value
        if model is not None and "model" in fields:
            settings_kwargs["model"] = model
        settings_kwargs.update({k: v for k, v in settings_overrides.items() if k in fields})
        unknown = set(settings_overrides) - fields
        if unknown:
            raise SystemExit(f"{settings_cls.__name__} has no field(s): {sorted(unknown)}")
        kwargs["settings"] = settings_cls(**settings_kwargs)
    elif model is not None and "model" in params:
        kwargs["model"] = model

    return kwargs, missing


def _read_wav(path: Path) -> tuple[bytes, int, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit PCM")
        return wav.readframes(wav.getnframes()), wav.getframerate(), wav.getnchannels()


# ---------------------------------------------------------------------- harness


async def _run_pipeline(
    service: Any,
    *,
    frames_to_send: list[Any],
    target: type | None,
    timeout: float,
    start_timeout: float,
    audio_in_sample_rate: int,
    settle: float = 0.0,
) -> tuple[dict[str, int], Timings, list[str], str | None]:
    """Push frames through ``[source, service, sink]``.

    Returns frame counts by type, the timings (service metrics plus wall-clock to
    the first ``target`` frame), upstream error messages, and the ``text`` of
    that first target frame when it has one.
    """
    from pipecat.frames.frames import EndFrame, ErrorFrame, Frame, MetricsFrame
    from pipecat.metrics.metrics import TTFATMetricsData, TTFBMetricsData
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.workers.runner import WorkerRunner

    counts: dict[str, int] = {}
    timings = Timings()
    errors: list[str] = []
    hit = asyncio.Event()
    failed = asyncio.Event()
    started = asyncio.Event()
    first_at: float | None = None
    first_text: str | None = None
    sent_at: float | None = None

    class Source(FrameProcessor):
        def __init__(self):
            super().__init__(enable_direct_mode=True)

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if direction == FrameDirection.UPSTREAM and isinstance(frame, ErrorFrame):
                errors.append(str(frame.error))
                failed.set()
            await self.push_frame(frame, direction)

    class Sink(FrameProcessor):
        def __init__(self):
            super().__init__(enable_direct_mode=True)

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            nonlocal first_at, first_text
            await super().process_frame(frame, direction)
            if direction == FrameDirection.DOWNSTREAM:
                name = type(frame).__name__
                counts[name] = counts.get(name, 0) + 1
                if isinstance(frame, MetricsFrame):
                    # Services emit a zeroed TTFB when they start; the measurement
                    # is the first non-zero one.
                    for data in frame.data:
                        if isinstance(data, TTFBMetricsData) and timings.ttfb is None:
                            timings.ttfb = data.value or None
                        elif isinstance(data, TTFATMetricsData) and timings.ttfat is None:
                            timings.ttfat = data.ttfat
                            timings.thinking = data.thinking_time
                if target is not None and isinstance(frame, target) and first_at is None:
                    first_at = time.monotonic()
                    first_text = getattr(frame, "text", None)
                    hit.set()
            await self.push_frame(frame, direction)

    pipeline = Pipeline([Source(), service, Sink()])
    worker = PipelineWorker(
        pipeline,
        cancel_on_idle_timeout=False,
        params=PipelineParams(audio_in_sample_rate=audio_in_sample_rate, enable_metrics=True),
    )

    @worker.event_handler("on_pipeline_started")
    async def _on_started(worker, frame):
        started.set()

    async def drive():
        nonlocal sent_at
        try:
            await asyncio.wait_for(started.wait(), timeout=start_timeout)
        except TimeoutError:
            errors.append(f"pipeline did not start within {start_timeout:.0f}s")
            await worker.cancel()
            return
        sent_at = time.monotonic()
        for frame in frames_to_send:
            if isinstance(frame, float):
                await asyncio.sleep(frame)
            else:
                await worker.queue_frame(frame)
        waiters = [asyncio.create_task(hit.wait()), asyncio.create_task(failed.wait())]
        if target is None:
            waiters.pop(0)
        done, pending = await asyncio.wait(
            waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if not done and target is not None:
            errors.append(f"no {target.__name__} within {timeout:.0f}s")
        if settle:
            await asyncio.sleep(settle)
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    try:
        await asyncio.wait_for(
            asyncio.gather(runner.run(), drive()), timeout=timeout + start_timeout + 30
        )
    except TimeoutError:
        errors.append("pipeline did not shut down; cancelled")
        await worker.cancel()

    if first_at and sent_at:
        timings.wall_clock_ttfb = first_at - sent_at
    return counts, timings, errors, first_text


def _frames_for(unit_type: str, args: argparse.Namespace) -> tuple[list[Any], type | None, float]:
    """Frames to send, the frame type that counts as success, and a post-hit settle time."""
    from pipecat.frames.frames import (
        InputAudioRawFrame,
        LLMContextFrame,
        LLMTextFrame,
        TranscriptionFrame,
        TTSAudioRawFrame,
        TTSSpeakFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.processors.aggregators.llm_context import LLMContext

    if unit_type == "llm":
        context = LLMContext(messages=[{"role": "user", "content": args.text or DEFAULT_TEXT}])
        return [LLMContextFrame(context=context)], LLMTextFrame, 0.0
    if unit_type == "tts":
        return [TTSSpeakFrame(text=args.text or DEFAULT_TTS_TEXT)], TTSAudioRawFrame, 0.0
    if unit_type == "stt":
        audio, sample_rate, channels = _read_wav(Path(args.wav or DEFAULT_WAV))
        chunk = int(sample_rate * channels * 2 * 0.02)  # 20 ms
        frames: list[Any] = [VADUserStartedSpeakingFrame()]
        for offset in range(0, len(audio), chunk):
            frames.append(InputAudioRawFrame(audio[offset : offset + chunk], sample_rate, channels))
        silence = bytes(chunk)
        frames.extend(InputAudioRawFrame(silence, sample_rate, channels) for _ in range(100))
        frames.append(VADUserStoppedSpeakingFrame())
        return frames, TranscriptionFrame, 0.0
    if unit_type == "realtime":
        return [], None, 0.0
    raise SystemExit(EXIT_UNSUPPORTED)


async def probe_one(
    unit: inventory.Unit, cls_info: inventory.ServiceClass, model: str | None, args
) -> ProbeResult:
    redact = _redactor()
    module = importlib.import_module(cls_info.module)
    cls = getattr(module, cls_info.name)
    entry = _registry_entry(unit, cls_info.name)
    try:
        kwargs, missing = build_kwargs(
            cls,
            unit,
            entry,
            model=model,
            settings_overrides=_kv_pairs(args.setting),
            kwarg_overrides=_kv_pairs(args.kwarg),
        )
    except TypeError as e:
        return ProbeResult(
            cls_info.name, model, False, None, None, None, {}, redact(f"settings: {e}")
        )
    if missing:
        print(f"missing environment variable(s): {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(EXIT_MISSING_ENV)

    try:
        service = cls(**kwargs)
    except Exception as e:  # constructor rejected something — that is a finding
        return ProbeResult(
            cls_info.name, model, False, None, None, None, {}, redact(f"constructor: {e!r}")
        )

    frames, target, settle = _frames_for(unit.type, args)
    sample_rate = 16000
    if unit.type == "stt":
        _, sample_rate, _ = _read_wav(Path(args.wav or DEFAULT_WAV))

    timeout = args.timeout
    note = None
    if unit.type == "realtime":
        timeout = min(args.timeout, 8.0)
        note = "connect-only: passes when no ErrorFrame arrives after StartFrame"

    try:
        counts, timings, errors, first_text = await _run_pipeline(
            service,
            frames_to_send=frames,
            target=target,
            timeout=timeout,
            start_timeout=args.start_timeout,
            audio_in_sample_rate=sample_rate,
            settle=settle,
        )
    except Exception as e:
        return ProbeResult(
            cls_info.name, model, False, None, None, None, {}, redact(f"pipeline: {e!r}")
        )

    def ms(value: float | None) -> int | None:
        return int(value * 1000) if value is not None else None

    ok = not errors and (target is None or counts.get(target.__name__, 0) > 0)
    return ProbeResult(
        service=cls_info.name,
        model=model,
        ok=ok,
        ttfb_ms=ms(timings.ttfb if timings.ttfb is not None else timings.wall_clock_ttfb),
        ttfat_ms=ms(timings.ttfat),
        thinking_ms=ms(timings.thinking),
        frames=counts,
        error=redact("; ".join(errors)) if errors else None,
        first_text=redact(first_text),
        note=note,
    )


def _load_env(args: argparse.Namespace) -> None:
    if args.no_dotenv:
        return
    from dotenv import load_dotenv

    load_dotenv(inventory.REPO_ROOT / ".env")


def cmd_run(args: argparse.Namespace) -> int:
    from loguru import logger

    _load_env(args)
    logger.remove()
    logger.add(sys.stderr, level="ERROR" if not args.verbose else "DEBUG")

    unit, cls_info = _find_unit(args.service)
    if unit.type not in {"llm", "tts", "stt", "realtime"}:
        print(f"{args.service} is a {unit.type} service: research-only, no probe", file=sys.stderr)
        return EXIT_UNSUPPORTED

    models = args.model or [None]
    if len(models) > MAX_MODELS_PER_RUN:
        raise SystemExit(f"at most {MAX_MODELS_PER_RUN} models per invocation")

    results = [asyncio.run(probe_one(unit, cls_info, model, args)) for model in models]
    for result in results:
        if args.json:
            print(json.dumps(asdict(result)))
        else:
            status = "ok " if result.ok else "FAIL"
            ttfb = f"{result.ttfb_ms} ms" if result.ttfb_ms is not None else "—"
            line = f"[{status}] {result.service} model={result.model or '(default)'} ttfb={ttfb}"
            if result.ttfat_ms is not None:
                line += f" ttfat={result.ttfat_ms} ms (thinking {result.thinking_ms} ms)"
            print(line)
            if result.first_text:
                print(f"       text: {result.first_text!r}")
            if result.note:
                print(f"       note: {result.note}")
            if result.error:
                print(f"       error: {result.error}")
            if result.frames:
                print(f"       frames: {result.frames}")
    return EXIT_OK if all(r.ok for r in results) else EXIT_FAILED


# ------------------------------------------------------------------ list-models


def _http_json(url: str, headers: dict[str, str]) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": "pipecat-provider-watch", **headers}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def _openai_compatible(base_url: str, api_key: str | None) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = _http_json(base_url.rstrip("/") + "/models", headers)
    items = data.get("data", data) if isinstance(data, dict) else data
    return sorted(str(item.get("id") or item.get("name")) for item in items)


def _anthropic(api_key: str) -> list[str]:
    ids: list[str] = []
    url = "https://api.anthropic.com/v1/models?limit=100"
    while url:
        data = _http_json(url, {"x-api-key": api_key, "anthropic-version": "2023-06-01"})
        ids.extend(item["id"] for item in data.get("data", []))
        url = (
            f"https://api.anthropic.com/v1/models?limit=100&after_id={data['last_id']}"
            if data.get("has_more") and data.get("last_id")
            else None
        )
    return sorted(ids)


def _google(api_key: str) -> list[str]:
    ids: list[str] = []
    token = None
    while True:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?pageSize=200&key={api_key}"
        if token:
            url += f"&pageToken={token}"
        data = _http_json(url, {})
        ids.extend(item["name"].removeprefix("models/") for item in data.get("models", []))
        token = data.get("nextPageToken")
        if not token:
            return sorted(ids)


def _deepgram(api_key: str) -> list[str]:
    data = _http_json("https://api.deepgram.com/v1/models", {"Authorization": f"Token {api_key}"})
    ids = {
        m.get("canonical_name") or m.get("name")
        for kind in ("stt", "tts")
        for m in data.get(kind, [])
    }
    return sorted(i for i in ids if i)


def _elevenlabs(api_key: str) -> list[str]:
    data = _http_json("https://api.elevenlabs.io/v1/models", {"xi-api-key": api_key})
    return sorted(item["model_id"] for item in data)


PROVIDER_FETCHERS = {
    "anthropic": ("ANTHROPIC_API_KEY", _anthropic),
    "google": ("GOOGLE_API_KEY", _google),
    "deepgram": ("DEEPGRAM_API_KEY", _deepgram),
    "elevenlabs": ("ELEVENLABS_API_KEY", _elevenlabs),
}


def cmd_list_models(args: argparse.Namespace) -> int:
    _load_env(args)
    redact = _redactor()
    provider = args.provider

    try:
        if provider in PROVIDER_FETCHERS:
            env_name, fetch = PROVIDER_FETCHERS[provider]
            api_key = os.environ.get(env_name)
            if not api_key:
                print(f"missing environment variable: {env_name}", file=sys.stderr)
                return EXIT_MISSING_ENV
            models = fetch(api_key)
        else:
            units = inventory.scan_services()
            inventory.enrich(units)
            unit = next((u for u in units if u.id == f"{provider}/llm"), None)
            if unit is None:
                print(f"no model catalogue known for {provider!r}", file=sys.stderr)
                return EXIT_UNSUPPORTED
            base_url = args.base_url or next((c.base_url for c in unit.classes if c.base_url), None)
            if base_url is None and provider == "openai":
                base_url = "https://api.openai.com/v1"
            if base_url is None:
                print(f"no OpenAI-compatible base_url known for {provider!r}", file=sys.stderr)
                return EXIT_UNSUPPORTED
            env_name = f"{_env_prefix(unit, _registry_entry(unit, unit.classes[0].name))}_API_KEY"
            api_key = os.environ.get(env_name)
            if not api_key and provider != "ollama":
                print(f"missing environment variable: {env_name}", file=sys.stderr)
                return EXIT_MISSING_ENV
            models = _openai_compatible(base_url, api_key)
    except urllib.error.HTTPError as e:
        print(redact(f"{provider}: HTTP {e.code} from {e.url}"), file=sys.stderr)
        return EXIT_UNSUPPORTED if e.code in (404, 405) else EXIT_FAILED
    except Exception as e:
        print(redact(f"{provider}: {e!r}"), file=sys.stderr)
        return EXIT_FAILED

    if args.json:
        print(json.dumps({"provider": provider, "models": models}))
    else:
        print("\n".join(models))
    return EXIT_OK


# ---------------------------------------------------------------------- signals


def _http_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "pipecat-provider-watch"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def provider_hints(provider: str) -> dict:
    """The provider's entry in providers.yaml, or an empty dict."""
    import yaml

    if not PROVIDERS_YAML.exists():
        return {}
    data = yaml.safe_load(PROVIDERS_YAML.read_text()) or {}
    return data.get(provider) or {}


def sdk_requirements(provider: str, units: list[inventory.Unit]) -> list[str]:
    """Requirement strings from pyproject extras backing this provider's services.

    The extras are the provider directory name and whatever the CLI registry lists
    for its units; providers with no SDK of their own (OpenAI-compatible wrappers
    and OpenAI itself) fall back to the core ``openai`` dependency.
    """
    project = tomllib.load(PYPROJECT.open("rb"))["project"]
    extras = project.get("optional-dependencies", {})
    names = {provider.replace("_", "-")}
    for unit in units:
        for entry in unit.registry:
            package = entry.get("package") or ""
            if "[" in package:
                names.update(e.strip() for e in package.split("[")[1].split("]")[0].split(","))
    requirements: list[str] = []
    for name in sorted(names):
        for requirement in extras.get(name, []):
            if requirement.startswith("pipecat-ai[") or requirement in requirements:
                continue
            requirements.append(requirement)
    if not requirements and (provider == "openai" or any(u.is_thin_wrapper for u in units)):
        requirements.extend(
            r
            for r in project["dependencies"]
            if r.split("[")[0].split(">")[0].split("<")[0].split("=")[0].strip() == "openai"
        )
    return requirements


def pypi_latest(requirement: str) -> dict:
    from packaging.requirements import Requirement

    name = Requirement(requirement).name
    try:
        data = json.loads(_http_bytes(f"https://pypi.org/pypi/{name}/json"))
    except Exception as e:  # network or unknown package
        return {"package": name, "requirement": requirement, "error": str(e)}
    version = data["info"]["version"]
    files = data.get("releases", {}).get(version) or []
    released = max(
        (f.get("upload_time_iso_8601") or f.get("upload_time") or "" for f in files), default=""
    )
    return {
        "package": name,
        "requirement": requirement,
        "latest": version,
        "released": released[:10],
    }


def spec_snapshot(name: str, url: str, snapshot_dir: Path | None) -> dict:
    """Fetch one API spec, hash it, and note whether it differs from the snapshot on disk."""
    try:
        content = _http_bytes(url)
    except Exception as e:
        return {"name": name, "url": url, "error": str(e)}
    digest = hashlib.sha256(content).hexdigest()[:12]
    result = {"name": name, "url": url, "sha256": digest, "bytes": len(content)}
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_dir / name
        previous = path.read_bytes() if path.exists() else None
        result["changed"] = previous != content
        result["new"] = previous is None
        if previous != content:
            path.write_bytes(content)
        result["path"] = str(path)
    return result


def cmd_signals(args: argparse.Namespace) -> int:
    provider = args.provider
    units = [u for u in inventory.scan_services() if u.provider == provider]
    if not units:
        print(f"unknown provider {provider!r}; see inventory.py --md", file=sys.stderr)
        return EXIT_UNSUPPORTED
    inventory.enrich(units)
    hints = provider_hints(provider)

    sdks = [pypi_latest(r) for r in sdk_requirements(provider, units)]

    specs = list(hints.get("specs") or [])
    for item in args.spec or []:
        name, sep, url = item.partition("=")
        if not sep:
            raise SystemExit(f"expected name=url, got {item!r}")
        specs.append({"name": name, "url": url})
    snapshot_dir = (
        None if args.no_snapshot else Path(args.snapshot_dir or DEFAULT_SPECS_DIR / provider)
    )
    spec_results = [spec_snapshot(spec["name"], spec["url"], snapshot_dir) for spec in specs]

    out = {"provider": provider, "sdks": sdks, "specs": spec_results}
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for sdk in sdks:
            latest = sdk.get("latest", "?")
            tail = (
                f" ({sdk['released']})"
                if sdk.get("released")
                else (f" — {sdk['error']}" if sdk.get("error") else "")
            )
            print(f"sdk   {sdk['package']:<28} pin {sdk['requirement']:<40} latest {latest}{tail}")
        for spec in spec_results:
            if spec.get("error"):
                print(f"spec  {spec['name']:<28} ERROR {spec['error']} ({spec['url']})")
                continue
            state = (
                "new" if spec.get("new") else ("CHANGED" if spec.get("changed") else "unchanged")
            )
            print(f"spec  {spec['name']:<28} sha256 {spec['sha256']} {spec['bytes']:>9} B  {state}")
    return EXIT_OK


# ------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--no-dotenv",
        action="store_true",
        help="ignore the repo .env; use only exported environment variables",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="push one real turn through a service class")
    run.add_argument("--service", required=True, help="service class name, e.g. CartesiaTTSService")
    run.add_argument(
        "--model", action="append", help="model to test (repeatable, max 3); omit for the default"
    )
    run.add_argument("--text", help="prompt (LLM) or text to synthesize (TTS)")
    run.add_argument("--wav", help="16-bit PCM wav for STT probes (default: bundled clip)")
    run.add_argument(
        "--setting",
        action="append",
        metavar="KEY=VALUE",
        help='extra Settings field; JSON for dict/list values, e.g. extra=\'{"reasoning_effort":"low"}\'',
    )
    run.add_argument(
        "--kwarg", action="append", metavar="KEY=VALUE", help="extra constructor argument"
    )
    run.add_argument(
        "--timeout", type=float, default=30.0, help="seconds to wait for the output frame"
    )
    run.add_argument(
        "--start-timeout",
        type=float,
        default=20.0,
        help="seconds to wait for the pipeline to start",
    )
    run.add_argument("--json", action="store_true", help="one JSON object per model on stdout")
    run.add_argument("--verbose", action="store_true", help="show service debug logs")
    run.set_defaults(func=cmd_run)

    lm = sub.add_parser("list-models", help="list the provider's model catalogue")
    lm.add_argument("--provider", required=True, help="provider directory name, e.g. groq")
    lm.add_argument("--base-url", help="override the OpenAI-compatible base URL")
    lm.add_argument("--json", action="store_true")
    lm.set_defaults(func=cmd_list_models)

    sg = sub.add_parser("signals", help="SDK versions on PyPI and API spec hashes for a provider")
    sg.add_argument("--provider", required=True, help="provider directory name, e.g. deepgram")
    sg.add_argument("--spec", action="append", metavar="NAME=URL", help="extra spec to fetch")
    sg.add_argument(
        "--snapshot-dir", help="where spec snapshots live (default: _reports/specs/<provider>)"
    )
    sg.add_argument(
        "--no-snapshot", action="store_true", help="hash only; do not read or write snapshots"
    )
    sg.add_argument("--json", action="store_true")
    sg.set_defaults(func=cmd_signals)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
