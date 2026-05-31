#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""TTS service registry + audio cache for the eval harness.

When a scenario defines a ``user_audio:`` block, the harness generates
raw audio for each user turn using that TTS config and sends it to the
bot as a ``user_audio`` message. The bot's STT then processes it for
real, exercising the full input audio path.

Audio is cached at ``.cache/evals/tts/<sha256>.wav`` keyed by
``(text, service, voice, model)``. The cache file's actual sample rate
is checked on load; if it doesn't match what the scenario requests,
the audio is regenerated and the cache file overwritten. This lets you
experiment with sample rates without bloating the cache.

Service support is via a small lazy-loaded registry. To add a new
service, append a factory to ``TTS_FACTORIES``. Optional escape hatch
via ``user_audio.factory: "my_pkg.my_func"`` (importable, called with
the voice config dict).
"""

import hashlib
import importlib
import io
import os
import wave
from pathlib import Path

from loguru import logger

# Where cached audio lives. Override via the PIPECAT_EVALS_CACHE_DIR env var.
DEFAULT_CACHE_DIR = Path(".cache/evals/tts")

# Default sample rate for generated audio (matches pipecat input default).
DEFAULT_SAMPLE_RATE = 16000


def _cache_dir() -> Path:
    """Resolve the cache directory, honoring the env var override."""
    override = os.environ.get("PIPECAT_EVALS_CACHE_DIR")
    base = Path(override) if override else DEFAULT_CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def _cache_key(text: str, service: str, voice: str, model: str) -> str:
    """Hash the semantic identity of the audio — sample rate excluded.

    Different SRs for the same text reuse the same cache slot; a mismatch
    just triggers regeneration in ``generate_or_load``.
    """
    h = hashlib.sha256()
    for part in (service, voice, model, text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _read_wav(path: Path) -> tuple[bytes, int]:
    """Read a mono 16-bit PCM WAV file. Returns (pcm_bytes, sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono, got {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit, got {wf.getsampwidth() * 8}-bit")
        return wf.readframes(wf.getnframes()), wf.getframerate()


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    """Write mono 16-bit PCM to a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


async def generate_or_load(text: str, voice_cfg: dict) -> tuple[bytes, int]:
    """Return audio bytes for ``text`` at the configured sample rate.

    Uses ``.cache/evals/tts/<hash>.wav`` if present and its sample rate
    matches; otherwise calls the configured TTS service to generate fresh
    audio and caches it.

    Args:
        text: The utterance to synthesize.
        voice_cfg: Mapping with at minimum ``service`` and ``voice`` keys.
            Optional: ``model`` (service-specific default), ``sample_rate``
            (default 16000), ``api_key_env`` (which env var holds the key).

    Returns:
        Tuple of ``(pcm_bytes, sample_rate)`` — raw 16-bit little-endian
        mono PCM.
    """
    service = str(voice_cfg.get("service", "")).lower()
    voice = str(voice_cfg.get("voice", ""))
    model = str(voice_cfg.get("model", ""))
    sample_rate = int(voice_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE))

    if not service or not voice:
        raise ValueError("user_audio config requires at least 'service' and 'voice'")

    cache_file = _cache_dir() / f"{_cache_key(text, service, voice, model)}.wav"

    if cache_file.exists():
        try:
            pcm, cached_sr = _read_wav(cache_file)
            if cached_sr == sample_rate:
                return pcm, sample_rate
            logger.info(
                f"Cache SR {cached_sr} ≠ requested {sample_rate} for {text!r} — regenerating"
            )
        except Exception as e:
            logger.warning(f"Cache read failed for {cache_file}: {e} — regenerating")

    factory = _resolve_factory(voice_cfg)
    pcm = await factory(text, voice_cfg, sample_rate)
    _write_wav(cache_file, pcm, sample_rate)
    return pcm, sample_rate


def _resolve_factory(voice_cfg: dict):
    """Look up the TTS factory in the registry, or import a user-provided one."""
    custom = voice_cfg.get("factory")
    if custom:
        module_name, _, attr = custom.rpartition(".")
        if not module_name:
            raise ValueError(f"user_audio.factory must be a dotted path: {custom!r}")
        return getattr(importlib.import_module(module_name), attr)

    service = str(voice_cfg.get("service", "")).lower()
    if service not in TTS_FACTORIES:
        known = ", ".join(sorted(TTS_FACTORIES))
        raise ValueError(
            f"Unknown TTS service: {service!r}. Known: {known}. "
            "Or specify user_audio.factory: 'module.func' for a custom one."
        )
    return TTS_FACTORIES[service]


async def _cartesia(text: str, voice_cfg: dict, sample_rate: int) -> bytes:
    """Cartesia HTTP TTS — returns raw PCM bytes at the requested sample rate."""
    import aiohttp  # lazy

    api_key_env = voice_cfg.get("api_key_env", "CARTESIA_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"${api_key_env} is not set — required for Cartesia TTS")

    payload = {
        "model_id": voice_cfg.get("model") or "sonic-2",
        "transcript": text,
        "voice": {"mode": "id", "id": voice_cfg["voice"]},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate,
        },
    }
    headers = {
        "X-API-Key": api_key,
        "Cartesia-Version": "2026-03-01",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.cartesia.ai/tts/bytes",
            headers=headers,
            json=payload,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Cartesia TTS HTTP {resp.status}: {body[:200]}")
            return await resp.read()


# Lazy: only imports its service module when the factory is actually called.
# Add new services here.
TTS_FACTORIES = {
    "cartesia": _cartesia,
}


def pcm_to_base64(pcm: bytes) -> str:
    """Encode raw PCM bytes as base64 for the ``user_audio`` WS message."""
    import base64

    return base64.b64encode(pcm).decode("ascii")


# Re-export this for use in BytesIO contexts (e.g. uploading to a service).
def pcm_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM in a WAV container, returning the full file bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
