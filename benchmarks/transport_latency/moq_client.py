"""MoQ connector for the transport latency benchmark.

A headless "user" endpoint built directly on the ``moq`` package (not
``MOQTransport`` — no pipeline/RTVI baggage in the measuring path). Mirrors
what the browser transport does:

- publishes an Opus audio track under ``<namespace>/<clientId>`` (the bot
  subscribes to the first audio track in that broadcast's catalog),
- subscribes to the bot's ``<namespace>/<botId>`` broadcast's audio track,
  decoding to S16 mono 48 kHz with a pinned ``latency_max_ms`` jitter buffer.

The bot handshake is the runner's ``POST /start``: its response carries the
relay URL (bot-served in serve mode, external relay in client mode), the
namespace, and both participant ids — so this connector works unchanged for
``moq-serve``, ``moq-relay-local``, and ``moq-relay-deployed``. TLS follows
the ``/start`` response: a self-signed endpoint (serve mode, local dev relay)
advertises a ``certHash`` and gets ``tls_verify=False``; a CA-signed deployed
relay does not and is verified normally.
"""

import asyncio
from collections.abc import AsyncIterator

import aiohttp
import moq

SAMPLE_RATE = 48000
CHUNK_US = 20_000  # 20 ms


class MoqConnector:
    def __init__(
        self,
        start_url: str = "http://localhost:7860/start",
        jitter_ms: int = 60,
        tls_verify: bool | None = None,
    ) -> None:
        self._start_url = start_url
        self._jitter_ms = jitter_ms
        self._tls_verify = tls_verify
        self._client: moq.Client | None = None
        self._producer = None
        self._pts_us = 0
        self.moq_config: dict = {}

    async def start(self) -> None:
        async with aiohttp.ClientSession() as http:
            async with http.post(self._start_url, json={}) as resp:
                resp.raise_for_status()
                body = await resp.json()
        cfg = body["moq"]
        self.moq_config = cfg

        # None = infer from the bot's config: a self-signed relay advertises a
        # certHash; a CA-signed one doesn't and gets full verification.
        tls_verify = self._tls_verify if self._tls_verify is not None else not cfg.get("certHash")
        self._client = moq.Client(cfg["relayUrl"], tls_verify=tls_verify)
        await self._client.__aenter__()

        broadcast = moq.BroadcastProducer()
        self._producer = broadcast.publish_audio(
            "audio",
            moq.AudioEncoderInput(format=moq.AudioFormat.S16, sample_rate=SAMPLE_RATE, channels=1),
            moq.AudioEncoderOutput(
                codec=moq.AudioCodec.OPUS,
                sample_rate=SAMPLE_RATE,
                channels=None,
                bitrate=None,
                frame_duration_ms=20,
            ),
        )
        self._client.publish(f"{cfg['namespace']}/{cfg['clientId']}", broadcast)

        # Bot broadcast may not be announced yet right after /start.
        announced = self._client.announced_broadcast(f"{cfg['namespace']}/{cfg['botId']}")
        self._bot_broadcast = await asyncio.wait_for(announced.available(), timeout=15.0)

    async def send_chunk(self, pcm: bytes) -> None:
        self._producer.write(moq.AudioFrame(timestamp_us=self._pts_us, data=pcm))
        self._pts_us += CHUNK_US

    async def recv_chunks(self) -> AsyncIterator[bytes]:
        catalog = await self._bot_broadcast.catalog()
        if not catalog.audio:
            raise RuntimeError("bot catalog has no audio track")
        track_name, audio = next(iter(catalog.audio.items()))
        consumer = self._bot_broadcast.subscribe_audio(
            track_name,
            audio,
            moq.AudioDecoderOutput(
                format=moq.AudioFormat.S16,
                sample_rate=SAMPLE_RATE,
                channels=1,
                latency_max_ms=self._jitter_ms,
            ),
        )
        async for frame in consumer:
            yield frame.data

    async def stop(self) -> None:
        if self._producer is not None:
            self._producer.finish()
        if self._client is not None:
            await self._client.__aexit__(None, None, None)


class MoqLocalConnector:
    """In-process Opus publish/consume via a local origin — no network.

    Measures the moq stack floor (FFI + Opus encode/decode + track queues) so
    scenario RTTs can be decomposed into client-stack vs bot-path time.
    """

    def __init__(self, jitter_ms: int = 60) -> None:
        self._jitter_ms = jitter_ms
        self._pts_us = 0

    async def start(self) -> None:
        self._origin = moq.OriginProducer()
        broadcast = moq.BroadcastProducer()
        self._producer = broadcast.publish_audio(
            "audio",
            moq.AudioEncoderInput(format=moq.AudioFormat.S16, sample_rate=SAMPLE_RATE, channels=1),
            moq.AudioEncoderOutput(
                codec=moq.AudioCodec.OPUS,
                sample_rate=SAMPLE_RATE,
                channels=None,
                bitrate=None,
                frame_duration_ms=20,
            ),
        )
        self._origin.publish("bench", broadcast)
        consumer = self._origin.consume()
        async for announcement in consumer.announced():
            self._broadcast = announcement.broadcast
            break

    async def send_chunk(self, pcm: bytes) -> None:
        self._producer.write(moq.AudioFrame(timestamp_us=self._pts_us, data=pcm))
        self._pts_us += CHUNK_US

    async def recv_chunks(self) -> AsyncIterator[bytes]:
        catalog = await self._broadcast.catalog()
        track_name, audio = next(iter(catalog.audio.items()))
        consumer = self._broadcast.subscribe_audio(
            track_name,
            audio,
            moq.AudioDecoderOutput(
                format=moq.AudioFormat.S16,
                sample_rate=SAMPLE_RATE,
                channels=1,
                latency_max_ms=self._jitter_ms,
            ),
        )
        async for frame in consumer:
            yield frame.data

    async def stop(self) -> None:
        self._producer.finish()


async def _main() -> None:
    import argparse

    import numpy as np
    from client_core import run_trial

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--jitter-ms", type=int, default=60)
    args = parser.parse_args()

    connector = MoqConnector(jitter_ms=args.jitter_ms)
    result = await run_trial(connector, duration_s=args.duration, warmup_s=5.0)
    arr = np.array(result.rtts_ms)
    print(f"relay url: {connector.moq_config.get('relayUrl')}")
    if len(arr) == 0:
        print(f"moq: NO MEASUREMENTS (drops={result.drops})")
        raise SystemExit(1)
    print(
        f"moq: n={len(arr)} drops={result.drops} ambiguous={result.ambiguous} "
        f"p50={np.percentile(arr, 50):.2f} ms p95={np.percentile(arr, 95):.2f} ms "
        f"max={arr.max():.2f} ms"
    )


if __name__ == "__main__":
    asyncio.run(_main())
