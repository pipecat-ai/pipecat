"""MLX backend temporarily disabled.

This stub keeps import paths intact but prevents accidental usage of the
unstable MLX backend. Use the default `ollama` engine instead.
"""

from __future__ import annotations

import asyncio
import os
# Disable fork-safety warning & potential crashes when using HuggingFace tokenizers in forked processes.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from types import SimpleNamespace
from typing import AsyncGenerator, List

from loguru import logger


import mlx_lm  # type: ignore
from pipecat.services.openai.llm import OpenAILLMService

# ----------------------------------------------------------------------------
# Cache already-loaded models to avoid duplicate loads and crashes
# ----------------------------------------------------------------------------
_MODEL_CACHE: dict[str, tuple[object, object]] = {}

# (all previous implementation removed)
class _FakeUsage:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0


class _FakeChoiceDelta:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None  # keep API parity


class _FakeChoice:
    def __init__(self, delta: _FakeChoiceDelta):
        self.delta = delta


class _FakeChunk:
    def __init__(self, content: str):
        self.choices: List[_FakeChoice] = [_FakeChoice(_FakeChoiceDelta(content))]
        self.usage = None  # usage metrics not supported for local models


class MLXLLMService(OpenAILLMService):
    """
    Drop-in replacement for OLLamaLLMService that runs a local
    mlx-lm language model (e.g. llama-3-8B-instruct-Q4, phi-3-mini, etc.)
    """

    def __init__(self, *_, **__):
        # MLX backend is currently disabled due to instability.
        raise RuntimeError("MLX backend is disabled. Use --engine ollama instead.")
        # OpenAI flow remains intact.
        super().__init__(model=model, api_key="local-mlx", base_url="http://localhost/ignore", **kwargs)

        if model in _MODEL_CACHE:
            logger.info(f"Re-using cached MLX model '{model}'")
            self._mlx_model, self._mlx_tokenizer = _MODEL_CACHE[model]
        else:
            logger.info(f"Loading MLX model '{model}' …")
            self._mlx_model, self._mlx_tokenizer = mlx_lm.load(model)
            _MODEL_CACHE[model] = (self._mlx_model, self._mlx_tokenizer)

    class _PseudoAsyncStream:
        """Minimal wrapper emulating openai.AsyncStream interface needed by BaseOpenAILLMService."""

        def __init__(self, agen):
            self._agen = agen

        def __aiter__(self):  # noqa: D401
            return self

        async def __anext__(self):
            try:
                return await self._agen.__anext__()
            except StopAsyncIteration:
                raise StopAsyncIteration

    # ------------------------------------------------------------------
    async def _run_generate(self, *_, **__):
        return ""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: mlx_lm.generate(
                self._mlx_model,
                self._mlx_tokenizer,
                prompt,
                max_tokens=gen_kwargs.get("max_tokens", 512),
            ),
        )

    # ------------------------------------------------------------------
    async def get_chat_completions(self, *_, **__):
        raise RuntimeError("MLX backend is disabled.")
        """Return a pseudo AsyncStream of ChatCompletionChunk-like objects."""
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        text = await self._run_generate(prompt)

        async def _stream():
            for word in text.split():
                yield _FakeChunk(word + " ")

        # Return wrapper; BaseOpenAILLMService awaits this and then iterates.
        return self._PseudoAsyncStream(_stream())
