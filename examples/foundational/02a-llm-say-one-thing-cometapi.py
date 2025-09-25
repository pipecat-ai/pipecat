#!/usr/bin/env python3
#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""
Minimal CometAPI LLM example.

This mirrors the style of other foundational examples and only depends on
COMETAPI_KEY. It sends a single prompt and streams back the response.
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.services.cometapi.llm import CometAPILLMService

load_dotenv(override=True)

async def run():
    parser = argparse.ArgumentParser(description="Minimal CometAPI example")
    parser.add_argument("--model", default="gpt-4o-mini", help="CometAPI model ID (must be chat capable)")
    args = parser.parse_args()
    api_key = os.getenv("COMETAPI_KEY")
    if not api_key:
        raise SystemExit("Set COMETAPI_KEY first.")

    llm = CometAPILLMService(api_key=api_key, model=args.model)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Reply tersely."},
        {"role": "user", "content": "Say: CometAPI integration OK."},
    ]

    # Stats
    recommended = CometAPILLMService.recommended_models()
    logger.info(f"Recommended models count: {len(recommended)} (showing first 5) {recommended[:5]}")
    try:
        fetched = await llm.fetch_chat_models()
        logger.info(f"Fetched chat models (filtered) count: {len(fetched)}")
    except Exception as e:  # pragma: no cover
        logger.warning(f"Could not fetch models list: {e}")

    logger.info(f"Requesting completion from CometAPI via OpenAI-compatible client (model={args.model})")
    # Use underlying OpenAI compatible async client directly (single-shot, non-streaming)
    response = await llm._client.chat.completions.create(  # pylint: disable=protected-access
        model=llm._model_name,  # noqa: SLF001
        messages=messages,
        temperature=0.2,
        max_tokens=50,
    )
    text = response.choices[0].message.content
    print(text)
    logger.info("Done.")

if __name__ == "__main__":
    import asyncio
    # Directly run the coroutine like other minimal examples.
    asyncio.run(run())