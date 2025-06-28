"""
ws_handler.py
WebSocket handler for live captions (importable module for Pipecat multi-session backend).
"""

import asyncio
from loguru import logger


async def _ws_handler(websocket, path):
    logger.info(f"Live captions WebSocket connection established: {path}")
    try:
        async for message in websocket:
            # Echo or handle live captions here if needed
            logger.debug(f"Received WS message: {message}")
            await websocket.send(message)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("Live captions WebSocket connection closed")
