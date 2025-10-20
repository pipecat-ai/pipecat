#!/usr/bin/env -S uv run
"""Utilities for creating Daily.co rooms with retry logic.

This module provides functions to create Daily rooms via REST API
with robust error handling, rate limiting, and exponential backoff retry logic.
"""

import asyncio
import os
import time
from typing import Dict, Optional

from httpx import AsyncClient, HTTPStatusError
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


async def periodic_progress_logger(
    progress_dict: Dict[str, int],
    total: int,
    interval_seconds: float = 5.0,
    stop_event: Optional[asyncio.Event] = None,
):
    """Log progress periodically in the background.

    Args:
        progress_dict: Shared dict with 'completed' and 'failed' counts
        total: Total number of items being processed
        interval_seconds: How often to log progress (default 5 seconds)
        stop_event: Event to signal when to stop logging
    """
    if stop_event is None:
        stop_event = asyncio.Event()

    while not stop_event.is_set():
        await asyncio.sleep(interval_seconds)

        if stop_event.is_set():
            break

        total_processed = progress_dict["completed"] + progress_dict["failed"]
        if total_processed > 0:
            percentage = (total_processed / total) * 100
            rate = total_processed / interval_seconds if interval_seconds > 0 else 0

            logger.info(
                f"⏳ Progress: {total_processed}/{total} ({percentage:.1f}%) - "
                f"✅ {progress_dict['completed']} succeeded, "
                f"❌ {progress_dict['failed']} failed"
            )


async def create_daily_room(
    name: Optional[str] = None,
    privacy: str = "public",
    exp_minutes: int = 10,
    max_retries: int = 5,
) -> Optional[Dict]:
    """Create a Daily room with automatic retry on rate limit errors.

    Uses tenacity library to handle rate limiting (429 errors) with
    exponential backoff and automatic retries.

    Args:
        name: Room name (auto-generated if None). Must match /[A-Za-z0-9_-]+/ and be <= 128 chars
        privacy: Room privacy setting ("public" or "private")
        exp_minutes: Minutes until room expires (default 10)
        max_retries: Maximum number of retry attempts on rate limit (default 5)

    Returns:
        Room object dict with 'name', 'url', 'id', 'config', etc., or None on failure
    """
    # Calculate expiration timestamp (unix timestamp in seconds)
    exp_timestamp = int(time.time()) + (exp_minutes * 60)

    # Build request body
    body = {
        "privacy": privacy,
        "properties": {
            "exp": exp_timestamp,
        },
    }

    if name:
        body["name"] = name

    try:
        # Use tenacity's AsyncRetrying for automatic retry with exponential backoff
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(HTTPStatusError),
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            reraise=True,
        ):
            with attempt:
                async with AsyncClient() as client:
                    response = await client.post(
                        url="https://api.daily.co/v1/rooms",
                        headers={
                            "Authorization": f"Bearer {os.getenv('DAILY_API_KEY')}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                        timeout=30,
                    )

                    response.raise_for_status()

                room_data = response.json()
                return room_data

    except RetryError as e:
        # All retries exhausted
        last_exception = e.last_attempt.exception()
        if isinstance(last_exception, HTTPStatusError):
            if last_exception.response.status_code == 429:
                logger.error(f"Rate limit exceeded after {max_retries} retries")
            else:
                logger.error(
                    f"HTTP {last_exception.response.status_code} error creating room: "
                    f"{last_exception.response.text}"
                )
        else:
            logger.exception(f"Failed to create room after {max_retries} retries: {last_exception}")
        return None

    except Exception as e:
        logger.exception(f"Unexpected error creating room: {e}")
        return None


async def create_room_with_progress(
    index: int, total: int, progress_dict: Dict[str, int], **kwargs
) -> Optional[Dict]:
    """Wrapper for create_daily_room that tracks progress.

    Args:
        index: Index of this room creation (0-based)
        total: Total number of rooms being created
        progress_dict: Shared dict for tracking progress {"completed": 0, "failed": 0}
        **kwargs: Arguments passed to create_daily_room

    Returns:
        Room object dict or None
    """
    result = await create_daily_room(**kwargs)

    # Update progress
    if result is not None:
        progress_dict["completed"] += 1
    else:
        progress_dict["failed"] += 1

    # Log progress periodically (every 10% or every 100 rooms, whichever is smaller)
    total_processed = progress_dict["completed"] + progress_dict["failed"]
    log_interval = min(100, max(1, total // 10))

    if total_processed % log_interval == 0 or total_processed == total:
        logger.info(
            f"Progress: {total_processed}/{total} "
            f"({(total_processed / total) * 100:.1f}%) - "
            f"✅ {progress_dict['completed']} succeeded, "
            f"❌ {progress_dict['failed']} failed"
        )

    return result


async def test_create_rooms(
    num_rooms: int = 1000,
    progress_interval: float = 5.0,
) -> Dict[str, int | float]:
    """Attempt to create multiple Daily rooms concurrently.

    This function demonstrates concurrent room creation and tracks
    success/failure statistics. Rate limiting will likely occur when
    creating many rooms quickly.

    Args:
        num_rooms: Number of rooms to attempt to create (default 1000)
        progress_interval: How often to log progress in seconds (default 5.0)

    Returns:
        Dict with statistics: {'success': int, 'failed': int, 'total': int, 'elapsed_seconds': float}
    """
    logger.info(f"Starting bulk room creation: attempting to create {num_rooms} rooms")
    start_time = time.time()

    # Shared progress tracking dictionary
    progress_dict = {"completed": 0, "failed": 0}

    # Create tasks for concurrent room creation with progress tracking
    tasks = []
    for i in range(num_rooms):
        task = create_room_with_progress(
            index=i,
            total=num_rooms,
            progress_dict=progress_dict,
            name=None,  # Auto-generate names
            privacy="public",
            exp_minutes=10,
            max_retries=5,
        )
        tasks.append(task)

    # Execute all tasks concurrently with periodic progress logging
    logger.info(f"Executing {num_rooms} concurrent room creation requests...")

    # Start background progress logger
    stop_event = asyncio.Event()
    progress_task = asyncio.create_task(
        periodic_progress_logger(progress_dict, num_rooms, progress_interval, stop_event)
    )

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        # Stop the progress logger
        stop_event.set()
        await progress_task

    # Count successes and failures
    success_count = 0
    failed_count = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Room {i + 1} failed with exception: {result}")
            failed_count += 1
        elif result is None:
            failed_count += 1
        else:
            success_count += 1

    elapsed_time = time.time() - start_time

    # Log statistics
    logger.info("=" * 60)
    logger.info("BULK ROOM CREATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total rooms attempted: {num_rooms}")
    logger.info(f"Successfully created: {success_count}")
    logger.info(f"Failed to create: {failed_count}")
    logger.info(f"Success rate: {(success_count / num_rooms * 100):.2f}%")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Average time per room: {(elapsed_time / num_rooms):.3f} seconds")
    logger.info("=" * 60)

    return {
        "success": success_count,
        "failed": failed_count,
        "total": num_rooms,
        "elapsed_seconds": elapsed_time,
    }


# Example usage
async def main():
    """Example usage of the room creation functions."""
    # Test creating a single room
    logger.info("Testing single room creation...")
    room = await create_daily_room(exp_minutes=10)
    if room:
        logger.info(f"Created room: {room['name']} at {room['url']}")
    else:
        logger.error("Failed to create room")

    # Uncomment to test bulk creation (warning: may hit rate limits!)
    logger.info("\nTesting bulk room creation...")
    stats = await test_create_rooms(num_rooms=1000)
    logger.info(f"Final stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
