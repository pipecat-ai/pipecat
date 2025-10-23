#!/usr/bin/env -S uv run
"""Utilities for fetching Daily.co recording URLs with retry logic.

This module provides functions to retrieve recording download links from Daily's REST API
with robust error handling, rate limiting, and exponential backoff retry logic.
"""

import asyncio
import os
from typing import Optional, Tuple

from httpx import AsyncClient, HTTPStatusError
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


async def get_recording_s3_url_with_retry(
    room_id: str,
    max_retries: int = 5,
) -> Tuple[Optional[str], Optional[str]]:
    """Retrieve recording URL with exponential backoff and retry logic.

    Uses tenacity library to handle rate limiting (429 errors) and other
    transient errors with automatic exponential backoff.

    Args:
        room_id: Daily.co room identifier
        max_retries: Maximum number of retry attempts (default 5)

    Returns:
        Tuple of (recording_url, recording_signed_url)
        Returns (None, None) if no recording exists for the room.
    """
    try:
        # Use tenacity's AsyncRetrying for automatic retry with exponential backoff
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((HTTPStatusError, Exception)),
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            reraise=True,
        ):
            with attempt:
                recording_url, recording_signed_url, status = await get_recording_s3_url(
                    room_id=room_id
                )

                # If no recording exists (status is None), return immediately - no retry
                if status is None:
                    logger.debug(f"No recording found for room {room_id}")
                    return None, None

                # If recording exists but is not finished yet, retry
                if status != "finished":
                    logger.warning(
                        f"Recording not finished for room {room_id}, status: {status} "
                        f"(attempt {attempt.retry_state.attempt_number}/{max_retries})"
                    )
                    raise Exception(f"Recording not ready, status: {status}")

                # Recording is finished, return the URLs
                return recording_url, recording_signed_url

        # This line should never be reached due to reraise=True, but satisfies type checker
        return None, None

    except RetryError as e:
        # All retries exhausted
        last_exception = e.last_attempt.exception()
        logger.error(
            f"Failed to retrieve recording URL for room {room_id} after {max_retries} attempts: "
            f"{last_exception}"
        )
        return None, None

    except Exception as e:
        logger.exception(f"Unexpected error retrieving recording for room {room_id}: {e}")
        return None, None


async def get_recording_s3_url(
    room_id: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Get recording URL using Daily's REST API.

    Args:
        room_id: Daily.co room identifier

    Returns:
        Tuple of (recording_url, recording_signed_url, status)
        - recording_url: The download link for the recording
        - recording_signed_url: Same as recording_url (kept for backward compatibility)
        - status: Recording status from Daily API

    Raises:
        HTTPStatusError: When HTTP errors occur (including rate limits)
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('DAILY_API_KEY')}",
        "Content-Type": "application/json",
    }

    async with AsyncClient(timeout=180) as client:
        # List recordings for the room
        list_response = await client.get(
            url=f"https://api.daily.co/v1/recordings?room_name={room_id}",
            headers=headers,
        )
        list_response.raise_for_status()
        list_data = list_response.json()

        # Check if recording exists and is finished
        if not list_data.get("data") or len(list_data["data"]) == 0:
            return (None, None, None)

        recording_id = list_data["data"][0].get("id")
        status = list_data["data"][0].get("status")

        if not recording_id or status != "finished":
            return (None, None, status)

        # Get the recording access link
        link_response = await client.get(
            url=f"https://api.daily.co/v1/recordings/{recording_id}/access-link",
            headers=headers,
        )
        link_response.raise_for_status()
        link_data = link_response.json()

    recording_url = link_data.get("download_link")
    if not recording_url:
        logger.warning(f"No download link found for recording {recording_id}")
        return (None, None, status)

    # Return the same URL for both fields for backward compatibility
    return (recording_url, recording_url, status)


async def get_recent_rooms(limit: int = 100) -> list[str]:
    """Get list of recent room names from Daily API.

    Args:
        limit: Maximum number of rooms to retrieve (default 100, max 100)

    Returns:
        List of room names (strings)
    """
    try:
        async with AsyncClient() as client:
            response = await client.get(
                url=f"https://api.daily.co/v1/rooms?limit={min(limit, 100)}",
                headers={
                    "Authorization": f"Bearer {os.getenv('DAILY_API_KEY')}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()

        data = response.json()
        rooms = data.get("data", [])
        room_names = [room.get("name") for room in rooms if room.get("name")]

        logger.info(f"Retrieved {len(room_names)} room names from Daily API")
        return room_names

    except Exception as e:
        logger.exception(f"Failed to get rooms from Daily API: {e}")
        return []


async def main():
    """Test get_recording_s3_url_with_retry with recent rooms."""
    logger.info("Starting recording fetch test...")

    # Step 1: Get the most recent 100 rooms
    logger.info("Fetching recent rooms...")
    room_names = await get_recent_rooms(limit=100)

    if not room_names:
        logger.error("No rooms found. Cannot proceed with test.")
        return

    logger.info(f"Found {len(room_names)} rooms to check for recordings")

    # Call get_recording_s3_url_with_retry on each room concurrently
    logger.info(f"Attempting to fetch recordings for {len(room_names)} rooms concurrently...")

    # Create tasks for all rooms
    tasks = [
        get_recording_s3_url_with_retry(room_id=room_name, max_retries=3)
        for room_name in room_names
    ]

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    success_count = 0
    not_found_count = 0
    failed_count = 0

    for i, (room_name, result) in enumerate(zip(room_names, results), 1):
        if isinstance(result, Exception):
            failed_count += 1
            logger.error(f"❌ [{i}/{len(room_names)}] Failed for {room_name}: {result}")
        elif isinstance(result, tuple) and len(result) == 2:
            recording_url, recording_signed_url = result
            if recording_url:
                success_count += 1
                logger.info(f"✅ [{i}/{len(room_names)}] Found recording for {room_name}")
                logger.debug(f"   URL: {recording_url[:80]}...")
            else:
                not_found_count += 1
                logger.debug(f"ℹ️  [{i}/{len(room_names)}] No recording for {room_name}")
        else:
            failed_count += 1
            logger.error(f"❌ [{i}/{len(room_names)}] Unexpected result type for {room_name}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RECORDING FETCH TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total rooms checked: {len(room_names)}")
    logger.info(f"✅ Recordings found: {success_count}")
    logger.info(f"ℹ️  No recordings: {not_found_count}")
    logger.info(f"❌ Failed: {failed_count}")
    logger.info(f"Success rate: {(success_count / len(room_names) * 100):.2f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
