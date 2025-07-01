#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini File API client for uploading and managing files.

This module provides a client for Google's Gemini File API, enabling file
uploads, metadata retrieval, listing, and deletion. Files uploaded through
this API can be referenced in Gemini generative model calls.
"""

import mimetypes
from typing import Any, Dict, Optional

import aiohttp
from google import genai
from loguru import logger


class GeminiFileAPI:
    """Client for the Gemini File API.

    This class provides methods for uploading, fetching, listing, and deleting files
    through Google's Gemini File API.

    Files uploaded through this API remain available for 48 hours and can be referenced
    in calls to the Gemini generative models. Maximum file size is 2GB, with total
    project storage limited to 20GB.
    """

    def __init__(
        self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta/files"
    ):
        """Initialize the Gemini File API client.

        Args:
            api_key: Google AI API key
            base_url: Base URL for the Gemini File API (default is the v1beta endpoint)
        """
        self._api_key = api_key
        self._client = genai.Client(api_key=self._api_key)
        self.base_url = base_url
        # Upload URL uses the /upload/ path
        self.upload_base_url = "https://generativelanguage.googleapis.com/upload/v1beta/files"

    async def upload_file(self, file_path: str) -> Dict[str, Any]:
        """Upload a file to the Gemini File API using the correct resumable upload protocol.

        Args:
            file_path: Path to the file to upload

        Returns:
            File metadata including uri, name, and display_name
        """
        logger.info(f"Uploading file: {file_path}")

        file_info = self._client.files.upload(file=file_path)
        return file_info

    async def get_file(self, name: str) -> Dict[str, Any]:
        """Get metadata for a file.

        Args:
            name: File name (or full path)

        Returns:
            File metadata
        """
        # Extract just the name part if a full path is provided
        # client.files.get(name=file_name)
        if "/" in name:
            name = name.split("/")[-1]

        file_info = self._client.files.get(name=f"files/{name}")
        return file_info

    async def list_files(
        self, page_size: int = 10, page_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """List uploaded files.

        Args:
            page_size: Number of files to return per page
            page_token: Token for pagination

        Returns:
            List of files and next page token if available
        """
        # Maximum pageSize is 100.
        page_size = page_size % 101
        params = {"key": self._api_key, "pageSize": page_size}

        if page_token:
            params["pageToken"] = page_token

        async with aiohttp.ClientSession() as session:
            async with session.get(self._base_url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error listing files: {error_text}")
                    raise Exception(f"Failed to list files: {response.status}")

                result = await response.json()
                return result

    async def delete_file(self, name: str) -> bool:
        """Delete a file.

        Args:
            name: File name (or full path)

        Returns:
            True if deleted successfully
        """
        # Extract just the name part if a full path is provided
        if "/" in name:
            name = name.split("/")[-1]

        async with aiohttp.ClientSession() as session:
            async with session.delete(f"{self.base_url}/{name}?key={self._api_key}") as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error deleting file: {error_text}")
                    raise Exception(f"Failed to delete file: {response.status}")

                return True
