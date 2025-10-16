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
        self._base_url = base_url
        # Upload URL uses the /upload/ path
        self.upload_base_url = "https://generativelanguage.googleapis.com/upload/v1beta/files"

    async def upload_file(
        self, file_path: str, display_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload a file to the Gemini File API using the correct resumable upload protocol.

        Args:
            file_path: Path to the file to upload
            display_name: Optional display name for the file

        Returns:
            File metadata including uri, name, and display_name
        """
        logger.info(f"Uploading file: {file_path}")

        async with aiohttp.ClientSession() as session:
            # Determine the file's MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "application/octet-stream"

            # Read the file
            with open(file_path, "rb") as f:
                file_data = f.read()

            # Create the metadata payload
            metadata = {}
            if display_name:
                metadata = {"file": {"display_name": display_name}}

            # Step 1: Initial resumable request to get upload URL
            headers = {
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(file_data)),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": "application/json",
            }

            logger.debug(f"Step 1: Getting upload URL from {self.upload_base_url}")
            async with session.post(
                f"{self.upload_base_url}?key={self._api_key}", headers=headers, json=metadata
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error initiating file upload: {error_text}")
                    raise Exception(f"Failed to initiate upload: {response.status} - {error_text}")

                # Get the upload URL from the response header
                upload_url = response.headers.get("X-Goog-Upload-URL")
                if not upload_url:
                    logger.error(f"Response headers: {dict(response.headers)}")
                    raise Exception("No upload URL in response headers")

                logger.debug(f"Got upload URL: {upload_url}")

            # Step 2: Upload the actual file data
            upload_headers = {
                "Content-Length": str(len(file_data)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            }

            logger.debug(f"Step 2: Uploading file data to {upload_url}")
            async with session.post(upload_url, headers=upload_headers, data=file_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error uploading file data: {error_text}")
                    raise Exception(f"Failed to upload file: {response.status} - {error_text}")

                file_info = await response.json()
                logger.info(f"File uploaded successfully: {file_info.get('file', {}).get('name')}")
                return file_info

    async def get_file(self, name: str) -> Dict[str, Any]:
        """Get metadata for a file.

        Args:
            name: File name (or full path)

        Returns:
            File metadata
        """
        # Extract just the name part if a full path is provided
        if "/" in name:
            name = name.split("/")[-1]

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self._base_url}/{name}?key={self._api_key}") as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error getting file metadata: {error_text}")
                    raise Exception(f"Failed to get file metadata: {response.status}")

                file_info = await response.json()
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
            async with session.delete(f"{self._base_url}/{name}?key={self._api_key}") as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error deleting file: {error_text}")
                    raise Exception(f"Failed to delete file: {response.status}")

                return True
