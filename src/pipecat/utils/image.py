#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Image encoding helpers."""

import asyncio
import io

from PIL import Image


async def encode_image(
    image: bytes, size: tuple[int, int], format: str | None
) -> tuple[bytes, str]:
    """Encode an image for a model to look at.

    An image whose format is a MIME type is already encoded and keeps its
    bytes. Raw pixels are encoded as a JPEG in a thread.

    Args:
        image: The image bytes, raw pixels or already encoded.
        size: The image's width and height.
        format: The pixel format, such as ``"RGB"`` or ``"RGBA"``, or the MIME
            type of an encoded image. Raw pixels are taken as RGB when not given.

    Returns:
        The encoded bytes and their MIME type.
    """
    if format and format.startswith("image/"):
        return image, format

    def encode() -> bytes:
        buffer = io.BytesIO()
        Image.frombytes(format or "RGB", size, image).convert("RGB").save(buffer, format="JPEG")
        return buffer.getvalue()

    return await asyncio.to_thread(encode), "image/jpeg"
