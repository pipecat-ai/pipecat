#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

import aiohttp
import modal
from bot import _voice_bot_process
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

MAX_SESSION_TIME = 15 * 60  # 15 minutes

app = modal.App("pipecat-modal")


image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(
    "requirements.txt"
)


@app.function(
    image=image,
    cpu=1.0,
    secrets=[modal.Secret.from_dotenv()],
    keep_warm=1,
    enable_memory_snapshot=True,
    max_inputs=1,  # Do not reuse instances across requests
    retries=0,
)
def launch_bot_process(room_url: str, token: str):
    _voice_bot_process(room_url, token)


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
)
@modal.web_endpoint(method="POST")
async def start():
    from pipecat.transports.services.helpers.daily_rest import (
        DailyRESTHelper,
        DailyRoomParams,
    )

    logger.info("Request received")

    async with aiohttp.ClientSession() as session:
        daily_rest_helper = DailyRESTHelper(
            daily_api_key=os.getenv("DAILY_API_KEY", ""),
            daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
            aiohttp_session=session,
        )

        # Create new Daily room
        room = await daily_rest_helper.create_room(DailyRoomParams())
        if not room.url:
            raise HTTPException(
                status_code=500,
                detail="Unable to create room",
            )
        logger.info(f"Created room: {room.url}")

        # Create bot token for room
        token = await daily_rest_helper.get_token(room.url, MAX_SESSION_TIME)
        if not token:
            raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room.url}")

        logger.info(f"Bot token created: {token}")

        # Spawn a new bot process
        launch_bot_process.spawn(room_url=room.url, token=token)

        # Return room URL to the user to join
        # Note: in production, you would want to return a token to the user
        return JSONResponse(content={"room_url": room.url, token: token})
