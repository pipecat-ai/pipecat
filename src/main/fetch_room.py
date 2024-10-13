# 
# Copyright (c) 2024, Daily 
# 
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse, RedirectResponse
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper

app = FastAPI()

class ConfigRequest(BaseModel):
    room_url: str = None
    apikey: str = None
    room_id: str = None

async def configure(aiohttp_session: aiohttp.ClientSession, request: ConfigRequest):
    url = request.room_url or os.getenv("DAILY_SAMPLE_ROOM_URL")
    key = request.apikey or os.getenv("DAILY_API_KEY")

    if not url:
        raise HTTPException(
            status_code=400,
            detail="No Daily room specified. Provide the 'url' parameter or set DAILY_SAMPLE_ROOM_URL in your environment."
        )

    if not key:
        raise HTTPException(
            status_code=400,
            detail="No Daily API key specified. Provide the 'apikey' parameter or set DAILY_API_KEY in your environment."
        )

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session
    )

    # Create a meeting token for the given room with an expiration of 1 hour.
    expiry_time: float = 60 * 60

    token = await daily_rest_helper.get_token(url, expiry_time)

    return {"room_url": url, "token": token}


