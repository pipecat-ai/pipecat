#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import json

import uvicorn
from bot import run_bot
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.responses import HTMLResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def start_call():
    print("POST Plivo XML")
    return HTMLResponse(content=open("templates/streams.xml").read(), media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Plivo sends a start event when the stream begins
    start_data = websocket.iter_text()
    start_message = json.loads(await start_data.__anext__())

    print("Received start message:", start_message, flush=True)

    # Extract stream_id and call_id from the start event
    start_info = start_message.get("start", {})
    stream_id = start_info.get("streamId")
    call_id = start_info.get("callId")

    if not stream_id:
        logger.error("No streamId found in start message")
        await websocket.close()
        return

    print(f"WebSocket connection accepted for stream: {stream_id}, call: {call_id}")
    await run_bot(websocket, stream_id, call_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
