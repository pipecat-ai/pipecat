#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json

import uvicorn
from bot import run_bot
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
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
    print("POST TeXML")
    return HTMLResponse(content=open("templates/streams.xml").read(), media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    start_data = websocket.iter_text()
    await start_data.__anext__()
    call_data = json.loads(await start_data.__anext__())
    print(call_data, flush=True)
    stream_id = call_data["stream_id"]
    outbound_encoding = call_data["start"]["media_format"]["encoding"]
    print("WebSocket connection accepted")
    await run_bot(websocket, stream_id, outbound_encoding, "PCMU")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
