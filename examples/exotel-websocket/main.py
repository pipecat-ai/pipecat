import json

import uvicorn

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

from bot import run_bot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/phonebot")
async def websocket_endpoint(websocket: WebSocket):
    print(f"Incoming connection from {websocket.client}")
    try:
        await websocket.accept()
        print("Connection accepted")

        # Wait for the "connected" event
        connected_data = await websocket.receive_json()
        if connected_data.get("event") != "connected":
            raise ValueError("Expected 'connected' event")

        # Wait for the "start" event
        start_data = await websocket.receive_json()
        if start_data.get("event") != "start":
            raise ValueError("Expected 'start' event")

        # Extract stream_sid from the start event
        stream_sid = start_data.get("stream_sid")
        if not stream_sid:
            raise ValueError("Missing stream_sid in start event")

        print(f"Starting bot with stream_sid: {stream_sid}")
        await run_bot(websocket, stream_sid)

    except Exception as e:
        print(f"Error during websocket handling: {e}")
        raise


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
