#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv(override=True)

NUMBER_OF_ROOMS = 1

from bot import run_bot


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles FastAPI startup and shutdown."""
    yield  # Run app


# Initialize FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")
    try:
        await run_bot(websocket)
    except Exception as e:
        print(f"Exception in run_bot: {e}")


@app.post("/connect")
async def bot_connect(request: Request) -> Dict[Any, Any]:
    return {
        "ws_host": "localhost",
        "ws_port": 7860,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
