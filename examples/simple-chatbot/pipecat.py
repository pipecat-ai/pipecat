import os
import argparse
import uvicorn

from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from dotenv import load_dotenv
load_dotenv(override=True)


# ------------ Configuration ------------ #

MAX_SESSION_TIME = 5 * 1000
BOT_CAN_IDLE = True
SERVE_STATIC = True
STATIC_DIR = "client/dist"
STATIC_ROUTE = "/static"
STATIC_INDEX = "index.html"


# ----------------- API ----------------- #

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Optionally serve client static files
if SERVE_STATIC:
    app.mount(STATIC_ROUTE, StaticFiles(
        directory=STATIC_DIR, html=True), name="static")

    @app.get("/{path_name:path}", response_class=FileResponse)
    async def catch_all(path_name: Optional[str] = ""):
        if path_name == "":
            return FileResponse(f"{STATIC_DIR}/{STATIC_INDEX}")

        file_path = Path(STATIC_DIR) / (path_name or "")

        if file_path.is_file():
            return file_path

        html_file_path = file_path.with_suffix(".html")
        if html_file_path.is_file():
            return FileResponse(html_file_path)

        raise HTTPException(
            status_code=404, detail="Page not found")


@app.post("/start_bot")
async def start_bot(request: Request):
    return JSONResponse({"bot_id": 123, "user_token": "abc", "room_url": "https://jpt.daily.co/hello"})


# ----------------- Main ----------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument("--host", type=str,
                        default=os.getenv("HOST", "localhost"), help="Host address")
    parser.add_argument("--port", type=int,
                        default=os.getenv("PORT", 7860), help="Port number")
    parser.add_argument("--reload", action="store_true",
                        default=True, help="Reload code on change")

    config = parser.parse_args()

    try:
        uvicorn.run(
            "pipecat:app",
            host=config.host,
            port=config.port,
            reload=config.reload
        )

    except KeyboardInterrupt:
        print("Pipecat runner shutting down...")
