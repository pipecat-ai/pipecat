from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.database import create_db_and_tables
from app.api import router
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    # RECORDINGS_DIR is ensured in global scope or here
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    yield

app = FastAPI(title="Voice Agent Freeze Detector", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api")

# Static files for recordings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
