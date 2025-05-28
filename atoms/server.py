import argparse
from contextlib import asynccontextmanager

import uvicorn
from call_manager import call_manager
from config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from router import outbound, webhooks, websocket, xml


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event."""
    yield
    await call_manager.initiate_shutdown()


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(outbound.router, prefix="/api/v1", tags=["outbound"])
app.include_router(websocket.router, tags=["websocket"])
app.include_router(webhooks.router, tags=["webhooks"])
app.include_router(xml.router, tags=["xml"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "pipecat-production-app"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness check endpoint."""
    return {"status": "ready", "service": "pipecat-production-app"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Production Server")
    parser.add_argument(
        "--env",
        choices=["development", "production"],
        default="development",
        help="Environment to run in",
    )
    parser.add_argument("--host", default=settings.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to bind to")
    args = parser.parse_args()

    if args.env == "production":
        logger.remove()
        logger.add("logs/app.log", rotation="1 day", retention="30 days", level="INFO")
        logger.add("sys.stderr", level="ERROR")
    else:
        logger.add("sys.stderr", level="DEBUG")

    logger.info(f"Starting pipetoms in {args.env} mode")

    if args.env == "development":
        uvicorn.run("server:app", host=args.host, port=args.port, reload=True, access_log=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port, access_log=True)
