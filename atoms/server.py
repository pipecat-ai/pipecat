import argparse
import asyncio
from contextlib import asynccontextmanager

import uvicorn
from call_manager import call_manager
from config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from router import outbound, webhooks, websocket, xml
from services.telemetry import instrument_app

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


async def main():
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

    instrument_app(args.env)

    logger.info(f"Starting pipetoms in {args.env} mode")

    if args.env == "development":
        config = uvicorn.Config(app, host=args.host, port=args.port, reload=True, access_log=True, log_level="info")
    else:
        config = uvicorn.Config(app, host=args.host, port=args.port, access_log=True, log_level="info")
    
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
