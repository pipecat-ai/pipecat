"""
Pipecat AI Backend Server
Optimized for local AI rig with Ollama and Cartesia

A production-ready voice AI backend that provides:
- WebSocket-based voice conversations
- Twilio phone integration
- User management and authentication
- Session tracking and analytics
- Low-latency local LLM processing
"""

import sys
from pathlib import Path

# Add parent directory to path for pipecat imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn

from .config import settings
from .routes import admin_router, voice_router, twilio_router

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL,
)

# Create FastAPI app
app = FastAPI(
    title="Pipecat AI Backend",
    description="Voice AI backend with Ollama and Cartesia for ultra-low latency",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred",
        },
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=" * 60)
    logger.info("Starting Pipecat AI Backend Server")
    logger.info("=" * 60)
    logger.info(f"Environment: {'Development' if settings.DEBUG else 'Production'}")
    logger.info(f"Host: {settings.HOST}:{settings.PORT}")
    logger.info(f"Ollama URL: {settings.OLLAMA_BASE_URL}")
    logger.info(f"Ollama Model: {settings.OLLAMA_MODEL}")
    logger.info(f"Cartesia API: {'Configured' if settings.CARTESIA_API_KEY else 'Not configured'}")
    logger.info(f"Twilio: {'Configured' if settings.TWILIO_ACCOUNT_SID else 'Not configured'}")
    logger.info("=" * 60)

    # Initialize services
    from .services.user_service import get_user_service
    from .services.session_service import get_session_service
    from .services.pipeline_service import get_pipeline_service

    user_service = get_user_service()
    session_service = get_session_service()
    pipeline_service = get_pipeline_service()

    logger.info("✓ User service initialized")
    logger.info("✓ Session service initialized")
    logger.info("✓ Pipeline service initialized")

    # Log default credentials
    logger.warning("=" * 60)
    logger.warning("DEFAULT ADMIN CREDENTIALS (change in production!):")
    logger.warning("  Username: admin")
    logger.warning("  Password: admin123")
    logger.warning("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Pipecat AI Backend Server")

    from .services.pipeline_service import get_pipeline_service

    pipeline_service = get_pipeline_service()

    # Stop all active pipelines
    active_sessions = pipeline_service.get_active_sessions()
    for session_id in active_sessions:
        logger.info(f"Stopping pipeline for session {session_id}")
        await pipeline_service.stop_pipeline(session_id)
        await pipeline_service.cleanup_pipeline(session_id)

    logger.info("✓ All pipelines stopped")
    logger.info("Goodbye!")


# Include routers
app.include_router(admin_router)
app.include_router(voice_router)
app.include_router(twilio_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Pipecat AI Backend",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "admin": "/api/admin",
            "voice_ws": "/api/voice/ws",
            "voice_ws_auth": "/api/voice/ws/auth",
            "twilio": "/api/twilio",
        },
        "features": {
            "local_llm": "Ollama",
            "tts_stt": "Cartesia",
            "telephony": "Twilio",
            "authentication": "JWT + API Keys",
        },
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from .services.pipeline_service import get_pipeline_service
    from .services.session_service import get_session_service

    pipeline_service = get_pipeline_service()
    session_service = get_session_service()

    return {
        "status": "healthy",
        "active_pipelines": pipeline_service.get_session_count(),
        "active_sessions": session_service.get_active_session_count(),
        "total_sessions": session_service.get_session_count(),
    }


# Main entry point
def main():
    """Run the server"""
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.ENABLE_LOGGING,
    )


if __name__ == "__main__":
    main()
