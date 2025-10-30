"""Voice WebSocket routes for real-time voice conversations"""

import asyncio
from typing import Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from loguru import logger

from ..models.session import SessionCreate, SessionStatus, SessionType
from ..models.user import User
from ..services.session_service import get_session_service, SessionService
from ..services.pipeline_service import get_pipeline_service, PipelineService
from ..middleware.auth import get_optional_user
from ..config import settings

router = APIRouter(prefix="/api/voice", tags=["voice"])


@router.websocket("/ws")
async def voice_websocket(
    websocket: WebSocket,
    voice_config: str = Query(default="conversational"),
    system_prompt: str = Query(default="default"),
    session_service: SessionService = Depends(get_session_service),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """
    WebSocket endpoint for voice conversations

    Args:
        websocket: WebSocket connection
        voice_config: Voice configuration key (conversational, professional, assistant)
        system_prompt: System prompt key (default, customer_service, sales, appointment)
        session_service: Session service dependency
        pipeline_service: Pipeline service dependency

    Flow:
        1. Accept WebSocket connection
        2. Create session
        3. Create and run Pipecat pipeline
        4. Handle voice data streaming
        5. Clean up on disconnect
    """
    await websocket.accept()

    session_id = None
    session = None

    try:
        # Get client info
        client_host = websocket.client.host if websocket.client else "unknown"
        user_agent = websocket.headers.get("user-agent", "unknown")

        logger.info(
            f"WebSocket connection from {client_host}, "
            f"voice_config={voice_config}, system_prompt={system_prompt}"
        )

        # Create session
        session_create = SessionCreate(
            user_id=None,  # Anonymous for now, can be linked later
            session_type=SessionType.WEBSOCKET,
            voice_config=voice_config,
            system_prompt=system_prompt,
            metadata={
                "client_host": client_host,
                "user_agent": user_agent,
            },
        )

        session = session_service.create_session(
            session_create,
            remote_addr=client_host,
            user_agent=user_agent,
        )
        session_id = session.id

        logger.info(f"Created session {session_id}")

        # Update session to active
        session_service.update_session_status(session_id, SessionStatus.ACTIVE)

        # Create Pipecat pipeline
        try:
            pipeline_task = await pipeline_service.create_pipeline(
                session_id=session_id,
                websocket=websocket,
                voice_config=voice_config,
                system_prompt=system_prompt,
            )

            logger.info(f"Pipeline created for session {session_id}")

            # Run pipeline (this will block until completion or error)
            await pipeline_service.run_pipeline(session_id, pipeline_task)

        except Exception as e:
            logger.error(f"Pipeline error for session {session_id}: {e}")
            session_service.add_session_error(session_id, str(e))
            session_service.update_session_status(session_id, SessionStatus.FAILED)
            raise

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
        if session_id:
            session_service.update_session_status(session_id, SessionStatus.COMPLETED)

    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        if session_id:
            session_service.add_session_error(session_id, str(e))
            session_service.update_session_status(session_id, SessionStatus.FAILED)
        await websocket.close(code=1011, reason=str(e))

    finally:
        # Cleanup
        if session_id:
            await pipeline_service.stop_pipeline(session_id)
            await pipeline_service.cleanup_pipeline(session_id)
            logger.info(f"Cleaned up session {session_id}")


@router.websocket("/ws/auth")
async def authenticated_voice_websocket(
    websocket: WebSocket,
    api_key: Optional[str] = Query(None),
    voice_config: str = Query(default="conversational"),
    system_prompt: str = Query(default="default"),
    session_service: SessionService = Depends(get_session_service),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """
    Authenticated WebSocket endpoint for voice conversations

    Args:
        websocket: WebSocket connection
        api_key: API key for authentication
        voice_config: Voice configuration key
        system_prompt: System prompt key
        session_service: Session service dependency
        pipeline_service: Pipeline service dependency

    Note:
        API key should be provided as a query parameter
        Example: ws://localhost:8000/api/voice/ws/auth?api_key=pk_xxx
    """
    # Authenticate via API key
    from ..services.user_service import get_user_service

    user_service = get_user_service()
    user = None

    if api_key:
        user = user_service.get_user_by_api_key(api_key)
        if not user or not user.is_active:
            await websocket.close(code=1008, reason="Invalid or inactive API key")
            return
    else:
        await websocket.close(code=1008, reason="API key required")
        return

    await websocket.accept()

    session_id = None

    try:
        # Get client info
        client_host = websocket.client.host if websocket.client else "unknown"
        user_agent = websocket.headers.get("user-agent", "unknown")

        logger.info(
            f"Authenticated WebSocket connection from user {user.username} ({client_host})"
        )

        # Create session linked to user
        session_create = SessionCreate(
            user_id=user.id,
            session_type=SessionType.WEBSOCKET,
            voice_config=voice_config,
            system_prompt=system_prompt,
            metadata={
                "client_host": client_host,
                "user_agent": user_agent,
                "username": user.username,
            },
        )

        session = session_service.create_session(
            session_create,
            remote_addr=client_host,
            user_agent=user_agent,
        )
        session_id = session.id

        logger.info(f"Created authenticated session {session_id} for user {user.username}")

        # Update session to active
        session_service.update_session_status(session_id, SessionStatus.ACTIVE)

        # Create and run pipeline
        try:
            pipeline_task = await pipeline_service.create_pipeline(
                session_id=session_id,
                websocket=websocket,
                voice_config=voice_config,
                system_prompt=system_prompt,
            )

            await pipeline_service.run_pipeline(session_id, pipeline_task)

        except Exception as e:
            logger.error(f"Pipeline error for session {session_id}: {e}")
            session_service.add_session_error(session_id, str(e))
            session_service.update_session_status(session_id, SessionStatus.FAILED)
            raise

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
        if session_id:
            session_service.update_session_status(session_id, SessionStatus.COMPLETED)

    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        if session_id:
            session_service.add_session_error(session_id, str(e))
            session_service.update_session_status(session_id, SessionStatus.FAILED)
        await websocket.close(code=1011, reason=str(e))

    finally:
        # Cleanup
        if session_id:
            await pipeline_service.stop_pipeline(session_id)
            await pipeline_service.cleanup_pipeline(session_id)
            logger.info(f"Cleaned up authenticated session {session_id}")


@router.get("/health")
async def voice_health_check():
    """Health check endpoint for voice service"""
    return {
        "status": "healthy",
        "service": "voice",
        "ollama_url": settings.OLLAMA_BASE_URL,
        "ollama_model": settings.OLLAMA_MODEL,
        "cartesia_configured": bool(settings.CARTESIA_API_KEY),
    }
