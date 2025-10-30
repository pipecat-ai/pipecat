"""Twilio integration routes for phone call handling"""

from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.responses import Response
from loguru import logger

from ..models.session import SessionCreate, SessionStatus, SessionType
from ..services.session_service import get_session_service, SessionService
from ..services.pipeline_service import get_pipeline_service, PipelineService
from ..config import settings

router = APIRouter(prefix="/api/twilio", tags=["twilio"])


@router.post("/incoming")
async def handle_incoming_call(request: Request):
    """
    Handle incoming Twilio voice call

    This endpoint receives the initial webhook from Twilio when a call comes in.
    It returns TwiML to connect the call to our WebSocket endpoint.

    Args:
        request: FastAPI request

    Returns:
        TwiML response to connect to WebSocket
    """
    # Get form data from Twilio webhook
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")
    from_number = form_data.get("From", "unknown")
    to_number = form_data.get("To", "unknown")

    logger.info(
        f"Incoming Twilio call: CallSid={call_sid}, From={from_number}, To={to_number}"
    )

    # Build WebSocket URL
    protocol = "wss" if request.url.scheme == "https" else "ws"
    ws_url = f"{protocol}://{request.url.netloc}/api/twilio/ws"

    # Return TwiML to connect call to WebSocket
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="callSid" value="{call_sid}" />
            <Parameter name="from" value="{from_number}" />
            <Parameter name="to" value="{to_number}" />
        </Stream>
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@router.websocket("/ws")
async def twilio_websocket(
    websocket: WebSocket,
    session_service: SessionService = Depends(get_session_service),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """
    WebSocket endpoint for Twilio Media Streams

    Twilio will connect to this endpoint and stream audio data.
    The audio format is mulaw encoded at 8kHz.

    Args:
        websocket: WebSocket connection from Twilio
        session_service: Session service dependency
        pipeline_service: Pipeline service dependency

    Flow:
        1. Accept WebSocket from Twilio
        2. Receive start message with call metadata
        3. Create session
        4. Create Pipecat pipeline
        5. Handle bidirectional audio streaming
        6. Clean up on disconnect
    """
    await websocket.accept()

    session_id = None
    call_sid = None
    stream_sid = None

    try:
        logger.info("Twilio WebSocket connected")

        # Wait for the start message from Twilio
        data = await websocket.receive_json()

        if data.get("event") == "start":
            stream_sid = data["streamSid"]
            start_data = data.get("start", {})
            call_sid = start_data.get("callSid", "unknown")
            from_number = start_data.get("customParameters", {}).get("from", "unknown")
            to_number = start_data.get("customParameters", {}).get("to", "unknown")

            logger.info(
                f"Twilio stream started: StreamSid={stream_sid}, "
                f"CallSid={call_sid}, From={from_number}, To={to_number}"
            )

            # Create session for this call
            session_create = SessionCreate(
                user_id=None,  # Anonymous phone call
                session_type=SessionType.TWILIO,
                voice_config="conversational",
                system_prompt="default",
                metadata={
                    "from_number": from_number,
                    "to_number": to_number,
                    "call_sid": call_sid,
                    "stream_sid": stream_sid,
                },
            )

            session = session_service.create_session(
                session_create,
                remote_addr=from_number,
                user_agent="Twilio",
            )
            session_id = session.id

            # Update Twilio-specific info
            session_service.update_twilio_info(session_id, call_sid, stream_sid)

            logger.info(f"Created Twilio session {session_id}")

            # Update session to active
            session_service.update_session_status(session_id, SessionStatus.ACTIVE)

            # Create Pipecat pipeline with Twilio-optimized settings
            try:
                pipeline_task = await pipeline_service.create_pipeline(
                    session_id=session_id,
                    websocket=websocket,
                    voice_config="conversational",
                    system_prompt="default",
                    custom_config={
                        "audio_format": "mulaw",
                        "sample_rate": 8000,
                    },
                )

                logger.info(f"Twilio pipeline created for session {session_id}")

                # Run pipeline
                await pipeline_service.run_pipeline(session_id, pipeline_task)

            except Exception as e:
                logger.error(f"Twilio pipeline error for session {session_id}: {e}")
                session_service.add_session_error(session_id, str(e))
                session_service.update_session_status(session_id, SessionStatus.FAILED)
                raise

        else:
            logger.warning(f"Unexpected Twilio message: {data.get('event')}")

    except WebSocketDisconnect:
        logger.info(f"Twilio WebSocket disconnected for session {session_id}")
        if session_id:
            session_service.update_session_status(session_id, SessionStatus.COMPLETED)

    except Exception as e:
        logger.error(f"Twilio WebSocket error for session {session_id}: {e}")
        if session_id:
            session_service.add_session_error(session_id, str(e))
            session_service.update_session_status(session_id, SessionStatus.FAILED)

    finally:
        # Cleanup
        if session_id:
            await pipeline_service.stop_pipeline(session_id)
            await pipeline_service.cleanup_pipeline(session_id)
            logger.info(f"Cleaned up Twilio session {session_id}")


@router.post("/status")
async def handle_call_status(request: Request):
    """
    Handle Twilio call status callbacks

    Twilio sends status updates to this endpoint during the call lifecycle.

    Args:
        request: FastAPI request with Twilio status data

    Returns:
        Success response
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")
    call_status = form_data.get("CallStatus", "unknown")

    logger.info(f"Twilio call status: CallSid={call_sid}, Status={call_status}")

    # You can update session status based on call status here
    # For example, mark session as completed when call ends

    return {"status": "ok"}


@router.get("/health")
async def twilio_health_check():
    """Health check endpoint for Twilio service"""
    return {
        "status": "healthy",
        "service": "twilio",
        "account_configured": bool(settings.TWILIO_ACCOUNT_SID),
    }
