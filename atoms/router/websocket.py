import json

from fastapi import APIRouter, WebSocket
from loguru import logger
from services.redis import redis_service

router = APIRouter()


@router.websocket("/ws/twilio")
async def twilio_websocket(websocket: WebSocket):
    """Handle Twilio WebSocket connections."""
    await websocket.accept()
    logger.info("Twilio WebSocket connection accepted")

    try:
        await websocket.receive_text()

        start_message = await websocket.receive_text()
        call_data = json.loads(start_message)

        stream_sid = call_data["start"]["streamSid"]
        call_sid = call_data["start"]["callSid"]

        logger.info(f"Twilio call started - Stream: {stream_sid}, Call: {call_sid}")

        try:
            call_details = await redis_service.get_call_details(call_sid)
            if call_details is None:
                logger.warning(
                    f"No call details found in Redis for call_sid: {call_sid}. Using defaults."
                )
                call_details = {}
        except Exception as e:
            logger.warning(
                f"Failed to retrieve call details from Redis for call_sid: {call_sid}. Error: {e}. Using defaults."
            )
            call_details = {}

        await redis_service.delete_call_details(call_id=call_sid)

        from bot import run_bot

        await run_bot(websocket, stream_sid, call_sid, provider="twilio", call_details=call_details)

    except Exception as e:
        logger.error(f"Error in Twilio WebSocket: {e}")
    finally:
        logger.info("Twilio WebSocket connection closed")


@router.websocket("/ws/plivo")
async def plivo_websocket(websocket: WebSocket):
    """Handle Plivo WebSocket connections."""
    await websocket.accept()
    logger.info("Plivo WebSocket connection accepted")

    try:
        start_message = await websocket.receive_text()
        call_data = json.loads(start_message)

        stream_sid = call_data["start"]["streamId"]
        call_sid = call_data["start"]["callId"]

        logger.info(f"Plivo call started - Stream: {stream_sid}, Call: {call_sid}")

        try:
            call_details = await redis_service.get_call_details(call_sid)
            if call_details is None:
                logger.warning(
                    f"No call details found in Redis for call_sid: {call_sid}. Using defaults."
                )
                call_details = {}
        except Exception as e:
            logger.warning(
                f"Failed to retrieve call details from Redis for call_sid: {call_sid}. Error: {e}. Using defaults."
            )
            call_details = {}

        await redis_service.delete_call_details(call_id=call_sid)

        from bot import run_bot

        await run_bot(websocket, stream_sid, call_sid, provider="plivo", call_details=call_details)

    except Exception as e:
        logger.error(f"Error in Plivo WebSocket: {e}")
    finally:
        logger.info("Plivo WebSocket connection closed")
