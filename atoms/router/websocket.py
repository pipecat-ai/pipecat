import json

from fastapi import APIRouter, WebSocket
from loguru import logger

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

        from bot import run_bot

        await run_bot(websocket, stream_sid, call_sid, provider="twilio")

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

        from bot import run_bot

        await run_bot(websocket, stream_sid, call_sid, provider="plivo")

    except Exception as e:
        logger.error(f"Error in Plivo WebSocket: {e}")
    finally:
        logger.info("Plivo WebSocket connection closed")
