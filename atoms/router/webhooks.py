from fastapi import APIRouter, Request
from loguru import logger

router = APIRouter()


@router.post("/webhooks/twilio/recording")
async def twilio_recording_webhook(request: Request):
    """Handle Twilio recording status callbacks."""
    try:
        form = await request.form()
        call_data = dict(form)

        call_sid = call_data.get("CallSid")
        recording_url = call_data.get("RecordingUrl")
        recording_status = call_data.get("RecordingStatus")

        logger.info(
            f"Twilio recording callback - Call: {call_sid}, Status: {recording_status}, URL: {recording_url}"
        )

        return {"status": "received"}

    except Exception as e:
        logger.error(f"Error handling Twilio recording webhook: {e}")
        return {"error": str(e)}


@router.post("/webhooks/plivo/recording")
async def plivo_recording_webhook(request: Request):
    """Handle Plivo recording status callbacks."""
    try:
        form = await request.form()
        call_data = dict(form)

        call_uuid = call_data.get("CallUUID")
        recording_url = call_data.get("RecordingUrl")
        recording_id = call_data.get("RecordingID")

        logger.info(
            f"Plivo recording callback - Call: {call_uuid}, ID: {recording_id}, URL: {recording_url}"
        )

        # Here you can add logic to save recording info to database
        # or trigger other processes

        return {"status": "received"}

    except Exception as e:
        logger.error(f"Error handling Plivo recording webhook: {e}")
        return {"error": str(e)}
