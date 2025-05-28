from fastapi import APIRouter, HTTPException
from loguru import logger
from models.OutboundService import OutboundCallRequest
from services.telephony import plivo_service, twilio_service

router = APIRouter()


@router.post("/outbound")
async def outbound(request: OutboundCallRequest):
    logger.info(
        f"Outbound call to {request.to_phone} from {request.from_phone} with provider {request.provider}"
    )
    try:
        if request.provider == "plivo":
            call_id = await plivo_service.make_outbound_call(
                to=request.to_phone, from_phone=request.from_phone
            )
            return {"call_id": call_id, "provider": "plivo", "status": "initiated"}
        elif request.provider == "twilio":
            call_id = await twilio_service.make_outbound_call(
                to=request.to_phone, from_phone=request.from_phone
            )
            return {"call_id": call_id, "provider": "twilio", "status": "initiated"}
        else:
            raise HTTPException(status_code=400, detail="Invalid provider")
    except Exception as e:
        logger.error(f"Error in outbound call: {e}")
        raise HTTPException(status_code=500, detail=str(e))
