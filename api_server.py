import os
import json
import asyncio
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from contract_verification_bot import start_verification_call, ContractVerificationPipeline

# Load environment variables
load_dotenv()

app = FastAPI(title="Contract Verificatie API")

@app.post("/webhook/salesdock")
async def salesdock_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Endpoint for receiving Salesdock webhooks when a new contract is created.
    It extracts contract data and starts a verification call.
    """
    try:
        # Parse the incoming webhook data
        request_data = await request.json()
        
        # Extract contract data
        contract_data = request_data.get("contract", {})
        
        if not contract_data:
            raise HTTPException(status_code=400, detail="No contract data found in the webhook payload")
        
        # Validate required fields
        required_fields = ["contract_id", "telefoonnummer", "postcode", "huisnummer", 
                          "bedrijfsnaam", "adres", "email", "pakket"]
        
        missing_fields = [field for field in required_fields if not contract_data.get(field)]
        
        if missing_fields:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required contract fields: {', '.join(missing_fields)}"
            )
        
        # Start the verification call in the background
        call_sid = start_verification_call(contract_data)
        
        # Return immediate response
        return {
            "status": "success",
            "message": "Verification call initiated",
            "call_sid": call_sid
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        # Log the error (in a production environment, use proper logging)
        print(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/twilio/voice")
async def twilio_voice_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Endpoint for Twilio voice webhooks.
    This is called when a call is answered.
    """
    try:
        # Parse the form data from Twilio
        form_data = await request.form()
        
        # Extract call data
        call_sid = form_data.get("CallSid")
        
        # Here you would typically:
        # 1. Retrieve contract data associated with this call
        # 2. Start the verification process
        
        # For demo purposes, using sample data
        sample_contract = {
            "contract_id": "CON12345",
            "naam": "Jan Jansen",
            "bedrijfsnaam": "Jansen IT Solutions",
            "postcode": "1234AB",
            "huisnummer": "42",
            "adres": "Hoofdstraat 42, Amsterdam",
            "telefoonnummer": "+31612345678",
            "email": "jan@jansen-it.nl",
            "pakket": "Premium Support Pakket"
        }
        
        # Start the verification process in the background
        background_tasks.add_task(start_verification_process, sample_contract)
        
        # Return TwiML to start the call
        return """
        <?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say language="nl-NL">Een moment geduld alstublieft terwijl we uw gegevens verifiëren.</Say>
            <Pause length="2"/>
        </Response>
        """
        
    except Exception as e:
        print(f"Error processing Twilio webhook: {str(e)}")
        # Return a TwiML error response
        return """
        <?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say language="nl-NL">Er is helaas een fout opgetreden. Probeer het later opnieuw.</Say>
            <Hangup/>
        </Response>
        """

@app.post("/webhook/twilio/recording")
async def recording_callback(request: Request):
    """
    Endpoint for Twilio recording status callbacks.
    This is called when a recording is completed.
    """
    try:
        form_data = await request.form()
        recording_url = form_data.get("RecordingUrl")
        recording_sid = form_data.get("RecordingSid")
        call_sid = form_data.get("CallSid")
        
        # Here you would typically:
        # 1. Update your database with the recording URL
        # 2. Process or store the recording
        
        print(f"Recording completed for call {call_sid}: {recording_url}")
        
        return {"status": "success", "recording_sid": recording_sid}
        
    except Exception as e:
        print(f"Error processing recording callback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/twilio/status")
async def call_status_callback(request: Request):
    """
    Endpoint for Twilio call status callbacks.
    This is called when a call status changes.
    """
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        call_status = form_data.get("CallStatus")
        
        print(f"Call {call_sid} status changed to: {call_status}")
        
        return {"status": "success", "call_status": call_status}
        
    except Exception as e:
        print(f"Error processing status callback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def start_verification_process(contract_data):
    """
    Start the verification process for a contract.
    This runs as a background task.
    """
    verification = await ContractVerificationPipeline(contract_data).setup()
    await verification.run()

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable or use default 8000
    port = int(os.getenv("PORT", 8000))
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=port) 