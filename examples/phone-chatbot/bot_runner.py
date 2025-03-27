import argparse
import json
import os
import shlex
import subprocess
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import aiohttp
from call_connection_manager import CallConfigManager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse

from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper,
    DailyRoomObject,
    DailyRoomParams,
    DailyRoomProperties,
    DailyRoomSipParams,
)

load_dotenv(override=True)

# ----------------- Constants ----------------- #

# Maximum session time
MAX_SESSION_TIME = 5 * 60  # 5 minutes

# Required environment variables
REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "DAILY_API_KEY",
    "CARTESIA_API_KEY",
    "DEEPGRAM_API_KEY",
]

# Default LLM to use when none is specified - this determines which bot file to execute
DEFAULT_LLM = "openai"

# Call transfer configuration constants
DEFAULT_MODE = "dialout"  # Call transfer dialout mode. Options: dialout, pstn_transfer, sip_transfer, dialout_warm_transfer, sip_refer
DEFAULT_SPEAK_SUMMARY = True  # Speak a summary of the call to the operator
DEFAULT_STORE_SUMMARY = False  # Store summary of the call (for future implementation)
DEFAULT_TEST_IN_PREBUILT = False  # Test in prebuilt mode (bypasses need to dial in/out)

daily_helpers = {}


# ----------------- Configuration Helpers ----------------- #


def determine_room_capabilities(config_body: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """Determine room capabilities based on the configuration.

    This function examines the configuration to determine which capabilities
    the Daily room should have enabled.

    Args:
        config_body: Configuration dictionary that determines room capabilities

    Returns:
        Dictionary of capability flags
    """
    capabilities = {
        "enable_dialin": False,
        "enable_dialout": False,
        # Add more capabilities here in the future as needed
    }

    if not config_body:
        return capabilities

    # Check for dialin capability
    capabilities["enable_dialin"] = "dialin_settings" in config_body

    # Check for dialout capability - needed for outbound calls or transfers
    has_dialout_settings = "dialout_settings" in config_body

    # Check if there's a transfer to an operator configured
    has_call_transfer = "call_transfer" in config_body
    has_operator_number = has_call_transfer and "operatorNumber" in config_body.get(
        "call_transfer", {}
    )

    # Enable dialout if any condition requires it
    capabilities["enable_dialout"] = has_dialout_settings or has_operator_number

    return capabilities


def ensure_dialout_settings_array(body: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures dialout_settings is an array of objects.

    Args:
        body: The configuration dictionary

    Returns:
        Updated configuration with dialout_settings as an array
    """
    if "dialout_settings" in body:
        # Convert to array if it's not already one
        if not isinstance(body["dialout_settings"], list):
            body["dialout_settings"] = [body["dialout_settings"]]

    return body


def validate_body(body: Dict[str, Any]) -> None:
    """Validates the body to ensure it doesn't contain contradictory options.

    Args:
        body: The configuration dictionary

    Raises:
        HTTPException: If the configuration is invalid or ambiguous
    """
    # Ensure dialout_settings is an array
    body = ensure_dialout_settings_array(body)

    # Check for incompatible scenario combinations
    has_call_transfer = "call_transfer" in body
    has_voicemail = "voicemail_detection" in body

    # Test in Prebuilt flags
    call_transfer_test = has_call_transfer and body.get("call_transfer", {}).get(
        "testInPrebuilt", False
    )
    voicemail_test = has_voicemail and body.get("voicemail_detection", {}).get(
        "testInPrebuilt", False
    )

    # Error scenarios
    errors = []

    # The restriction for having both dialin and dialout settings has been removed
    # This allows for future functionality that might use both simultaneously

    # Cannot have both call_transfer and voicemail_detection
    if has_call_transfer and has_voicemail:
        errors.append(
            "Cannot have both 'call_transfer' and 'voicemail_detection' in the same configuration"
        )

    # The restriction that dialin_settings can only be used with call_transfer has been removed
    # This allows greater flexibility for future scenarios

    # The restriction that dialout_settings can only be used with voicemail_detection has been removed
    # This allows call_transfer to work with dialout_settings for future functionality
    # where the bot might need to dial out to customers

    # If we have any errors, raise an exception with all the problems
    if errors:
        error_message = "Invalid configuration: " + "; ".join(errors)
        raise HTTPException(status_code=400, detail=error_message)


def ensure_prompt_config(body: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures the body has appropriate prompts settings, but doesn't add defaults.

    Only makes sure the prompt section exists, allowing the bot script to handle defaults.

    Args:
        body: The configuration dictionary

    Returns:
        Updated configuration with prompt settings section
    """
    if "prompts" not in body:
        body["prompts"] = []
    return body


def create_call_transfer_settings(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create call transfer settings based on configuration and customer mapping.

    Args:
        body: The configuration dictionary

    Returns:
        Call transfer settings dictionary
    """
    # Return existing settings if already specified
    if "call_transfer" in body:
        return body["call_transfer"]

    # Default transfer settings
    transfer_settings = {
        "mode": DEFAULT_MODE,
        "speakSummary": DEFAULT_SPEAK_SUMMARY,
        "storeSummary": DEFAULT_STORE_SUMMARY,
        "testInPrebuilt": DEFAULT_TEST_IN_PREBUILT,
    }

    # Check if we have dialin settings
    if "dialin_settings" in body:
        # Create a temporary routing manager just for customer lookup
        call_config_manager = CallConfigManager(body)

        # Get caller info
        caller_info = call_config_manager.get_caller_info()
        from_number = caller_info.get("caller_number")

        if from_number:
            # Get customer name from phone number
            customer_name = call_config_manager.get_customer_name(from_number)

            # If we know the customer name, add it to the config for the bot to use
            if customer_name:
                transfer_settings["customerName"] = customer_name

    return transfer_settings


# ----------------- Daily Room Management ----------------- #


async def create_daily_room(room_url: Optional[str], config_body: Optional[Dict[str, Any]] = None):
    """Create or retrieve a Daily room with appropriate properties based on the configuration.

    Args:
        room_url: Optional existing room URL
        config_body: Optional configuration that determines room capabilities

    Returns:
        Dict containing room URL, token, and SIP endpoint
    """
    if not room_url:
        # Get room capabilities based on the configuration
        capabilities = determine_room_capabilities(config_body)

        # Configure SIP parameters if dialin is needed
        sip_params = None
        if capabilities["enable_dialin"]:
            sip_params = DailyRoomSipParams(
                display_name="dialin-user", video=False, sip_mode="dial-in", num_endpoints=2
            )

        # Create the properties object with the appropriate settings
        properties = DailyRoomProperties(sip=sip_params)

        # Set dialout capability if needed
        if capabilities["enable_dialout"]:
            properties.enable_dialout = True

        # Log the capabilities being used
        capability_str = ", ".join([f"{k}={v}" for k, v in capabilities.items()])
        print(f"Creating room with capabilities: {capability_str}")

        params = DailyRoomParams(properties=properties)

        print("Creating new room...")
        room = await daily_helpers["rest"].create_room(params=params)
    else:
        # Check if passed room URL exists
        try:
            room = await daily_helpers["rest"].get_room_from_url(room_url)
        except Exception:
            raise HTTPException(status_code=500, detail=f"Room not found: {room_url}")

    print(f"Daily room: {room.url} {room.config.sip_endpoint}")

    # Get token for the agent
    token = await daily_helpers["rest"].get_token(room.url, MAX_SESSION_TIME)

    if not room or not token:
        raise HTTPException(status_code=500, detail="Failed to get room or token")

    return {"room": room.url, "token": token, "sip_endpoint": room.config.sip_endpoint}


# ----------------- Bot Process Management ----------------- #


async def process_dialin_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process incoming dial-in request data to create a properly formatted body.

    Converts camelCase fields received from webhook to snake_case format
    for internal consistency across the codebase.

    Args:
        data: Raw dialin data from webhook

    Returns:
        Properly formatted configuration with snake_case keys
    """
    # Create base body with dialin settings
    body = {
        "dialin_settings": {
            "to": data.get("To", ""),
            "from": data.get("From", ""),
            "call_id": data.get("callId", data.get("CallSid", "")),  # Convert to snake_case
            "call_domain": data.get("callDomain", ""),  # Convert to snake_case
        }
    }

    # Set the default LLM model - this determines which bot file to run
    body["llm"] = DEFAULT_LLM

    # Create call transfer settings (handled in bot_runner)
    body["call_transfer"] = create_call_transfer_settings(body)

    return body


async def start_bot(room_details: Dict[str, str], body: Dict[str, Any], example: str) -> bool:
    """Start a bot process with the given configuration.

    Args:
        room_details: Room URL and token
        body: Bot configuration
        example: Example script to run

    Returns:
        Boolean indicating success
    """
    llm_model = body.get("llm", DEFAULT_LLM)  # Use default if not specified
    room_url = room_details["room"]
    token = room_details["token"]

    # Properly format body as JSON string for command line
    body_json = json.dumps(body).replace('"', '\\"')
    print(f"++++ Body JSON: {body_json}")

    bot_proc = f'python3 -m {example}_{llm_model} -u {room_url} -t {token} -b "{body_json}"'
    print(f"Starting bot. Example: {example}, Model: {llm_model}, Room: {room_url}")

    try:
        command_parts = shlex.split(bot_proc)
        subprocess.Popen(command_parts, bufsize=1, cwd=os.path.dirname(os.path.abspath(__file__)))
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")


async def start_twilio_bot(room_details: Dict[str, str], call_id: str) -> bool:
    """Start a Twilio bot process with the given configuration.

    Args:
        room_details: Room URL, token, and SIP endpoint
        call_id: Twilio call ID (CallSid)

    Returns:
        Boolean indicating success
    """
    room_url = room_details["room"]
    token = room_details["token"]
    sip_endpoint = room_details["sip_endpoint"]

    # Format command for Twilio bot
    bot_proc = f"python3 -m bot_twilio -u {room_url} -t {token} -i {call_id} -s {sip_endpoint}"
    print(f"Starting Twilio bot. Room: {room_url}")

    try:
        command_parts = shlex.split(bot_proc)
        subprocess.Popen(command_parts, bufsize=1, cwd=os.path.dirname(os.path.abspath(__file__)))
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")


# ----------------- API Setup ----------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- API Endpoints ----------------- #


@app.post("/twilio_start_bot", response_class=PlainTextResponse)
async def twilio_start_bot(request: Request):
    """Handle incoming Twilio webhook calls and start a Twilio bot.

    This endpoint is called directly by Twilio as a webhook when a call is received.
    It puts the call on hold with music and starts a bot that will handle the call.
    """
    print("POST /twilio_start_bot")

    # Get form data from Twilio webhook
    try:
        form_data = await request.form()
        data = dict(form_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse Twilio form data: {str(e)}")

    # Get default room URL from environment
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", None)

    # Extract call ID from Twilio data
    call_id = data.get("CallSid")
    if not call_id:
        raise HTTPException(status_code=400, detail="Missing 'CallSid' in request")

    print(f"CallId: {call_id}")

    # Create Daily room for the Twilio call
    room_details = await create_daily_room(room_url, None)  # No special config for Twilio rooms

    # Start the Twilio bot
    await start_twilio_bot(room_details, call_id)

    # Put the call on hold until the bot is ready to handle it
    # The bot will update the call with the SIP URI when it's ready
    resp = VoiceResponse()
    resp.play(
        url="http://com.twilio.sounds.music.s3.amazonaws.com/MARKOVICHAMP-Borghestral.mp3", loop=10
    )
    return str(resp)


@app.post("/start")
async def handle_start_request(request: Request) -> JSONResponse:
    """Unified endpoint to handle bot configuration for different scenarios."""
    # Get default room URL from environment
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", None)

    try:
        # Check if this is form data (from Twilio) or JSON
        content_type = request.headers.get("content-type", "").lower()

        if "application/x-www-form-urlencoded" in content_type:
            # Handle form data from Twilio
            form_data = await request.form()
            data = dict(form_data)

            # Check for CallSid which indicates this is a Twilio webhook
            if "CallSid" in data:
                # Redirect to Twilio handler for backward compatibility
                return await twilio_start_bot(request)
        else:
            # Parse JSON request data
            data = await request.json()

        # Handle webhook test
        if "test" in data:
            return JSONResponse({"test": True})

        # Handle direct dialin webhook from Daily
        if all(key in data for key in ["From", "To", "callId", "callDomain"]):
            body = await process_dialin_request(data)
        # Handle body-based request
        elif "config" in data:
            body = data["config"]

            # Ensure dialout_settings is an array if present
            body = ensure_dialout_settings_array(body)

            # Set default LLM if not specified (this will be used to determine which bot file to run)
            if "llm" not in body:
                body["llm"] = DEFAULT_LLM

            # Add call_transfer if dealing with dialin settings - handled by bot_runner
            if "dialin_settings" in body and "call_transfer" not in body:
                body["call_transfer"] = create_call_transfer_settings(body)

            # Validate the body
            validate_body(body)
        else:
            raise HTTPException(status_code=400, detail="Invalid request format")

        # Process based on body type

        # Special case for call_transfer with testInPrebuilt
        if "call_transfer" in body and body["call_transfer"].get("testInPrebuilt", False):
            # Auto-add empty dialin_settings if not present
            if "dialin_settings" not in body:
                body["dialin_settings"] = {}

            # Ensure prompt configuration
            body = ensure_prompt_config(body)

            # Handle call transfer test scenario
            room_details = await create_daily_room(room_url, body)
            await start_bot(room_details, body, "call_transfer")

            return JSONResponse(
                {
                    "status": "Bot started",
                    "room_url": room_details["room"],
                    "bot_type": "call_transfer",
                    "llm_model": body["llm"],
                }
            )

        # Regular dialin with call transfer scenario
        elif "dialin_settings" in body and "call_transfer" in body:
            # Ensure prompt configuration
            body = ensure_prompt_config(body)

            # Handle dialin call transfer scenario
            room_details = await create_daily_room(room_url, body)
            await start_bot(room_details, body, "call_transfer")

            if body.get("call_transfer", {}).get("testInPrebuilt", False):
                return JSONResponse(
                    {
                        "status": "Bot started",
                        "room_url": room_details["room"],
                        "bot_type": "call_transfer",
                        "llm_model": body["llm"],
                    }
                )
            return JSONResponse({"status": "Bot started", "bot_type": "call_transfer"})

        # Special case for voicemail detection with testInPrebuilt
        elif "voicemail_detection" in body and body["voicemail_detection"].get(
            "testInPrebuilt", False
        ):
            # Auto-add empty dialout_settings if not present
            if "dialout_settings" not in body:
                body["dialout_settings"] = [{}]  # Empty array with one object

            # Ensure prompt configuration
            body = ensure_prompt_config(body)

            # Handle voicemail detection test scenario
            room_details = await create_daily_room(room_url, body)
            await start_bot(room_details, body, "voicemail_detection")

            return JSONResponse(
                {
                    "status": "Bot started",
                    "room_url": room_details["room"],
                    "bot_type": "voicemail_detection",
                    "llm_model": body["llm"],
                }
            )

        # Regular voicemail detection scenario
        elif "dialout_settings" in body and "voicemail_detection" in body:
            # Ensure prompt configuration
            body = ensure_prompt_config(body)

            # Handle dialout voicemail detection scenario
            room_details = await create_daily_room(room_url, body)
            await start_bot(room_details, body, "voicemail_detection")

            # Get the first dialout info for logging (if available)
            dialout_info = "unknown"
            if body["dialout_settings"] and len(body["dialout_settings"]) > 0:
                first_setting = body["dialout_settings"][0]
                if "phoneNumber" in first_setting:
                    dialout_info = f"phone:{first_setting['phoneNumber']}"
                elif "sipUri" in first_setting:
                    dialout_info = f"sip:{first_setting['sipUri']}"

            if body.get("voicemail_detection", {}).get("testInPrebuilt", False):
                return JSONResponse(
                    {
                        "status": "Bot started",
                        "room_url": room_details["room"],
                        "bot_type": "voicemail_detection",
                        "llm_model": body["llm"],
                    }
                )
            return JSONResponse(
                {
                    "status": "Bot started",
                    "dialing_to": dialout_info,
                    "bot_type": "voicemail_detection",
                }
            )

        # If we got here with a valid body but didn't match any scenario
        raise HTTPException(
            status_code=400, detail="Configuration doesn't match any supported scenario"
        )

    except json.JSONDecodeError:
        # Check if this might be form data from Twilio
        try:
            content_type = request.headers.get("content-type", "").lower()
            if "application/x-www-form-urlencoded" in content_type:
                return await twilio_start_bot(request)
        except Exception:
            pass

        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Request processing error: {str(e)}")


# ----------------- Main ----------------- #

if __name__ == "__main__":
    # Check environment variables
    for env_var in REQUIRED_ENV_VARS:
        if env_var not in os.environ:
            raise Exception(f"Missing environment variable: {env_var}.")

    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host address"
    )
    parser.add_argument("--port", type=int, default=os.getenv("PORT", 7860), help="Port number")
    parser.add_argument("--reload", action="store_true", default=True, help="Reload code on change")

    config = parser.parse_args()

    try:
        import uvicorn

        uvicorn.run("bot_runner:app", host=config.host, port=config.port, reload=config.reload)

    except KeyboardInterrupt:
        print("Pipecat runner shutting down...")
