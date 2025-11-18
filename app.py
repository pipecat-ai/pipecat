import os
import asyncio
import httpx
import uuid
from typing import Dict
from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Pipecat imports
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.transcriptions.language import Language
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.frames.frames import (
    EndFrame,
    LLMRunFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame
)

# -----------------------------------------------------
# Load environment variables
# -----------------------------------------------------
load_dotenv(override=True)

# -----------------------------------------------------
# Session Management
# -----------------------------------------------------
conversation_sessions: Dict[str, list] = {}

# -----------------------------------------------------
# Function/Tool Definitions
# -----------------------------------------------------
TRANSACTION_TOOLS = [
    {
        "type": "function",
        "name": "query_transactions",
        "description": "Query and retrieve transaction records from the database. Use this when users ask about client transactions, purchases, refunds, transaction status, approved transactions, declined transactions, or financial data. Always call this when a client ID is mentioned.",
        "parameters": {
            "type": "object",
            "properties": {
                "clientId": {
                    "type": "number",
                    "description": "The client ID to filter transactions by (e.g., 5001, 5002, 5003)"
                },
                "type": {
                    "type": "string",
                    "enum": ["PURCHASE", "REFUND"],
                    "description": "Filter by transaction type"
                },
                "status": {
                    "type": "string",
                    "enum": ["APPROVED", "DECLINED"],
                    "description": "Filter by transaction status"
                }
            }
        }
    },
    {
        "type": "function",
        "name": "search_documents",
        "description": "Search the knowledge base for relevant information from uploaded documents. Use this when users ask questions about specific topics, products, services, or information that might be in the documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information in documents"
                }
            },
            "required": ["query"]
        }
    },
    {
        "type": "function",
        "name": "web_search",
        "description": "Search the web for current information when the knowledge base does not contain relevant information. Use this for general knowledge questions or topics not covered in documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find information on the web"
                }
            },
            "required": ["query"]
        }
    },
    {
        "type": "function",
        "name": "send_email_report",
        "description": "Send a transaction report via email to a specified recipient. Use this when users ask to email or send transaction reports.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Email address of the recipient"
                },
                "subject": {
                    "type": "string",
                    "description": "Subject line for the email"
                },
                "clientId": {
                    "type": "number",
                    "description": "Client ID to generate the report for"
                }
            },
            "required": ["to", "clientId"]
        }
    },
    {
        "type": "function",
        "name": "generate_transaction_chart",
        "description": "Generate a chart visualization of transaction trends over time for a specific client.",
        "parameters": {
            "type": "object",
            "properties": {
                "clientId": {
                    "type": "number",
                    "description": "The client ID to generate chart for"
                }
            },
            "required": ["clientId"]
        }
    }
]

# -----------------------------------------------------
# Function Execution Handler
# -----------------------------------------------------
async def execute_function(function_name: str, arguments: dict):
    """Execute function calls against Supabase edge functions"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        logger.error("❌ Supabase credentials not configured")
        return {"error": "Service configuration error", "success": False}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {supabase_key}"
    }

    logger.info(f"🔧 Executing function: {function_name} with args: {arguments}")

    try:
        async with httpx.AsyncClient() as client:
            if function_name == "query_transactions":
                response = await client.post(
                    f"{supabase_url}/functions/v1/transaction-query",
                    headers=headers,
                    json={
                        **arguments,
                        "query": f"transactions for client {arguments.get('clientId', 'all')}"
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"✅ Transaction query result: {result}")
                return result

            elif function_name == "search_documents":
                response = await client.post(
                    f"{supabase_url}/functions/v1/rag-retrieval",
                    headers=headers,
                    json={"query": arguments["query"]},
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"✅ Document search result: {result}")
                return result

            elif function_name == "web_search":
                response = await client.post(
                    f"{supabase_url}/functions/v1/web-search-tool",
                    headers=headers,
                    json={"query": arguments["query"]},
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"✅ Web search result: {result}")
                return result

            elif function_name == "send_email_report":
                # First get transaction data
                logger.info(f"📧 Fetching transaction data for client {arguments['clientId']}")
                query_response = await client.post(
                    f"{supabase_url}/functions/v1/transaction-query",
                    headers=headers,
                    json={
                        "clientId": arguments["clientId"],
                        "query": f"transactions for client {arguments['clientId']}"
                    },
                    timeout=30.0
                )
                query_response.raise_for_status()
                query_data = query_response.json()

                # Then send email
                logger.info(f"📧 Sending email to {arguments['to']}")
                email_response = await client.post(
                    f"{supabase_url}/functions/v1/transaction-email",
                    headers=headers,
                    json={
                        "to": arguments["to"],
                        "subject": arguments.get("subject", "Transaction Intelligence Report"),
                        "transactionSummary": query_data.get("summary", "")
                    },
                    timeout=30.0
                )
                email_response.raise_for_status()
                result = email_response.json()
                logger.info(f"✅ Email sent successfully: {result}")
                return result

            elif function_name == "generate_transaction_chart":
                response = await client.post(
                    f"{supabase_url}/functions/v1/transaction-chart",
                    headers=headers,
                    json=arguments,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"✅ Chart generated: {result}")
                return result

            else:
                logger.error(f"❌ Unknown function: {function_name}")
                return {"error": f"Unknown function: {function_name}", "success": False}

    except httpx.HTTPStatusError as e:
        logger.error(f"❌ HTTP error calling {function_name}: {e.response.status_code} - {e.response.text}")
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}", "success": False}
    except Exception as e:
        logger.error(f"❌ Function execution error for {function_name}: {e}")
        return {"error": str(e), "success": False}

# -----------------------------------------------------
# Function Call Processor
# -----------------------------------------------------
class FunctionCallProcessor(FrameProcessor):
    """Handles function call execution and returns results to LLM"""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, FunctionCallInProgressFrame):
            logger.info(f"🔧 Function call detected: {frame.function_name}")

            try:
                # Execute the function
                result = await execute_function(frame.function_name, frame.arguments)

                # Create result frame
                result_frame = FunctionCallResultFrame(
                    function_name=frame.function_name,
                    call_id=frame.call_id,
                    result=result
                )

                await self.push_frame(result_frame)
                logger.info(f"✅ Function result pushed to pipeline")

            except Exception as e:
                logger.error(f"❌ Error processing function call: {e}")
                error_result = FunctionCallResultFrame(
                    function_name=frame.function_name,
                    call_id=frame.call_id,
                    result={"error": str(e), "success": False}
                )
                await self.push_frame(error_result)
        else:
            await self.push_frame(frame)

# -----------------------------------------------------
# FastAPI Server Setup
# -----------------------------------------------------
app = FastAPI(title="Pipecat Speech2Speech", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "🎙️ Nova Voice Assistant is running!",
        "websocket": "/ws",
        "health": "/health",
        "sessions": len(conversation_sessions)
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(conversation_sessions)
    }

@app.get("/test_client.html")
async def test_client():
    """Serve the test client HTML file"""
    import pathlib
    file_path = pathlib.Path(__file__).parent / "test_client.html"
    return FileResponse(str(file_path))

# -----------------------------------------------------
# Speech2Speech Pipeline 
# -----------------------------------------------------
async def run_pipeline(transport, session_id: str, handle_sigint: bool = False):
    logger.info(f"🎤 Starting pipeline for session: {session_id}")

    # --- Services setup ---
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("VOICE_ID", "829ccd10-f8b3-43cd-b8a0-4aeaa81f3b30"),
    )

    stt = SpeechmaticsSTTService(
        api_key=os.getenv("SPEECHMATICS_API_KEY"),
        language=Language.EN,
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        params=BaseOpenAILLMService.InputParams(
            temperature=0.7,
            tools=TRANSACTION_TOOLS,
            tool_choice="auto"
        ),
    )

    # --- Get or create conversation context for this session ---
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = [
            {
                "role": "system",
                "content": (
                    "You are Julia, a helpful AI assistant for a financial transaction intelligence system. "
                    "You help users with:\n"
                    "- Querying client transactions (use query_transactions when users mention client IDs or ask about transactions)\n"
                    "- Searching documents in the knowledge base (use search_documents for document-related questions)\n"
                    "- Web searches for general information (use web_search when documents don't have the answer)\n"
                    "- Sending email reports (use send_email_report when users ask to email reports)\n"
                    "- Generating transaction charts (use generate_transaction_chart for visualizations)\n\n"
                    "IMPORTANT: When users mention a client ID or ask about transactions, you MUST call the query_transactions function first. "
                    "Always explain what you found in simple, conversational terms suitable for voice interaction. "
                    "Keep responses concise and natural for real-time speech. "
                    "Maintain context throughout the conversation - remember what was discussed earlier."
                ),
            }
        ]

    messages = conversation_sessions[session_id]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # --- Audio Buffer (records all user audio) ---
    audiobuffer = AudioBufferProcessor(user_continuous_stream=True)

    # --- Function call processor ---
    function_processor = FunctionCallProcessor()

    # --- Define processing pipeline ---
    pipeline = Pipeline([
        transport.input(),               # Mic input
        stt,                             # Speech → Text
        context_aggregator.user(),       # Update user context
        llm,                             # Generate response (with function calling)
        function_processor,              # Handle function calls
        tts,                             # Text → Speech
        transport.output(),              # Send audio back to user
        audiobuffer,                     # Record audio (optional)
        context_aggregator.assistant(),  # Update assistant context
    ])

    # --- Pipeline task parameters ---
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            allow_interruptions=True,
        ),
    )

    # --- Event Hooks ---
    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info(f"✅ Client connected - Session: {session_id}")
        # Trigger LLM startup message
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info(f"❌ Client disconnected - Session: {session_id}")
        await task.queue_frames([EndFrame()])

        # Clean up session after some time (optional)
        # You might want to keep it for reconnection purposes
        # await asyncio.sleep(300)  # Keep for 5 minutes
        # if session_id in conversation_sessions:
        #     del conversation_sessions[session_id]

    # --- Run the pipeline ---
    runner = PipelineRunner(handle_sigint=handle_sigint, force_gc=True)
    await runner.run(task)

# -----------------------------------------------------
# WebSocket endpoint
# -----------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Create unique session ID for this connection
    session_id = str(uuid.uuid4())
    logger.info(f"🔌 WebSocket client connected - Session: {session_id}")

    params = FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        add_wav_header=False,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
        vad_audio_passthrough=True,
        serializer=ProtobufFrameSerializer(),
    )

    transport = FastAPIWebsocketTransport(websocket, params)

    try:
        await run_pipeline(transport, session_id, handle_sigint=False)
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket client disconnected - Session: {session_id}")
    except Exception as e:
        logger.exception(f"❗ Error in WebSocket pipeline for session {session_id}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        # Optional: Clean up session after disconnect
        # You might want to keep sessions for a while to handle reconnections
        logger.info(f"🧹 Cleaning up session: {session_id}")
