import os
import tempfile
import uuid
import asyncio
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
import whisper
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask

app = FastAPI(title="Pipecat Speech↔Speech Server", version="2.0")

# Allow CORS for Bolt.new or browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health Route ---
@app.get("/")
async def root():
    return {"message": "✅ Pipecat Speech2Speech Server running!"}

# --- Existing HTTP Speech2Speech route (batch mode) ---
@app.post("/v1/speech2speech")
async def speech_to_speech(file: UploadFile = File(...), source_language: str = Form("en")):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(await file.read())
            input_path = tmp_audio.name

        model = whisper.load_model("small")
        result = model.transcribe(input_path, language=source_language)
        recognized_text = result["text"]

        output_file = f"speech_{uuid.uuid4()}.mp3"
        tts = gTTS(text=recognized_text, lang=source_language)
        tts.save(output_file)

        return FileResponse(output_file, media_type="audio/mpeg")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# --- NEW: WebSocket Streaming Route ---
@app.websocket("/v1/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("🔊 Connected to Pipecat Real-Time Speech Stream")

    # Simple Pipecat placeholder (expand to full STT→LLM→TTS pipeline if desired)
    try:
        while True:
            data = await websocket.receive_bytes()
            # TODO: In production, decode and stream this audio into STT→LLM→TTS pipeline
            await asyncio.sleep(0.2)
            await websocket.send_text("Streaming response chunk...")

    except Exception as e:
        await websocket.send_text(f"❌ Error: {str(e)}")
    finally:
        await websocket.close()
