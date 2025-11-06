from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from gtts import gTTS
import whisper
import os, uuid, tempfile

app = FastAPI(title="Pipecat Speech↔Speech", version="1.0")

@app.get("/")
def home():
    return {"message": "🎧 Pipecat Speech-to-Speech API is running!"}

@app.post("/v1/speech2speech")
async def speech_to_speech(file: UploadFile = File(...), source_language: str = Form("en")):
    try:
        # Step 1: Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(await file.read())
            input_path = tmp_audio.name

        # Step 2: Speech → Text using Whisper
        model = whisper.load_model("small")
        result = model.transcribe(input_path, language=source_language)
        recognized_text = result["text"]

        # Step 3: Text → Speech using gTTS
        output_file = f"speech_{uuid.uuid4()}.mp3"
        tts = gTTS(text=recognized_text, lang=source_language)
        tts.save(output_file)

        # Step 4: Return generated audio
        return FileResponse(output_file, media_type="audio/mpeg")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
