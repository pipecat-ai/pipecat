#!/bin/bash

# Pipecat AI Backend Startup Script

set -e

echo "================================="
echo "Pipecat AI Backend Server"
echo "================================="
echo ""

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv ../venv"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ../venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "Copying .env.example to .env..."
    cp .env.example .env
    echo "✅ Please edit .env and add your API keys!"
    echo ""
fi

# Check if Ollama is running
echo "🔍 Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not running!"
    echo "Please start Ollama: ollama serve"
    exit 1
fi

# Check if required model is available
echo "🔍 Checking for Ollama model..."
OLLAMA_MODEL=$(grep OLLAMA_MODEL .env | cut -d '=' -f2)
if ollama list | grep -q "$OLLAMA_MODEL"; then
    echo "✅ Model $OLLAMA_MODEL is available"
else
    echo "⚠️  Model $OLLAMA_MODEL not found!"
    echo "Pulling model... (this may take a few minutes)"
    ollama pull "$OLLAMA_MODEL"
fi

# Check if Cartesia API key is set
echo "🔍 Checking Cartesia API key..."
CARTESIA_KEY=$(grep CARTESIA_API_KEY .env | cut -d '=' -f2)
if [ "$CARTESIA_KEY" = "your_cartesia_api_key_here" ] || [ -z "$CARTESIA_KEY" ]; then
    echo "⚠️  Cartesia API key not set!"
    echo "Please add your Cartesia API key to .env"
    echo "Get it from: https://cartesia.ai"
    echo ""
fi

echo ""
echo "================================="
echo "🚀 Starting server..."
echo "================================="
echo ""

# Start the server
python -m backend.main

# If the above doesn't work, try:
# uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
