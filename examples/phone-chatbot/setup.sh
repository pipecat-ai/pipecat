#!/bin/bash
# Setup script for phone-chatbot

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install pipecat if it's not already installed
if ! python -c "import pipecat" &> /dev/null; then
    echo "Installing pipecat..."
    pip install pipecat-ai
fi

echo "Setup complete. Activate the virtual environment with:"
echo "source venv/bin/activate" 