# AwaazAI Chatbot

This project is a FastAPI-based chatbot that integrates with AwaazAI to handle WebSocket connections and provide real-time communication. The project includes endpoints for starting a call and handling WebSocket connections.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configure Twilio URLs](#configure-twilio-urls)
- [Running the Application](#running-the-application)
- [Usage](#usage)

## Features

- **FastAPI**: A modern, fast (high-performance), web framework for building APIs with Python 3.6+.
- **WebSocket Support**: Real-time communication using WebSockets.
- **CORS Middleware**: Allowing cross-origin requests for testing.
- **Dockerized**: Easily deployable using Docker.

## Requirements

- Python 3.10
- Docker (for containerized deployment)
- ngrok (for tunneling)
- AwaazAI number setup

## Installation

1. **Set up a virtual environment** (optional but recommended):

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

3. **Create .env**:
   Copy the example environment file and update with your settings:

   ```sh
   cp env.example .env
   ```

4. **Install ngrok**:
   Follow the instructions on the [ngrok website](https://ngrok.com/download) to download and install ngrok.

## Configure AwaazAI URLs

1. **Start ngrok**:
   In a new terminal, start ngrok to tunnel the local server:

   ```sh
   ngrok http 8765
   ```

2. **Get number configured by AwaazAI at given ngrok url**:
    Contact AwaazAI team to get a number configured for given ngrok url 
## Running the Application

Choose one of these two methods to run the application:

### Using Python (Option 1)

**Run the FastAPI application**:

```sh
# Make sure you’re in the project directory and your virtual environment is activated
python server.py
```

### Using Docker (Option 2)

1. **Build the Docker image**:

   ```sh
   docker build -t awaazai-chatbot .
   ```

2. **Run the Docker container**:
   ```sh
   docker run -it --rm -p 8765:8765 awaazai-chatbot
   ```

The server will start on port 8765. Keep this running while you test with Twilio.

## Usage

To start a call, simply make a call to your configured AwaazAI phone number. The webhook URL will direct the call to your FastAPI application, which will handle it accordingly.
