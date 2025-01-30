# Telnyx Chatbot

This project is a FastAPI-based chatbot that integrates with Telnyx to handle WebSocket connections and provide real-time communication. The project includes endpoints for starting a call and handling WebSocket connections.

## Table of Contents

- [Telnyx Chatbot](#telnyx-chatbot)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Configure Telnyx TeXML application](#configure-telnyx-texml-application)
  - [Running the Application](#running-the-application)
    - [Using Python (Option 1)](#using-python-option-1)
    - [Using Docker (Option 2)](#using-docker-option-2)
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
- Telnyx Account

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

## Configure Telnyx TeXML application

1. **Start ngrok**:
   In a new terminal, start ngrok to tunnel the local server:

   ```sh
   ngrok http 8765
   ```

2. **Update the Telnyx TeXML applications Webhook**:

   - Go to your TeXML configuration page
   - Provide the ngrok URL to the Webhook URL field and ensure the POST method is selected
   - Click Save at the bottom of the page

3. **Configure streams.xml**:
   - Copy the template file to create your local version:
     ```sh
     cp templates/streams.xml.template templates/streams.xml
     ```
   - In `templates/streams.xml`, replace `<your server url>` with your ngrok URL (without `https://`)
   - The final URL should look like: `wss://abc123.ngrok.io/ws`
   - The encoding (`bidirectionalCodec`) should be `PCMU` or `PCMA` depending on your needs. Based on selected encoding, set the outbound_encoding in `server.py` when the bot is initialized.
   - The inbound encoding can be controlled from the application configuration for inbound calls and dial/transfer commands for outbound calls.

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
   docker build -t telnyx-chatbot .
   ```

2. **Run the Docker container**:
   ```sh
   docker run -it --rm -p 8765:8765 telnyx-chatbot
   ```

The server will start on port 8765. Keep this running while you test with Telnyx.

## Usage

To start a call, simply make a call to your configured Telnyx phone number. The webhook URL will direct the call to your FastAPI application, which will handle it accordingly.
