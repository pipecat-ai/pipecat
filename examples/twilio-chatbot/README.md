# Twilio Chatbot

This project is a FastAPI-based chatbot that integrates with Twilio to handle WebSocket connections and provide real-time communication. The project includes endpoints for starting a call and handling WebSocket connections.

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
- Twilio Account

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

## Configure Twilio URLs

1. **Start ngrok**:
   In a new terminal, start ngrok to tunnel the local server:

   ```sh
   ngrok http 8765
   ```

2. **Update the Twilio Webhook**:

   - Go to your Twilio phone number's configuration page
   - Under "Voice Configuration", in the "A call comes in" section:
     - Select "Webhook" from the dropdown
     - Enter your ngrok URL (e.g., http://<ngrok_url>)
     - Ensure "HTTP POST" is selected
   - Click Save at the bottom of the page

3. **Configure streams.xml**:
   - Copy the template file to create your local version:
     ```sh
     cp templates/streams.xml.template templates/streams.xml
     ```
   - In `templates/streams.xml`, replace `<your server url>` with your ngrok URL (without `https://`)
   - The final URL should look like: `wss://abc123.ngrok.io/ws`

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
   docker build -t twilio-chatbot .
   ```

2. **Run the Docker container**:
   ```sh
   docker run -it --rm -p 8765:8765 twilio-chatbot
   ```

The server will start on port 8765. Keep this running while you test with Twilio.

## Usage

To start a call, simply make a call to your configured Twilio phone number. The webhook URL will direct the call to your FastAPI application, which will handle it accordingly.

## Testing

It is also possible to automatically test the server without making phone calls by using a software client.

First, update `templates/streams.xml` to point to your server's websocket endpoint. For example:

```
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="ws://localhost:8765/ws"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>
```

Then, start the server with `-t` to indicate we are testing:

```sh
# Make sure you’re in the project directory and your virtual environment is activated
python server.py -t
```

Finally, just point the client to the server's URL:

```sh
python client.py -u http://localhost:8765 -c 2
```

where `-c` allows you to create multiple concurrent clients.
