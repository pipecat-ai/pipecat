# Python Client for Server Testing

This Python client enables automated testing of the **Vonage Pipecat WebSocket server** . It opens a WS connection to your Pipecat endpoint, streams test audio (microphone) and plays back the audio received from the server.

## Setup Instructions

1. **Clone the repo and enter it**
    ```sh
    git clone https://github.com/opentok/vonage-pipecat.git
    cd vonage-pipecat/examples/vonage-speech-to-speech/client
    ```

2. **Set up a virtual environment** (optional but recommended):
    ```sh
    python -m venv .venv-client
    source .venv-client/bin/activate      # Windows: .venv-client\Scripts\activate
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Create .env**:
    Copy the example environment file and update with your settings:

    ```sh
    cp env.example .env
    ```

5. **Start an Opentok Session and Publish a stream**
    The OpenTok Session ID is required. 
    Note: You can open the OpenTok Playground (or your own app) to create an opentok session and publish a stream.
    Copy the Session ID and set it in `.env` file:
    ```sh
    VONAGE_SESSION_ID=<paste-your-session-id-here>
    ```

    Set OPENTOK_API_URL in your .env:
    ```sh
    OPENTOK_API_URL=https://api.opentok.com
    ```

    Use the **API key** and **secret** from the **same project** that created the `sessionId`.

6. **Set the Keys in .env**:
    ```sh
    # Vonage (OpenTok) credentials
    VONAGE_API_KEY=YOUR_API_KEY
    VONAGE_API_SECRET=YOUR_API_SECRET

    # Your Pipecat WebSocket endpoint (ngrok or prod)
    WS_URI=wss://<your-ngrok-domain>

    # Put existing session from playground or app which you want to connect pipecat-ai
    VONAGE_SESSION_ID=1_MX4....

    # API base
    OPENTOK_API_URL=https://api.opentok.com

    # Keep rest as same.
    ```

7. **Start your Pipecat WS server**:
    Make sure the Vonage Pipecat server is running locally and exposes a WS endpoint via ngrok

8. **Running the Client**:
    Below program will connect the opentok session created above to the pipecat-ai pipeline.
    ```sh
    python connect_and_stream.py
    ```

**Note** 
The script reads everything from .env via os.getenv().
You can still override via flags if you want, e.g.:

    ```sh
    # Example
    python connect_and_stream.py --ws-uri wss://my-ngrok/ws --audio-rate 16000
    ```
