# Pipecat Foundational Examples

This directory contains foundational examples showing how to use Pipecat to build voice and multimodal agents. Each example demonstrates specific features of the framework, building from basic to more complex concepts.

## Prerequisites

1. If you haven't already, set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install Pipecat with the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up an `.env` file with the API keys of services you'll run.

## Running the examples

Each example is a self-contained bot that runs using our built-in `run.py` FastAPI server. The server automatically starts a web interface ([Pipecat SmallWebRTC Prebuilt](https://pypi.org/project/pipecat-ai-small-webrtc-prebuilt/)) that allows you to interact with the bot via WebRTC in your browser.

1. **Run a specific example**:

   ```bash
   python <example-name>
   ```

   For example:

   ```bash
   python 07-interruptible.py
   ```

2. **Open the web app** at the URL displayed in the console:

   ```
   Open your browser to: http://localhost:7860
   ```

3. **Start the example**:

   Click the "Connect" button in the web interface, grant camera/microphone permissions when prompted, and start interacting with the bot.

## Troubleshooting

- **No audio or video**: Make sure your browser permissions for microphone and camera are granted
- **Connection errors**: Check that your API keys are correctly set in the `.env` file
- **Missing dependencies**: Ensure you've installed all required dependencies with `pip install -r requirements.txt`
- **Port already in use**: Change the port with `--port <number>` if the default port is unavailable

### Customizing the network interface

If you have conflict on a host or port, you can customize using:

```bash
python <example-name> --host 0.0.0.0 --port 8080
```
