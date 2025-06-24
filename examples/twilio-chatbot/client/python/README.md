# Python Client for Server Testing

This Python client enables automated testing of the server via WebSocket without the need to make actual phone calls.

## Setup Instructions

### 1. Configure the Stream Template

Edit the `templates/streams.xml` file to point to your server’s WebSocket endpoint. For example:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="ws://localhost:8765/ws" />
  </Connect>
  <Pause length="40"/>
</Response>
```

### 2. Start the Server in Test Mode

Run the server with the `-t` flag to indicate test mode:

```sh
# Ensure you're in the project directory and your virtual environment is activated
python server.py -t
```

### 3. Run the Client

Start the client and point it to the server URL:

```sh
python client.py -u http://localhost:8765 -c 2
```

- `-u`: Server URL (default is `http://localhost:8765`)
- `-c`: Number of concurrent client connections (e.g., 2)
