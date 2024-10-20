# Transports Guide

Transports are a crucial component of the Pipecat framework, serving as the interface between your pipeline and the outside world. They handle the input and output of frames, allowing your pipeline to receive and send data.

## What are Transports?

Transports provide input and output frame processors to receive or send frames respectively. They act as the entry and exit points for data in your Pipecat pipeline.


## Built-in Transports

Pipecat comes with several built-in transports to cover common use cases:

1. **DailyTransport**: Integrates with WebRTC sessions using Daily.co rooms.
2. **WebSocketTransport**: Allows communication over WebSocket connections.
3. **HTTPTransport**: Enables HTTP-based communication for RESTful APIs.
4. **FileTransport**: Reads from and writes to files on the local filesystem.

### DailyTransport

The DailyTransport is designed for real-time communication using WebRTC through Daily.co rooms. It's ideal for applications requiring audio/video streaming and real-time data exchange.

Usage example:
```python
from pipecat.transports import WebSocketTransport
transport = WebSocketTransport(url="ws://example.com/websocket")```


### WebSocketTransport

WebSocketTransport is suitable for bi-directional, full-duplex communication over a single TCP connection. It's great for real-time applications that require low-latency data exchange.

Usage example:

```python
from pipecat.transports import HTTPTransport
transport = HTTPTransport(base_url="https://api.example.com")```


### HTTPTransport

HTTPTransport is useful for RESTful API interactions or when you need to communicate with HTTP-based services.

Usage example:
```python
from pipecat.transports import HTTPTransport
transport = HTTPTransport(base_url="https://api.example.com")
```



### FileTransport

FileTransport is used for reading from and writing to files on the local filesystem. It's helpful for processing local data or storing output locally.

Usage example:
python
from pipecat.transports import FileTransport
transport = FileTransport(file_path="path/to/file")


## Creating Custom Transports

To create a custom transport, you need to implement the `Transport` interface. Here's a basic template for creating a custom transport:

```python
from pipecat.transports import Transport
from pipecat.frames import Frame

class CustomTransport(Transport):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Initialize your transport-specific attributes here

    async def connect(self):
        # Implement connection logic here
        pass

    async def disconnect(self):
        # Implement disconnection logic here
        pass

    async def receive_frame(self) -> Frame:
        # Implement logic to receive a frame and return a Frame object
        pass

    async def send_frame(self, frame: Frame):
        # Implement logic to send a frame here
        pass

    async def is_connected(self) -> bool:
        # Implement logic to check the connection status here
        pass

```

When implementing a custom transport, ensure that you handle errors gracefully and maintain a consistent state.

## Use Cases for Different Types of Transports

1. **Real-time Communication**: Use DailyTransport or WebSocketTransport for applications requiring real-time, bi-directional communication, such as chat applications or live collaboration tools.

2. **API Integration**: Use HTTPTransport when integrating with RESTful APIs or services that communicate over HTTP.

3. **Local Data Processing**: Use FileTransport for batch processing of local files or when you need to store output locally.

4. **IoT and Sensor Data**: Create a custom transport for specific protocols used in IoT devices or sensor networks.

5. **Database Integration**: Implement a custom transport to directly interface with databases for real-time data processing.

6. **Message Queues**: Create a custom transport to integrate with message queue systems like RabbitMQ or Apache Kafka for distributed systems.

## Best Practices

1. **Error Handling**: Implement robust error handling in your transports to manage connection issues, timeouts, and other potential failures.

2. **Asynchronous Design**: Design your transports to work asynchronously to prevent blocking operations and improve performance.

3. **Configuration**: Allow for easy configuration of your transports through constructor parameters or configuration files.

4. **Logging**: Implement comprehensive logging in your transports to aid in debugging and monitoring.

5. **Testing**: Create unit tests for your custom transports to ensure they behave correctly under various conditions.

6. **Documentation**: Provide clear documentation for your custom transports, including usage examples and any specific requirements or limitations.

By following these guidelines and understanding the various transport options available, you can effectively integrate Pipecat into your data processing workflows and create robust, efficient pipelines.
