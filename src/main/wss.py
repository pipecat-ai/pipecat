from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List

class ConnectionManager:
    _instance = None  # This will hold the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConnectionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Ensure that init doesn't re-initialize on subsequent calls
        if not hasattr(self, "initialized"):
            self.active_connections: Dict[str, List[WebSocket]] = {}
            self.initialized = True

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            self.active_connections[room_id].remove(websocket)
            if len(self.active_connections[room_id]) == 0:
                del self.active_connections[room_id]

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)  # Send JSON instead of plain text

    async def broadcast(self, message: str, room_id: str):
        print("broadcasting.....", message)
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                await connection.send_text(message)

    async def broadcast_json(self, message: dict, room_id: str):
        print("broadcasting JSON.....", message)
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                await connection.send_json(message)  # Send JSON to all clients
