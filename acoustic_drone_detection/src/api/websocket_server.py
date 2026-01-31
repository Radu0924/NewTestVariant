"""
WebSocket Server Module

Real-time WebSocket streaming for drone detection events:
- Detection event streaming
- Track updates
- System status updates
- Audio level streaming
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Set, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue


class MessageType(str, Enum):
    """WebSocket message types."""
    DETECTION = "detection"
    TRACK_UPDATE = "track_update"
    TRACK_LOST = "track_lost"
    STATUS = "status"
    METRICS = "metrics"
    AUDIO_LEVEL = "audio_level"
    ERROR = "error"
    CONNECTED = "connected"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: str
    timestamp: float
    data: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "timestamp": self.timestamp,
            "data": self.data
        })


class WebSocketManager:
    """
    Manages WebSocket connections and message broadcasting.

    Supports multiple clients with subscription-based filtering.
    """

    def __init__(self):
        """Initialize WebSocket manager."""
        self._clients: Set = set()
        self._subscriptions: Dict[Any, Set[str]] = {}
        self._message_queue: queue.Queue = queue.Queue()
        self._running = False
        self._broadcast_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Statistics
        self._messages_sent = 0
        self._connections_total = 0

    async def connect(self, websocket) -> None:
        """Handle new WebSocket connection."""
        with self._lock:
            self._clients.add(websocket)
            self._subscriptions[websocket] = {"detection", "status"}  # Default subscriptions
            self._connections_total += 1

        # Send welcome message
        welcome = WebSocketMessage(
            type=MessageType.CONNECTED.value,
            timestamp=time.time(),
            data={
                "message": "Connected to Drone Detection WebSocket",
                "subscriptions": list(self._subscriptions[websocket])
            }
        )
        await websocket.send_text(welcome.to_json())

    def disconnect(self, websocket) -> None:
        """Handle WebSocket disconnection."""
        with self._lock:
            self._clients.discard(websocket)
            self._subscriptions.pop(websocket, None)

    async def subscribe(self, websocket, channels: list) -> None:
        """Subscribe client to specific channels."""
        valid_channels = {mt.value for mt in MessageType}

        with self._lock:
            if websocket not in self._subscriptions:
                self._subscriptions[websocket] = set()

            for channel in channels:
                if channel in valid_channels:
                    self._subscriptions[websocket].add(channel)

        # Confirm subscription
        msg = WebSocketMessage(
            type=MessageType.SUBSCRIBED.value,
            timestamp=time.time(),
            data={"channels": channels}
        )
        await websocket.send_text(msg.to_json())

    async def unsubscribe(self, websocket, channels: list) -> None:
        """Unsubscribe client from channels."""
        with self._lock:
            if websocket in self._subscriptions:
                for channel in channels:
                    self._subscriptions[websocket].discard(channel)

        msg = WebSocketMessage(
            type=MessageType.UNSUBSCRIBED.value,
            timestamp=time.time(),
            data={"channels": channels}
        )
        await websocket.send_text(msg.to_json())

    async def broadcast(self, message: WebSocketMessage) -> None:
        """Broadcast message to subscribed clients."""
        with self._lock:
            clients = list(self._clients)
            subscriptions = dict(self._subscriptions)

        json_message = message.to_json()

        for client in clients:
            # Check subscription
            client_subs = subscriptions.get(client, set())
            if message.type in client_subs or "all" in client_subs:
                try:
                    await client.send_text(json_message)
                    self._messages_sent += 1
                except Exception:
                    # Client disconnected
                    self.disconnect(client)

    def queue_message(self, message: WebSocketMessage) -> None:
        """Queue a message for broadcast (thread-safe)."""
        self._message_queue.put(message)

    async def process_queue(self) -> None:
        """Process queued messages."""
        while not self._message_queue.empty():
            try:
                message = self._message_queue.get_nowait()
                await self.broadcast(message)
            except queue.Empty:
                break

    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        with self._lock:
            return len(self._clients)

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "connected_clients": self.client_count,
            "total_connections": self._connections_total,
            "messages_sent": self._messages_sent
        }


class DetectionStreamer:
    """
    Streams detection events to WebSocket clients.

    Integrates with the detection system to provide real-time updates.
    """

    def __init__(self, websocket_manager: WebSocketManager):
        """Initialize detection streamer."""
        self._ws_manager = websocket_manager
        self._detection_system = None
        self._running = False
        self._stream_thread: Optional[threading.Thread] = None
        self._last_track_ids: Set[int] = set()

    def set_detection_system(self, system) -> None:
        """Set the detection system to stream from."""
        self._detection_system = system
        system.add_callback(self._on_detection)

    def _on_detection(self, event) -> None:
        """Handle detection event from system."""
        message = WebSocketMessage(
            type=MessageType.DETECTION.value,
            timestamp=event.timestamp,
            data={
                "azimuth": event.azimuth,
                "elevation": event.elevation,
                "distance": event.distance,
                "confidence": event.confidence,
                "classification": event.classification,
                "threat_level": event.threat_level,
                "track_id": event.track_id,
                "snr": event.snr,
                "dominant_frequencies": event.dominant_frequencies or []
            }
        )
        self._ws_manager.queue_message(message)

    def start(self) -> None:
        """Start streaming."""
        if self._running:
            return

        self._running = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            daemon=True
        )
        self._stream_thread.start()

    def stop(self) -> None:
        """Stop streaming."""
        self._running = False
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)

    def _stream_loop(self) -> None:
        """Background streaming loop for tracks and metrics."""
        while self._running:
            if self._detection_system and self._detection_system.is_running:
                # Stream track updates
                self._stream_tracks()

                # Stream metrics periodically
                self._stream_metrics()

            time.sleep(0.1)  # 10 Hz update rate

    def _stream_tracks(self) -> None:
        """Stream track updates."""
        if not self._detection_system:
            return

        tracks = self._detection_system.get_tracks()
        current_track_ids = set()

        for track in tracks:
            current_track_ids.add(track.track_id)
            state = track.state

            message = WebSocketMessage(
                type=MessageType.TRACK_UPDATE.value,
                timestamp=time.time(),
                data={
                    "track_id": track.track_id,
                    "azimuth": state.azimuth,
                    "elevation": state.elevation,
                    "distance": state.distance,
                    "velocity_azimuth": state.velocity_azimuth,
                    "velocity_elevation": state.velocity_elevation,
                    "velocity_radial": state.velocity_radial,
                    "confidence": state.confidence,
                    "classification": track.classification,
                    "age": track.age
                }
            )
            self._ws_manager.queue_message(message)

        # Check for lost tracks
        lost_tracks = self._last_track_ids - current_track_ids
        for track_id in lost_tracks:
            message = WebSocketMessage(
                type=MessageType.TRACK_LOST.value,
                timestamp=time.time(),
                data={"track_id": track_id}
            )
            self._ws_manager.queue_message(message)

        self._last_track_ids = current_track_ids

    def _stream_metrics(self) -> None:
        """Stream performance metrics."""
        if not self._detection_system:
            return

        metrics = self._detection_system.performance_metrics
        if metrics:
            message = WebSocketMessage(
                type=MessageType.METRICS.value,
                timestamp=time.time(),
                data={
                    "fps": metrics.fps,
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "gpu_percent": metrics.gpu_percent,
                    "processing_latency_ms": metrics.processing_latency_ms
                }
            )
            self._ws_manager.queue_message(message)


# FastAPI WebSocket integration
try:
    from fastapi import WebSocket, WebSocketDisconnect

    async def websocket_endpoint(websocket: WebSocket, manager: WebSocketManager):
        """WebSocket endpoint handler."""
        await websocket.accept()
        await manager.connect(websocket)

        try:
            while True:
                # Handle incoming messages
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=0.1
                    )

                    # Parse command
                    try:
                        command = json.loads(data)
                        cmd_type = command.get("type", "")

                        if cmd_type == "subscribe":
                            channels = command.get("channels", [])
                            await manager.subscribe(websocket, channels)

                        elif cmd_type == "unsubscribe":
                            channels = command.get("channels", [])
                            await manager.unsubscribe(websocket, channels)

                        elif cmd_type == "ping":
                            pong = WebSocketMessage(
                                type="pong",
                                timestamp=time.time(),
                                data={}
                            )
                            await websocket.send_text(pong.to_json())

                    except json.JSONDecodeError:
                        error = WebSocketMessage(
                            type=MessageType.ERROR.value,
                            timestamp=time.time(),
                            data={"message": "Invalid JSON"}
                        )
                        await websocket.send_text(error.to_json())

                except asyncio.TimeoutError:
                    pass

                # Process queued messages
                await manager.process_queue()

        except WebSocketDisconnect:
            manager.disconnect(websocket)

except ImportError:
    # FastAPI not available
    pass


# Standalone WebSocket server using websockets library
class StandaloneWebSocketServer:
    """
    Standalone WebSocket server without FastAPI.

    Uses the websockets library for basic WebSocket support.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        """Initialize standalone server."""
        self._host = host
        self._port = port
        self._manager = WebSocketManager()
        self._streamer = DetectionStreamer(self._manager)
        self._server = None
        self._running = False

    def set_detection_system(self, system) -> None:
        """Set detection system for streaming."""
        self._streamer.set_detection_system(system)

    async def _handler(self, websocket, path):
        """Handle WebSocket connections."""
        await self._manager.connect(websocket)

        try:
            async for message in websocket:
                try:
                    command = json.loads(message)
                    cmd_type = command.get("type", "")

                    if cmd_type == "subscribe":
                        channels = command.get("channels", [])
                        await self._manager.subscribe(websocket, channels)

                    elif cmd_type == "unsubscribe":
                        channels = command.get("channels", [])
                        await self._manager.unsubscribe(websocket, channels)

                    elif cmd_type == "ping":
                        pong = WebSocketMessage(
                            type="pong",
                            timestamp=time.time(),
                            data={}
                        )
                        await websocket.send(pong.to_json())

                except json.JSONDecodeError:
                    error = WebSocketMessage(
                        type=MessageType.ERROR.value,
                        timestamp=time.time(),
                        data={"message": "Invalid JSON"}
                    )
                    await websocket.send(error.to_json())

        finally:
            self._manager.disconnect(websocket)

    async def _broadcast_loop(self):
        """Background task for processing message queue."""
        while self._running:
            await self._manager.process_queue()
            await asyncio.sleep(0.05)

    async def start_async(self):
        """Start server asynchronously."""
        try:
            import websockets

            self._running = True
            self._streamer.start()

            self._server = await websockets.serve(
                self._handler,
                self._host,
                self._port
            )

            # Start broadcast loop
            asyncio.create_task(self._broadcast_loop())

            print(f"WebSocket server started on ws://{self._host}:{self._port}")

            await self._server.wait_closed()

        except ImportError:
            print("websockets library not installed")

    def start(self):
        """Start server (blocking)."""
        asyncio.run(self.start_async())

    async def stop_async(self):
        """Stop server asynchronously."""
        self._running = False
        self._streamer.stop()

        if self._server:
            self._server.close()
            await self._server.wait_closed()

    def stop(self):
        """Stop server."""
        asyncio.run(self.stop_async())

    @property
    def manager(self) -> WebSocketManager:
        """Get the WebSocket manager."""
        return self._manager


def create_websocket_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    detection_system=None
) -> StandaloneWebSocketServer:
    """Create a WebSocket server instance."""
    server = StandaloneWebSocketServer(host, port)
    if detection_system:
        server.set_detection_system(detection_system)
    return server


# For running standalone
if __name__ == "__main__":
    server = create_websocket_server()
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()
