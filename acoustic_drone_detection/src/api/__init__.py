"""
API Module

REST API and WebSocket interfaces:
- FastAPI-based REST API
- Real-time WebSocket streaming
"""

from .rest_api import app, create_app, set_detection_system, add_detection
from .websocket_server import (
    WebSocketManager,
    WebSocketMessage,
    MessageType,
    DetectionStreamer,
    StandaloneWebSocketServer,
    create_websocket_server
)

__all__ = [
    'app',
    'create_app',
    'set_detection_system',
    'add_detection',
    'WebSocketManager',
    'WebSocketMessage',
    'MessageType',
    'DetectionStreamer',
    'StandaloneWebSocketServer',
    'create_websocket_server'
]
