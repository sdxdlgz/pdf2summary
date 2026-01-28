"""
WebSocket handler for real-time progress updates.

This module provides WebSocket connection management and progress broadcasting
for the Research Report Processor.

Requirements:
- 8.2: Broadcast progress updates via WebSocket
- 8.7: Broadcast error messages via WebSocket
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.models import ProgressUpdate


logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and broadcasts progress updates.
    
    This class handles:
    - Connection management (connect/disconnect) per task_id
    - Broadcasting progress updates to all connected clients for a task
    - Graceful handling of connection errors
    
    Attributes:
        _connections: Dictionary mapping task_id to set of connected WebSockets
        _lock: Asyncio lock for thread-safe connection management
    """
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        # Dictionary mapping task_id to set of connected WebSockets
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        # Lock for thread-safe connection management
        self._lock = asyncio.Lock()
    
    async def connect(self, task_id: str, websocket: WebSocket) -> None:
        """
        Establish a WebSocket connection for a task.
        
        Accepts the WebSocket connection and adds it to the set of
        connections for the specified task_id.
        
        Args:
            task_id: The unique task identifier
            websocket: The WebSocket connection to add
        """
        await websocket.accept()
        async with self._lock:
            self._connections[task_id].add(websocket)
        logger.info(
            "WebSocket connected for task %s (total connections: %d)",
            task_id,
            len(self._connections[task_id]),
        )
    
    async def disconnect(self, task_id: str, websocket: WebSocket) -> None:
        """
        Disconnect a WebSocket connection for a task.
        
        Removes the WebSocket from the set of connections for the
        specified task_id. Cleans up empty task entries.
        
        Args:
            task_id: The unique task identifier
            websocket: The WebSocket connection to remove
        """
        async with self._lock:
            if task_id in self._connections:
                self._connections[task_id].discard(websocket)
                # Clean up empty task entries
                if not self._connections[task_id]:
                    del self._connections[task_id]
        logger.info(
            "WebSocket disconnected for task %s (remaining connections: %d)",
            task_id,
            len(self._connections.get(task_id, set())),
        )
    
    async def broadcast_progress(
        self, 
        task_id: str, 
        progress: ProgressUpdate
    ) -> None:
        """
        Broadcast a progress update to all connected clients for a task.
        
        Sends the progress update as JSON to all WebSocket connections
        associated with the specified task_id. Handles disconnected
        clients gracefully by removing them from the connection set.
        
        Args:
            task_id: The unique task identifier
            progress: The progress update to broadcast
            
        Requirement: 8.2 - Broadcast progress updates via WebSocket
        """
        async with self._lock:
            connections = self._connections.get(task_id, set()).copy()
        
        if not connections:
            logger.debug("No WebSocket connections for task %s", task_id)
            return
        
        # Serialize progress update to JSON-compatible dict
        message = progress.model_dump(mode="json")
        
        # Track disconnected clients for cleanup
        disconnected: list[WebSocket] = []
        
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(websocket)
                logger.debug(
                    "WebSocket disconnected during broadcast for task %s",
                    task_id,
                )
            except Exception as e:
                disconnected.append(websocket)
                logger.warning(
                    "Error broadcasting to WebSocket for task %s: %s",
                    task_id,
                    e,
                )
        
        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                for ws in disconnected:
                    self._connections[task_id].discard(ws)
                # Clean up empty task entries
                if task_id in self._connections and not self._connections[task_id]:
                    del self._connections[task_id]
        
        logger.debug(
            "Broadcast progress to %d clients for task %s: stage=%s, progress=%d/%d",
            len(connections) - len(disconnected),
            task_id,
            progress.stage.value,
            progress.progress,
            progress.total,
        )
    
    async def broadcast_error(
        self, 
        task_id: str, 
        error_message: str
    ) -> None:
        """
        Broadcast an error message to all connected clients for a task.
        
        Creates a ProgressUpdate with FAILED stage and broadcasts it
        to all connected clients.
        
        Args:
            task_id: The unique task identifier
            error_message: The error message to broadcast
            
        Requirement: 8.7 - Broadcast error messages via WebSocket
        """
        from backend.models import TaskStage
        
        error_progress = ProgressUpdate(
            task_id=task_id,
            stage=TaskStage.FAILED,
            progress=0,
            total=0,
            percentage=0.0,
            message=error_message,
        )
        await self.broadcast_progress(task_id, error_progress)
        logger.info(
            "Broadcast error for task %s: %s",
            task_id,
            error_message,
        )
    
    def get_connection_count(self, task_id: str) -> int:
        """
        Get the number of active connections for a task.
        
        Args:
            task_id: The unique task identifier
            
        Returns:
            int: Number of active WebSocket connections
        """
        return len(self._connections.get(task_id, set()))
    
    def get_all_task_ids(self) -> list[str]:
        """
        Get all task IDs with active connections.
        
        Returns:
            list[str]: List of task IDs with active WebSocket connections
        """
        return list(self._connections.keys())


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
