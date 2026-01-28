"""
Tests for the WebSocket handler.

Tests cover:
- WebSocket connection management (connect/disconnect)
- Progress broadcasting to connected clients
- Error broadcasting
- Multiple clients per task
- Graceful handling of disconnections

Requirements:
- 8.2: Broadcast progress updates via WebSocket
- 8.7: Broadcast error messages via WebSocket
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

from backend.api.websocket import WebSocketManager, websocket_manager
from backend.models import ProgressUpdate, TaskStage


class TestWebSocketManager:
    """Tests for WebSocketManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh WebSocketManager instance for each test."""
        return WebSocketManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        ws = AsyncMock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.receive_text = AsyncMock()
        return ws
    
    @pytest.mark.asyncio
    async def test_connect_accepts_websocket(self, manager, mock_websocket):
        """Test that connect accepts the WebSocket connection."""
        task_id = "task-123"
        
        await manager.connect(task_id, mock_websocket)
        
        mock_websocket.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_adds_to_connections(self, manager, mock_websocket):
        """Test that connect adds WebSocket to connections."""
        task_id = "task-123"
        
        await manager.connect(task_id, mock_websocket)
        
        assert manager.get_connection_count(task_id) == 1
    
    @pytest.mark.asyncio
    async def test_connect_multiple_clients_same_task(self, manager):
        """Test connecting multiple clients to the same task."""
        task_id = "task-123"
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        ws3 = AsyncMock(spec=WebSocket)
        
        await manager.connect(task_id, ws1)
        await manager.connect(task_id, ws2)
        await manager.connect(task_id, ws3)
        
        assert manager.get_connection_count(task_id) == 3
    
    @pytest.mark.asyncio
    async def test_connect_different_tasks(self, manager):
        """Test connecting clients to different tasks."""
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        
        await manager.connect("task-1", ws1)
        await manager.connect("task-2", ws2)
        
        assert manager.get_connection_count("task-1") == 1
        assert manager.get_connection_count("task-2") == 1
        assert len(manager.get_all_task_ids()) == 2
    
    @pytest.mark.asyncio
    async def test_disconnect_removes_from_connections(self, manager, mock_websocket):
        """Test that disconnect removes WebSocket from connections."""
        task_id = "task-123"
        
        await manager.connect(task_id, mock_websocket)
        assert manager.get_connection_count(task_id) == 1
        
        await manager.disconnect(task_id, mock_websocket)
        assert manager.get_connection_count(task_id) == 0
    
    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_empty_task(self, manager, mock_websocket):
        """Test that disconnect cleans up empty task entries."""
        task_id = "task-123"
        
        await manager.connect(task_id, mock_websocket)
        await manager.disconnect(task_id, mock_websocket)
        
        assert task_id not in manager.get_all_task_ids()
    
    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_task(self, manager, mock_websocket):
        """Test that disconnect handles nonexistent task gracefully."""
        # Should not raise an error
        await manager.disconnect("nonexistent-task", mock_websocket)
    
    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_websocket(self, manager):
        """Test that disconnect handles nonexistent WebSocket gracefully."""
        task_id = "task-123"
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        
        await manager.connect(task_id, ws1)
        # Disconnect a WebSocket that was never connected
        await manager.disconnect(task_id, ws2)
        
        # Original connection should still exist
        assert manager.get_connection_count(task_id) == 1
    
    @pytest.mark.asyncio
    async def test_broadcast_progress_sends_to_all_clients(self, manager):
        """Test that broadcast_progress sends to all connected clients."""
        task_id = "task-123"
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        
        await manager.connect(task_id, ws1)
        await manager.connect(task_id, ws2)
        
        progress = ProgressUpdate(
            task_id=task_id,
            stage=TaskStage.PARSING,
            progress=50,
            total=100,
            percentage=25.0,
            message="Parsing in progress",
        )
        
        await manager.broadcast_progress(task_id, progress)
        
        ws1.send_json.assert_called_once()
        ws2.send_json.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_broadcast_progress_sends_correct_data(self, manager, mock_websocket):
        """Test that broadcast_progress sends correct JSON data."""
        task_id = "task-123"
        
        await manager.connect(task_id, mock_websocket)
        
        progress = ProgressUpdate(
            task_id=task_id,
            stage=TaskStage.TRANSLATING,
            progress=30,
            total=60,
            percentage=50.0,
            message="Translating document",
        )
        
        await manager.broadcast_progress(task_id, progress)
        
        # Get the data that was sent
        call_args = mock_websocket.send_json.call_args
        sent_data = call_args[0][0]
        
        assert sent_data["task_id"] == task_id
        assert sent_data["stage"] == "translating"
        assert sent_data["progress"] == 30
        assert sent_data["total"] == 60
        assert sent_data["percentage"] == 50.0
        assert sent_data["message"] == "Translating document"
    
    @pytest.mark.asyncio
    async def test_broadcast_progress_no_connections(self, manager):
        """Test that broadcast_progress handles no connections gracefully."""
        progress = ProgressUpdate(
            task_id="task-123",
            stage=TaskStage.PARSING,
            progress=50,
            total=100,
            percentage=25.0,
            message="Parsing",
        )
        
        # Should not raise an error
        await manager.broadcast_progress("task-123", progress)
    
    @pytest.mark.asyncio
    async def test_broadcast_progress_handles_disconnected_client(self, manager):
        """Test that broadcast_progress handles disconnected clients."""
        task_id = "task-123"
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        
        # ws2 will raise WebSocketDisconnect when sending
        ws2.send_json.side_effect = WebSocketDisconnect()
        
        await manager.connect(task_id, ws1)
        await manager.connect(task_id, ws2)
        
        progress = ProgressUpdate(
            task_id=task_id,
            stage=TaskStage.PARSING,
            progress=50,
            total=100,
            percentage=25.0,
            message="Parsing",
        )
        
        await manager.broadcast_progress(task_id, progress)
        
        # ws1 should have received the message
        ws1.send_json.assert_called_once()
        # ws2 should be removed from connections
        assert manager.get_connection_count(task_id) == 1
    
    @pytest.mark.asyncio
    async def test_broadcast_progress_handles_send_error(self, manager):
        """Test that broadcast_progress handles send errors gracefully."""
        task_id = "task-123"
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        
        # ws2 will raise an exception when sending
        ws2.send_json.side_effect = Exception("Connection error")
        
        await manager.connect(task_id, ws1)
        await manager.connect(task_id, ws2)
        
        progress = ProgressUpdate(
            task_id=task_id,
            stage=TaskStage.PARSING,
            progress=50,
            total=100,
            percentage=25.0,
            message="Parsing",
        )
        
        # Should not raise an error
        await manager.broadcast_progress(task_id, progress)
        
        # ws1 should have received the message
        ws1.send_json.assert_called_once()
        # ws2 should be removed from connections
        assert manager.get_connection_count(task_id) == 1
    
    @pytest.mark.asyncio
    async def test_broadcast_error_sends_failed_stage(self, manager, mock_websocket):
        """Test that broadcast_error sends a FAILED stage progress update."""
        task_id = "task-123"
        error_message = "Processing failed: file corrupted"
        
        await manager.connect(task_id, mock_websocket)
        
        await manager.broadcast_error(task_id, error_message)
        
        # Get the data that was sent
        call_args = mock_websocket.send_json.call_args
        sent_data = call_args[0][0]
        
        assert sent_data["task_id"] == task_id
        assert sent_data["stage"] == "failed"
        assert sent_data["message"] == error_message
        assert sent_data["progress"] == 0
        assert sent_data["total"] == 0
        assert sent_data["percentage"] == 0.0
    
    @pytest.mark.asyncio
    async def test_broadcast_error_to_multiple_clients(self, manager):
        """Test that broadcast_error sends to all connected clients."""
        task_id = "task-123"
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        
        await manager.connect(task_id, ws1)
        await manager.connect(task_id, ws2)
        
        await manager.broadcast_error(task_id, "Error occurred")
        
        ws1.send_json.assert_called_once()
        ws2.send_json.assert_called_once()
    
    def test_get_connection_count_empty(self, manager):
        """Test get_connection_count returns 0 for unknown task."""
        assert manager.get_connection_count("unknown-task") == 0
    
    def test_get_all_task_ids_empty(self, manager):
        """Test get_all_task_ids returns empty list initially."""
        assert manager.get_all_task_ids() == []
    
    @pytest.mark.asyncio
    async def test_get_all_task_ids_with_connections(self, manager):
        """Test get_all_task_ids returns all task IDs with connections."""
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        ws3 = AsyncMock(spec=WebSocket)
        
        await manager.connect("task-1", ws1)
        await manager.connect("task-2", ws2)
        await manager.connect("task-3", ws3)
        
        task_ids = manager.get_all_task_ids()
        
        assert len(task_ids) == 3
        assert "task-1" in task_ids
        assert "task-2" in task_ids
        assert "task-3" in task_ids


class TestWebSocketEndpoint:
    """Tests for the WebSocket endpoint in main.py."""
    
    @pytest.fixture
    def test_app(self):
        """Create a test FastAPI app with WebSocket endpoint."""
        from backend.api.main import websocket_endpoint
        
        app = FastAPI()
        app.websocket("/ws/{task_id}")(websocket_endpoint)
        
        return app
    
    def test_websocket_connection(self, test_app):
        """Test that WebSocket connection can be established."""
        with TestClient(test_app) as client:
            with client.websocket_connect("/ws/task-123") as websocket:
                # Connection should be established
                # Send a ping to verify connection is working
                websocket.send_text("ping")
                response = websocket.receive_text()
                assert response == "pong"
    
    def test_websocket_ping_pong(self, test_app):
        """Test that WebSocket responds to ping with pong."""
        with TestClient(test_app) as client:
            with client.websocket_connect("/ws/task-456") as websocket:
                websocket.send_text("ping")
                response = websocket.receive_text()
                assert response == "pong"
    
    def test_websocket_multiple_pings(self, test_app):
        """Test multiple ping-pong exchanges."""
        with TestClient(test_app) as client:
            with client.websocket_connect("/ws/task-789") as websocket:
                for _ in range(3):
                    websocket.send_text("ping")
                    response = websocket.receive_text()
                    assert response == "pong"

    def test_websocket_forwards_redis_progress(self, test_app, monkeypatch):
        """Test that WebSocket forwards progress updates published to Redis."""
        import backend.api.main as main_module
        from backend.models import FileInfo, ProgressUpdate, TaskStage, TaskStatus

        task_id = "task-redis-123"

        initial_progress = ProgressUpdate(
            task_id=task_id,
            stage=TaskStage.UPLOADING,
            progress=0,
            total=1,
            percentage=0.0,
            message="Task created",
        )
        initial_status = TaskStatus(
            task_id=task_id,
            stage=TaskStage.UPLOADING,
            files=[FileInfo(name="example.pdf", data_id="data-1", size=123)],
            progress=initial_progress,
        )

        completed_progress = ProgressUpdate(
            task_id=task_id,
            stage=TaskStage.COMPLETED,
            progress=100,
            total=100,
            percentage=100.0,
            message="Processing complete",
        )

        class FakePubSub:
            def __init__(self, messages: list[dict]):
                self._messages = list(messages)

            async def subscribe(self, _channel: str) -> None:
                return None

            async def unsubscribe(self, _channel: str) -> None:
                return None

            async def close(self) -> None:
                return None

            async def get_message(self, *args, **kwargs):  # noqa: ANN001,ANN002
                if self._messages:
                    return self._messages.pop(0)
                return None

        class FakeRedis:
            def __init__(self, task_data: bytes, pubsub: FakePubSub):
                self._task_data = task_data
                self._pubsub = pubsub

            async def get(self, _key: str) -> bytes:
                return self._task_data

            def pubsub(self) -> FakePubSub:
                return self._pubsub

        pubsub = FakePubSub(
            messages=[
                {"data": completed_progress.model_dump_json().encode("utf-8")},
            ]
        )
        fake_redis = FakeRedis(
            task_data=initial_status.model_dump_json().encode("utf-8"),
            pubsub=pubsub,
        )

        monkeypatch.setattr(main_module, "_redis_client", fake_redis)

        with TestClient(test_app) as client:
            with client.websocket_connect(f"/ws/{task_id}") as websocket:
                first = websocket.receive_json()
                assert first["stage"] == "uploading"

                second = websocket.receive_json()
                assert second["stage"] == "completed"


class TestGlobalWebSocketManager:
    """Tests for the global websocket_manager instance."""
    
    def test_global_manager_exists(self):
        """Test that global websocket_manager is available."""
        assert websocket_manager is not None
        assert isinstance(websocket_manager, WebSocketManager)
    
    @pytest.mark.asyncio
    async def test_global_manager_can_be_used(self):
        """Test that global websocket_manager can be used."""
        # Should not raise any errors
        count = websocket_manager.get_connection_count("test-task")
        assert count == 0
