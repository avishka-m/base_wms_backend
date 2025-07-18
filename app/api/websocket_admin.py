"""
WebSocket administration endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from ..services.websocket_service import websocket_manager
from ..auth.dependencies import get_current_user
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/websocket/stats")
async def get_websocket_stats(current_user = Depends(get_current_user)):
    """Get WebSocket connection statistics"""
    if current_user.get("role") not in ["Manager", "Admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only managers can view WebSocket statistics"
        )
    
    return websocket_manager.get_connection_stats()

@router.post("/websocket/cleanup")
async def cleanup_stale_connections(current_user = Depends(get_current_user)):
    """Clean up stale WebSocket connections"""
    if current_user.get("role") not in ["Manager", "Admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only managers can cleanup WebSocket connections"
        )
    
    await websocket_manager.cleanup_stale_connections()
    stats = websocket_manager.get_connection_stats()
    
    return {
        "message": "Stale connections cleaned up",
        "current_stats": stats
    }

@router.post("/websocket/ping-all")
async def ping_all_connections(current_user = Depends(get_current_user)):
    """Send ping to all WebSocket connections"""
    if current_user.get("role") not in ["Manager", "Admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only managers can ping WebSocket connections"
        )
    
    await websocket_manager.send_ping_to_all()
    stats = websocket_manager.get_connection_stats()
    
    return {
        "message": "Ping sent to all connections",
        "current_stats": stats
    }

@router.post("/websocket/broadcast")
async def broadcast_test_message(
    message: str,
    current_user = Depends(get_current_user)
):
    """Broadcast a test message to all WebSocket connections"""
    if current_user.get("role") not in ["Manager", "Admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only managers can broadcast WebSocket messages"
        )
    
    import json
    import asyncio
    
    test_message = {
        "type": "admin_broadcast",
        "message": message,
        "sender": current_user.get("username", "Admin"),
        "timestamp": asyncio.get_event_loop().time()
    }
    
    await websocket_manager.broadcast_message(json.dumps(test_message))
    stats = websocket_manager.get_connection_stats()
    
    return {
        "message": "Test message broadcasted",
        "sent_message": test_message,
        "current_stats": stats
    }
