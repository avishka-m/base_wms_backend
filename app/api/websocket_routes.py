"""
WebSocket endpoints for real-time updates
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from ..services.websocket_service import websocket_manager
from ..auth.dependencies import get_current_user_from_token
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws/orders")
async def websocket_orders_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
):
    """
    WebSocket endpoint for real-time order updates
    
    Requires authentication token as query parameter
    """
    try:
        # Authenticate user using token
        user = await get_current_user_from_token(token)
        if not user:
            await websocket.close(code=4001, reason="Authentication failed")
            return
        
        user_id = str(user.get("workerID", user.get("_id", "unknown")))
        
        # Connect to WebSocket manager
        await websocket_manager.connect(websocket, user_id)
        
        # Send initial connection confirmation
        await websocket.send_text(f'{{"type": "connection_established", "user_id": "{user_id}"}}')
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client (heartbeat, etc.)
                data = await websocket.receive_text()
                
                # Handle client messages if needed
                if data == "ping":
                    await websocket.send_text('{"type": "pong"}')
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {e}")
                break
                
    except Exception as e:
        logger.error(f"Error establishing WebSocket connection: {e}")
        await websocket.close(code=4000, reason="Connection error")
    finally:
        # Clean up connection
        websocket_manager.disconnect(websocket)
