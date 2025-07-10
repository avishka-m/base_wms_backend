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
            logger.error("WebSocket authentication failed - no user found")
            await websocket.close(code=4001, reason="Authentication failed")
            return
        
        user_id = str(user.get("workerID", user.get("_id", "unknown")))
        logger.info(f"WebSocket user authenticated: {user_id}")
        
        # Connect to WebSocket manager
        await websocket_manager.connect(websocket, user_id)
        logger.info(f"WebSocket connected for user: {user_id}")
        
        # Send initial connection confirmation
        await websocket.send_text(f'{{"type": "connection_established", "user_id": "{user_id}"}}')
        logger.info(f"WebSocket connection confirmation sent to user: {user_id}")
        
        # Keep connection alive and handle messages
        try:
            while True:
                try:
                    # Wait for messages from client (heartbeat, etc.)
                    data = await websocket.receive_text()
                    logger.debug(f"WebSocket received message from {user_id}: {data}")
                    
                    # Handle client messages if needed
                    if data == "ping":
                        await websocket.send_text('{"type": "pong"}')
                        logger.debug(f"WebSocket pong sent to {user_id}")
                    
                except WebSocketDisconnect:
                    logger.info(f"WebSocket client {user_id} disconnected normally")
                    break
                except Exception as e:
                    logger.error(f"Error in WebSocket message handling for {user_id}: {e}")
                    break
        except Exception as e:
            logger.error(f"Error in WebSocket message loop for {user_id}: {e}")
                
    except Exception as e:
        logger.error(f"Error establishing WebSocket connection: {e}")
        try:
            await websocket.close(code=4000, reason="Connection error")
        except Exception:
            pass  # Connection might already be closed
    finally:
        # Clean up connection
        logger.info(f"WebSocket cleanup for connection")
        websocket_manager.disconnect(websocket)
