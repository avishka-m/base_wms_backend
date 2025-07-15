"""
WebSocket endpoints for real-time updates
"""
import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from ..services.websocket_service import websocket_manager
from ..auth.dependencies import get_current_user_from_token
import logging
from ..config import DEV_MODE

logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws/orders")
async def websocket_orders_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
):
    """WebSocket endpoint for real-time order updates"""
    user_id = None
    
    try:
        # Accept connection first
        await websocket.accept()
        logger.info("WebSocket connection accepted, authenticating...")
        
        # Authenticate user
        user = await get_current_user_from_token(token)
        
        if not user:
            if DEV_MODE:
                user = {"_id": "dev", "username": "dev_manager", "role": "Manager", "workerID": 1}
                logger.warning("DEV_MODE: Using development user for WebSocket")
            else:
                logger.error("WebSocket authentication failed - no user found")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": "Authentication failed"
                }))
                await websocket.close(code=4001, reason="Authentication failed")
                return
        
        user_id = str(user.get("workerID", user.get("_id", "unknown")))
        user_role = user.get("role", "Unknown")
        logger.info(f"WebSocket user authenticated: {user_id} (role: {user_role})")
        
        # Connect to WebSocket manager
        await websocket_manager.connect(websocket, user_id)
        logger.info(f"WebSocket connected for user: {user_id}")
        
        # Send initial connection confirmation
        confirmation_message = {
            "type": "connection_established", 
            "user_id": user_id,
            "role": user_role,
            "timestamp": asyncio.get_event_loop().time()
        }
        await websocket.send_text(json.dumps(confirmation_message))
        logger.info(f"WebSocket connection confirmation sent to user: {user_id}")
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                logger.debug(f"WebSocket received message from {user_id}: {data}")
                
                # Handle different client message types
                try:
                    message = json.loads(data) if data.startswith('{') else {"type": "text", "data": data}
                except json.JSONDecodeError:
                    message = {"type": "text", "data": data}
                
                # Handle ping/pong for heartbeat
                if data == "ping" or message.get("type") == "ping":
                    pong_response = {"type": "pong", "timestamp": asyncio.get_event_loop().time()}
                    await websocket.send_text(json.dumps(pong_response))
                    logger.debug(f"WebSocket pong sent to {user_id}")
                
                # Handle status requests
                elif message.get("type") == "status":
                    status_response = {
                        "type": "status_response",
                        "user_id": user_id,
                        "role": user_role,
                        "connected": True,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await websocket.send_text(json.dumps(status_response))
                    logger.debug(f"WebSocket status response sent to {user_id}")
                
            except asyncio.TimeoutError:
                # Send ping to check if client is still alive
                ping_message = {"type": "ping", "timestamp": asyncio.get_event_loop().time()}
                await websocket.send_text(json.dumps(ping_message))
                logger.debug(f"WebSocket ping sent to {user_id} (timeout check)")
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket client {user_id} disconnected normally")
                break
                
            except Exception as e:
                logger.error(f"Error in WebSocket message handling for {user_id}: {e}")
                # Send error message to client
                error_message = {
                    "type": "error",
                    "message": "Message handling error",
                    "timestamp": asyncio.get_event_loop().time()
                }
                try:
                    await websocket.send_text(json.dumps(error_message))
                except:
                    break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed during setup for user: {user_id or 'unknown'}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection for user {user_id or 'unknown'}: {e}")
        try:
            error_message = {
                "type": "error",
                "message": "Connection error",
                "timestamp": asyncio.get_event_loop().time()
            }
            await websocket.send_text(json.dumps(error_message))
        except:
            pass
    finally:
        # Clean up connection
        if user_id:
            logger.info(f"WebSocket cleanup for user: {user_id}")
            websocket_manager.disconnect(websocket)
        else:
            logger.info("WebSocket cleanup for unauthenticated connection")
