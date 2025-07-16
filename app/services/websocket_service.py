"""
WebSocket service for real-time order updates
"""
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Set
from fastapi import WebSocket, WebSocketDisconnect
from bson import ObjectId
from ..auth.dependencies import get_current_user_from_token
import logging

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle MongoDB ObjectId and datetime objects"""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class OrderUpdateWebSocketManager:
    def __init__(self):
        # Dictionary to store active connections by user ID
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Dictionary to store connection to user mapping
        self.connection_users: Dict[WebSocket, str] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """Add WebSocket connection to active connections (assumes already accepted)"""
        # Initialize user connections list if not exists
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        
        # Add connection to user's list
        self.active_connections[user_id].append(websocket)
        self.connection_users[websocket] = user_id
        
        logger.info(f"WebSocket connection established for user: {user_id}")
        
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.connection_users:
            user_id = self.connection_users[websocket]
            
            # Remove from user's connections
            if user_id in self.active_connections:
                self.active_connections[user_id].remove(websocket)
                
                # Clean up empty user lists
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
            
            # Remove from connection mapping
            del self.connection_users[websocket]
            
            logger.info(f"WebSocket connection closed for user: {user_id}")
    
    async def send_personal_message(self, message: str, user_id: str):
        """Send a message to a specific user"""
        if user_id in self.active_connections:
            # Send to all connections for this user
            broken_connections = []
            for connection in self.active_connections[user_id]:
                try:
                    if connection.client_state.value in [1, 2]:  # CONNECTING or OPEN
                        await connection.send_text(message)
                        logger.debug(f"Message sent to user {user_id}: {message[:100]}...")
                    else:
                        logger.warning(f"Connection for user {user_id} is not open (state: {connection.client_state.value})")
                        broken_connections.append(connection)
                except Exception as e:
                    logger.error(f"Error sending message to user {user_id}: {e}")
                    broken_connections.append(connection)
            
            # Clean up broken connections
            for broken_conn in broken_connections:
                self.disconnect(broken_conn)
    
    async def broadcast_message(self, message: str):
        """Send a message to all connected users"""
        all_broken_connections = []
        
        for user_id, connections in list(self.active_connections.items()):
            broken_connections = []
            for connection in connections:
                try:
                    if connection.client_state.value in [1, 2]:  # CONNECTING or OPEN
                        await connection.send_text(message)
                        logger.debug(f"Broadcast message sent to user {user_id}")
                    else:
                        logger.warning(f"Connection for user {user_id} is not open (state: {connection.client_state.value})")
                        broken_connections.append(connection)
                except Exception as e:
                    logger.error(f"Error broadcasting message to user {user_id}: {e}")
                    broken_connections.append(connection)
            
            # Clean up broken connections for this user
            for broken_conn in broken_connections:
                self.disconnect(broken_conn)
    
    async def broadcast_to_roles(self, message: str, roles: List[str]):
        """Send a message to users with specific roles"""
        # Note: This would need to be enhanced with role checking
        # For now, we'll broadcast to all (can be improved later)
        await self.broadcast_message(message)
    
    def get_connection_stats(self) -> Dict[str, int]:
        """Get statistics about active connections"""
        total_connections = sum(len(connections) for connections in self.active_connections.values())
        return {
            "total_users": len(self.active_connections),
            "total_connections": total_connections,
            "user_details": {
                user_id: len(connections) 
                for user_id, connections in self.active_connections.items()
            }
        }
    
    async def cleanup_stale_connections(self):
        """Remove stale/closed connections"""
        stale_connections = []
        
        for user_id, connections in list(self.active_connections.items()):
            for connection in connections:
                try:
                    # Check if connection is still alive
                    if connection.client_state.value not in [1, 2]:  # Not CONNECTING or OPEN
                        stale_connections.append(connection)
                except Exception:
                    stale_connections.append(connection)
        
        # Clean up stale connections
        for stale_conn in stale_connections:
            self.disconnect(stale_conn)
        
        if stale_connections:
            logger.info(f"Cleaned up {len(stale_connections)} stale connections")
    
    async def send_ping_to_all(self):
        """Send ping to all connections to check health"""
        ping_message = json.dumps({
            "type": "ping",
            "timestamp": asyncio.get_event_loop().time()
        })
        await self.broadcast_message(ping_message)
    
    async def notify_order_update(self, order_id: int, order_status: str, user_roles: List[str] = None):
        """Notify relevant users about order updates"""
        # Import here to avoid circular imports
        from ..services.orders_service import OrdersService
        
        # Fetch complete order data
        try:
            order_data = await OrdersService.get_order(order_id)
        except Exception as e:
            logger.error(f"Failed to fetch order data for WebSocket notification: {e}")
            order_data = None
        
        message = {
            "type": "order_update",
            "data": {
                "order_id": order_id,
                "order_status": order_status,
                "order_data": order_data,
                "timestamp": asyncio.get_event_loop().time()
            }
        }
        
        message_json = json.dumps(message, cls=CustomJSONEncoder)
        
        if user_roles:
            await self.broadcast_to_roles(message_json, user_roles)
        else:
            await self.broadcast_message(message_json)

# Global instance
websocket_manager = OrderUpdateWebSocketManager()
