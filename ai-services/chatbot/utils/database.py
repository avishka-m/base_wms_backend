# app/utils/database.py
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
from typing import Dict, Any, Optional
from datetime import datetime
import jwt
from fastapi import HTTPException, status
from app.auth.utils import verify_password, get_password_hash
from app.config import settings

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URL)
db = client["warehouse_management"]

# User session store
active_sessions = {}

async def validate_token(token: str) -> Dict[str, Any]:
    """
    Validate JWT token and return user data.
    
    Args:
        token: JWT token to validate
        
    Returns:
        Dict containing user data if valid
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_user_session(username: str) -> Optional[Dict[str, Any]]:
    """
    Get user session data.
    
    Args:
        username: Username to get session for
        
    Returns:
        Session data if exists, else None
    """
    return active_sessions.get(username)

async def create_user_session(user_data: Dict[str, Any]) -> None:
    """
    Create or update user session.
    
    Args:
        user_data: User data to store in session
    """
    username = user_data.get("username")
    if username:
        active_sessions[username] = {
            "user_data": user_data,
            "last_activity": datetime.now().isoformat(),
            "conversation_history": []
        }

async def update_session_activity(username: str) -> None:
    """
    Update last activity timestamp for user session.
    
    Args:
        username: Username to update activity for
    """
    if username in active_sessions:
        active_sessions[username]["last_activity"] = datetime.now().isoformat()
