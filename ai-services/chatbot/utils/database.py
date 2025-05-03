# app/utils/database.py
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
from typing import Dict, Any, Optional
from datetime import datetime

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URL)
db = client["warehouse_management"]

# Simple in-memory conversation store
conversations = {}

async def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Get conversation data.
    
    Args:
        conversation_id: ID of the conversation to retrieve
        
    Returns:
        Conversation data if exists, else None
    """
    return conversations.get(conversation_id)

async def save_conversation(conversation_id: str, data: Dict[str, Any]) -> None:
    """
    Save conversation data.
    
    Args:
        conversation_id: ID of the conversation
        data: Conversation data to save
    """
    conversations[conversation_id] = {
        **data,
        "last_updated": datetime.now().isoformat()
    }

async def append_to_conversation(conversation_id: str, message: Dict[str, Any]) -> None:
    """
    Append a message to an existing conversation.
    
    Args:
        conversation_id: ID of the conversation
        message: Message to append
    """
    if conversation_id not in conversations:
        conversations[conversation_id] = {
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    conversations[conversation_id]["messages"].append(message)
    conversations[conversation_id]["last_updated"] = datetime.now().isoformat()
