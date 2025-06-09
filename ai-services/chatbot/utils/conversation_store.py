"""Conversation store utilities."""

from typing import Dict, Any, List
from datetime import datetime
from fastapi import HTTPException, status
from config import MAX_CONVERSATION_MESSAGES

# Simple in-memory conversation store
# In production, this should be replaced with a database
user_conversations: Dict[str, Dict[str, Any]] = {}

def create_conversation(user_id: str, role: str, title: str) -> str:
    """
    Create a new conversation.
    
    Args:
        user_id: User identifier
        role: Chatbot role for this conversation
        title: Conversation title
        
    Returns:
        Conversation ID
    """
    # Create conversation ID with timestamp to make it unique
    conversation_id = f"{role}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Initialize user's conversations dictionary if it doesn't exist
    if user_id not in user_conversations:
        user_conversations[user_id] = {}
    
    # Store the new conversation with metadata
    user_conversations[user_id][conversation_id] = {
        "metadata": {
            "title": title,
            "created_at": datetime.now().isoformat(),
            "role": role,
        },
        "messages": []
    }
    
    return conversation_id

def add_message(user_id: str, conversation_id: str, role: str, content: str):
    """
    Add a message to a conversation.
    
    Args:
        user_id: User identifier
        conversation_id: Conversation identifier
        role: Message sender role (user/assistant)
        content: Message content
    """
    # Check if conversation exists
    if user_id not in user_conversations or conversation_id not in user_conversations[user_id]:
        # Create conversation if it doesn't exist
        role_from_id = conversation_id.split('_')[0]
        user_conversations[user_id][conversation_id] = {
            "metadata": {
                "title": f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "created_at": datetime.now().isoformat(),
                "role": role_from_id,
            },
            "messages": []
        }
    
    # Add the message
    user_conversations[user_id][conversation_id]["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    # Trim conversation history if it gets too long
    if len(user_conversations[user_id][conversation_id]["messages"]) > MAX_CONVERSATION_MESSAGES:
        user_conversations[user_id][conversation_id]["messages"] = \
            user_conversations[user_id][conversation_id]["messages"][-MAX_CONVERSATION_MESSAGES:]

def get_conversation(user_id: str, conversation_id: str) -> Dict[str, Any]:
    """
    Get a conversation by ID.
    
    Args:
        user_id: User identifier
        conversation_id: Conversation identifier
        
    Returns:
        Conversation data
    """
    if user_id not in user_conversations or conversation_id not in user_conversations[user_id]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )
    
    conv = user_conversations[user_id][conversation_id]
    return {
        "conversation_id": conversation_id,
        "metadata": conv["metadata"],
        "messages": conv["messages"]
    }

def get_user_conversations(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all conversations for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        List of conversation summaries
    """
    if user_id not in user_conversations:
        return []
    
    return [
        {
            "conversation_id": conv_id,
            "title": data["metadata"].get("title", f"Chat {conv_id}"),
            "role": data["metadata"].get("role", "unknown"),
            "created_at": data["metadata"].get("created_at", ""),
            "message_count": len(data["messages"]),
            "last_updated": data["messages"][-1]["timestamp"] if data["messages"] else data["metadata"].get("created_at", "")
        }
        for conv_id, data in user_conversations[user_id].items()
    ]

def update_conversation_title(user_id: str, conversation_id: str, title: str):
    """
    Update a conversation's title.
    
    Args:
        user_id: User identifier
        conversation_id: Conversation identifier
        title: New title
    """
    if user_id not in user_conversations or conversation_id not in user_conversations[user_id]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )
    
    user_conversations[user_id][conversation_id]["metadata"]["title"] = title

def delete_conversation(user_id: str, conversation_id: str):
    """
    Delete a conversation.
    
    Args:
        user_id: User identifier
        conversation_id: Conversation identifier
    """
    if user_id not in user_conversations or conversation_id not in user_conversations[user_id]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )
    
    del user_conversations[user_id][conversation_id] 