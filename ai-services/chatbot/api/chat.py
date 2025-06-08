from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
import logging

from chatbot.models.schemas import (
    ChatMessage, 
    ChatResponse, 
    NewConversation,
    ConversationResponse,
    ConversationListItem
)
from chatbot.dependencies.auth import get_optional_current_user, get_allowed_chatbot_roles
from chatbot.utils.conversation_store import (
    create_conversation,
    add_message,
    get_conversation,
    get_user_conversations,
    update_conversation_title,
    delete_conversation as delete_conv
)

# Get reference to the global agents dictionary
from chatbot.main import agents

logger = logging.getLogger("wms_chatbot")
router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/conversations", response_model=Dict[str, Any])
async def create_new_conversation(
    data: NewConversation,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Create a new conversation for the user.
    
    Args:
        data: New conversation data
        current_user: Current authenticated user
        
    Returns:
        New conversation information
    """
    # Get user ID for storage
    user_id = current_user.get("username", "anonymous")
    
    # Validate role exists
    if data.role.lower() not in agents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {data.role}. Must be one of {list(agents.keys())}"
        )
    
    # Create the conversation
    conversation_id = create_conversation(user_id, data.role.lower(), data.title)
    
    return {
        "conversation_id": conversation_id,
        "title": data.title,
        "role": data.role.lower(),
        "created_at": datetime.now().isoformat()
    }

@router.post("/message", response_model=ChatResponse)
async def chat(
    message: ChatMessage, 
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Chat with the WMS chatbot.
    
    Args:
        message: User message
        current_user: Current authenticated user or dev user in dev mode
        
    Returns:
        Chatbot response
    """
    role = message.role.lower()
    user_id = current_user.get("username", "anonymous")
    
    # Validate role exists
    if role not in agents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {role}. Must be one of {list(agents.keys())}"
        )
    
    # Validate user has access to this chatbot role
    allowed_roles = get_allowed_chatbot_roles(current_user.get("role", ""))
    if role not in allowed_roles and current_user.get("role") != "Manager":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User with role '{current_user.get('role')}' cannot access the '{role}' chatbot"
        )
    
    try:
        # Get the appropriate agent
        agent = agents[role]
        
        # Log the incoming message
        logger.info(f"Received message from {role}: {message.message[:100]}")
        
        # Process the message through the agent
        response_text = agent.run(message.message)
        
        # Create or update conversation
        conversation_id = message.conversation_id or f"{role}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Add messages to conversation
        add_message(user_id, conversation_id, "user", message.message)
        add_message(user_id, conversation_id, "assistant", response_text)
        
        # Return the response
        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )

@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation_history(
    conversation_id: str, 
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get a conversation history.
    
    Args:
        conversation_id: ID of the conversation
        current_user: Current authenticated user or dev user in dev mode
        
    Returns:
        Conversation history
    """
    return get_conversation(current_user.get("username", "anonymous"), conversation_id)

@router.put("/conversations/{conversation_id}", response_model=Dict[str, Any])
async def update_conversation(
    conversation_id: str,
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Update conversation metadata (title)
    
    Args:
        conversation_id: ID of the conversation
        data: Updated conversation data
        current_user: Current authenticated user
        
    Returns:
        Updated conversation information
    """
    if "title" in data:
        update_conversation_title(
            current_user.get("username", "anonymous"),
            conversation_id,
            data["title"]
        )
    
    return {
        "conversation_id": conversation_id,
        "success": True
    }

@router.get("/conversations", response_model=List[ConversationListItem])
async def list_conversations(
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get all conversations for the current user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List of all user conversations
    """
    return get_user_conversations(current_user.get("username", "anonymous"))

@router.delete("/conversations/{conversation_id}", response_model=Dict[str, Any])
async def delete_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Delete a user conversation.
    
    Args:
        conversation_id: ID of the conversation
        current_user: Current authenticated user
        
    Returns:
        Success message
    """
    delete_conv(current_user.get("username", "anonymous"), conversation_id)
    
    return {
        "success": True,
        "message": f"Conversation {conversation_id} deleted successfully"
    } 