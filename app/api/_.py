"""
Chat routes for the WMS Chatbot API.
"""

from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status, Depends

from app.models.chatbot.chat_models import (
    ChatMessage,
    ChatResponse,
    NewConversation,
    ConversationResponse,
    UserRoleResponse,
    ConversationUpdateRequest,
    DeleteResponse
)
from app.services.chatbot.auth_service import get_optional_current_user, get_allowed_chatbot_roles
from app.services.chatbot.agent_service import AgentService
from app.services.chatbot.conversation_service import ConversationService

router = APIRouter()

# Initialize services
agent_service = AgentService()
conversation_service = ConversationService()


@router.post("/conversations", response_model=Dict[str, Any])
async def create_conversation(
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
    
    # Get available roles for this user
    available_roles = agent_service.get_available_roles()
    
    try:
        return conversation_service.create_conversation(
            user_id=user_id,
            title=data.title,
            role=data.role,
            available_roles=available_roles
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/chat", response_model=ChatResponse)
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
    
    # Validate user has access to this chatbot role
    allowed_roles = get_allowed_chatbot_roles(current_user.get("role", ""))
    if role not in allowed_roles and current_user.get("role") != "Manager":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User with role '{current_user.get('role')}' cannot access the '{role}' chatbot"
        )
    
    try:
        # Process the message through the agent
        response_text = agent_service.process_message(role, message.message)
        
        # Create or update conversation
        conversation_id = message.conversation_id or f"{role}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Store the conversation
        conversation_service.add_message(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=message.message,
            bot_response=response_text
        )
        
        # Return the response
        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
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
    user_id = current_user.get("username", "anonymous")
    
    try:
        return conversation_service.get_conversation(user_id, conversation_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.put("/conversations/{conversation_id}", response_model=Dict[str, Any])
async def update_conversation(
    conversation_id: str,
    data: ConversationUpdateRequest,
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
    user_id = current_user.get("username", "anonymous")
    
    if not data.title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Title is required for update"
        )
    
    try:
        return conversation_service.update_conversation(user_id, conversation_id, data.title)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/conversations", response_model=List[Dict[str, Any]])
async def get_user_conversations(
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get all conversations for the current user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List of all user conversations
    """
    user_id = current_user.get("username", "anonymous")
    return conversation_service.get_user_conversations(user_id)


@router.delete("/conversations/{conversation_id}", response_model=DeleteResponse)
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
    user_id = current_user.get("username", "anonymous")
    
    try:
        return conversation_service.delete_conversation(user_id, conversation_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/user/role", response_model=UserRoleResponse)
async def get_user_role(current_user: Dict[str, Any] = Depends(get_optional_current_user)):
    """
    Get the current user's role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User role information
    """
    return {
        "username": current_user.get("username"),
        "role": current_user.get("role"),
        "allowed_chatbot_roles": get_allowed_chatbot_roles(current_user.get("role", ""))
    }


@router.post("/logout", response_model=Dict[str, Any])
async def logout_user(current_user: Dict[str, Any] = Depends(get_optional_current_user)):
    """
    Clear all user session data and conversation memories on logout.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Logout confirmation
    """
    try:
        user_id = current_user.get("username", "anonymous")
        
        # Clear all conversation memories for this user
        await conversation_service.memory_service.clear_user_sessions(user_id)
        
        return {
            "success": True,
            "message": "User session cleared successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear user session: {str(e)}"
        )
