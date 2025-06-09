"""Conversation management API endpoints"""

from datetime import datetime
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, status

from dependencies.auth import get_optional_current_user
from models.schemas import NewConversation, ConversationResponse, ConversationListItem
from services.conversation import conversation_service
from services.agent import agent_service
from services.auth import auth_service

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

@router.post("", response_model=Dict[str, Any])
async def create_conversation(
    data: NewConversation,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """Create a new conversation for the user."""
    # Validate role exists
    if not agent_service.validate_role(data.role):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {data.role}. Must be one of {agent_service.get_available_agents()}"
        )
    
    # Get user ID
    user_id = auth_service.get_user_id(current_user)
    
    # Create conversation
    return conversation_service.create_conversation(
        user_id=user_id,
        role=data.role.lower(),
        title=data.title
    )

@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str, 
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """Get a conversation history."""
    user_id = auth_service.get_user_id(current_user)
    
    conversation = conversation_service.get_conversation(user_id, conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )
    
    return conversation

@router.put("/{conversation_id}", response_model=Dict[str, Any])
async def update_conversation(
    conversation_id: str,
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """Update conversation metadata."""
    user_id = auth_service.get_user_id(current_user)
    
    # Update title if provided
    if "title" in data:
        success = conversation_service.update_conversation_title(
            user_id, conversation_id, data["title"]
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {conversation_id}"
            )
    
    return {
        "conversation_id": conversation_id,
        "success": True
    }

@router.get("", response_model=List[ConversationListItem])
async def list_conversations(
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """Get all conversations for the current user."""
    user_id = auth_service.get_user_id(current_user)
    return conversation_service.get_user_conversations(user_id)

@router.delete("/{conversation_id}", response_model=Dict[str, Any])
async def delete_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """Delete a user conversation."""
    user_id = auth_service.get_user_id(current_user)
    
    success = conversation_service.delete_conversation(user_id, conversation_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )
    
    return {
        "success": True,
        "message": f"Conversation {conversation_id} deleted successfully"
    } 