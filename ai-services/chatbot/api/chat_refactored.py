"""Chat API endpoints"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status

from dependencies.auth import get_optional_current_user
from models.schemas import ChatMessage, ChatResponse
from services.conversation import conversation_service
from services.agent import agent_service
from services.auth import auth_service
from core.logging import logger

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat(
    message: ChatMessage, 
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Chat with the WMS chatbot.
    
    Args:
        message: User message
        current_user: Current authenticated user
        
    Returns:
        Chatbot response
    """
    role = message.role.lower()
    
    # Validate role exists
    if not agent_service.validate_role(role):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {role}. Must be one of {agent_service.get_available_agents()}"
        )
    
    # Check user access to this chatbot role
    auth_service.check_role_access(current_user, role)
    
    try:
        # Process message through agent
        response_text = agent_service.process_message(role, message.message)
        
        # Get user ID
        user_id = auth_service.get_user_id(current_user)
        
        # Create or update conversation
        conversation_id = message.conversation_id
        if not conversation_id:
            # Create new conversation
            conv_data = conversation_service.create_conversation(
                user_id=user_id,
                role=role,
                title=f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            conversation_id = conv_data["conversation_id"]
        
        # Add messages to conversation
        conversation_service.add_message(user_id, conversation_id, "user", message.message)
        conversation_service.add_message(user_id, conversation_id, "assistant", response_text)
        
        # Return the response
        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.error(f"Value error in chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        ) 