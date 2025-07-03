"""
Enhanced chat routes for the WMS Chatbot API with persistent storage.
Supports advanced features including search, analytics, file uploads, and data export.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Query, BackgroundTasks
import logging

from app.models.chatbot.chat_models import (
    ChatMessage as ChatMessageRequest,
    ChatResponse,
    UserRoleResponse
)
from app.models.chatbot.enhanced_chat_models import (
    ConversationCreateRequest,
    MessageCreateRequest,
    ConversationSearchRequest,
    ChatMessageType
)
from app.services.chatbot.auth_service import get_optional_current_user, get_allowed_chatbot_roles
from app.services.chatbot.agent_service import AgentService
from app.services.chatbot.enhanced_conversation_service import EnhancedConversationService

logger = logging.getLogger("wms_chatbot.enhanced_chat_routes")

router = APIRouter()

# Initialize services
agent_service = AgentService()
conversation_service = EnhancedConversationService()


@router.post("/conversations", response_model=Dict[str, Any])
async def create_conversation(
    data: ConversationCreateRequest,
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
    user_id = current_user.get("username", "anonymous")
    available_roles = agent_service.get_available_roles()
    
    try:
        result = await conversation_service.create_conversation(
            user_id=user_id,
            title=data.title,
            agent_role=data.agent_role,
            available_roles=available_roles,
            initial_context=data.initial_context
        )
        
        logger.info(f"Created conversation {result['conversation_id']} for user {user_id}")
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation"
        )


@router.post("/conversations/{conversation_id}/messages", response_model=ChatResponse)
async def send_message(
    conversation_id: str,
    data: MessageCreateRequest,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Send a message in a conversation and get AI response.
    
    Args:
        conversation_id: Conversation identifier
        data: Message data
        current_user: Current authenticated user
        
    Returns:
        AI response to the message
    """
    user_id = current_user.get("username", "anonymous")
    
    try:
        # Add user message
        start_time = datetime.now()
        
        await conversation_service.add_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_content=data.content,
            message_type=ChatMessageType.USER,
            context=data.context,
            metadata={"has_attachments": bool(data.attachments)}
        )
        
        # Get conversation history for context
        conversation_history = await conversation_service.get_conversation_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=10,  # Last 10 messages for context
            include_context=True
        )
        
        if not conversation_history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Get AI response using agent service
        agent_role = conversation_history["agent_role"]
        context = {
            "conversation_history": conversation_history["messages"][-10:],
            "user_context": data.context or {},
            "conversation_metadata": conversation_history["metadata"]
        }
        
        ai_response_data = await agent_service.process_message(
            message=data.content,
            role=agent_role,
            context=context,
            conversation_id=conversation_id
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add assistant message
        await conversation_service.add_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_content=ai_response_data["response"],
            message_type=ChatMessageType.ASSISTANT,
            context=ai_response_data.get("context", {}),
            metadata=ai_response_data.get("metadata", {}),
            tokens_used=ai_response_data.get("tokens_used"),
            processing_time=processing_time,
            model_used=ai_response_data.get("model_used")
        )
        
        return ChatResponse(
            response=ai_response_data["response"],
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message"
        )


@router.get("/conversations/{conversation_id}", response_model=Dict[str, Any])
async def get_conversation(
    conversation_id: str,
    include_context: bool = Query(False, description="Include message context"),
    limit: Optional[int] = Query(None, description="Maximum number of messages"),
    offset: int = Query(0, description="Number of messages to skip"),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get conversation history with messages.
    
    Args:
        conversation_id: Conversation identifier
        include_context: Whether to include message context
        limit: Maximum number of messages to return
        offset: Number of messages to skip
        current_user: Current authenticated user
        
    Returns:
        Conversation data with messages
    """
    user_id = current_user.get("username", "anonymous")
    
    try:
        conversation = await conversation_service.get_conversation_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
            include_context=include_context
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get conversation"
        )


@router.get("/conversations", response_model=Dict[str, Any])
async def list_conversations(
    limit: int = Query(20, description="Maximum number of conversations"),
    offset: int = Query(0, description="Number of conversations to skip"),
    status: Optional[str] = Query(None, description="Filter by conversation status"),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get user's conversations with pagination.
    
    Args:
        limit: Maximum number of conversations to return
        offset: Number of conversations to skip
        status: Filter by conversation status
        current_user: Current authenticated user
        
    Returns:
        User conversations with pagination info
    """
    user_id = current_user.get("username", "anonymous")
    
    try:
        result = await conversation_service.get_user_conversations(
            user_id=user_id,
            limit=limit,
            offset=offset,
            status=status
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to list conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list conversations"
        )


@router.post("/conversations/search", response_model=Dict[str, Any])
async def search_conversations(
    search_request: ConversationSearchRequest,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Search conversations based on criteria.
    
    Args:
        search_request: Search parameters
        current_user: Current authenticated user
        
    Returns:
        Search results with pagination
    """
    user_id = current_user.get("username", "anonymous")
    
    try:
        result = await conversation_service.search_conversations(
            user_id=user_id,
            search_request=search_request
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to search conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search conversations"
        )


@router.patch("/conversations/{conversation_id}/archive")
async def archive_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Archive a conversation.
    
    Args:
        conversation_id: Conversation identifier
        current_user: Current authenticated user
        
    Returns:
        Success message
    """
    user_id = current_user.get("username", "anonymous")
    
    try:
        success = await conversation_service.archive_conversation(
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {"message": "Conversation archived successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to archive conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to archive conversation"
        )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    hard_delete: bool = Query(False, description="Permanently delete the conversation"),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Delete a conversation (soft delete by default).
    
    Args:
        conversation_id: Conversation identifier
        hard_delete: Whether to permanently delete the conversation
        current_user: Current authenticated user
        
    Returns:
        Success message
    """
    user_id = current_user.get("username", "anonymous")
    
    try:
        success = await conversation_service.delete_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            hard_delete=hard_delete
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {"message": f"Conversation {'permanently deleted' if hard_delete else 'deleted'} successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation"
        )


@router.get("/roles", response_model=UserRoleResponse)
async def get_user_roles(
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get available chatbot roles for the current user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Available roles and current role
    """
    try:
        available_roles = agent_service.get_available_roles()
        allowed_roles = get_allowed_chatbot_roles(current_user)
        
        # Filter available roles by what the user is allowed to use
        user_roles = [role for role in available_roles if role.lower() in [r.lower() for r in allowed_roles]]
        
        return UserRoleResponse(
            available_roles=user_roles,
            current_role=allowed_roles[0] if allowed_roles else "clerk"
        )
        
    except Exception as e:
        logger.error(f"Failed to get user roles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user roles"
        )


# Legacy endpoint for backward compatibility
@router.post("/chat", response_model=ChatResponse)
async def chat_message(
    data: ChatMessageRequest,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Legacy chat endpoint for backward compatibility.
    
    Args:
        data: Chat message data
        current_user: Current authenticated user
        
    Returns:
        Chat response
    """
    user_id = current_user.get("username", "anonymous")
    
    try:
        # If conversation_id is provided, use it; otherwise create a new conversation
        conversation_id = data.conversation_id
        
        if not conversation_id:
            # Create a new conversation
            available_roles = agent_service.get_available_roles()
            conversation_result = await conversation_service.create_conversation(
                user_id=user_id,
                title=f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                agent_role=data.role,
                available_roles=available_roles
            )
            conversation_id = conversation_result["conversation_id"]
        
        # Send message using the enhanced endpoint logic
        
        # Add user message
        start_time = datetime.now()
        
        await conversation_service.add_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_content=data.message,
            message_type=ChatMessageType.USER,
            metadata={"legacy_api": True}
        )
        
        # Get AI response
        context = {"legacy_api": True}
        ai_response_data = await agent_service.process_message(
            message=data.message,
            role=data.role,
            context=context,
            conversation_id=conversation_id
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add assistant message
        await conversation_service.add_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_content=ai_response_data["response"],
            message_type=ChatMessageType.ASSISTANT,
            metadata={**ai_response_data.get("metadata", {}), "legacy_api": True},
            tokens_used=ai_response_data.get("tokens_used"),
            processing_time=processing_time,
            model_used=ai_response_data.get("model_used")
        )
        
        return ChatResponse(
            response=ai_response_data["response"],
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to process legacy chat message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message"
        )


# Migration endpoint for moving from in-memory to persistent storage
@router.post("/admin/migrate-conversations")
async def migrate_conversations(
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Migrate conversations from in-memory storage to MongoDB.
    This is an admin endpoint for system migration.
    
    Args:
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        
    Returns:
        Migration status
    """
    # Check if user has admin privileges (implement your auth logic)
    user_role = current_user.get("role", "").lower()
    if user_role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges for migration"
        )
    
    try:
        # Import the old conversation service to get in-memory data
        from app.services.chatbot.conversation_service import ConversationService
        old_service = ConversationService()
        
        if not old_service.user_conversations:
            return {"message": "No conversations to migrate", "migrated": 0}
        
        # Perform migration in background
        async def perform_migration():
            try:
                migrated_count = await conversation_service.migrate_from_memory(
                    old_service.user_conversations
                )
                logger.info(f"Migration completed: {migrated_count} conversations migrated")
            except Exception as e:
                logger.error(f"Migration failed: {str(e)}")
        
        background_tasks.add_task(perform_migration)
        
        return {
            "message": "Migration started in background",
            "status": "in_progress",
            "conversations_to_migrate": sum(len(convs) for convs in old_service.user_conversations.values())
        }
        
    except Exception as e:
        logger.error(f"Failed to start migration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start migration"
        )
