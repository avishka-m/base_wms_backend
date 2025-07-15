"""
chat routes for the WMS Chatbot API with persistent storage.
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
from app.services.chatbot.agent_service import EnhancedAgentService
from app.services.chatbot.enhanced_conversation_service import EnhancedConversationService

logger = logging.getLogger("wms_chatbot.enhanced_chat_routes")

router = APIRouter()

# Initialize services
agent_service = EnhancedAgentService()
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
        # Debug: Log the types and values
        logger.info("Getting available roles from agent service...")
        available_roles = agent_service.get_available_roles()
        logger.info(f"Available roles: {available_roles}, type: {type(available_roles)}")
        
        # Extract user info from current_user dict
        username = current_user.get("username", "anonymous") if current_user else "anonymous"
        user_role = current_user.get("role", "Clerk") if current_user else "Clerk"
        logger.info(f"User: {username}, role: {user_role}")
        
        allowed_roles = get_allowed_chatbot_roles(user_role)
        logger.info(f"Allowed roles: {allowed_roles}, type: {type(allowed_roles)}")
        
        return UserRoleResponse(
            username=username,
            role=user_role,
            allowed_chatbot_roles=allowed_roles
        )
        
    except Exception as e:
        logger.error(f"Failed to get user roles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user roles"
        )


@router.get("/agents/capabilities", response_model=Dict[str, Any])
async def get_agent_capabilities(
    role: Optional[str] = Query(None, description="Specific agent role to get capabilities for"),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get capabilities for agents.
    
    Args:
        role: Optional specific agent role
        current_user: Current authenticated user
        
    Returns:
        Agent capabilities information
    """
    try:
        if role:
            capabilities = agent_service.get_agent_capabilities(role)
            return {
                "role": role,
                "capabilities": capabilities
            }
        else:
            # Get capabilities for all agents
            all_capabilities = {}
            for agent_role in agent_service.get_available_roles():
                all_capabilities[agent_role] = agent_service.get_agent_capabilities(agent_role)
            
            return {
                "all_capabilities": all_capabilities,
                "available_roles": agent_service.get_available_roles()
            }
    except Exception as e:
        logger.error(f"Failed to get agent capabilities: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get agent capabilities"
        )


@router.get("/agents/performance", response_model=Dict[str, Any])
async def get_agent_performance(
    role: Optional[str] = Query(None, description="Specific agent role to get performance for"),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get performance metrics for agents.
    
    Args:
        role: Optional specific agent role
        current_user: Current authenticated user
        
    Returns:
        Agent performance metrics
    """
    try:
        if role:
            performance = agent_service.get_agent_performance(role)
            return {
                "role": role,
                "performance": performance
            }
        else:
            # Get performance for all agents
            all_performance = {}
            for agent_role in agent_service.agents.keys():
                all_performance[agent_role] = agent_service.get_agent_performance(agent_role)
            
            return {
                "all_performance": all_performance,
                "total_agents": len(agent_service.agents)
            }
    except Exception as e:
        logger.error(f"Failed to get agent performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get agent performance"
        )


@router.post("/agents/select", response_model=Dict[str, Any])
async def select_best_agent(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Select the best agent for a query.
    
    Args:
        data: Query data with message and optional context
        current_user: Current authenticated user
        
    Returns:
        Best agent selection with suitability scores
    """
    try:
        query = data.get("query", "")
        user_role = data.get("user_role") or current_user.get("role")
        context = data.get("context", {})
        
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query is required"
            )
        
        # Get best agent
        best_agent = agent_service.select_best_agent(query, user_role, context)
        
        # Get all suitable agents with scores
        suitable_agents = agent_service.get_suitable_agents(query, user_role, context)
        
        # Get query classifications
        query_classifications = agent_service.classify_query(query)
        
        return {
            "best_agent": best_agent,
            "suitable_agents": [
                {"role": role, "suitability_score": score}
                for role, score in suitable_agents
            ],
            "query_classifications": [
                {"type": qt.value, "confidence": conf}
                for qt, conf in query_classifications
            ],
            "user_role": user_role,
            "query": query[:100] + "..." if len(query) > 100 else query
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to select best agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to select best agent"
        )


@router.get("/agents/status", response_model=Dict[str, Any])
async def get_agent_system_status(
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get overall agent system status and metrics.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        System status information
    """
    try:
        status_info = agent_service.get_system_status()
        return status_info
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system status"
        )


@router.post("/agents/feedback", response_model=Dict[str, Any])
async def provide_agent_feedback(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Provide feedback on agent performance.
    
    Args:
        data: Feedback data including agent role and feedback type
        current_user: Current authenticated user
        
    Returns:
        Feedback acknowledgment
    """
    try:
        agent_role = data.get("agent_role")
        positive_feedback = data.get("positive_feedback", True)
        user_role = current_user.get("role")
        
        if not agent_role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Agent role is required"
            )
        
        if agent_role not in agent_service.agents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown agent role: {agent_role}"
            )
        
        # Update user preferences
        if user_role:
            agent_service.update_user_preferences(user_role, agent_role, positive_feedback)
        
        return {
            "message": "Feedback received",
            "agent_role": agent_role,
            "positive_feedback": positive_feedback,
            "user_role": user_role
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process feedback"
        )


@router.post("/conversations/semantic-search", response_model=Dict[str, Any])
async def semantic_search_conversations(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Perform semantic search across conversations using AI embeddings.
    
    Args:
        data: Search parameters including query and filters
        current_user: Current authenticated user
        
    Returns:
        Search results with similarity scores
    """
    try:
        from app.services.chatbot.advanced_conversation_service import AdvancedConversationService
        
        # Initialize advanced service
        advanced_service = AdvancedConversationService()
        
        user_id = current_user.get("username", "anonymous")
        query = data.get("query", "")
        limit = data.get("limit", 20)
        similarity_threshold = data.get("similarity_threshold", 0.7)
        content_types = data.get("content_types")
        agent_roles = data.get("agent_roles")
        
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query is required"
            )
        
        # Parse date range if provided
        date_range = None
        if data.get("date_from") or data.get("date_to"):
            date_from = datetime.fromisoformat(data["date_from"]) if data.get("date_from") else None
            date_to = datetime.fromisoformat(data["date_to"]) if data.get("date_to") else None
            if date_from or date_to:
                date_range = (date_from, date_to)
        
        # Perform semantic search
        results = await advanced_service.semantic_search(
            user_id=user_id,
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
            content_types=content_types,
            agent_roles=agent_roles,
            date_range=date_range
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Semantic search failed"
        )


@router.post("/conversations/bulk-operations", response_model=Dict[str, Any])
async def bulk_conversation_operations(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Perform bulk operations on conversations (archive, delete, tag, etc.).
    
    Args:
        data: Operation parameters including conversation IDs and operation type
        current_user: Current authenticated user
        
    Returns:
        Operation results with success/failure counts
    """
    try:
        from app.services.chatbot.advanced_conversation_service import AdvancedConversationService, ArchiveOperation
        
        # Initialize advanced service
        advanced_service = AdvancedConversationService()
        
        user_id = current_user.get("username", "anonymous")
        conversation_ids = data.get("conversation_ids", [])
        operation = data.get("operation", "")
        operation_data = data.get("operation_data", {})
        
        if not conversation_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Conversation IDs are required"
            )
        
        if not operation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Operation type is required"
            )
        
        # Validate operation type
        try:
            operation_enum = ArchiveOperation(operation.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid operation: {operation}"
            )
        
        # Perform bulk operation
        results = await advanced_service.bulk_archive_operations(
            user_id=user_id,
            conversation_ids=conversation_ids,
            operation=operation_enum,
            operation_data=operation_data
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk operation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk operation failed"
        )


@router.post("/conversations/templates", response_model=Dict[str, Any])
async def create_conversation_template(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Create a conversation template for reuse.
    
    Args:
        data: Template data including name, description, and configuration
        current_user: Current authenticated user
        
    Returns:
        Created template information
    """
    try:
        from app.services.chatbot.advanced_conversation_service import AdvancedConversationService
        
        # Initialize advanced service
        advanced_service = AdvancedConversationService()
        
        user_id = current_user.get("username", "anonymous")
        
        # Validate required fields
        if not data.get("name"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Template name is required"
            )
        
        # Create template
        template = await advanced_service.create_conversation_template(
            template_data=data,
            user_id=user_id
        )
        
        return template
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create conversation template: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation template"
        )


@router.get("/conversations/templates", response_model=list[Dict[str, Any]])
async def get_conversation_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    agent_role: Optional[str] = Query(None, description="Filter by agent role"),
    limit: int = Query(20, description="Maximum number of templates"),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get conversation templates.
    
    Args:
        category: Optional category filter
        agent_role: Optional agent role filter
        limit: Maximum number of templates
        current_user: Current authenticated user
        
    Returns:
        List of conversation templates
    """
    try:
        from app.services.chatbot.advanced_conversation_service import AdvancedConversationService
        
        # Initialize advanced service
        advanced_service = AdvancedConversationService()
        
        user_id = current_user.get("username", "anonymous")
        
        # Get templates
        templates = await advanced_service.get_conversation_templates(
            user_id=user_id,
            category=category,
            agent_role=agent_role,
            limit=limit
        )
        
        return templates
        
    except Exception as e:
        logger.error(f"Failed to get conversation templates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get conversation templates"
        )


@router.get("/conversations/insights", response_model=Dict[str, Any])
async def get_conversation_insights(
    period_days: int = Query(30, description="Analysis period in days"),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Generate comprehensive insights about user's conversations.
    
    Args:
        period_days: Period in days to analyze
        current_user: Current authenticated user
        
    Returns:
        Conversation insights and analytics
    """
    try:
        from app.services.chatbot.advanced_conversation_service import AdvancedConversationService
        
        # Initialize advanced service
        advanced_service = AdvancedConversationService()
        
        user_id = current_user.get("username", "anonymous")
        
        # Generate insights
        insights = await advanced_service.generate_conversation_insights(
            user_id=user_id,
            period_days=period_days
        )
        
        # Convert to dict for JSON response
        insights_dict = {
            "total_conversations": insights.total_conversations,
            "active_conversations": insights.active_conversations,
            "archived_conversations": insights.archived_conversations,
            "deleted_conversations": insights.deleted_conversations,
            "average_messages_per_conversation": insights.average_messages_per_conversation,
            "most_active_agent": insights.most_active_agent,
            "most_common_topics": insights.most_common_topics,
            "conversation_trends": insights.conversation_trends,
            "user_engagement_score": insights.user_engagement_score,
            "context_usage_patterns": insights.context_usage_patterns,
            "period_days": period_days,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return insights_dict
        
    except Exception as e:
        logger.error(f"Failed to generate conversation insights: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate conversation insights"
        )


@router.post("/conversations/export", response_model=Dict[str, Any])
async def export_conversations(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Export conversations to various formats.
    
    Args:
        data: Export parameters including format and options
        current_user: Current authenticated user
        
    Returns:
        Export data and metadata
    """
    try:
        from app.services.chatbot.advanced_conversation_service import AdvancedConversationService
        
        # Initialize advanced service
        advanced_service = AdvancedConversationService()
        
        user_id = current_user.get("username", "anonymous")
        conversation_ids = data.get("conversation_ids")  # None for all conversations
        format = data.get("format", "json")
        include_metadata = data.get("include_metadata", True)
        include_context = data.get("include_context", True)
        
        # Validate format
        if format.lower() not in ["json", "csv", "txt"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid format. Must be one of: json, csv, txt"
            )
        
        # Export conversations
        export_result = await advanced_service.export_conversations(
            user_id=user_id,
            conversation_ids=conversation_ids,
            format=format,
            include_metadata=include_metadata,
            include_context=include_context
        )
        
        return export_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export conversations"
        )


@router.get("/conversations/analytics/search", response_model=Dict[str, Any])
async def get_search_analytics(
    days: int = Query(7, description="Number of days to analyze"),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get search analytics for the user.
    
    Args:
        days: Number of days to analyze
        current_user: Current authenticated user
        
    Returns:
        Search analytics data
    """
    try:
        user_id = current_user.get("username", "anonymous")
        
        # Get search analytics from database
        search_analytics_col = await get_async_collection("chat_search_analytics")
        
        # Date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get analytics data
        analytics = []
        async for doc in search_analytics_col.find({
            "user_id": user_id,
            "timestamp": {"$gte": start_date, "$lte": end_date}
        }).sort("timestamp", DESCENDING):
            analytics.append({
                "query": doc.get("query", ""),
                "search_type": doc.get("search_type", ""),
                "result_count": doc.get("result_count", 0),
                "processing_time": doc.get("processing_time", 0),
                "timestamp": doc.get("timestamp", "").isoformat() if doc.get("timestamp") else ""
            })
        
        # Calculate summary statistics
        total_searches = len(analytics)
        avg_results = sum(a["result_count"] for a in analytics) / total_searches if total_searches > 0 else 0
        avg_processing_time = sum(a["processing_time"] for a in analytics) / total_searches if total_searches > 0 else 0
        
        search_types = Counter(a["search_type"] for a in analytics)
        popular_queries = Counter(a["query"] for a in analytics).most_common(10)
        
        return {
            "total_searches": total_searches,
            "average_results_per_search": avg_results,
            "average_processing_time": avg_processing_time,
            "search_types": dict(search_types),
            "popular_queries": popular_queries,
            "recent_searches": analytics[:20],  # Last 20 searches
            "period_days": days
        }
        
    except Exception as e:
        logger.error(f"Failed to get search analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get search analytics"
        )


@router.get("/context/current", response_model=Dict[str, Any])
async def get_current_context(
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get current workplace context for the user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current workplace context
    """
    try:
        from app.services.chatbot.context_awareness_service import ContextAwarenessService
        
        context_service = ContextAwarenessService()
        user_id = current_user.get("username", "anonymous")
        
        context = await context_service.get_current_context(user_id)
        
        return {
            "user_id": context.user_id,
            "current_location": context.current_location,
            "current_task": context.current_task,
            "active_orders": context.active_orders,
            "current_role": context.current_role,
            "shift_info": context.shift_info,
            "inventory_focus": context.inventory_focus,
            "recent_activities": context.recent_activities[-10:],  # Last 10 activities
            "performance_metrics": context.performance_metrics,
            "preferences": context.preferences,
            "context_score": context.context_score,
            "last_updated": context.last_updated.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get current context: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get current context"
        )


@router.post("/context/signal", response_model=Dict[str, Any])
async def update_context_signal(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Update a context signal for the user.
    
    Args:
        data: Context signal data
        current_user: Current authenticated user
        
    Returns:
        Update confirmation
    """
    try:
        from app.services.chatbot.context_awareness_service import ContextAwarenessService, ContextType
        
        context_service = ContextAwarenessService()
        user_id = current_user.get("username", "anonymous")
        
        signal_type = data.get("signal_type")
        source = data.get("source", "api")
        signal_data = data.get("data", {})
        confidence = data.get("confidence", 1.0)
        expires_in_minutes = data.get("expires_in_minutes")
        
        if not signal_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Signal type is required"
            )
        
        # Validate signal type
        try:
            context_type = ContextType(signal_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid signal type: {signal_type}"
            )
        
        await context_service.update_context_signal(
            user_id=user_id,
            signal_type=context_type,
            source=source,
            data=signal_data,
            confidence=confidence,
            expires_in_minutes=expires_in_minutes
        )
        
        return {
            "message": "Context signal updated successfully",
            "signal_type": signal_type,
            "source": source,
            "confidence": confidence
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update context signal: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update context signal"
        )


@router.get("/context/suggestions", response_model=list[Dict[str, Any]])
async def get_contextual_suggestions(
    query: Optional[str] = Query(None, description="Optional query filter"),
    limit: int = Query(5, description="Maximum number of suggestions"),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get contextual suggestions based on current context.
    
    Args:
        query: Optional query for filtering suggestions
        limit: Maximum number of suggestions
        current_user: Current authenticated user
        
    Returns:
        List of contextual suggestions
    """
    try:
        from app.services.chatbot.context_awareness_service import ContextAwarenessService
        
        context_service = ContextAwarenessService()
        user_id = current_user.get("username", "anonymous")
        
        suggestions = await context_service.get_contextual_suggestions(
            user_id=user_id,
            query=query,
            limit=limit
        )
        
        return [
            {
                "suggestion_id": s.suggestion_id,
                "type": s.type,
                "title": s.title,
                "content": s.content,
                "action": s.action,
                "priority": s.priority,
                "confidence": s.confidence,
                "context_match": s.context_match
            } for s in suggestions
        ]
        
    except Exception as e:
        logger.error(f"Failed to get contextual suggestions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get contextual suggestions"
        )


@router.post("/context/detect", response_model=Dict[str, Any])
async def detect_context_from_message(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Detect context from a message.
    
    Args:
        data: Message data for context detection
        current_user: Current authenticated user
        
    Returns:
        Detected context information
    """
    try:
        from app.services.chatbot.context_awareness_service import ContextAwarenessService
        
        context_service = ContextAwarenessService()
        user_id = current_user.get("username", "anonymous")
        
        message = data.get("message", "")
        conversation_context = data.get("conversation_context")
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message is required"
            )
        
        detected_context = await context_service.detect_context_from_message(
            user_id=user_id,
            message=message,
            conversation_context=conversation_context
        )
        
        return {
            "detected_context": detected_context,
            "message": message,
            "confidence_scores": {
                key: 0.8 if key in ["location", "order_id"] else 0.6
                for key in detected_context.keys()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to detect context from message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to detect context from message"
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
    user_role = current_user.get("role", "")
    
    # Validate user has access to this chatbot role
    allowed_roles = get_allowed_chatbot_roles(user_role)
    if data.role not in allowed_roles and user_role != "Manager":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User with role '{user_role}' cannot access the '{data.role}' chatbot"
        )
    
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


# Add these new endpoints after the existing ones, before the migration endpoint
@router.get("/users/{user_id}/history/overview", response_model=Dict[str, Any])
async def get_user_history_overview(
    user_id: str,
    period_days: int = Query(30, description="Number of days to analyze"),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get comprehensive history overview for a specific user.
    
    Args:
        user_id: User identifier (must match current user unless admin)
        period_days: Number of days to analyze
        current_user: Current authenticated user
        
    Returns:
        User's conversation overview with analytics
    """
    requesting_user = current_user.get("username", "anonymous")
    user_role = current_user.get("role", "").lower()
    
    # Security check: users can only access their own data unless admin/manager
    if requesting_user != user_id and user_role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Can only view your own conversation history"
        )
    
    try:
        # Get user's conversation statistics
        conversations = await conversation_service.get_user_conversations(
            user_id=user_id,
            limit=1000,  # Get all conversations for analytics
            offset=0
        )
        
        # Calculate analytics
        total_conversations = len(conversations.get("conversations", []))
        today = datetime.now()
        recent_conversations = []
        agent_usage = {}
        conversation_lengths = []
        
        for conv in conversations.get("conversations", []):
            conv_date = datetime.fromisoformat(conv.get("created_at", today.isoformat()))
            days_ago = (today - conv_date).days
            
            if days_ago <= period_days:
                recent_conversations.append(conv)
                
                # Track agent role usage
                agent_role = conv.get("agent_role", "general")
                agent_usage[agent_role] = agent_usage.get(agent_role, 0) + 1
                
                # Track conversation length
                msg_count = conv.get("message_count", 0)
                if msg_count > 0:
                    conversation_lengths.append(msg_count)
        
        # Calculate average conversation length
        avg_length = sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
        
        # Find most active agent
        most_used_agent = max(agent_usage.items(), key=lambda x: x[1]) if agent_usage else ("general", 0)
        
        # Get conversation categories (based on first message content)
        categories = {}
        for conv in recent_conversations[:20]:  # Analyze recent 20 conversations
            # Simple categorization based on keywords
            last_msg = conv.get("last_message", "").lower()
            if any(word in last_msg for word in ["inventory", "stock", "item"]):
                categories["inventory"] = categories.get("inventory", 0) + 1
            elif any(word in last_msg for word in ["order", "shipping", "delivery"]):
                categories["orders"] = categories.get("orders", 0) + 1
            elif any(word in last_msg for word in ["analytics", "report", "data"]):
                categories["analytics"] = categories.get("analytics", 0) + 1
            else:
                categories["general"] = categories.get("general", 0) + 1
        
        return {
            "user_id": user_id,
            "period_days": period_days,
            "overview": {
                "total_conversations": total_conversations,
                "recent_conversations": len(recent_conversations),
                "average_conversation_length": round(avg_length, 1),
                "most_used_agent": {
                    "role": most_used_agent[0],
                    "count": most_used_agent[1]
                },
                "agent_usage_breakdown": agent_usage,
                "conversation_categories": categories,
                "activity_trend": len(recent_conversations) > 0
            },
            "recent_conversations": recent_conversations[:10]  # Return 10 most recent
        }
        
    except Exception as e:
        logger.error(f"Failed to get user history overview: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get history overview"
        )


@router.get("/users/current/quick-history", response_model=Dict[str, Any])
async def get_current_user_quick_history(
    limit: int = Query(5, description="Number of recent conversations"),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get quick history for current user - optimized for UI sidebar.
    
    Args:
        limit: Number of recent conversations to return
        current_user: Current authenticated user
        
    Returns:
        Quick access history data
    """
    user_id = current_user.get("username", "anonymous")
    
    try:
        # Get recent conversations with minimal data
        conversations = await conversation_service.get_user_conversations(
            user_id=user_id,
            limit=limit,
            offset=0,
            status="active"
        )
        
        quick_history = []
        for conv in conversations.get("conversations", []):
            # Create quick summary
            last_msg = conv.get("last_message", "")
            preview = last_msg[:50] + "..." if len(last_msg) > 50 else last_msg
            
            quick_history.append({
                "conversation_id": conv["conversation_id"],
                "title": conv.get("title", preview),
                "preview": preview,
                "agent_role": conv.get("agent_role", "general"),
                "message_count": conv.get("message_count", 0),
                "last_activity": conv.get("last_message_at", conv.get("created_at")),
                "created_at": conv.get("created_at"),
                "is_recent": True  # All returned conversations are recent
            })
        
        return {
            "user_id": user_id,
            "conversations": quick_history,
            "total_available": conversations.get("total", 0),
            "has_more": conversations.get("total", 0) > limit
        }
        
    except Exception as e:
        logger.error(f"Failed to get quick history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get quick history"
        )


@router.post("/conversations/smart-search", response_model=Dict[str, Any])
async def smart_conversation_search(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Smart search across user's conversations with AI-powered relevance.
    
    Args:
        data: Search parameters including query, filters, and options
        current_user: Current authenticated user
        
    Returns:
        Relevant conversations with context highlights
    """
    user_id = current_user.get("username", "anonymous")
    query = data.get("query", "").strip()
    
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query is required"
        )
    
    try:
        # Get user's conversations
        all_conversations = await conversation_service.get_user_conversations(
            user_id=user_id,
            limit=200,  # Search through recent 200 conversations
            offset=0
        )
        
        search_results = []
        query_lower = query.lower()
        
        for conv in all_conversations.get("conversations", []):
            relevance_score = 0
            matched_context = []
            
            # Check title match
            title = conv.get("title", "")
            if query_lower in title.lower():
                relevance_score += 10
                matched_context.append(f"Title: {title}")
            
            # Check last message match
            last_msg = conv.get("last_message", "")
            if query_lower in last_msg.lower():
                relevance_score += 8
                # Find the matching snippet
                start_idx = max(0, last_msg.lower().find(query_lower) - 20)
                end_idx = min(len(last_msg), start_idx + 60)
                snippet = last_msg[start_idx:end_idx]
                matched_context.append(f"Message: ...{snippet}...")
            
            # Check agent role match
            agent_role = conv.get("agent_role", "")
            if query_lower in agent_role.lower():
                relevance_score += 5
                matched_context.append(f"Agent: {agent_role}")
            
            # Add conversation if relevant
            if relevance_score > 0:
                search_results.append({
                    "conversation_id": conv["conversation_id"],
                    "title": title,
                    "relevance_score": relevance_score,
                    "matched_context": matched_context,
                    "agent_role": agent_role,
                    "message_count": conv.get("message_count", 0),
                    "last_activity": conv.get("last_message_at"),
                    "created_at": conv.get("created_at")
                })
        
        # Sort by relevance score
        search_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "query": query,
            "total_results": len(search_results),
            "results": search_results[:20],  # Return top 20 results
            "search_suggestions": [
                "inventory status",
                "order tracking", 
                "analytics report",
                "shipping updates"
            ] if not search_results else []
        }
        
    except Exception as e:
        logger.error(f"Failed to perform smart search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform smart search"
        )


@router.post("/conversations/{conversation_id}/summarize", response_model=Dict[str, Any])
async def summarize_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Generate an AI summary of a conversation.
    
    Args:
        conversation_id: Conversation identifier
        current_user: Current authenticated user
        
    Returns:
        Conversation summary with key topics and outcomes
    """
    user_id = current_user.get("username", "anonymous")
    
    try:
        # Get conversation with all messages
        conversation = await conversation_service.get_conversation_history(
            user_id=user_id,
            conversation_id=conversation_id,
            include_context=True
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        messages = conversation.get("messages", [])
        if len(messages) < 2:
            return {
                "conversation_id": conversation_id,
                "summary": "Conversation too short to summarize",
                "key_topics": [],
                "outcome": "No significant outcome"
            }
        
        # Simple summarization logic (in production, use AI model)
        user_messages = [msg for msg in messages if msg.get("message_type") == "USER"]
        ai_messages = [msg for msg in messages if msg.get("message_type") == "ASSISTANT"]
        
        # Extract key topics from user messages
        key_topics = []
        common_words = ["inventory", "order", "shipping", "analytics", "item", "stock", "delivery"]
        
        all_text = " ".join([msg.get("content", "") for msg in user_messages]).lower()
        for word in common_words:
            if word in all_text:
                key_topics.append(word.title())
        
        # Create simple summary
        summary_parts = []
        if user_messages:
            first_msg = user_messages[0].get("content", "")[:100]
            summary_parts.append(f"Started with: {first_msg}...")
        
        if len(messages) > 4:
            summary_parts.append(f"Extended conversation with {len(messages)} messages")
        
        if ai_messages:
            last_ai_msg = ai_messages[-1].get("content", "")[:100]
            summary_parts.append(f"Concluded with: {last_ai_msg}...")
        
        return {
            "conversation_id": conversation_id,
            "summary": " | ".join(summary_parts),
            "key_topics": key_topics[:5],  # Top 5 topics
            "message_count": len(messages),
            "duration_estimation": f"~{max(1, len(messages) // 2)} minutes",
            "agent_role": conversation.get("agent_role", "general"),
            "outcome": "Information provided" if ai_messages else "In progress"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to summarize conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to summarize conversation"
        )


@router.post("/conversations/bulk-actions", response_model=Dict[str, Any])
async def bulk_conversation_actions(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Perform bulk actions on multiple conversations.
    
    Args:
        data: Bulk action data including conversation IDs and action type
        current_user: Current authenticated user
        
    Returns:
        Results of bulk operations
    """
    user_id = current_user.get("username", "anonymous")
    conversation_ids = data.get("conversation_ids", [])
    action = data.get("action", "")
    
    if not conversation_ids or not action:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conversation IDs and action type are required"
        )
    
    try:
        results = {
            "action": action,
            "total_requested": len(conversation_ids),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        for conv_id in conversation_ids:
            try:
                if action == "delete":
                    success = await conversation_service.delete_conversation(
                        user_id=user_id,
                        conversation_id=conv_id,
                        hard_delete=False
                    )
                elif action == "archive":
                    success = await conversation_service.archive_conversation(
                        user_id=user_id,
                        conversation_id=conv_id
                    )
                else:
                    results["errors"].append(f"Unknown action: {action}")
                    results["failed"] += 1
                    continue
                
                if success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Failed to {action} conversation {conv_id}")
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Error with {conv_id}: {str(e)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to perform bulk actions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform bulk actions"
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


# Add this new endpoint after the existing user history overview endpoint
@router.get("/users/current/history/overview", response_model=Dict[str, Any])
async def get_current_user_history_overview(
    period_days: int = Query(30, description="Number of days to analyze"),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get comprehensive history overview for the current user.
    
    Args:
        period_days: Number of days to analyze
        current_user: Current authenticated user
        
    Returns:
        Current user's conversation overview with analytics
    """
    user_id = current_user.get("username", "anonymous")
    
    try:
        # Get user's conversation statistics
        conversations = await conversation_service.get_user_conversations(
            user_id=user_id,
            limit=1000,  # Get all conversations for analytics
            offset=0
        )
        
        # Calculate analytics
        total_conversations = len(conversations.get("conversations", []))
        today = datetime.now()
        recent_conversations = []
        agent_usage = {}
        conversation_lengths = []
        
        for conv in conversations.get("conversations", []):
            conv_date = datetime.fromisoformat(conv.get("created_at", today.isoformat()))
            days_ago = (today - conv_date).days
            
            if days_ago <= period_days:
                recent_conversations.append(conv)
                
                # Track agent role usage
                agent_role = conv.get("agent_role", "general")
                agent_usage[agent_role] = agent_usage.get(agent_role, 0) + 1
                
                # Track conversation length
                msg_count = conv.get("message_count", 0)
                if msg_count > 0:
                    conversation_lengths.append(msg_count)
        
        # Calculate average conversation length
        avg_length = sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
        
        # Find most active agent
        most_used_agent = max(agent_usage.items(), key=lambda x: x[1]) if agent_usage else ("general", 0)
        
        # Get conversation categories (based on first message content)
        categories = {}
        for conv in recent_conversations[:20]:  # Analyze recent 20 conversations
            # Simple categorization based on keywords
            last_msg = conv.get("last_message", "").lower()
            if any(word in last_msg for word in ["inventory", "stock", "item"]):
                categories["inventory"] = categories.get("inventory", 0) + 1
            elif any(word in last_msg for word in ["order", "shipping", "delivery"]):
                categories["orders"] = categories.get("orders", 0) + 1
            elif any(word in last_msg for word in ["analytics", "report", "data"]):
                categories["analytics"] = categories.get("analytics", 0) + 1
            else:
                categories["general"] = categories.get("general", 0) + 1
        
        return {
            "user_id": user_id,
            "period_days": period_days,
            "overview": {
                "total_conversations": total_conversations,
                "recent_conversations": len(recent_conversations),
                "average_conversation_length": round(avg_length, 1),
                "most_used_agent": {
                    "role": most_used_agent[0],
                    "count": most_used_agent[1]
                },
                "agent_usage_breakdown": agent_usage,
                "conversation_categories": categories,
                "activity_trend": len(recent_conversations) > 0
            },
            "recent_conversations": recent_conversations[:10]  # Return 10 most recent
        }
        
    except Exception as e:
        logger.error(f"Failed to get current user history overview: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get history overview"
        )


@router.get("/users/current/quick-history", response_model=Dict[str, Any])
async def get_current_user_quick_history(
    limit: int = Query(5, description="Number of recent conversations"),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get quick history for current user - optimized for UI sidebar.
    
    Args:
        limit: Number of recent conversations to return
        current_user: Current authenticated user
        
    Returns:
        Quick access history data
    """
    user_id = current_user.get("username", "anonymous")
    
    try:
        # Get recent conversations with minimal data
        conversations = await conversation_service.get_user_conversations(
            user_id=user_id,
            limit=limit,
            offset=0,
            status="active"
        )
        
        quick_history = []
        for conv in conversations.get("conversations", []):
            # Create quick summary
            last_msg = conv.get("last_message", "")
            preview = last_msg[:50] + "..." if len(last_msg) > 50 else last_msg
            
            quick_history.append({
                "conversation_id": conv["conversation_id"],
                "title": conv.get("title", preview),
                "preview": preview,
                "agent_role": conv.get("agent_role", "general"),
                "message_count": conv.get("message_count", 0),
                "last_activity": conv.get("last_message_at", conv.get("created_at")),
                "created_at": conv.get("created_at"),
                "is_recent": True  # All returned conversations are recent
            })
        
        return {
            "user_id": user_id,
            "conversations": quick_history,
            "total_available": conversations.get("total", 0),
            "has_more": conversations.get("total", 0) > limit
        }
        
    except Exception as e:
        logger.error(f"Failed to get quick history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get quick history"
        )
