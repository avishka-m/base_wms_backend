# """
# Role-based enhanced chatbot API routes.
# Provides controlled access to enhanced chatbot features based on user roles.
# """

# from datetime import datetime
# from typing import Dict, Any, Optional, List
# from fastapi import APIRouter, HTTPException, status, Depends, Query, BackgroundTasks
# import logging

# from app.models.chatbot.chat_models import (
#     ChatMessage as ChatMessageRequest,
#     ChatResponse,
#     UserRoleResponse
# )
# from app.services.chatbot.auth_service import get_optional_current_user
# from app.services.chatbot.role_based_chatbot_service import RoleBasedChatbotService, ChatbotFeature
# from app.services.chatbot.manager_analytics_service import ManagerAnalyticsService
# from app.services.chatbot.enhanced_conversation_service import EnhancedConversationService
# from app.services.chatbot.agent_service import EnhancedAgentService

# logger = logging.getLogger("wms_chatbot.role_based_routes")

# router = APIRouter()

# # Initialize services
# role_based_service = RoleBasedChatbotService()
# manager_analytics_service = ManagerAnalyticsService()
# conversation_service = EnhancedConversationService()
# agent_service = EnhancedAgentService()


# # Dependency to check feature access
# def require_feature_access(feature: ChatbotFeature):
#     """
#     Dependency to check if user has access to a specific feature.
    
#     Args:
#         feature: Required feature for access
        
#     Returns:
#         Dependency function
#     """
#     async def check_access(current_user: Dict[str, Any] = Depends(get_optional_current_user)):
#         user_role = current_user.get("role", "")
        
#         if not await role_based_service.validate_feature_access(user_role, feature):
#             raise HTTPException(
#                 status_code=status.HTTP_403_FORBIDDEN,
#                 detail=f"Access denied: {role_based_service.get_feature_description(feature)}"
#             )
        
#         return current_user
    
#     return check_access


# # Dashboard and Overview Endpoints
# @router.get("/dashboard", response_model=Dict[str, Any])
# async def get_role_based_dashboard(
#     current_user: Dict[str, Any] = Depends(get_optional_current_user)
# ):
#     """
#     Get role-based dashboard data.
    
#     Args:
#         current_user: Current authenticated user
        
#     Returns:
#         Dashboard data tailored to user's role
#     """
#     try:
#         user_id = current_user.get("username", "anonymous")
#         user_role = current_user.get("role", "")
        
#         dashboard_data = await role_based_service.get_role_specific_dashboard_data(user_id, user_role)
        
#         return dashboard_data
        
#     except Exception as e:
#         logger.error(f"Failed to get dashboard data: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to get dashboard data"
#         )


# @router.get("/permissions", response_model=Dict[str, Any])
# async def get_user_permissions(
#     current_user: Dict[str, Any] = Depends(get_optional_current_user)
# ):
#     """
#     Get user's permissions and available features.
    
#     Args:
#         current_user: Current authenticated user
        
#     Returns:
#         User permissions and features
#     """
#     try:
#         user_role = current_user.get("role", "")
#         permissions = role_based_service.get_user_permissions(user_role)
        
#         return permissions
        
#     except Exception as e:
#         logger.error(f"Failed to get user permissions: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to get user permissions"
#         )


# @router.get("/activity-summary", response_model=Dict[str, Any])
# async def get_user_activity_summary(
#     days: int = Query(7, description="Number of days to analyze"),
#     current_user: Dict[str, Any] = Depends(get_optional_current_user)
# ):
#     """
#     Get user's activity summary.
    
#     Args:
#         days: Number of days to analyze
#         current_user: Current authenticated user
        
#     Returns:
#         User activity summary
#     """
#     try:
#         user_id = current_user.get("username", "anonymous")
#         user_role = current_user.get("role", "")
        
#         activity_summary = await role_based_service.get_user_activity_summary(user_id, user_role, days)
        
#         return activity_summary
        
#     except Exception as e:
#         logger.error(f"Failed to get activity summary: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to get activity summary"
#         )


# # Manager-Only Analytics Endpoints
# @router.get("/analytics/system-overview", response_model=Dict[str, Any])
# async def get_system_overview(
#     period_days: int = Query(30, description="Number of days to analyze"),
#     current_user: Dict[str, Any] = Depends(require_feature_access(ChatbotFeature.SYSTEM_ANALYTICS))
# ):
#     """
#     Get comprehensive system overview (Manager only).
    
#     Args:
#         period_days: Number of days to analyze
#         current_user: Current authenticated user (must be Manager)
        
#     Returns:
#         System overview data
#     """
#     try:
#         overview = await manager_analytics_service.get_system_overview(period_days)
#         return overview
        
#     except Exception as e:
#         logger.error(f"Failed to get system overview: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to get system overview"
#         )


# @router.get("/analytics/user/{user_id}", response_model=Dict[str, Any])
# async def get_user_analytics(
#     user_id: str,
#     period_days: int = Query(30, description="Number of days to analyze"),
#     current_user: Dict[str, Any] = Depends(require_feature_access(ChatbotFeature.USER_ANALYTICS))
# ):
#     """
#     Get analytics for a specific user (Manager only).
    
#     Args:
#         user_id: User identifier to analyze
#         period_days: Number of days to analyze
#         current_user: Current authenticated user (must be Manager)
        
#     Returns:
#         User analytics data
#     """
#     try:
#         analytics = await manager_analytics_service.get_user_analytics(user_id, period_days)
#         return analytics
        
#     except Exception as e:
#         logger.error(f"Failed to get user analytics: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to get user analytics"
#         )


# @router.get("/analytics/performance", response_model=Dict[str, Any])
# async def get_performance_metrics(
#     period_days: int = Query(30, description="Number of days to analyze"),
#     current_user: Dict[str, Any] = Depends(require_feature_access(ChatbotFeature.PERFORMANCE_METRICS))
# ):
#     """
#     Get performance metrics (Manager only).
    
#     Args:
#         period_days: Number of days to analyze
#         current_user: Current authenticated user (must be Manager)
        
#     Returns:
#         Performance metrics data
#     """
#     try:
#         comparison = await manager_analytics_service.get_performance_comparison(period_days)
#         return comparison
        
#     except Exception as e:
#         logger.error(f"Failed to get performance metrics: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to get performance metrics"
#         )


# @router.get("/analytics/alerts", response_model=List[Dict[str, Any]])
# async def get_system_alerts(
#     current_user: Dict[str, Any] = Depends(require_feature_access(ChatbotFeature.SYSTEM_ANALYTICS))
# ):
#     """
#     Get system alerts (Manager only).
    
#     Args:
#         current_user: Current authenticated user (must be Manager)
        
#     Returns:
#         List of system alerts
#     """
#     try:
#         alerts = await manager_analytics_service.get_system_alerts()
#         return alerts
        
#     except Exception as e:
#         logger.error(f"Failed to get system alerts: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to get system alerts"
#         )


# @router.get("/analytics/executive-summary", response_model=Dict[str, Any])
# async def get_executive_summary(
#     period_days: int = Query(30, description="Number of days to analyze"),
#     current_user: Dict[str, Any] = Depends(require_feature_access(ChatbotFeature.SYSTEM_ANALYTICS))
# ):
#     """
#     Get executive summary (Manager only).
    
#     Args:
#         period_days: Number of days to analyze
#         current_user: Current authenticated user (must be Manager)
        
#     Returns:
#         Executive summary data
#     """
#     try:
#         summary = await manager_analytics_service.generate_executive_summary(period_days)
#         return summary
        
#     except Exception as e:
#         logger.error(f"Failed to get executive summary: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to get executive summary"
#         )


# # Agent Management Endpoints (Manager only)
# @router.get("/agents/management/overview", response_model=Dict[str, Any])
# async def get_agent_management_overview(
#     current_user: Dict[str, Any] = Depends(require_feature_access(ChatbotFeature.AGENT_MANAGEMENT))
# ):
#     """
#     Get agent management overview (Manager only).
    
#     Args:
#         current_user: Current authenticated user (must be Manager)
        
#     Returns:
#         Agent management overview
#     """
#     try:
#         # Get system status
#         system_status = agent_service.get_system_status()
        
#         # Get all agent capabilities
#         all_capabilities = {}
#         for agent_role in agent_service.get_available_roles():
#             all_capabilities[agent_role] = agent_service.get_agent_capabilities(agent_role)
        
#         # Get all agent performance
#         all_performance = {}
#         for agent_role in agent_service.get_available_roles():
#             all_performance[agent_role] = agent_service.get_agent_performance(agent_role)
        
#         return {
#             "system_status": system_status,
#             "agent_capabilities": all_capabilities,
#             "agent_performance": all_performance,
#             "total_agents": len(agent_service.get_available_roles()),
#             "available_roles": agent_service.get_available_roles()
#         }
        
#     except Exception as e:
#         logger.error(f"Failed to get agent management overview: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to get agent management overview"
#         )


# @router.post("/agents/feedback", response_model=Dict[str, Any])
# async def provide_agent_feedback(
#     data: Dict[str, Any],
#     current_user: Dict[str, Any] = Depends(require_feature_access(ChatbotFeature.AGENT_MANAGEMENT))
# ):
#     """
#     Provide feedback on agent performance (Manager only).
    
#     Args:
#         data: Feedback data
#         current_user: Current authenticated user (must be Manager)
        
#     Returns:
#         Feedback acknowledgment
#     """
#     try:
#         agent_role = data.get("agent_role")
#         positive_feedback = data.get("positive_feedback", True)
#         user_role = current_user.get("role")
        
#         if not agent_role:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Agent role is required"
#             )
        
#         # Update user preferences
#         if user_role:
#             agent_service.update_user_preferences(user_role, agent_role, positive_feedback)
        
#         return {
#             "message": "Feedback received",
#             "agent_role": agent_role,
#             "positive_feedback": positive_feedback,
#             "user_role": user_role
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Failed to process feedback: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to process feedback"
#         )


# # Enhanced Search Endpoints (Available to all roles with semantic search)
# @router.post("/search/smart", response_model=Dict[str, Any])
# async def smart_search_conversations(
#     data: Dict[str, Any],
#     current_user: Dict[str, Any] = Depends(require_feature_access(ChatbotFeature.SEARCH_CONVERSATIONS))
# ):
#     """
#     Smart search through user's conversations.
    
#     Args:
#         data: Search parameters
#         current_user: Current authenticated user
        
#     Returns:
#         Search results
#     """
#     try:
#         user_id = current_user.get("username", "anonymous")
#         query = data.get("query", "").strip()
        
#         if not query:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Search query is required"
#             )
        
#         # Get user's conversations
#         conversations = await conversation_service.get_user_conversations(
#             user_id=user_id,
#             limit=200,
#             offset=0
#         )
        
#         # Implement smart search logic
#         search_results = []
#         query_lower = query.lower()
        
#         for conv in conversations.get("conversations", []):
#             relevance_score = 0
#             matched_context = []
            
#             # Check title match
#             title = conv.get("title", "")
#             if query_lower in title.lower():
#                 relevance_score += 10
#                 matched_context.append(f"Title: {title}")
            
#             # Check last message match
#             last_msg = conv.get("last_message", "")
#             if query_lower in last_msg.lower():
#                 relevance_score += 8
#                 start_idx = max(0, last_msg.lower().find(query_lower) - 20)
#                 end_idx = min(len(last_msg), start_idx + 60)
#                 snippet = last_msg[start_idx:end_idx]
#                 matched_context.append(f"Message: ...{snippet}...")
            
#             # Add if relevant
#             if relevance_score > 0:
#                 search_results.append({
#                     "conversation_id": conv["conversation_id"],
#                     "title": title,
#                     "relevance_score": relevance_score,
#                     "matched_context": matched_context,
#                     "agent_role": conv.get("agent_role"),
#                     "last_activity": conv.get("last_message_at")
#                 })
        
#         # Sort by relevance
#         search_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
#         return {
#             "query": query,
#             "total_results": len(search_results),
#             "results": search_results[:20]
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Failed to perform smart search: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to perform smart search"
#         )


# @router.post("/search/semantic", response_model=Dict[str, Any])
# async def semantic_search_conversations(
#     data: Dict[str, Any],
#     current_user: Dict[str, Any] = Depends(require_feature_access(ChatbotFeature.SEMANTIC_SEARCH))
# ):
#     """
#     Semantic search through conversations (requires semantic search access).
    
#     Args:
#         data: Search parameters
#         current_user: Current authenticated user
        
#     Returns:
#         Semantic search results
#     """
#     try:
#         from app.services.chatbot.advanced_conversation_service import AdvancedConversationService
        
#         # Initialize advanced service
#         advanced_service = AdvancedConversationService()
        
#         user_id = current_user.get("username", "anonymous")
#         query = data.get("query", "")
#         limit = data.get("limit", 20)
#         similarity_threshold = data.get("similarity_threshold", 0.7)
        
#         if not query:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Search query is required"
#             )
        
#         # Perform semantic search
#         results = await advanced_service.semantic_search(
#             user_id=user_id,
#             query=query,
#             limit=limit,
#             similarity_threshold=similarity_threshold
#         )
        
#         return results
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Semantic search failed: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Semantic search failed"
#         )


# # Export and Bulk Operations
# @router.post("/export/conversations", response_model=Dict[str, Any])
# async def export_user_conversations(
#     data: Dict[str, Any],
#     current_user: Dict[str, Any] = Depends(require_feature_access(ChatbotFeature.EXPORT_CONVERSATIONS))
# ):
#     """
#     Export user's conversations.
    
#     Args:
#         data: Export parameters
#         current_user: Current authenticated user
        
#     Returns:
#         Export data
#     """
#     try:
#         from app.services.chatbot.advanced_conversation_service import AdvancedConversationService
        
#         # Initialize advanced service
#         advanced_service = AdvancedConversationService()
        
#         user_id = current_user.get("username", "anonymous")
#         conversation_ids = data.get("conversation_ids")  # None for all
#         format = data.get("format", "json")
#         include_metadata = data.get("include_metadata", True)
#         include_context = data.get("include_context", True)
        
#         # Export conversations
#         export_result = await advanced_service.export_conversations(
#             user_id=user_id,
#             conversation_ids=conversation_ids,
#             format=format,
#             include_metadata=include_metadata,
#             include_context=include_context
#         )
        
#         return export_result
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Failed to export conversations: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to export conversations"
#         )


# @router.post("/bulk/operations", response_model=Dict[str, Any])
# async def bulk_conversation_operations(
#     data: Dict[str, Any],
#     current_user: Dict[str, Any] = Depends(require_feature_access(ChatbotFeature.BULK_OPERATIONS))
# ):
#     """
#     Perform bulk operations on conversations (Manager only).
    
#     Args:
#         data: Bulk operation parameters
#         current_user: Current authenticated user (must be Manager)
        
#     Returns:
#         Operation results
#     """
#     try:
#         from app.services.chatbot.advanced_conversation_service import AdvancedConversationService, ArchiveOperation
        
#         # Initialize advanced service
#         advanced_service = AdvancedConversationService()
        
#         user_id = current_user.get("username", "anonymous")
#         conversation_ids = data.get("conversation_ids", [])
#         operation = data.get("operation", "")
#         operation_data = data.get("operation_data", {})
        
#         if not conversation_ids:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Conversation IDs are required"
#             )
        
#         if not operation:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Operation type is required"
#             )
        
#         # Validate operation type
#         try:
#             operation_enum = ArchiveOperation(operation.lower())
#         except ValueError:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Invalid operation: {operation}"
#             )
        
#         # Perform bulk operation
#         results = await advanced_service.bulk_archive_operations(
#             user_id=user_id,
#             conversation_ids=conversation_ids,
#             operation=operation_enum,
#             operation_data=operation_data
#         )
        
#         return results
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Bulk operation failed: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Bulk operation failed"
#         )


# # Quick Access Endpoints for All Users
# @router.get("/quick/history", response_model=Dict[str, Any])
# async def get_quick_history(
#     limit: int = Query(5, description="Number of recent conversations"),
#     current_user: Dict[str, Any] = Depends(require_feature_access(ChatbotFeature.CONVERSATION_HISTORY))
# ):
#     """
#     Get quick access to recent conversations.
    
#     Args:
#         limit: Number of conversations to return
#         current_user: Current authenticated user
        
#     Returns:
#         Quick history data
#     """
#     try:
#         user_id = current_user.get("username", "anonymous")
        
#         # Get recent conversations
#         conversations = await conversation_service.get_user_conversations(
#             user_id=user_id,
#             limit=limit,
#             offset=0,
#             status="active"
#         )
        
#         quick_history = []
#         for conv in conversations.get("conversations", []):
#             last_msg = conv.get("last_message", "")
#             preview = last_msg[:50] + "..." if len(last_msg) > 50 else last_msg
            
#             quick_history.append({
#                 "conversation_id": conv["conversation_id"],
#                 "title": conv.get("title", preview),
#                 "preview": preview,
#                 "agent_role": conv.get("agent_role", "general"),
#                 "message_count": conv.get("message_count", 0),
#                 "last_activity": conv.get("last_message_at"),
#                 "created_at": conv.get("created_at")
#             })
        
#         return {
#             "user_id": user_id,
#             "conversations": quick_history,
#             "total_available": conversations.get("total", 0),
#             "has_more": conversations.get("total", 0) > limit
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Failed to get quick history: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to get quick history"
#         )


# @router.get("/features", response_model=Dict[str, Any])
# async def get_available_features(
#     current_user: Dict[str, Any] = Depends(get_optional_current_user)
# ):
#     """
#     Get available features for the current user.
    
#     Args:
#         current_user: Current authenticated user
        
#     Returns:
#         Available features and their descriptions
#     """
#     try:
#         user_role = current_user.get("role", "")
#         features = role_based_service.get_user_features(user_role)
        
#         feature_list = []
#         for feature in features:
#             feature_list.append({
#                 "feature": feature.value,
#                 "description": role_based_service.get_feature_description(feature),
#                 "enabled": True
#             })
        
#         return {
#             "user_role": user_role,
#             "total_features": len(feature_list),
#             "features": feature_list
#         }
        
#     except Exception as e:
#         logger.error(f"Failed to get available features: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to get available features"
#         ) 