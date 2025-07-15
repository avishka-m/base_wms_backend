"""
Role-based chatbot service for WMS.
Manages feature access and permissions based on user roles.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

from app.services.chatbot.enhanced_conversation_service import EnhancedConversationService
from app.services.chatbot.agent_service import EnhancedAgentService
from app.services.chatbot.auth_service import get_allowed_chatbot_roles

logger = logging.getLogger("wms_chatbot.role_based_service")


class ChatbotFeature(Enum):
    """Available chatbot features"""
    BASIC_CHAT = "basic_chat"
    CONVERSATION_HISTORY = "conversation_history"
    SEARCH_CONVERSATIONS = "search_conversations"
    EXPORT_CONVERSATIONS = "export_conversations"
    ARCHIVE_CONVERSATIONS = "archive_conversations"
    BULK_OPERATIONS = "bulk_operations"
    
    # Analytics features (Manager only)
    SYSTEM_ANALYTICS = "system_analytics"
    USER_ANALYTICS = "user_analytics"
    PERFORMANCE_METRICS = "performance_metrics"
    AGENT_MANAGEMENT = "agent_management"
    
    # Context features
    CONTEXT_AWARENESS = "context_awareness"
    CONTEXTUAL_SUGGESTIONS = "contextual_suggestions"
    
    # Advanced features
    SEMANTIC_SEARCH = "semantic_search"
    CONVERSATION_INSIGHTS = "conversation_insights"
    TEMPLATES = "templates"
    MIGRATION_TOOLS = "migration_tools"


class RoleBasedChatbotService:
    """
    Service for managing role-based access to chatbot features.
    """
    
    def __init__(self):
        self.conversation_service = EnhancedConversationService()
        self.agent_service = EnhancedAgentService()
        
        # Define role-based feature access
        self.role_features = {
            "Manager": [
                ChatbotFeature.BASIC_CHAT,
                ChatbotFeature.CONVERSATION_HISTORY,
                ChatbotFeature.SEARCH_CONVERSATIONS,
                ChatbotFeature.EXPORT_CONVERSATIONS,
                ChatbotFeature.ARCHIVE_CONVERSATIONS,
                ChatbotFeature.BULK_OPERATIONS,
                ChatbotFeature.SYSTEM_ANALYTICS,
                ChatbotFeature.USER_ANALYTICS,
                ChatbotFeature.PERFORMANCE_METRICS,
                ChatbotFeature.AGENT_MANAGEMENT,
                ChatbotFeature.CONTEXT_AWARENESS,
                ChatbotFeature.CONTEXTUAL_SUGGESTIONS,
                ChatbotFeature.SEMANTIC_SEARCH,
                ChatbotFeature.CONVERSATION_INSIGHTS,
                ChatbotFeature.TEMPLATES,
                ChatbotFeature.MIGRATION_TOOLS
            ],
            "ReceivingClerk": [
                ChatbotFeature.BASIC_CHAT,
                ChatbotFeature.CONVERSATION_HISTORY,
                ChatbotFeature.SEARCH_CONVERSATIONS,
                ChatbotFeature.EXPORT_CONVERSATIONS,
                ChatbotFeature.ARCHIVE_CONVERSATIONS,
                ChatbotFeature.CONTEXT_AWARENESS,
                ChatbotFeature.CONTEXTUAL_SUGGESTIONS,
                ChatbotFeature.SEMANTIC_SEARCH
            ],
            "Picker": [
                ChatbotFeature.BASIC_CHAT,
                ChatbotFeature.CONVERSATION_HISTORY,
                ChatbotFeature.SEARCH_CONVERSATIONS,
                ChatbotFeature.EXPORT_CONVERSATIONS,
                ChatbotFeature.ARCHIVE_CONVERSATIONS,
                ChatbotFeature.CONTEXT_AWARENESS,
                ChatbotFeature.CONTEXTUAL_SUGGESTIONS,
                ChatbotFeature.SEMANTIC_SEARCH
            ],
            "Packer": [
                ChatbotFeature.BASIC_CHAT,
                ChatbotFeature.CONVERSATION_HISTORY,
                ChatbotFeature.SEARCH_CONVERSATIONS,
                ChatbotFeature.EXPORT_CONVERSATIONS,
                ChatbotFeature.ARCHIVE_CONVERSATIONS,
                ChatbotFeature.CONTEXT_AWARENESS,
                ChatbotFeature.CONTEXTUAL_SUGGESTIONS,
                ChatbotFeature.SEMANTIC_SEARCH
            ],
            "Driver": [
                ChatbotFeature.BASIC_CHAT,
                ChatbotFeature.CONVERSATION_HISTORY,
                ChatbotFeature.SEARCH_CONVERSATIONS,
                ChatbotFeature.EXPORT_CONVERSATIONS,
                ChatbotFeature.ARCHIVE_CONVERSATIONS,
                ChatbotFeature.CONTEXT_AWARENESS,
                ChatbotFeature.CONTEXTUAL_SUGGESTIONS,
                ChatbotFeature.SEMANTIC_SEARCH
            ]
        }
    
    def has_feature_access(self, user_role: str, feature: ChatbotFeature) -> bool:
        """
        Check if a user role has access to a specific feature.
        
        Args:
            user_role: User's role in the system
            feature: Feature to check access for
            
        Returns:
            True if user has access, False otherwise
        """
        allowed_features = self.role_features.get(user_role, [])
        return feature in allowed_features
    
    def get_user_features(self, user_role: str) -> List[ChatbotFeature]:
        """
        Get all features available to a user role.
        
        Args:
            user_role: User's role in the system
            
        Returns:
            List of available features
        """
        return self.role_features.get(user_role, [])
    
    def get_user_permissions(self, user_role: str) -> Dict[str, Any]:
        """
        Get comprehensive permissions for a user role.
        
        Args:
            user_role: User's role in the system
            
        Returns:
            Dictionary of permissions and capabilities
        """
        features = self.get_user_features(user_role)
        chatbot_roles = get_allowed_chatbot_roles(user_role)
        
        return {
            "user_role": user_role,
            "available_features": [f.value for f in features],
            "chatbot_roles": chatbot_roles,
            "is_manager": user_role == "Manager",
            "can_access_analytics": self.has_feature_access(user_role, ChatbotFeature.SYSTEM_ANALYTICS),
            "can_manage_agents": self.has_feature_access(user_role, ChatbotFeature.AGENT_MANAGEMENT),
            "can_use_advanced_search": self.has_feature_access(user_role, ChatbotFeature.SEMANTIC_SEARCH),
            "can_export_data": self.has_feature_access(user_role, ChatbotFeature.EXPORT_CONVERSATIONS),
            "can_bulk_operations": self.has_feature_access(user_role, ChatbotFeature.BULK_OPERATIONS)
        }
    
    async def get_role_specific_dashboard_data(self, user_id: str, user_role: str) -> Dict[str, Any]:
        """
        Get dashboard data specific to the user's role.
        
        Args:
            user_id: User identifier
            user_role: User's role in the system
            
        Returns:
            Role-specific dashboard data
        """
        dashboard_data = {
            "user_id": user_id,
            "user_role": user_role,
            "permissions": self.get_user_permissions(user_role),
            "timestamp": datetime.now().isoformat()
        }
        
        # Basic data for all users
        try:
            # Get recent conversations
            conversations = await self.conversation_service.get_user_conversations(
                user_id=user_id,
                limit=10,
                offset=0
            )
            dashboard_data["recent_conversations"] = conversations.get("conversations", [])
            dashboard_data["total_conversations"] = conversations.get("total", 0)
            
            # Get user's chatbot roles
            dashboard_data["available_chatbot_roles"] = get_allowed_chatbot_roles(user_role)
            
        except Exception as e:
            logger.error(f"Error getting basic dashboard data: {str(e)}")
            dashboard_data["recent_conversations"] = []
            dashboard_data["total_conversations"] = 0
        
        # Manager-specific data
        if user_role == "Manager":
            try:
                dashboard_data["manager_data"] = await self._get_manager_dashboard_data(user_id)
            except Exception as e:
                logger.error(f"Error getting manager dashboard data: {str(e)}")
                dashboard_data["manager_data"] = {}
        
        return dashboard_data
    
    async def _get_manager_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """
        Get manager-specific dashboard data.
        
        Args:
            user_id: Manager user identifier
            
        Returns:
            Manager dashboard data
        """
        manager_data = {}
        
        try:
            # System-wide analytics
            manager_data["system_status"] = self.agent_service.get_system_status()
            
            # Agent performance metrics
            manager_data["agent_performance"] = {}
            for agent_role in self.agent_service.get_available_roles():
                manager_data["agent_performance"][agent_role] = self.agent_service.get_agent_performance(agent_role)
            
            # Get insights for the manager
            try:
                from app.services.chatbot.advanced_conversation_service import AdvancedConversationService
                advanced_service = AdvancedConversationService()
                
                insights = await advanced_service.generate_conversation_insights(
                    user_id=user_id,
                    period_days=30
                )
                
                manager_data["insights"] = {
                    "total_conversations": insights.total_conversations,
                    "active_conversations": insights.active_conversations,
                    "most_active_agent": insights.most_active_agent,
                    "user_engagement_score": insights.user_engagement_score,
                    "most_common_topics": insights.most_common_topics
                }
            except Exception as e:
                logger.error(f"Error getting manager insights: {str(e)}")
                manager_data["insights"] = {}
            
        except Exception as e:
            logger.error(f"Error getting manager data: {str(e)}")
        
        return manager_data
    
    async def get_user_activity_summary(self, user_id: str, user_role: str, days: int = 7) -> Dict[str, Any]:
        """
        Get user activity summary for the specified period.
        
        Args:
            user_id: User identifier
            user_role: User's role in the system
            days: Number of days to analyze
            
        Returns:
            User activity summary
        """
        try:
            # Get user conversations from the period
            conversations = await self.conversation_service.get_user_conversations(
                user_id=user_id,
                limit=1000,
                offset=0
            )
            
            # Filter conversations within the date range
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_conversations = []
            
            for conv in conversations.get("conversations", []):
                conv_date = datetime.fromisoformat(conv.get("created_at", datetime.now().isoformat()))
                if conv_date >= cutoff_date:
                    recent_conversations.append(conv)
            
            # Calculate activity metrics
            total_messages = sum(conv.get("message_count", 0) for conv in recent_conversations)
            active_days = len(set(
                datetime.fromisoformat(conv.get("created_at", datetime.now().isoformat())).date()
                for conv in recent_conversations
            ))
            
            # Agent usage breakdown
            agent_usage = {}
            for conv in recent_conversations:
                agent_role = conv.get("agent_role", "general")
                agent_usage[agent_role] = agent_usage.get(agent_role, 0) + 1
            
            return {
                "user_id": user_id,
                "user_role": user_role,
                "period_days": days,
                "activity_summary": {
                    "total_conversations": len(recent_conversations),
                    "total_messages": total_messages,
                    "active_days": active_days,
                    "average_messages_per_conversation": total_messages / len(recent_conversations) if recent_conversations else 0,
                    "agent_usage_breakdown": agent_usage,
                    "most_used_agent": max(agent_usage.items(), key=lambda x: x[1]) if agent_usage else None
                },
                "recent_conversations": recent_conversations[:5]  # Last 5 conversations
            }
            
        except Exception as e:
            logger.error(f"Error getting user activity summary: {str(e)}")
            return {
                "user_id": user_id,
                "user_role": user_role,
                "period_days": days,
                "activity_summary": {},
                "recent_conversations": [],
                "error": str(e)
            }
    
    def get_feature_description(self, feature: ChatbotFeature) -> str:
        """
        Get human-readable description of a feature.
        
        Args:
            feature: Feature to describe
            
        Returns:
            Feature description
        """
        descriptions = {
            ChatbotFeature.BASIC_CHAT: "Basic chat functionality with AI assistants",
            ChatbotFeature.CONVERSATION_HISTORY: "View and manage conversation history",
            ChatbotFeature.SEARCH_CONVERSATIONS: "Search through conversation history",
            ChatbotFeature.EXPORT_CONVERSATIONS: "Export conversations to various formats",
            ChatbotFeature.ARCHIVE_CONVERSATIONS: "Archive old conversations",
            ChatbotFeature.BULK_OPERATIONS: "Perform bulk operations on conversations",
            ChatbotFeature.SYSTEM_ANALYTICS: "View system-wide analytics and metrics",
            ChatbotFeature.USER_ANALYTICS: "View user-specific analytics",
            ChatbotFeature.PERFORMANCE_METRICS: "View agent performance metrics",
            ChatbotFeature.AGENT_MANAGEMENT: "Manage AI agents and their capabilities",
            ChatbotFeature.CONTEXT_AWARENESS: "Context-aware conversations",
            ChatbotFeature.CONTEXTUAL_SUGGESTIONS: "Get contextual suggestions",
            ChatbotFeature.SEMANTIC_SEARCH: "Advanced semantic search capabilities",
            ChatbotFeature.CONVERSATION_INSIGHTS: "Generate insights from conversations",
            ChatbotFeature.TEMPLATES: "Use and manage conversation templates",
            ChatbotFeature.MIGRATION_TOOLS: "System migration and management tools"
        }
        
        return descriptions.get(feature, "Unknown feature")
    
    async def validate_feature_access(self, user_role: str, feature: ChatbotFeature) -> bool:
        """
        Validate if a user has access to a specific feature.
        
        Args:
            user_role: User's role in the system
            feature: Feature to validate access for
            
        Returns:
            True if access is granted, False otherwise
        """
        has_access = self.has_feature_access(user_role, feature)
        
        if not has_access:
            logger.warning(f"User with role '{user_role}' attempted to access restricted feature '{feature.value}'")
        
        return has_access 