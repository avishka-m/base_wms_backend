"""
Manager Analytics Service for WMS Chatbot.
Provides comprehensive analytics and performance metrics for managers.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import asyncio

from app.services.chatbot.enhanced_conversation_service import EnhancedConversationService
from app.services.chatbot.agent_service import EnhancedAgentService

logger = logging.getLogger("wms_chatbot.manager_analytics")


class ManagerAnalyticsService:
    """
    Service for providing manager-level analytics and insights.
    """
    
    def __init__(self):
        self.conversation_service = EnhancedConversationService()
        self.agent_service = EnhancedAgentService()
    
    async def get_system_overview(self, period_days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive system overview for managers.
        
        Args:
            period_days: Number of days to analyze
            
        Returns:
            System overview data
        """
        try:
            # Get system status
            system_status = self.agent_service.get_system_status()
            
            # Get agent performance metrics
            agent_metrics = await self._get_agent_performance_metrics()
            
            # Get usage statistics
            usage_stats = await self._get_usage_statistics(period_days)
            
            # Get user engagement metrics
            engagement_metrics = await self._get_user_engagement_metrics(period_days)
            
            # Get trending topics
            trending_topics = await self._get_trending_topics(period_days)
            
            return {
                "system_status": system_status,
                "agent_metrics": agent_metrics,
                "usage_statistics": usage_stats,
                "engagement_metrics": engagement_metrics,
                "trending_topics": trending_topics,
                "period_days": period_days,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system overview: {str(e)}")
            return {
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }
    
    async def _get_agent_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed agent performance metrics.
        
        Returns:
            Agent performance data
        """
        try:
            agent_metrics = {}
            available_roles = self.agent_service.get_available_roles()
            
            for role in available_roles:
                performance = self.agent_service.get_agent_performance(role)
                capabilities = self.agent_service.get_agent_capabilities(role)
                
                agent_metrics[role] = {
                    "performance": performance,
                    "capabilities": capabilities,
                    "total_interactions": performance.get("total_interactions", 0),
                    "success_rate": performance.get("success_rate", 0),
                    "average_response_time": performance.get("average_response_time", 0),
                    "user_satisfaction": performance.get("user_satisfaction", 0)
                }
            
            # Calculate overall metrics
            total_interactions = sum(metrics["total_interactions"] for metrics in agent_metrics.values())
            avg_success_rate = sum(metrics["success_rate"] for metrics in agent_metrics.values()) / len(agent_metrics) if agent_metrics else 0
            
            return {
                "individual_agents": agent_metrics,
                "overall_metrics": {
                    "total_interactions": total_interactions,
                    "average_success_rate": avg_success_rate,
                    "active_agents": len(agent_metrics),
                    "best_performing_agent": max(agent_metrics.items(), key=lambda x: x[1]["success_rate"]) if agent_metrics else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting agent performance metrics: {str(e)}")
            return {"error": str(e)}
    
    async def _get_usage_statistics(self, period_days: int) -> Dict[str, Any]:
        """
        Get usage statistics for the specified period.
        
        Args:
            period_days: Number of days to analyze
            
        Returns:
            Usage statistics
        """
        try:
            # This would typically involve querying the database
            # For now, we'll return mock data with the structure
            
            cutoff_date = datetime.now() - timedelta(days=period_days)
            
            # Mock data structure - in real implementation, query the database
            usage_stats = {
                "total_conversations": 0,
                "total_messages": 0,
                "active_users": 0,
                "conversations_per_day": [],
                "messages_per_day": [],
                "peak_usage_hours": {},
                "user_role_breakdown": {},
                "agent_usage_breakdown": {},
                "average_conversation_length": 0,
                "most_active_day": None,
                "growth_rate": 0  # Percentage change from previous period
            }
            
            # In real implementation, this would query the conversation database
            logger.info(f"Generating usage statistics for {period_days} days")
            
            return usage_stats
            
        except Exception as e:
            logger.error(f"Error getting usage statistics: {str(e)}")
            return {"error": str(e)}
    
    async def _get_user_engagement_metrics(self, period_days: int) -> Dict[str, Any]:
        """
        Get user engagement metrics.
        
        Args:
            period_days: Number of days to analyze
            
        Returns:
            User engagement metrics
        """
        try:
            engagement_metrics = {
                "total_active_users": 0,
                "new_users": 0,
                "returning_users": 0,
                "user_retention_rate": 0,
                "average_session_duration": 0,
                "messages_per_user": 0,
                "engagement_score": 0,
                "user_satisfaction_score": 0,
                "feature_adoption_rates": {},
                "drop_off_points": [],
                "user_journey_metrics": {}
            }
            
            # In real implementation, calculate these metrics from database
            logger.info(f"Generating engagement metrics for {period_days} days")
            
            return engagement_metrics
            
        except Exception as e:
            logger.error(f"Error getting user engagement metrics: {str(e)}")
            return {"error": str(e)}
    
    async def _get_trending_topics(self, period_days: int) -> List[Dict[str, Any]]:
        """
        Get trending topics from conversations.
        
        Args:
            period_days: Number of days to analyze
            
        Returns:
            List of trending topics
        """
        try:
            # Mock trending topics - in real implementation, use NLP on conversation content
            trending_topics = [
                {
                    "topic": "inventory_management",
                    "mentions": 45,
                    "growth_rate": 23.5,
                    "related_keywords": ["stock", "inventory", "items", "warehouse"],
                    "sentiment": "positive"
                },
                {
                    "topic": "order_processing",
                    "mentions": 38,
                    "growth_rate": 15.2,
                    "related_keywords": ["orders", "shipping", "fulfillment"],
                    "sentiment": "neutral"
                },
                {
                    "topic": "system_issues",
                    "mentions": 12,
                    "growth_rate": -8.3,
                    "related_keywords": ["error", "problem", "issue", "bug"],
                    "sentiment": "negative"
                }
            ]
            
            return trending_topics
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {str(e)}")
            return []
    
    async def get_user_analytics(self, user_id: str, period_days: int = 30) -> Dict[str, Any]:
        """
        Get detailed analytics for a specific user.
        
        Args:
            user_id: User identifier
            period_days: Number of days to analyze
            
        Returns:
            User-specific analytics
        """
        try:
            # Get user's conversations
            conversations = await self.conversation_service.get_user_conversations(
                user_id=user_id,
                limit=1000,
                offset=0
            )
            
            # Filter conversations for the period
            cutoff_date = datetime.now() - timedelta(days=period_days)
            recent_conversations = []
            
            for conv in conversations.get("conversations", []):
                conv_date = datetime.fromisoformat(conv.get("created_at", datetime.now().isoformat()))
                if conv_date >= cutoff_date:
                    recent_conversations.append(conv)
            
            # Calculate metrics
            total_messages = sum(conv.get("message_count", 0) for conv in recent_conversations)
            agent_usage = Counter(conv.get("agent_role", "general") for conv in recent_conversations)
            
            # Calculate engagement score
            engagement_score = min(100, (len(recent_conversations) * 10 + total_messages) / period_days)
            
            return {
                "user_id": user_id,
                "period_days": period_days,
                "conversation_count": len(recent_conversations),
                "total_messages": total_messages,
                "average_messages_per_conversation": total_messages / len(recent_conversations) if recent_conversations else 0,
                "agent_usage_breakdown": dict(agent_usage),
                "most_used_agent": agent_usage.most_common(1)[0] if agent_usage else None,
                "engagement_score": round(engagement_score, 2),
                "activity_trend": self._calculate_activity_trend(recent_conversations),
                "conversation_patterns": self._analyze_conversation_patterns(recent_conversations)
            }
            
        except Exception as e:
            logger.error(f"Error getting user analytics: {str(e)}")
            return {"error": str(e), "user_id": user_id}
    
    def _calculate_activity_trend(self, conversations: List[Dict[str, Any]]) -> str:
        """
        Calculate activity trend for a user.
        
        Args:
            conversations: List of user conversations
            
        Returns:
            Activity trend description
        """
        if not conversations:
            return "no_activity"
        
        # Group conversations by day
        daily_activity = defaultdict(int)
        for conv in conversations:
            conv_date = datetime.fromisoformat(conv.get("created_at", datetime.now().isoformat())).date()
            daily_activity[conv_date] += 1
        
        # Calculate trend
        if len(daily_activity) < 2:
            return "insufficient_data"
        
        sorted_days = sorted(daily_activity.keys())
        recent_half = sorted_days[len(sorted_days)//2:]
        earlier_half = sorted_days[:len(sorted_days)//2]
        
        recent_avg = sum(daily_activity[day] for day in recent_half) / len(recent_half)
        earlier_avg = sum(daily_activity[day] for day in earlier_half) / len(earlier_half)
        
        if recent_avg > earlier_avg * 1.2:
            return "increasing"
        elif recent_avg < earlier_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _analyze_conversation_patterns(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze conversation patterns.
        
        Args:
            conversations: List of user conversations
            
        Returns:
            Conversation pattern analysis
        """
        if not conversations:
            return {}
        
        # Analyze conversation lengths
        lengths = [conv.get("message_count", 0) for conv in conversations]
        
        # Analyze time patterns
        hours = []
        for conv in conversations:
            conv_time = datetime.fromisoformat(conv.get("created_at", datetime.now().isoformat()))
            hours.append(conv_time.hour)
        
        hour_distribution = Counter(hours)
        
        return {
            "average_length": sum(lengths) / len(lengths),
            "length_distribution": {
                "short": len([l for l in lengths if l <= 3]),
                "medium": len([l for l in lengths if 4 <= l <= 10]),
                "long": len([l for l in lengths if l > 10])
            },
            "peak_hours": hour_distribution.most_common(3),
            "conversation_frequency": len(conversations) / 30  # conversations per day
        }
    
    async def get_performance_comparison(self, period_days: int = 30) -> Dict[str, Any]:
        """
        Get performance comparison between different metrics.
        
        Args:
            period_days: Number of days to analyze
            
        Returns:
            Performance comparison data
        """
        try:
            # Get current period data
            current_period = await self._get_usage_statistics(period_days)
            
            # Get previous period data for comparison
            previous_period = await self._get_usage_statistics(period_days)  # Mock - would get previous period
            
            # Calculate comparisons
            comparison = {}
            
            metrics_to_compare = [
                "total_conversations",
                "total_messages",
                "active_users",
                "average_conversation_length"
            ]
            
            for metric in metrics_to_compare:
                current_value = current_period.get(metric, 0)
                previous_value = previous_period.get(metric, 0)
                
                if previous_value > 0:
                    change_percent = ((current_value - previous_value) / previous_value) * 100
                else:
                    change_percent = 0
                
                comparison[metric] = {
                    "current": current_value,
                    "previous": previous_value,
                    "change_percent": round(change_percent, 2),
                    "trend": "up" if change_percent > 0 else "down" if change_percent < 0 else "stable"
                }
            
            return {
                "period_days": period_days,
                "comparison": comparison,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance comparison: {str(e)}")
            return {"error": str(e)}
    
    async def get_system_alerts(self) -> List[Dict[str, Any]]:
        """
        Get system alerts and notifications for managers.
        
        Returns:
            List of system alerts
        """
        try:
            alerts = []
            
            # Check system status
            system_status = self.agent_service.get_system_status()
            
            # Check for performance issues
            if system_status.get("status") != "healthy":
                alerts.append({
                    "type": "system_health",
                    "severity": "warning",
                    "message": "System health check indicates potential issues",
                    "timestamp": datetime.now().isoformat(),
                    "action_required": True
                })
            
            # Check agent performance
            agent_metrics = await self._get_agent_performance_metrics()
            
            for agent_role, metrics in agent_metrics.get("individual_agents", {}).items():
                if metrics.get("success_rate", 0) < 0.8:  # 80% threshold
                    alerts.append({
                        "type": "agent_performance",
                        "severity": "warning",
                        "message": f"Agent '{agent_role}' has low success rate: {metrics.get('success_rate', 0):.2%}",
                        "timestamp": datetime.now().isoformat(),
                        "action_required": True,
                        "agent_role": agent_role
                    })
            
            # Check for unusual activity patterns
            # This would typically involve analyzing recent data
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting system alerts: {str(e)}")
            return []
    
    async def generate_executive_summary(self, period_days: int = 30) -> Dict[str, Any]:
        """
        Generate executive summary for managers.
        
        Args:
            period_days: Number of days to analyze
            
        Returns:
            Executive summary data
        """
        try:
            # Get all necessary data
            system_overview = await self.get_system_overview(period_days)
            performance_comparison = await self.get_performance_comparison(period_days)
            alerts = await self.get_system_alerts()
            
            # Generate key insights
            key_insights = []
            
            # Usage insights
            usage_stats = system_overview.get("usage_statistics", {})
            if usage_stats.get("total_conversations", 0) > 0:
                key_insights.append({
                    "category": "usage",
                    "insight": f"Total of {usage_stats.get('total_conversations', 0)} conversations in the last {period_days} days",
                    "impact": "positive" if usage_stats.get("growth_rate", 0) > 0 else "neutral"
                })
            
            # Agent performance insights
            agent_metrics = system_overview.get("agent_metrics", {})
            overall_metrics = agent_metrics.get("overall_metrics", {})
            
            if overall_metrics.get("average_success_rate", 0) > 0.9:
                key_insights.append({
                    "category": "performance",
                    "insight": f"High agent success rate of {overall_metrics.get('average_success_rate', 0):.2%}",
                    "impact": "positive"
                })
            
            # Trending topics insights
            trending_topics = system_overview.get("trending_topics", [])
            if trending_topics:
                top_topic = trending_topics[0]
                key_insights.append({
                    "category": "trends",
                    "insight": f"'{top_topic['topic']}' is the most discussed topic with {top_topic['mentions']} mentions",
                    "impact": "neutral"
                })
            
            return {
                "period_days": period_days,
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "total_conversations": usage_stats.get("total_conversations", 0),
                    "active_users": usage_stats.get("active_users", 0),
                    "system_health": system_overview.get("system_status", {}).get("status", "unknown"),
                    "agent_performance": overall_metrics.get("average_success_rate", 0),
                    "critical_alerts": len([a for a in alerts if a.get("severity") == "critical"]),
                    "warning_alerts": len([a for a in alerts if a.get("severity") == "warning"])
                },
                "key_insights": key_insights,
                "recommendations": self._generate_recommendations(system_overview, alerts),
                "next_actions": self._generate_next_actions(alerts)
            }
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, system_overview: Dict[str, Any], alerts: List[Dict[str, Any]]) -> List[str]:
        """
        Generate recommendations based on system data.
        
        Args:
            system_overview: System overview data
            alerts: List of system alerts
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check for performance issues
        if any(alert.get("type") == "agent_performance" for alert in alerts):
            recommendations.append("Review and optimize underperforming agents")
        
        # Check usage patterns
        usage_stats = system_overview.get("usage_statistics", {})
        if usage_stats.get("growth_rate", 0) < 0:
            recommendations.append("Investigate declining usage and implement user engagement strategies")
        
        # Default recommendations
        recommendations.extend([
            "Regular monitoring of agent performance metrics",
            "User feedback collection and analysis",
            "System capacity planning for peak usage periods"
        ])
        
        return recommendations
    
    def _generate_next_actions(self, alerts: List[Dict[str, Any]]) -> List[str]:
        """
        Generate next actions based on alerts.
        
        Args:
            alerts: List of system alerts
            
        Returns:
            List of next actions
        """
        next_actions = []
        
        # Priority actions based on alerts
        critical_alerts = [a for a in alerts if a.get("severity") == "critical"]
        warning_alerts = [a for a in alerts if a.get("severity") == "warning"]
        
        if critical_alerts:
            next_actions.append("Address critical system issues immediately")
        
        if warning_alerts:
            next_actions.append("Review and resolve warning alerts")
        
        # Standard actions
        next_actions.extend([
            "Review weekly performance metrics",
            "Update agent training data if needed",
            "Plan capacity adjustments for next month"
        ])
        
        return next_actions 