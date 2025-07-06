"""
Advanced Conversation Management Service with sophisticated search, archive, and context features.
Extends the enhanced conversation service with AI-powered capabilities.
"""

import logging
import json
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
import re
from pymongo import ASCENDING, DESCENDING, TEXT
import numpy as np
import pickle
import os

# Try to import sentence_transformers, but handle gracefully if not installed
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from app.utils.database import get_async_collection
from app.models.chatbot.enhanced_chat_models import (
    ChatConversation,
    ChatMessage,
    ChatAuditLog,
    ChatConversationStatus,
    ChatMessageType,
    ConversationSearchRequest
)
from app.services.chatbot.enhanced_conversation_service import EnhancedConversationService

logger = logging.getLogger("wms_chatbot.advanced_conversation_service")


class SearchResultType(Enum):
    """Types of search results."""
    CONVERSATION = "conversation"
    MESSAGE = "message"
    CONTEXT = "context"
    ATTACHMENT = "attachment"


class ArchiveOperation(Enum):
    """Archive operation types."""
    ARCHIVE = "archive"
    UNARCHIVE = "unarchive"
    DELETE = "delete"
    TAG = "tag"
    UNTAG = "untag"
    MOVE = "move"
    EXPORT = "export"


@dataclass
class ConversationCluster:
    """Represents a cluster of similar conversations."""
    cluster_id: str
    conversations: List[str]
    topic: str
    keywords: List[str]
    similarity_score: float
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConversationTemplate:
    """Template for creating conversations."""
    template_id: str
    name: str
    description: str
    agent_role: str
    initial_context: Dict[str, Any]
    suggested_prompts: List[str]
    tags: List[str]
    category: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SearchAnalytics:
    """Analytics for search operations."""
    query: str
    user_id: str
    search_type: str
    result_count: int
    selected_result: Optional[str]
    response_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConversationInsights:
    """Insights about conversations."""
    total_conversations: int
    active_conversations: int
    archived_conversations: int
    deleted_conversations: int
    average_messages_per_conversation: float
    most_active_agent: str
    most_common_topics: List[Tuple[str, int]]
    conversation_trends: Dict[str, List[int]]  # Daily, weekly, monthly trends
    user_engagement_score: float
    context_usage_patterns: Dict[str, Any]


class AdvancedConversationService(EnhancedConversationService):
    """Advanced conversation management service with AI-powered features."""
    
    def __init__(self):
        """Initialize the advanced conversation service."""
        super().__init__()
        self._embeddings_model = None
        self._embeddings_cache = {}
        self._search_analytics = []
        self._conversation_clusters = {}
        self._conversation_templates = {}
        self._context_patterns = {}
        self._advanced_initialized = False
    
    async def _ensure_advanced_collections_and_indexes(self):
        """Ensure advanced collections and indexes are set up."""
        if self._advanced_initialized:
            return
            
        await self._ensure_collections_and_indexes()  # Base collections
        
        try:
            # Get advanced collections
            embeddings_col = await get_async_collection("chat_embeddings")
            clusters_col = await get_async_collection("chat_clusters")
            templates_col = await get_async_collection("chat_templates")
            search_analytics_col = await get_async_collection("chat_search_analytics")
            insights_col = await get_async_collection("chat_insights")
            context_patterns_col = await get_async_collection("chat_context_patterns")
            
            # Create indexes for embeddings
            await embeddings_col.create_index([("conversation_id", ASCENDING)])
            await embeddings_col.create_index([("message_id", ASCENDING)])
            await embeddings_col.create_index([("user_id", ASCENDING)])
            await embeddings_col.create_index([("content_type", ASCENDING)])
            
            # Create indexes for clusters
            await clusters_col.create_index([("user_id", ASCENDING)])
            await clusters_col.create_index([("topic", ASCENDING)])
            await clusters_col.create_index([("created_at", DESCENDING)])
            
            # Create indexes for templates
            await templates_col.create_index([("template_id", ASCENDING)], unique=True)
            await templates_col.create_index([("category", ASCENDING)])
            await templates_col.create_index([("agent_role", ASCENDING)])
            await templates_col.create_index([("tags", ASCENDING)])
            
            # Create indexes for search analytics
            await search_analytics_col.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
            await search_analytics_col.create_index([("search_type", ASCENDING)])
            
            # Create indexes for insights
            await insights_col.create_index([("user_id", ASCENDING), ("period", ASCENDING)])
            await insights_col.create_index([("generated_at", DESCENDING)])
            
            # Create indexes for context patterns
            await context_patterns_col.create_index([("pattern_id", ASCENDING)], unique=True)
            await context_patterns_col.create_index([("user_id", ASCENDING)])
            await context_patterns_col.create_index([("context_type", ASCENDING)])
            
            self._advanced_initialized = True
            logger.info("Advanced conversation service collections and indexes initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced collections: {str(e)}")
            raise
    
    def _get_embeddings_model(self):
        """Get or initialize the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available, semantic search will use fallback")
            return None
            
        if self._embeddings_model is None:
            try:
                # Use a lightweight model for embeddings
                self._embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized sentence transformer model")
            except Exception as e:
                logger.error(f"Failed to initialize embeddings model: {str(e)}")
                # Fallback to None - semantic search will be disabled
                self._embeddings_model = None
        return self._embeddings_model
    
    async def _generate_embeddings(self, text: str, content_type: str = "message") -> Optional[List[float]]:
        """Generate embeddings for text content."""
        try:
            model = self._get_embeddings_model()
            if model is None:
                return None
            
            # Create a cache key
            cache_key = hashlib.md5(f"{text}:{content_type}".encode()).hexdigest()
            
            if cache_key in self._embeddings_cache:
                return self._embeddings_cache[cache_key]
            
            # Generate embedding
            embedding = model.encode(text).tolist()
            
            # Cache the embedding (limit cache size)
            if len(self._embeddings_cache) > 1000:
                # Remove oldest entries
                keys_to_remove = list(self._embeddings_cache.keys())[:100]
                for key in keys_to_remove:
                    del self._embeddings_cache[key]
            
            self._embeddings_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            return None
    
    async def _store_embeddings(self, conversation_id: str, message_id: str, user_id: str, 
                              content: str, content_type: str, embedding: List[float]):
        """Store embeddings in the database."""
        try:
            embeddings_col = await get_async_collection("chat_embeddings")
            
            embedding_doc = {
                "conversation_id": conversation_id,
                "message_id": message_id,
                "user_id": user_id,
                "content": content,
                "content_type": content_type,
                "embedding": embedding,
                "created_at": datetime.utcnow()
            }
            
            await embeddings_col.insert_one(embedding_doc)
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
    
    async def semantic_search(
        self,
        user_id: str,
        query: str,
        limit: int = 20,
        similarity_threshold: float = 0.7,
        content_types: Optional[List[str]] = None,
        agent_roles: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Perform semantic search across conversations using embeddings.
        
        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            content_types: Types of content to search
            agent_roles: Agent roles to filter by
            date_range: Date range for filtering
            
        Returns:
            Search results with similarity scores
        """
        await self._ensure_advanced_collections_and_indexes()
        
        start_time = datetime.utcnow()
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_embeddings(query, "query")
            if query_embedding is None:
                # Fallback to regular text search
                return await self._fallback_text_search(user_id, query, limit)
            
            # Get embeddings from database
            embeddings_col = await get_async_collection("chat_embeddings")
            
            # Build filter query
            filter_query = {"user_id": user_id}
            
            if content_types:
                filter_query["content_type"] = {"$in": content_types}
            
            if date_range:
                filter_query["created_at"] = {
                    "$gte": date_range[0],
                    "$lte": date_range[1]
                }
            
            # Get all embeddings for the user
            embeddings = []
            async for doc in embeddings_col.find(filter_query):
                embeddings.append(doc)
            
            # Calculate similarities
            results = []
            for embedding_doc in embeddings:
                stored_embedding = embedding_doc["embedding"]
                
                # Calculate cosine similarity
                similarity = self._calculate_cosine_similarity(query_embedding, stored_embedding)
                
                if similarity >= similarity_threshold:
                    results.append({
                        "conversation_id": embedding_doc["conversation_id"],
                        "message_id": embedding_doc["message_id"],
                        "content": embedding_doc["content"],
                        "content_type": embedding_doc["content_type"],
                        "similarity_score": similarity,
                        "created_at": embedding_doc["created_at"]
                    })
            
            # Sort by similarity score
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            results = results[:limit]
            
            # Enrich results with conversation metadata
            enriched_results = await self._enrich_search_results(results, agent_roles)
            
            # Log search analytics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            await self._log_search_analytics(user_id, query, "semantic", len(enriched_results), processing_time)
            
            return {
                "results": enriched_results,
                "total": len(enriched_results),
                "search_type": "semantic",
                "query": query,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            # Fallback to regular text search
            return await self._fallback_text_search(user_id, query, limit)
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm_a = np.linalg.norm(vec1_np)
            norm_b = np.linalg.norm(vec2_np)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    async def _enrich_search_results(self, results: List[Dict], agent_roles: Optional[List[str]] = None) -> List[Dict]:
        """Enrich search results with conversation metadata."""
        try:
            conversations_col = await get_async_collection("chat_conversations")
            
            enriched_results = []
            conversation_cache = {}
            
            for result in results:
                conversation_id = result["conversation_id"]
                
                # Get conversation metadata from cache or database
                if conversation_id not in conversation_cache:
                    conv_doc = await conversations_col.find_one({"conversation_id": conversation_id})
                    if conv_doc:
                        conversation_cache[conversation_id] = conv_doc
                
                if conversation_id in conversation_cache:
                    conv_data = conversation_cache[conversation_id]
                    
                    # Filter by agent roles if specified
                    if agent_roles and conv_data.get("agent_role") not in agent_roles:
                        continue
                    
                    enriched_result = {
                        **result,
                        "conversation_title": conv_data.get("title", "Untitled"),
                        "agent_role": conv_data.get("agent_role", "unknown"),
                        "conversation_status": conv_data.get("status", "active"),
                        "conversation_created_at": conv_data.get("created_at"),
                        "conversation_tags": conv_data.get("tags", [])
                    }
                    
                    enriched_results.append(enriched_result)
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Failed to enrich search results: {str(e)}")
            return results
    
    async def _fallback_text_search(self, user_id: str, query: str, limit: int) -> Dict[str, Any]:
        """Fallback text search when semantic search is unavailable."""
        try:
            # Use existing text search from parent class
            search_request = ConversationSearchRequest(
                query=query,
                limit=limit,
                offset=0
            )
            
            return await self.search_conversations(user_id, search_request)
            
        except Exception as e:
            logger.error(f"Fallback text search failed: {str(e)}")
            return {"results": [], "total": 0, "search_type": "fallback", "query": query}
    
    async def _log_search_analytics(self, user_id: str, query: str, search_type: str, 
                                  result_count: int, processing_time: float):
        """Log search analytics for performance monitoring."""
        try:
            search_analytics_col = await get_async_collection("chat_search_analytics")
            
            analytics_doc = {
                "user_id": user_id,
                "query": query,
                "search_type": search_type,
                "result_count": result_count,
                "processing_time": processing_time,
                "timestamp": datetime.utcnow()
            }
            
            await search_analytics_col.insert_one(analytics_doc)
            
        except Exception as e:
            logger.error(f"Failed to log search analytics: {str(e)}")
    
    async def bulk_archive_operations(
        self,
        user_id: str,
        conversation_ids: List[str],
        operation: ArchiveOperation,
        operation_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform bulk archive operations on conversations.
        
        Args:
            user_id: User identifier
            conversation_ids: List of conversation IDs
            operation: Operation to perform
            operation_data: Additional data for the operation
            
        Returns:
            Operation results
        """
        await self._ensure_advanced_collections_and_indexes()
        
        try:
            conversations_col = await get_async_collection("chat_conversations")
            
            successful_operations = []
            failed_operations = []
            
            for conversation_id in conversation_ids:
                try:
                    if operation == ArchiveOperation.ARCHIVE:
                        result = await conversations_col.update_one(
                            {"conversation_id": conversation_id, "user_id": user_id},
                            {
                                "$set": {
                                    "status": ChatConversationStatus.ARCHIVED,
                                    "archived_at": datetime.utcnow(),
                                    "updated_at": datetime.utcnow()
                                }
                            }
                        )
                        
                    elif operation == ArchiveOperation.UNARCHIVE:
                        result = await conversations_col.update_one(
                            {"conversation_id": conversation_id, "user_id": user_id},
                            {
                                "$set": {
                                    "status": ChatConversationStatus.ACTIVE,
                                    "updated_at": datetime.utcnow()
                                },
                                "$unset": {"archived_at": ""}
                            }
                        )
                        
                    elif operation == ArchiveOperation.DELETE:
                        hard_delete = operation_data.get("hard_delete", False) if operation_data else False
                        
                        if hard_delete:
                            result = await conversations_col.delete_one({
                                "conversation_id": conversation_id,
                                "user_id": user_id
                            })
                            
                            if result.deleted_count > 0:
                                # Delete associated messages and embeddings
                                await self._hard_delete_conversation_data(conversation_id, user_id)
                        else:
                            result = await conversations_col.update_one(
                                {"conversation_id": conversation_id, "user_id": user_id},
                                {
                                    "$set": {
                                        "status": ChatConversationStatus.DELETED,
                                        "deleted_at": datetime.utcnow(),
                                        "updated_at": datetime.utcnow()
                                    }
                                }
                            )
                    
                    elif operation == ArchiveOperation.TAG:
                        tags = operation_data.get("tags", []) if operation_data else []
                        result = await conversations_col.update_one(
                            {"conversation_id": conversation_id, "user_id": user_id},
                            {
                                "$addToSet": {"tags": {"$each": tags}},
                                "$set": {"updated_at": datetime.utcnow()}
                            }
                        )
                        
                    elif operation == ArchiveOperation.UNTAG:
                        tags = operation_data.get("tags", []) if operation_data else []
                        result = await conversations_col.update_one(
                            {"conversation_id": conversation_id, "user_id": user_id},
                            {
                                "$pull": {"tags": {"$in": tags}},
                                "$set": {"updated_at": datetime.utcnow()}
                            }
                        )
                    
                    else:
                        failed_operations.append({
                            "conversation_id": conversation_id,
                            "error": f"Unsupported operation: {operation}"
                        })
                        continue
                    
                    if hasattr(result, 'modified_count') and result.modified_count > 0:
                        successful_operations.append(conversation_id)
                    elif hasattr(result, 'deleted_count') and result.deleted_count > 0:
                        successful_operations.append(conversation_id)
                    else:
                        failed_operations.append({
                            "conversation_id": conversation_id,
                            "error": "No documents modified"
                        })
                        
                except Exception as e:
                    failed_operations.append({
                        "conversation_id": conversation_id,
                        "error": str(e)
                    })
            
            # Log audit event for bulk operation
            await self._log_audit_event(
                user_id=user_id,
                action_type=f"bulk_{operation.value}",
                resource_type="conversation",
                resource_id="bulk_operation",
                action_details={
                    "operation": operation.value,
                    "total_conversations": len(conversation_ids),
                    "successful": len(successful_operations),
                    "failed": len(failed_operations),
                    "operation_data": operation_data
                }
            )
            
            return {
                "operation": operation.value,
                "total_conversations": len(conversation_ids),
                "successful": successful_operations,
                "failed": failed_operations,
                "success_rate": len(successful_operations) / len(conversation_ids) if conversation_ids else 0
            }
            
        except Exception as e:
            logger.error(f"Bulk archive operation failed: {str(e)}")
            raise
    
    async def _hard_delete_conversation_data(self, conversation_id: str, user_id: str):
        """Hard delete all data associated with a conversation."""
        try:
            # Delete messages
            messages_col = await get_async_collection("chat_messages")
            await messages_col.delete_many({
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            
            # Delete attachments
            attachments_col = await get_async_collection("chat_attachments")
            await attachments_col.delete_many({
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            
            # Delete embeddings
            embeddings_col = await get_async_collection("chat_embeddings")
            await embeddings_col.delete_many({
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            
        except Exception as e:
            logger.error(f"Failed to hard delete conversation data: {str(e)}")
    
    async def create_conversation_template(
        self,
        template_data: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Create a conversation template for reuse.
        
        Args:
            template_data: Template data
            user_id: User creating the template
            
        Returns:
            Created template information
        """
        await self._ensure_advanced_collections_and_indexes()
        
        try:
            template_id = f"template_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
            
            template = ConversationTemplate(
                template_id=template_id,
                name=template_data.get("name", "Untitled Template"),
                description=template_data.get("description", ""),
                agent_role=template_data.get("agent_role", "general"),
                initial_context=template_data.get("initial_context", {}),
                suggested_prompts=template_data.get("suggested_prompts", []),
                tags=template_data.get("tags", []),
                category=template_data.get("category", "general")
            )
            
            templates_col = await get_async_collection("chat_templates")
            
            template_doc = {
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "agent_role": template.agent_role,
                "initial_context": template.initial_context,
                "suggested_prompts": template.suggested_prompts,
                "tags": template.tags,
                "category": template.category,
                "created_by": user_id,
                "created_at": template.created_at,
                "usage_count": 0
            }
            
            await templates_col.insert_one(template_doc)
            
            logger.info(f"Created conversation template {template_id}")
            
            return {
                "template_id": template_id,
                "name": template.name,
                "description": template.description,
                "created_at": template.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create conversation template: {str(e)}")
            raise
    
    async def get_conversation_templates(
        self,
        user_id: str,
        category: Optional[str] = None,
        agent_role: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get conversation templates.
        
        Args:
            user_id: User identifier
            category: Filter by category
            agent_role: Filter by agent role
            limit: Maximum number of templates
            
        Returns:
            List of templates
        """
        await self._ensure_advanced_collections_and_indexes()
        
        try:
            templates_col = await get_async_collection("chat_templates")
            
            # Build query
            query = {}
            if category:
                query["category"] = category
            if agent_role:
                query["agent_role"] = agent_role
            
            # Get templates
            templates = []
            cursor = templates_col.find(query).sort("usage_count", DESCENDING).limit(limit)
            
            async for doc in cursor:
                template_data = {
                    "template_id": doc["template_id"],
                    "name": doc["name"],
                    "description": doc["description"],
                    "agent_role": doc["agent_role"],
                    "initial_context": doc["initial_context"],
                    "suggested_prompts": doc["suggested_prompts"],
                    "tags": doc["tags"],
                    "category": doc["category"],
                    "created_by": doc["created_by"],
                    "created_at": doc["created_at"].isoformat(),
                    "usage_count": doc["usage_count"]
                }
                templates.append(template_data)
            
            return templates
            
        except Exception as e:
            logger.error(f"Failed to get conversation templates: {str(e)}")
            raise
    
    async def generate_conversation_insights(
        self,
        user_id: str,
        period_days: int = 30
    ) -> ConversationInsights:
        """
        Generate comprehensive insights about user's conversations.
        
        Args:
            user_id: User identifier
            period_days: Period in days to analyze
            
        Returns:
            Conversation insights
        """
        await self._ensure_advanced_collections_and_indexes()
        
        try:
            conversations_col = await get_async_collection("chat_conversations")
            messages_col = await get_async_collection("chat_messages")
            
            # Date range for analysis
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=period_days)
            
            # Get conversations in the period
            conversations = []
            async for doc in conversations_col.find({
                "user_id": user_id,
                "created_at": {"$gte": start_date, "$lte": end_date}
            }):
                conversations.append(doc)
            
            # Calculate basic metrics
            total_conversations = len(conversations)
            active_conversations = len([c for c in conversations if c.get("status") == ChatConversationStatus.ACTIVE])
            archived_conversations = len([c for c in conversations if c.get("status") == ChatConversationStatus.ARCHIVED])
            deleted_conversations = len([c for c in conversations if c.get("status") == ChatConversationStatus.DELETED])
            
            # Calculate average messages per conversation
            total_messages = sum(c.get("message_count", 0) for c in conversations)
            avg_messages = total_messages / total_conversations if total_conversations > 0 else 0
            
            # Find most active agent
            agent_counts = Counter(c.get("agent_role", "unknown") for c in conversations)
            most_active_agent = agent_counts.most_common(1)[0][0] if agent_counts else "unknown"
            
            # Analyze conversation topics (simplified - based on titles and tags)
            topics = []
            for conv in conversations:
                title = conv.get("title", "").lower()
                tags = conv.get("tags", [])
                topics.extend(tags)
                
                # Extract keywords from titles
                title_words = re.findall(r'\b\w+\b', title)
                topics.extend([word for word in title_words if len(word) > 3])
            
            most_common_topics = Counter(topics).most_common(10)
            
            # Calculate conversation trends (daily counts)
            daily_counts = defaultdict(int)
            for conv in conversations:
                date_key = conv.get("created_at", datetime.utcnow()).strftime("%Y-%m-%d")
                daily_counts[date_key] += 1
            
            # Generate trend data for the last 30 days
            trend_data = []
            for i in range(30):
                date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
                trend_data.append(daily_counts.get(date, 0))
            
            conversation_trends = {
                "daily": list(reversed(trend_data))
            }
            
            # Calculate user engagement score (simplified)
            engagement_factors = [
                min(total_conversations / 10, 1.0),  # Conversation frequency
                min(avg_messages / 5, 1.0),  # Message depth
                min(len(set(c.get("agent_role") for c in conversations)) / 5, 1.0),  # Agent diversity
                min(len(most_common_topics) / 10, 1.0)  # Topic diversity
            ]
            user_engagement_score = sum(engagement_factors) / len(engagement_factors)
            
            # Analyze context usage patterns
            context_patterns = await self._analyze_context_patterns(user_id, conversations)
            
            insights = ConversationInsights(
                total_conversations=total_conversations,
                active_conversations=active_conversations,
                archived_conversations=archived_conversations,
                deleted_conversations=deleted_conversations,
                average_messages_per_conversation=avg_messages,
                most_active_agent=most_active_agent,
                most_common_topics=most_common_topics,
                conversation_trends=conversation_trends,
                user_engagement_score=user_engagement_score,
                context_usage_patterns=context_patterns
            )
            
            # Store insights
            await self._store_conversation_insights(user_id, insights, period_days)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate conversation insights: {str(e)}")
            raise
    
    async def _analyze_context_patterns(self, user_id: str, conversations: List[Dict]) -> Dict[str, Any]:
        """Analyze context usage patterns in conversations."""
        try:
            context_types = defaultdict(int)
            context_values = defaultdict(int)
            
            for conv in conversations:
                context = conv.get("conversation_context", {})
                
                for key, value in context.items():
                    context_types[key] += 1
                    if isinstance(value, str):
                        context_values[f"{key}:{value}"] += 1
            
            return {
                "most_used_context_types": dict(context_types.most_common(10)),
                "most_used_context_values": dict(context_values.most_common(10)),
                "total_context_keys": len(context_types),
                "average_context_per_conversation": sum(context_types.values()) / len(conversations) if conversations else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze context patterns: {str(e)}")
            return {}
    
    async def _store_conversation_insights(self, user_id: str, insights: ConversationInsights, period_days: int):
        """Store conversation insights in the database."""
        try:
            insights_col = await get_async_collection("chat_insights")
            
            insights_doc = {
                "user_id": user_id,
                "period_days": period_days,
                "insights": {
                    "total_conversations": insights.total_conversations,
                    "active_conversations": insights.active_conversations,
                    "archived_conversations": insights.archived_conversations,
                    "deleted_conversations": insights.deleted_conversations,
                    "average_messages_per_conversation": insights.average_messages_per_conversation,
                    "most_active_agent": insights.most_active_agent,
                    "most_common_topics": insights.most_common_topics,
                    "conversation_trends": insights.conversation_trends,
                    "user_engagement_score": insights.user_engagement_score,
                    "context_usage_patterns": insights.context_usage_patterns
                },
                "generated_at": datetime.utcnow()
            }
            
            await insights_col.insert_one(insights_doc)
            
        except Exception as e:
            logger.error(f"Failed to store conversation insights: {str(e)}")
    
    async def export_conversations(
        self,
        user_id: str,
        conversation_ids: Optional[List[str]] = None,
        format: str = "json",
        include_metadata: bool = True,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Export conversations to various formats.
        
        Args:
            user_id: User identifier
            conversation_ids: Specific conversations to export (None for all)
            format: Export format (json, csv, txt)
            include_metadata: Whether to include metadata
            include_context: Whether to include context
            
        Returns:
            Export data and metadata
        """
        await self._ensure_advanced_collections_and_indexes()
        
        try:
            conversations_col = await get_async_collection("chat_conversations")
            messages_col = await get_async_collection("chat_messages")
            
            # Build query
            query = {"user_id": user_id}
            if conversation_ids:
                query["conversation_id"] = {"$in": conversation_ids}
            
            # Get conversations
            conversations = []
            async for doc in conversations_col.find(query):
                conversations.append(doc)
            
            # Get messages for each conversation
            export_data = []
            for conv in conversations:
                conversation_data = {
                    "conversation_id": conv["conversation_id"],
                    "title": conv.get("title", ""),
                    "agent_role": conv.get("agent_role", ""),
                    "status": conv.get("status", ""),
                    "created_at": conv.get("created_at", "").isoformat() if conv.get("created_at") else "",
                    "messages": []
                }
                
                if include_metadata:
                    conversation_data["metadata"] = {
                        "message_count": conv.get("message_count", 0),
                        "total_tokens_used": conv.get("total_tokens_used", 0),
                        "last_message_at": conv.get("last_message_at", "").isoformat() if conv.get("last_message_at") else "",
                        "tags": conv.get("tags", [])
                    }
                
                if include_context:
                    conversation_data["context"] = conv.get("conversation_context", {})
                
                # Get messages
                messages = []
                async for msg_doc in messages_col.find({"conversation_id": conv["conversation_id"]}).sort("timestamp", ASCENDING):
                    message_data = {
                        "message_type": msg_doc.get("message_type", ""),
                        "content": msg_doc.get("content", ""),
                        "timestamp": msg_doc.get("timestamp", "").isoformat() if msg_doc.get("timestamp") else ""
                    }
                    
                    if include_metadata:
                        message_data["metadata"] = msg_doc.get("metadata", {})
                    
                    if include_context:
                        message_data["context"] = msg_doc.get("message_context", {})
                    
                    messages.append(message_data)
                
                conversation_data["messages"] = messages
                export_data.append(conversation_data)
            
            # Format the export data
            if format.lower() == "json":
                export_content = json.dumps(export_data, indent=2, default=str)
                content_type = "application/json"
            elif format.lower() == "csv":
                export_content = self._convert_to_csv(export_data)
                content_type = "text/csv"
            elif format.lower() == "txt":
                export_content = self._convert_to_txt(export_data)
                content_type = "text/plain"
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            # Log export operation
            await self._log_audit_event(
                user_id=user_id,
                action_type="conversations_exported",
                resource_type="conversation",
                resource_id="bulk_export",
                action_details={
                    "format": format,
                    "conversation_count": len(export_data),
                    "include_metadata": include_metadata,
                    "include_context": include_context
                }
            )
            
            return {
                "export_data": export_content,
                "content_type": content_type,
                "conversation_count": len(export_data),
                "format": format,
                "exported_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to export conversations: {str(e)}")
            raise
    
    def _convert_to_csv(self, export_data: List[Dict]) -> str:
        """Convert export data to CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "conversation_id", "title", "agent_role", "status", "created_at",
            "message_type", "content", "message_timestamp"
        ])
        
        # Write data
        for conv in export_data:
            for message in conv["messages"]:
                writer.writerow([
                    conv["conversation_id"],
                    conv["title"],
                    conv["agent_role"],
                    conv["status"],
                    conv["created_at"],
                    message["message_type"],
                    message["content"],
                    message["timestamp"]
                ])
        
        return output.getvalue()
    
    def _convert_to_txt(self, export_data: List[Dict]) -> str:
        """Convert export data to plain text format."""
        output = []
        
        for conv in export_data:
            output.append(f"=== Conversation: {conv['title']} ===")
            output.append(f"ID: {conv['conversation_id']}")
            output.append(f"Agent: {conv['agent_role']}")
            output.append(f"Status: {conv['status']}")
            output.append(f"Created: {conv['created_at']}")
            output.append("")
            
            for message in conv["messages"]:
                output.append(f"[{message['timestamp']}] {message['message_type'].upper()}: {message['content']}")
            
            output.append("")
            output.append("-" * 50)
            output.append("")
        
        return "\n".join(output)
    
    # Override parent methods to add embeddings generation
    async def add_message(
        self,
        user_id: str,
        conversation_id: str,
        message_content: str,
        message_type: str = ChatMessageType.USER,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tokens_used: Optional[int] = None,
        processing_time: Optional[float] = None,
        model_used: Optional[str] = None
    ) -> str:
        """
        Add a message to a conversation with embeddings generation.
        
        This overrides the parent method to add embeddings generation.
        """
        await self._ensure_advanced_collections_and_indexes()
        
        # Call parent method
        message_id = await super().add_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_content=message_content,
            message_type=message_type,
            context=context,
            metadata=metadata,
            tokens_used=tokens_used,
            processing_time=processing_time,
            model_used=model_used
        )
        
        # Generate and store embeddings
        try:
            embedding = await self._generate_embeddings(message_content, message_type)
            if embedding:
                await self._store_embeddings(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    user_id=user_id,
                    content=message_content,
                    content_type=message_type,
                    embedding=embedding
                )
        except Exception as e:
            logger.error(f"Failed to generate embeddings for message {message_id}: {str(e)}")
            # Don't fail the entire operation for embedding errors
        
        return message_id 