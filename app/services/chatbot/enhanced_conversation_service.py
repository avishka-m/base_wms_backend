"""
Enhanced conversation service with MongoDB persistence.
Supports advanced chatbot features including multi-modal content,
analytics, search, and comprehensive conversation management.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pymongo import ASCENDING, DESCENDING, TEXT

from app.utils.database import get_async_collection
from app.models.chatbot.enhanced_chat_models import (
    ChatConversation,
    ChatMessage,
    ChatAuditLog,
    ChatConversationStatus,
    ChatMessageType,
    ConversationSearchRequest
)

logger = logging.getLogger("wms_chatbot.enhanced_conversation_service")


class EnhancedConversationService:
    """Enhanced service for managing user conversations with MongoDB persistence."""
    
    def __init__(self):
        """Initialize the enhanced conversation service."""
        self._collections_initialized = False
        self._context_service = None
    
    async def _ensure_collections_and_indexes(self):
        """Ensure database collections and indexes are set up."""
        if self._collections_initialized:
            return
            
        try:
            # Get collections
            conversations_col = await get_async_collection("chat_conversations")
            messages_col = await get_async_collection("chat_messages")
            attachments_col = await get_async_collection("chat_attachments")
            preferences_col = await get_async_collection("chat_user_preferences")
            analytics_col = await get_async_collection("chat_analytics")
            audit_col = await get_async_collection("chat_audit_logs")
            
            # Create indexes for conversations
            await conversations_col.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
            await conversations_col.create_index([("conversation_id", ASCENDING)], unique=True)
            await conversations_col.create_index([("status", ASCENDING)])
            await conversations_col.create_index([("agent_role", ASCENDING)])
            await conversations_col.create_index([("tags", ASCENDING)])
            
            # Create indexes for messages
            await messages_col.create_index([("conversation_id", ASCENDING), ("timestamp", ASCENDING)])
            await messages_col.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
            await messages_col.create_index([("message_type", ASCENDING)])
            await messages_col.create_index([("content", TEXT)])  # Full-text search
            
            # Create indexes for attachments
            await attachments_col.create_index([("message_id", ASCENDING)])
            await attachments_col.create_index([("conversation_id", ASCENDING)])
            await attachments_col.create_index([("user_id", ASCENDING)])
            
            # Create indexes for preferences
            await preferences_col.create_index([("user_id", ASCENDING)], unique=True)
            
            # Create indexes for analytics
            await analytics_col.create_index([("user_id", ASCENDING), ("period_start", DESCENDING)])
            await analytics_col.create_index([("period_type", ASCENDING)])
            
            # Create indexes for audit logs
            await audit_col.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
            await audit_col.create_index([("action_type", ASCENDING)])
            await audit_col.create_index([("resource_type", ASCENDING)])
            
            self._collections_initialized = True
            logger.info("Database collections and indexes initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database collections: {str(e)}")
            raise
    
    def _get_context_service(self):
        """Get or initialize the context awareness service."""
        if self._context_service is None:
            try:
                from app.services.chatbot.context_awareness_service import ContextAwarenessService
                self._context_service = ContextAwarenessService()
            except ImportError:
                logger.warning("Context awareness service not available")
                self._context_service = None
        return self._context_service
    
    async def enrich_response_with_context(
        self,
        user_id: str,
        message: str,
        base_response: str,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enrich AI response with contextual information.
        
        Args:
            user_id: User identifier
            message: User message
            base_response: Base AI response
            conversation_context: Current conversation context
            
        Returns:
            Enriched response with context
        """
        try:
            context_service = self._get_context_service()
            if context_service:
                return await context_service.get_context_enriched_response(
                    user_id=user_id,
                    message=message,
                    base_response=base_response,
                    conversation_context=conversation_context
                )
            else:
                # Fallback if context service is not available
                return {
                    "response": base_response,
                    "context": {},
                    "suggestions": []
                }
        except Exception as e:
            logger.error(f"Failed to enrich response with context: {str(e)}")
            return {
                "response": base_response,
                "context": {},
                "suggestions": []
            }
    
    async def create_conversation(
        self,
        user_id: str,
        title: str,
        agent_role: str,
        available_roles: List[str],
        initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new conversation for a user.
        
        Args:
            user_id: User identifier
            title: Conversation title
            agent_role: Role for the conversation
            available_roles: List of roles available to the user
            initial_context: Initial conversation context
            
        Returns:
            New conversation information
            
        Raises:
            ValueError: If role is invalid
        """
        await self._ensure_collections_and_indexes()
        
        # Validate role exists
        if agent_role.lower() not in [role.lower() for role in available_roles]:
            raise ValueError(f"Invalid role: {agent_role}. Must be one of {available_roles}")
        
        # Generate unique conversation ID
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # Include milliseconds
        conversation_id = f"{agent_role.lower()}_{timestamp}"
        
        # Create conversation document
        conversation = ChatConversation(
            conversation_id=conversation_id,
            user_id=user_id,
            title=title,
            agent_role=agent_role.lower(),
            conversation_context=initial_context or {},
            metadata={
                "created_by": "api",
                "initial_role": agent_role,
                "available_roles": available_roles
            }
        )
        
        try:
            # Insert conversation into database
            conversations_col = await get_async_collection("chat_conversations")
            await conversations_col.insert_one(conversation.to_mongo())
            
            # Log audit event
            await self._log_audit_event(
                user_id=user_id,
                action_type="conversation_created",
                resource_type="conversation",
                resource_id=conversation_id,
                action_details={
                    "title": title,
                    "agent_role": agent_role,
                    "conversation_id": conversation_id
                }
            )
            
            logger.info(f"Created conversation {conversation_id} for user {user_id}")
            
            return {
                "conversation_id": conversation_id,
                "title": title,
                "agent_role": agent_role.lower(),
                "created_at": conversation.created_at.isoformat(),
                "status": ChatConversationStatus.ACTIVE
            }
            
        except Exception as e:
            logger.error(f"Failed to create conversation: {str(e)}")
            raise
    
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
        Add a message to a conversation.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            message_content: Message content
            message_type: Type of message (user, assistant, system)
            context: Message context
            metadata: Additional metadata
            tokens_used: Number of tokens used for processing
            processing_time: Time taken to process message
            model_used: AI model used for response
            
        Returns:
            Message ID
        """
        await self._ensure_collections_and_indexes()
        
        try:
            # Create message document
            message = ChatMessage(
                conversation_id=conversation_id,
                user_id=user_id,
                message_type=message_type,
                content=message_content,
                context=context or {},
                metadata=metadata or {},
                tokens_used=tokens_used,
                processing_time=processing_time,
                model_used=model_used
            )
            
            # Insert message into database
            messages_col = await get_async_collection("chat_messages")
            result = await messages_col.insert_one(message.to_mongo())
            message_id = str(result.inserted_id)
            
            # Update conversation statistics
            await self._update_conversation_stats(conversation_id, message_type, tokens_used or 0)
            
            logger.debug(f"Added {message_type} message to conversation {conversation_id}")
            
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to add message: {str(e)}")
            raise
    
    async def get_conversation_history(
        self,
        user_id: str,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        include_context: bool = False
    ) -> Dict[str, Any]:
        """
        Get conversation history with messages.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            include_context: Whether to include message context
            
        Returns:
            Conversation data with messages
        """
        await self._ensure_collections_and_indexes()
        
        try:
            # Get conversation metadata
            conversations_col = await get_async_collection("chat_conversations")
            conversation_doc = await conversations_col.find_one({
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            
            if not conversation_doc:
                return None
            
            conversation = ChatConversation.from_mongo(conversation_doc)
            
            # Get messages
            messages_col = await get_async_collection("chat_messages")
            query = {"conversation_id": conversation_id, "user_id": user_id}
            
            cursor = messages_col.find(query).sort("timestamp", ASCENDING)
            if offset > 0:
                cursor = cursor.skip(offset)
            if limit:
                cursor = cursor.limit(limit)
            
            messages = []
            async for msg_doc in cursor:
                message = ChatMessage.from_mongo(msg_doc)
                msg_data = {
                    "id": str(message.id),
                    "type": message.message_type,
                    "content": message.content,
                    "timestamp": message.timestamp.isoformat(),
                    "has_attachments": message.has_attachments,
                    "attachment_count": message.attachment_count
                }
                
                if include_context:
                    msg_data["context"] = message.context
                    msg_data["metadata"] = message.metadata
                
                messages.append(msg_data)
            
            return {
                "conversation_id": conversation_id,
                "title": conversation.title,
                "agent_role": conversation.agent_role,
                "status": conversation.status,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "last_message_at": conversation.last_message_at.isoformat() if conversation.last_message_at else None,
                "message_count": conversation.message_count,
                "messages": messages,
                "metadata": conversation.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {str(e)}")
            raise
    
    async def search_conversations(
        self,
        user_id: str,
        search_request: ConversationSearchRequest
    ) -> Dict[str, Any]:
        """
        Search conversations based on criteria.
        
        Args:
            user_id: User identifier
            search_request: Search parameters
            
        Returns:
            Search results with pagination
        """
        await self._ensure_collections_and_indexes()
        
        try:
            conversations_col = await get_async_collection("chat_conversations")
            
            # Build query
            query = {"user_id": user_id}
            
            if search_request.agent_role:
                query["agent_role"] = search_request.agent_role.lower()
            
            if search_request.status:
                query["status"] = search_request.status
            
            if search_request.date_from or search_request.date_to:
                date_query = {}
                if search_request.date_from:
                    date_query["$gte"] = search_request.date_from
                if search_request.date_to:
                    date_query["$lte"] = search_request.date_to
                query["created_at"] = date_query
            
            if search_request.tags:
                query["tags"] = {"$in": search_request.tags}
            
            # Text search if query provided
            if search_request.query:
                # Search in conversation titles and message content
                text_results = []
                
                # Search conversation titles
                title_query = {**query, "title": {"$regex": search_request.query, "$options": "i"}}
                async for doc in conversations_col.find(title_query):
                    text_results.append(doc["conversation_id"])
                
                # Search message content
                messages_col = await get_async_collection("chat_messages")
                msg_query = {"user_id": user_id, "$text": {"$search": search_request.query}}
                async for msg_doc in messages_col.find(msg_query):
                    if msg_doc["conversation_id"] not in text_results:
                        text_results.append(msg_doc["conversation_id"])
                
                if text_results:
                    query["conversation_id"] = {"$in": text_results}
                else:
                    # No text matches found
                    return {"conversations": [], "total": 0, "has_more": False}
            
            # Get total count
            total = await conversations_col.count_documents(query)
            
            # Get conversations with pagination
            cursor = conversations_col.find(query).sort("last_message_at", DESCENDING)
            if search_request.offset > 0:
                cursor = cursor.skip(search_request.offset)
            cursor = cursor.limit(search_request.limit)
            
            conversations = []
            async for doc in cursor:
                conversation = ChatConversation.from_mongo(doc)
                conversations.append({
                    "conversation_id": conversation.conversation_id,
                    "title": conversation.title,
                    "agent_role": conversation.agent_role,
                    "status": conversation.status,
                    "created_at": conversation.created_at.isoformat(),
                    "last_message_at": conversation.last_message_at.isoformat() if conversation.last_message_at else None,
                    "message_count": conversation.message_count,
                    "has_attachments": conversation.has_attachments,
                    "tags": conversation.tags
                })
            
            return {
                "conversations": conversations,
                "total": total,
                "has_more": (search_request.offset + search_request.limit) < total
            }
            
        except Exception as e:
            logger.error(f"Failed to search conversations: {str(e)}")
            raise
    
    async def archive_conversation(
        self,
        user_id: str,
        conversation_id: str
    ) -> bool:
        """
        Archive a conversation.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            
        Returns:
            True if successful
        """
        await self._ensure_collections_and_indexes()
        
        try:
            conversations_col = await get_async_collection("chat_conversations")
            
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
            
            if hasattr(result, 'modified_count') and result.modified_count > 0:
                # Log audit event
                await self._log_audit_event(
                    user_id=user_id,
                    action_type="conversation_archived",
                    resource_type="conversation",
                    resource_id=conversation_id
                )
                
                logger.info(f"Archived conversation {conversation_id} for user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to archive conversation: {str(e)}")
            raise
    
    async def delete_conversation(
        self,
        user_id: str,
        conversation_id: str,
        hard_delete: bool = False
    ) -> bool:
        """
        Delete a conversation (soft delete by default).
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            hard_delete: Whether to permanently delete the conversation
            
        Returns:
            True if successful
        """
        await self._ensure_collections_and_indexes()
        
        try:
            conversations_col = await get_async_collection("chat_conversations")
            
            if hard_delete:
                # Hard delete: remove conversation and all messages
                result = await conversations_col.delete_one({
                    "conversation_id": conversation_id,
                    "user_id": user_id
                })
                
                if result.deleted_count > 0:
                    # Delete associated messages
                    messages_col = await get_async_collection("chat_messages")
                    await messages_col.delete_many({
                        "conversation_id": conversation_id,
                        "user_id": user_id
                    })
                    
                    # Delete associated attachments
                    attachments_col = await get_async_collection("chat_attachments")
                    await attachments_col.delete_many({
                        "conversation_id": conversation_id,
                        "user_id": user_id
                    })
            else:
                # Soft delete: mark as deleted
                result = await conversations_col.update_one(
                    {"conversation_id": conversation_id, "user_id": user_id},
                    {
                        "$set": {
                            "status": ChatConversationStatus.DELETED,
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
            
            if (hasattr(result, 'modified_count') and result.modified_count > 0) or (hard_delete and hasattr(result, 'deleted_count') and result.deleted_count > 0):
                # Log audit event
                await self._log_audit_event(
                    user_id=user_id,
                    action_type="conversation_deleted" if hard_delete else "conversation_soft_deleted",
                    resource_type="conversation",
                    resource_id=conversation_id,
                    action_details={"hard_delete": hard_delete}
                )
                
                logger.info(f"{'Hard' if hard_delete else 'Soft'} deleted conversation {conversation_id} for user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete conversation: {str(e)}")
            raise
    
    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get user's conversations with pagination.
        
        Args:
            user_id: User identifier
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            status: Filter by conversation status
            
        Returns:
            User conversations with pagination info
        """
        await self._ensure_collections_and_indexes()
        
        try:
            conversations_col = await get_async_collection("chat_conversations")
            
            # Build query
            query = {"user_id": user_id}
            if status:
                query["status"] = status
            else:
                # Default: exclude deleted conversations
                query["status"] = {"$ne": ChatConversationStatus.DELETED}
            
            # Get total count
            total = await conversations_col.count_documents(query)
            
            # Get conversations
            cursor = conversations_col.find(query).sort("last_message_at", DESCENDING)
            if offset > 0:
                cursor = cursor.skip(offset)
            cursor = cursor.limit(limit)
            
            conversations = []
            async for doc in cursor:
                conversation = ChatConversation.from_mongo(doc)
                conversations.append({
                    "conversation_id": conversation.conversation_id,
                    "title": conversation.title,
                    "agent_role": conversation.agent_role,
                    "status": conversation.status,
                    "created_at": conversation.created_at.isoformat(),
                    "last_message_at": conversation.last_message_at.isoformat() if conversation.last_message_at else None,
                    "message_count": conversation.message_count,
                    "has_attachments": conversation.has_attachments
                })
            
            return {
                "conversations": conversations,
                "total": total,
                "has_more": (offset + limit) < total
            }
            
        except Exception as e:
            logger.error(f"Failed to get user conversations: {str(e)}")
            raise
    
    async def _update_conversation_stats(
        self,
        conversation_id: str,
        message_type: str,
        tokens_used: int
    ):
        """Update conversation statistics."""
        try:
            conversations_col = await get_async_collection("chat_conversations")
            
            # Separate $set and $inc operations
            update_data = {
                "$set": {
                    "updated_at": datetime.utcnow(),
                    "last_message_at": datetime.utcnow()
                },
                "$inc": {
                    "message_count": 1,
                    "total_tokens_used": tokens_used
                }
            }
            
            if message_type == ChatMessageType.USER:
                update_data["$inc"]["user_message_count"] = 1
            elif message_type == ChatMessageType.ASSISTANT:
                update_data["$inc"]["assistant_message_count"] = 1
            
            await conversations_col.update_one(
                {"conversation_id": conversation_id},
                update_data
            )
            
        except Exception as e:
            logger.error(f"Failed to update conversation stats: {str(e)}")
    
    async def _log_audit_event(
        self,
        user_id: str,
        action_type: str,
        resource_type: str,
        resource_id: str,
        action_details: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None
    ):
        """Log an audit event."""
        try:
            audit_log = ChatAuditLog(
                user_id=user_id,
                action_type=action_type,
                resource_type=resource_type,
                resource_id=resource_id,
                action_details=action_details or {},
                conversation_id=conversation_id
            )
            
            audit_col = await get_async_collection("chat_audit_logs")
            await audit_col.insert_one(audit_log.to_mongo())
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")
    
    # Migration method for existing in-memory conversations
    async def migrate_from_memory(self, memory_conversations: Dict[str, Any]):
        """
        Migrate conversations from in-memory storage to MongoDB.
        
        Args:
            memory_conversations: Dictionary of in-memory conversations
        """
        await self._ensure_collections_and_indexes()
        
        try:
            conversations_col = await get_async_collection("chat_conversations")
            messages_col = await get_async_collection("chat_messages")
            
            migrated_count = 0
            
            for user_id, user_convs in memory_conversations.items():
                for conv_id, conv_data in user_convs.items():
                    metadata = conv_data.get("metadata", {})
                    messages = conv_data.get("messages", [])
                    
                    # Create conversation document
                    conversation = ChatConversation(
                        conversation_id=conv_id,
                        user_id=user_id,
                        title=metadata.get("title", f"Migrated Conversation {conv_id}"),
                        agent_role=metadata.get("role", "general"),
                        message_count=len(messages),
                        created_at=datetime.fromisoformat(metadata.get("created_at", datetime.utcnow().isoformat())),
                        metadata={**metadata, "migrated": True}
                    )
                    
                    # Insert conversation
                    await conversations_col.insert_one(conversation.to_mongo())
                    
                    # Migrate messages
                    for i, msg in enumerate(messages):
                        message = ChatMessage(
                            conversation_id=conv_id,
                            user_id=user_id,
                            message_type=msg.get("role", "user"),
                            content=msg.get("content", ""),
                            timestamp=datetime.fromisoformat(msg.get("timestamp", datetime.utcnow().isoformat())),
                            metadata={"migrated": True, "original_index": i}
                        )
                        
                        await messages_col.insert_one(message.to_mongo())
                    
                    migrated_count += 1
            
            logger.info(f"Successfully migrated {migrated_count} conversations from memory to MongoDB")
            return migrated_count
            
        except Exception as e:
            logger.error(f"Failed to migrate conversations: {str(e)}")
            raise
