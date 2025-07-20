"""
Advanced Conversation Memory Service for WMS Chatbot

This service implements LangChain's ConversationSummaryBufferMemory to provide:
- Recent conversation details (buffer)
- Summarized older conversations (summary) 
- Token management to stay within limits
- Good balance of context retention and performance
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_KEY, LLM_MODEL
from app.utils.database import get_async_collection

logger = logging.getLogger("wms_chatbot.conversation_memory_service")


class PersistentChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history that persists to MongoDB.
    Implements BaseChatMessageHistory for LangChain compatibility.
    """
    
    def __init__(self, conversation_id: str, user_id: str):
        """
        Initialize persistent chat history.
        
        Args:
            conversation_id: Unique conversation identifier
            user_id: User identifier
        """
        self.conversation_id = conversation_id
        self.user_id = user_id
        self._messages: List[BaseMessage] = []
        self._loaded = False
    
    async def _ensure_loaded(self):
        """Load messages from database if not already loaded."""
        if self._loaded:
            return
            
        try:
            messages_col = await get_async_collection("chat_conversation_memory")
            
            # Find conversation memory
            memory_doc = await messages_col.find_one({
                "conversation_id": self.conversation_id,
                "user_id": self.user_id
            })
            
            if memory_doc and "messages" in memory_doc:
                # Reconstruct messages from stored data
                self._messages = []
                for msg_data in memory_doc["messages"]:
                    if msg_data["type"] == "human":
                        self._messages.append(HumanMessage(content=msg_data["content"]))
                    elif msg_data["type"] == "ai":
                        self._messages.append(AIMessage(content=msg_data["content"]))
            
            self._loaded = True
            logger.debug(f"Loaded {len(self._messages)} messages for conversation {self.conversation_id}")
            
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            self._loaded = True  # Prevent infinite retry
    
    async def _save_to_database(self):
        """Save current messages to database."""
        try:
            messages_col = await get_async_collection("chat_conversation_memory")
            
            # Convert messages to serializable format
            messages_data = []
            for msg in self._messages:
                if isinstance(msg, HumanMessage):
                    messages_data.append({
                        "type": "human",
                        "content": msg.content,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif isinstance(msg, AIMessage):
                    messages_data.append({
                        "type": "ai", 
                        "content": msg.content,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Upsert conversation memory
            await messages_col.update_one(
                {
                    "conversation_id": self.conversation_id,
                    "user_id": self.user_id
                },
                {
                    "$set": {
                        "messages": messages_data,
                        "updated_at": datetime.utcnow(),
                        "message_count": len(messages_data)
                    },
                    "$setOnInsert": {
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            
            logger.debug(f"Saved {len(messages_data)} messages for conversation {self.conversation_id}")
            
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages in the conversation."""
        return self._messages
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the conversation."""
        self._messages.append(message)
    
    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self._messages.clear()


class ConversationMemoryService:
    """
    Service for managing conversation memory using LangChain's ConversationSummaryBufferMemory.
    
    This provides:
    - Recent conversation details (buffer)
    - Summarized older conversations (summary)
    - Token management to stay within limits
    - Persistent storage in MongoDB
    """
    
    def __init__(self, max_token_limit: int = 2000):
        """
        Initialize the conversation memory service.
        
        Args:
            max_token_limit: Maximum tokens to keep in memory buffer
        """
        self.max_token_limit = max_token_limit
        self.conversation_memories: Dict[str, ConversationSummaryBufferMemory] = {}
        
        # Initialize LLM for summarization
        if OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model=LLM_MODEL,
                temperature=0.1,  # Low temperature for consistent summaries
                tags=["conversation-summary", "wms-chatbot"]
            )
        else:
            self.llm = None
            logger.warning("OPENAI_API_KEY not set. Memory service will use simple buffer memory.")
    
    async def get_memory(
        self, 
        conversation_id: str, 
        user_id: str,
        agent_role: str = "assistant"
    ) -> ConversationSummaryBufferMemory:
        """
        Get or create conversation memory for a specific conversation.
        
        Args:
            conversation_id: Unique conversation identifier
            user_id: User identifier
            agent_role: Role of the agent (for context)
            
        Returns:
            ConversationSummaryBufferMemory instance
        """
        memory_key = f"{user_id}:{conversation_id}"
        
        if memory_key not in self.conversation_memories:
            # Create persistent chat history
            chat_history = PersistentChatMessageHistory(conversation_id, user_id)
            await chat_history._ensure_loaded()
            
            if self.llm:
                # Create ConversationSummaryBufferMemory with LLM for summarization
                memory = ConversationSummaryBufferMemory(
                    llm=self.llm,
                    chat_memory=chat_history,
                    max_token_limit=self.max_token_limit,
                    return_messages=True,
                    human_prefix="User",
                    ai_prefix=f"{agent_role.title()} Assistant"
                )
            else:
                # Fallback to basic buffer memory without summarization
                from langchain.memory import ConversationBufferMemory
                memory = ConversationBufferMemory(
                    chat_memory=chat_history,
                    return_messages=True
                )
            
            self.conversation_memories[memory_key] = memory
            logger.info(f"Created new conversation memory for {memory_key}")
        
        return self.conversation_memories[memory_key]
    
    async def add_message(
        self,
        conversation_id: str,
        user_id: str,
        user_message: str,
        ai_response: str,
        agent_role: str = "assistant"
    ):
        """
        Add a user message and AI response to conversation memory.
        
        Args:
            conversation_id: Unique conversation identifier
            user_id: User identifier
            user_message: User's message
            ai_response: AI's response
            agent_role: Role of the agent
        """
        try:
            memory = await self.get_memory(conversation_id, user_id, agent_role)
            
            # Add messages to memory
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(ai_response)
            
            # Save to database
            if isinstance(memory.chat_memory, PersistentChatMessageHistory):
                await memory.chat_memory._save_to_database()
            
            logger.debug(f"Added message pair to conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error adding message to conversation memory: {e}")
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        user_id: str,
        agent_role: str = "assistant"
    ) -> Dict[str, Any]:
        """
        Get conversation context including summary and recent messages.
        
        Args:
            conversation_id: Unique conversation identifier
            user_id: User identifier
            agent_role: Role of the agent
            
        Returns:
            Dictionary containing conversation context
        """
        try:
            memory = await self.get_memory(conversation_id, user_id, agent_role)
            
            # Get memory variables (includes summary and recent messages)
            memory_vars = memory.load_memory_variables({})
            
            # Calculate conversation stats
            total_messages = len(memory.chat_memory.messages)
            recent_messages = len(memory_vars.get("history", []))
            
            context = {
                "total_messages": total_messages,
                "recent_messages": recent_messages,
                "memory_variables": memory_vars,
                "has_summary": hasattr(memory, "moving_summary_buffer") and bool(memory.moving_summary_buffer),
                "conversation_id": conversation_id,
                "user_id": user_id
            }
            
            # Add summary information if available
            if hasattr(memory, "moving_summary_buffer") and memory.moving_summary_buffer:
                context["summary"] = memory.moving_summary_buffer
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return {
                "total_messages": 0,
                "recent_messages": 0,
                "memory_variables": {"history": []},
                "has_summary": False,
                "conversation_id": conversation_id,
                "user_id": user_id
            }
    
    async def clear_conversation(self, conversation_id: str, user_id: str):
        """
        Clear all memory for a specific conversation.
        
        Args:
            conversation_id: Unique conversation identifier
            user_id: User identifier
        """
        try:
            memory_key = f"{user_id}:{conversation_id}"
            
            # Clear from memory cache
            if memory_key in self.conversation_memories:
                self.conversation_memories[memory_key].clear()
                del self.conversation_memories[memory_key]
            
            # Clear from database
            messages_col = await get_async_collection("chat_conversation_memory")
            await messages_col.delete_one({
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            
            logger.info(f"Cleared conversation memory for {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error clearing conversation memory: {e}")
    
    async def clear_user_sessions(self, user_id: str):
        """
        Clear all conversation memories for a specific user (for logout).
        
        Args:
            user_id: User identifier
        """
        try:
            # Clear from memory cache - find all keys that start with this user_id
            keys_to_remove = [key for key in self.conversation_memories.keys() 
                             if key.startswith(f"{user_id}:")]
            
            for key in keys_to_remove:
                self.conversation_memories[key].clear()
                del self.conversation_memories[key]
            
            # Clear from database
            messages_col = await get_async_collection("chat_conversation_memory")
            await messages_col.delete_many({"user_id": user_id})
            
            logger.info(f"Cleared all conversation memories for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error clearing user sessions: {e}")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about conversation memory usage.
        
        Returns:
            Dictionary containing memory statistics
        """
        try:
            messages_col = await get_async_collection("chat_conversation_memory")
            
            # Get total conversations and messages
            total_conversations = await messages_col.count_documents({})
            
            # Get active memory instances
            active_memories = len(self.conversation_memories)
            
            # Calculate memory distribution
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_messages": {"$sum": "$message_count"},
                        "avg_messages_per_conversation": {"$avg": "$message_count"},
                        "max_messages_per_conversation": {"$max": "$message_count"}
                    }
                }
            ]
            
            stats_result = await messages_col.aggregate(pipeline).to_list(1)
            
            stats = {
                "total_conversations": total_conversations,
                "active_memory_instances": active_memories,
                "max_token_limit": self.max_token_limit,
                "llm_available": self.llm is not None
            }
            
            if stats_result:
                stats.update({
                    "total_messages": stats_result[0].get("total_messages", 0),
                    "avg_messages_per_conversation": round(stats_result[0].get("avg_messages_per_conversation", 0), 2),
                    "max_messages_per_conversation": stats_result[0].get("max_messages_per_conversation", 0)
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {
                "total_conversations": 0,
                "active_memory_instances": len(self.conversation_memories),
                "max_token_limit": self.max_token_limit,
                "llm_available": self.llm is not None
            }


# Create a singleton instance
conversation_memory_service = ConversationMemoryService()