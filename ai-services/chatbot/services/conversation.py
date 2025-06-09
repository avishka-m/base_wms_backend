"""Conversation management service"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from core.logging import logger

class ConversationService:
    """Service for managing user conversations"""
    
    def __init__(self):
        # In-memory storage for conversations
        # In production, this should be replaced with a database
        self.user_conversations: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.max_messages_per_conversation = 100
    
    def create_conversation(
        self, 
        user_id: str, 
        role: str, 
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new conversation for a user.
        
        Args:
            user_id: User identifier
            role: Chatbot role for this conversation
            title: Optional title for the conversation
            
        Returns:
            Conversation details
        """
        # Generate conversation ID
        conversation_id = f"{role}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize user's conversations if needed
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = {}
        
        # Create conversation
        if title is None:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
        self.user_conversations[user_id][conversation_id] = {
            "metadata": {
                "title": title,
                "created_at": datetime.now().isoformat(),
                "role": role,
            },
            "messages": []
        }
        
        logger.info(f"Created conversation {conversation_id} for user {user_id}")
        
        return {
            "conversation_id": conversation_id,
            "title": title,
            "role": role,
            "created_at": datetime.now().isoformat()
        }
    
    def add_message(
        self, 
        user_id: str, 
        conversation_id: str, 
        role: str, 
        content: str
    ) -> None:
        """
        Add a message to a conversation.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            role: Message role (user/assistant)
            content: Message content
        """
        # Ensure conversation exists
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = {}
            
        if conversation_id not in self.user_conversations[user_id]:
            # Create conversation if it doesn't exist
            self.user_conversations[user_id][conversation_id] = {
                "metadata": {
                    "title": f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "created_at": datetime.now().isoformat(),
                    "role": "unknown",
                },
                "messages": []
            }
        
        # Add message
        self.user_conversations[user_id][conversation_id]["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim conversation history if too long
        messages = self.user_conversations[user_id][conversation_id]["messages"]
        if len(messages) > self.max_messages_per_conversation:
            self.user_conversations[user_id][conversation_id]["messages"] = messages[-self.max_messages_per_conversation:]
    
    def get_conversation(self, user_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            
        Returns:
            Conversation data or None if not found
        """
        if user_id in self.user_conversations:
            if conversation_id in self.user_conversations[user_id]:
                return {
                    "conversation_id": conversation_id,
                    "metadata": self.user_conversations[user_id][conversation_id]["metadata"],
                    "messages": self.user_conversations[user_id][conversation_id]["messages"]
                }
        return None
    
    def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all conversations for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of conversation summaries
        """
        if user_id not in self.user_conversations:
            return []
        
        conversations = []
        for conv_id, data in self.user_conversations[user_id].items():
            conversations.append({
                "conversation_id": conv_id,
                "title": data["metadata"].get("title", f"Chat {conv_id}"),
                "role": data["metadata"].get("role", "unknown"),
                "created_at": data["metadata"].get("created_at", ""),
                "message_count": len(data["messages"]),
                "last_updated": data["messages"][-1]["timestamp"] if data["messages"] else data["metadata"].get("created_at", "")
            })
        
        return conversations
    
    def update_conversation_title(
        self, 
        user_id: str, 
        conversation_id: str, 
        title: str
    ) -> bool:
        """
        Update a conversation's title.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            title: New title
            
        Returns:
            True if updated, False if not found
        """
        if user_id in self.user_conversations:
            if conversation_id in self.user_conversations[user_id]:
                self.user_conversations[user_id][conversation_id]["metadata"]["title"] = title
                logger.info(f"Updated title for conversation {conversation_id}")
                return True
        return False
    
    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            
        Returns:
            True if deleted, False if not found
        """
        if user_id in self.user_conversations:
            if conversation_id in self.user_conversations[user_id]:
                del self.user_conversations[user_id][conversation_id]
                logger.info(f"Deleted conversation {conversation_id} for user {user_id}")
                return True
        return False

# Create a singleton instance
conversation_service = ConversationService() 