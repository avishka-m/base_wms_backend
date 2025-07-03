"""
Conversation service for managing chat conversations in the WMS Chatbot.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger("wms_chatbot.conversation_service")


class ConversationService:
    """Service for managing user conversations."""
    
    def __init__(self):
        """Initialize the conversation service."""
        # Simple in-memory conversation store
        # In production, this should be replaced with a database
        self.user_conversations = {}
    
    def create_conversation(
        self, 
        user_id: str, 
        title: str, 
        role: str,
        available_roles: List[str]
    ) -> Dict[str, Any]:
        """
        Create a new conversation for a user.
        
        Args:
            user_id: User identifier
            title: Conversation title
            role: Role for the conversation
            available_roles: List of roles available to the user
            
        Returns:
            New conversation information
            
        Raises:
            ValueError: If role is invalid
        """
        # Validate role exists
        if role.lower() not in available_roles:
            raise ValueError(f"Invalid role: {role}. Must be one of {available_roles}")
        
        # Create conversation ID with timestamp to make it unique
        conversation_id = f"{role.lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize user's conversations dictionary if it doesn't exist
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = {}
        
        # Store the new conversation with metadata
        self.user_conversations[user_id][conversation_id] = {
            "metadata": {
                "title": title,
                "created_at": datetime.now().isoformat(),
                "role": role.lower(),
            },
            "messages": []
        }
        
        logger.info(f"Created conversation {conversation_id} for user {user_id}")
        
        return {
            "conversation_id": conversation_id,
            "title": title,
            "role": role.lower(),
            "created_at": datetime.now().isoformat()
        }
    
    def add_message(
        self, 
        user_id: str, 
        conversation_id: str, 
        user_message: str, 
        bot_response: str
    ) -> None:
        """
        Add messages to a conversation.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            user_message: User's message
            bot_response: Bot's response
        """
        # Initialize user's conversations dictionary if it doesn't exist
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = {}
            
        # Create conversation if it doesn't exist
        if conversation_id not in self.user_conversations[user_id]:
            self.user_conversations[user_id][conversation_id] = {
                "metadata": {
                    "title": f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "created_at": datetime.now().isoformat(),
                    "role": conversation_id.split('_')[0],  # Extract role from ID
                },
                "messages": []
            }
            
        # Store the message and response
        messages = self.user_conversations[user_id][conversation_id]["messages"]
        
        messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        messages.append({
            "role": "assistant",
            "content": bot_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim conversation history if it gets too long
        if len(messages) > 100:
            self.user_conversations[user_id][conversation_id]["messages"] = messages[-100:]
    
    def get_conversation(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        Get a specific conversation.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            
        Returns:
            Conversation data
            
        Raises:
            ValueError: If conversation not found
        """
        if user_id not in self.user_conversations or conversation_id not in self.user_conversations[user_id]:
            raise ValueError(f"Conversation not found: {conversation_id}")
            
        return {
            "conversation_id": conversation_id,
            "metadata": self.user_conversations[user_id][conversation_id]["metadata"],
            "messages": self.user_conversations[user_id][conversation_id]["messages"]
        }
    
    def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all conversations for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of conversations with metadata
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
    
    def update_conversation(self, user_id: str, conversation_id: str, title: str) -> Dict[str, Any]:
        """
        Update conversation metadata.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            title: New conversation title
            
        Returns:
            Updated conversation information
            
        Raises:
            ValueError: If conversation not found
        """
        if user_id not in self.user_conversations or conversation_id not in self.user_conversations[user_id]:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        # Update title
        self.user_conversations[user_id][conversation_id]["metadata"]["title"] = title
        
        return {
            "conversation_id": conversation_id,
            "metadata": self.user_conversations[user_id][conversation_id]["metadata"],
            "success": True
        }
    
    def delete_conversation(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        Delete a conversation.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            
        Returns:
            Success response
            
        Raises:
            ValueError: If conversation not found
        """
        if user_id not in self.user_conversations or conversation_id not in self.user_conversations[user_id]:
            raise ValueError(f"Conversation not found: {conversation_id}")
            
        # Delete the conversation
        del self.user_conversations[user_id][conversation_id]
        
        logger.info(f"Deleted conversation {conversation_id} for user {user_id}")
        
        return {
            "success": True,
            "message": f"Conversation {conversation_id} deleted successfully"
        }
