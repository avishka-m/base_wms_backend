from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

class ChatMessage(BaseModel):
    """Chat message sent by a user."""
    role: str = Field(..., description="Role of the user sending the message (clerk, picker, packer, driver, manager)")
    message: str = Field(..., description="Message content")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for continued conversations")

class ChatResponse(BaseModel):
    """Response from the chatbot."""
    response: str = Field(..., description="Chatbot response message")
    conversation_id: str = Field(..., description="Conversation ID for continued conversations")
    timestamp: str = Field(..., description="Timestamp of the response")

class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Status of the service")
    version: str = Field(..., description="Version of the API")
    timestamp: str = Field(..., description="Timestamp of the health check")

class NewConversation(BaseModel):
    """Create a new conversation"""
    title: str = Field(..., description="Title of the conversation")
    role: str = Field(..., description="Role to use for this conversation")

class ConversationMetadata(BaseModel):
    """Metadata for a conversation"""
    title: str = Field(..., description="Title of the conversation")
    role: str = Field(..., description="Role used for this conversation")
    created_at: str = Field(..., description="Creation timestamp")

class ConversationMessage(BaseModel):
    """A message in a conversation"""
    role: str = Field(..., description="Role of the sender (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="Message timestamp")

class ConversationResponse(BaseModel):
    """Full conversation details"""
    conversation_id: str = Field(..., description="Unique conversation ID")
    metadata: ConversationMetadata
    messages: List[ConversationMessage]

class ConversationListItem(BaseModel):
    """Summary of a conversation for listing"""
    conversation_id: str = Field(..., description="Unique conversation ID")
    title: str = Field(..., description="Conversation title")
    role: str = Field(..., description="Chatbot role")
    created_at: str = Field(..., description="Creation timestamp")
    message_count: int = Field(..., description="Number of messages")
    last_updated: str = Field(..., description="Last message timestamp") 