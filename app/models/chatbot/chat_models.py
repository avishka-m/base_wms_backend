"""
Chat-related Pydantic models for the WMS Chatbot API.
"""

from typing import Optional, List, Dict, Any
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


class NewConversation(BaseModel):
    """Create a new conversation"""
    title: str = Field(..., description="Title of the conversation")
    role: str = Field(..., description="Role to use for this conversation")


class ConversationMetadata(BaseModel):
    """Conversation metadata"""
    title: str = Field(..., description="Conversation title")
    role: str = Field(..., description="Role associated with conversation")
    created_at: str = Field(..., description="Creation timestamp")
    message_count: int = Field(..., description="Number of messages in conversation")
    last_updated: str = Field(..., description="Last update timestamp")


class ConversationMessage(BaseModel):
    """Individual conversation message"""
    role: str = Field(..., description="Message role (user or assistant)")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="Message timestamp")


class ConversationResponse(BaseModel):
    """Full conversation response with messages"""
    conversation_id: str = Field(..., description="Conversation ID")
    metadata: ConversationMetadata = Field(..., description="Conversation metadata")
    messages: List[ConversationMessage] = Field(..., description="Conversation messages")


class ConversationList(BaseModel):
    """List of conversations for a user"""
    conversations: List[Dict[str, Any]] = Field(..., description="List of user conversations")


class UserRoleResponse(BaseModel):
    """User role and permissions response"""
    username: str = Field(..., description="Username")
    role: str = Field(..., description="User role")
    allowed_chatbot_roles: List[str] = Field(..., description="Allowed chatbot roles for this user")


class ConversationUpdateRequest(BaseModel):
    """Request to update conversation metadata"""
    title: Optional[str] = Field(None, description="New conversation title")


class DeleteResponse(BaseModel):
    """Response for delete operations"""
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Response message")
