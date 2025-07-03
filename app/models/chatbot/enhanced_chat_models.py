"""
Enhanced chat models for persistent storage in MongoDB.
Supports advanced chatbot features including multi-modal content,
analytics, and comprehensive conversation management.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from app.models.base import BaseDBModel


class ChatConversationStatus:
    """Conversation status constants."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ChatMessageType:
    """Message type constants."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatAttachmentType:
    """Attachment type constants."""
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


class ChatAttachment(BaseDBModel):
    """Chat attachment model for multi-modal support."""
    message_id: str = Field(..., description="ID of the message this attachment belongs to")
    conversation_id: str = Field(..., description="ID of the conversation")
    user_id: str = Field(..., description="ID of the user who uploaded the attachment")
    
    # File information
    file_name: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Type of attachment (image, document, etc.)")
    file_path: str = Field(..., description="Path to stored file")
    file_size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="MIME type of the file")
    
    # Processing information
    processing_status: str = Field(default="pending", description="Processing status")
    extracted_text: Optional[str] = Field(None, description="Extracted text content (for documents/images)")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional file metadata")
    
    class Config:
        collection_name = "chat_attachments"


class ChatMessage(BaseDBModel):
    """Enhanced chat message model with multi-modal support."""
    conversation_id: str = Field(..., description="ID of the conversation")
    user_id: str = Field(..., description="ID of the user")
    
    # Message content
    message_type: str = Field(..., description="Type of message (user, assistant, system)")
    content: str = Field(..., description="Message text content")
    
    # Processing metadata
    tokens_used: Optional[int] = Field(None, description="Number of tokens used for AI processing")
    processing_time: Optional[float] = Field(None, description="Time taken to process message (seconds)")
    model_used: Optional[str] = Field(None, description="AI model used for response")
    
    # Context and state
    context: Dict[str, Any] = Field(default_factory=dict, description="Conversation context at time of message")
    agent_role: Optional[str] = Field(None, description="Agent role used for this message")
    
    # Attachments
    has_attachments: bool = Field(default=False, description="Whether message has attachments")
    attachment_count: int = Field(default=0, description="Number of attachments")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    
    class Config:
        collection_name = "chat_messages"


class ChatConversation(BaseDBModel):
    """Enhanced conversation model with comprehensive metadata."""
    conversation_id: str = Field(..., description="Unique conversation identifier")
    user_id: str = Field(..., description="ID of the user")
    
    # Basic information
    title: str = Field(..., description="Conversation title")
    agent_role: str = Field(..., description="Primary agent role for conversation")
    status: str = Field(default=ChatConversationStatus.ACTIVE, description="Conversation status")
    
    # Timestamps
    last_message_at: Optional[datetime] = Field(None, description="Timestamp of last message")
    archived_at: Optional[datetime] = Field(None, description="Timestamp when archived")
    
    # Statistics
    message_count: int = Field(default=0, description="Total number of messages")
    user_message_count: int = Field(default=0, description="Number of user messages")
    assistant_message_count: int = Field(default=0, description="Number of assistant messages")
    total_tokens_used: int = Field(default=0, description="Total tokens used in conversation")
    
    # Features and capabilities
    has_attachments: bool = Field(default=False, description="Whether conversation has any attachments")
    attachment_count: int = Field(default=0, description="Total number of attachments")
    languages_detected: List[str] = Field(default_factory=list, description="Languages detected in conversation")
    
    # Context and personalization
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences for this conversation")
    conversation_context: Dict[str, Any] = Field(default_factory=dict, description="Persistent conversation context")
    tags: List[str] = Field(default_factory=list, description="Conversation tags for categorization")
    
    # Analytics and insights
    satisfaction_score: Optional[float] = Field(None, description="User satisfaction score (1-5)")
    resolution_status: Optional[str] = Field(None, description="Whether user's issue was resolved")
    escalation_level: int = Field(default=0, description="Number of times conversation was escalated")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional conversation metadata")
    
    class Config:
        collection_name = "chat_conversations"


class ChatUserPreferences(BaseDBModel):
    """User preferences for chat experience."""
    user_id: str = Field(..., description="ID of the user")
    
    # Communication preferences
    preferred_language: str = Field(default="en", description="Preferred language code")
    response_style: str = Field(default="professional", description="Preferred response style")
    verbosity_level: str = Field(default="normal", description="Preferred response length")
    
    # Notification preferences
    enable_notifications: bool = Field(default=True, description="Enable chat notifications")
    notification_frequency: str = Field(default="immediate", description="Notification frequency")
    
    # Privacy preferences
    data_retention_days: Optional[int] = Field(None, description="Custom data retention period")
    allow_analytics: bool = Field(default=True, description="Allow conversation analytics")
    allow_ai_learning: bool = Field(default=True, description="Allow AI to learn from conversations")
    
    # Feature preferences
    enable_voice_input: bool = Field(default=False, description="Enable voice input")
    enable_file_uploads: bool = Field(default=True, description="Enable file uploads")
    auto_archive_days: int = Field(default=30, description="Auto-archive conversations after N days")
    
    # Context preferences
    remember_context: bool = Field(default=True, description="Remember conversation context")
    cross_conversation_context: bool = Field(default=False, description="Share context across conversations")
    
    # Accessibility preferences
    high_contrast_mode: bool = Field(default=False, description="Enable high contrast mode")
    large_text_mode: bool = Field(default=False, description="Enable large text mode")
    screen_reader_mode: bool = Field(default=False, description="Enable screen reader optimizations")
    
    class Config:
        collection_name = "chat_user_preferences"


class ChatAnalytics(BaseDBModel):
    """Analytics data for chat conversations and user behavior."""
    user_id: str = Field(..., description="ID of the user")
    period_type: str = Field(..., description="Analytics period (daily, weekly, monthly)")
    period_start: datetime = Field(..., description="Start of analytics period")
    period_end: datetime = Field(..., description="End of analytics period")
    
    # Conversation metrics
    total_conversations: int = Field(default=0)
    active_conversations: int = Field(default=0)
    completed_conversations: int = Field(default=0)
    average_conversation_length: float = Field(default=0.0)
    
    # Message metrics
    total_messages: int = Field(default=0)
    user_messages: int = Field(default=0)
    assistant_messages: int = Field(default=0)
    average_response_time: float = Field(default=0.0)
    
    # Usage patterns
    most_used_agent_roles: List[str] = Field(default_factory=list)
    peak_usage_hours: List[int] = Field(default_factory=list)
    common_topics: List[str] = Field(default_factory=list)
    
    # Performance metrics
    total_tokens_used: int = Field(default=0)
    average_tokens_per_message: float = Field(default=0.0)
    total_processing_time: float = Field(default=0.0)
    
    # Satisfaction metrics
    average_satisfaction_score: Optional[float] = Field(None)
    resolution_rate: Optional[float] = Field(None)
    escalation_rate: Optional[float] = Field(None)
    
    # Feature usage
    attachment_usage_count: int = Field(default=0)
    voice_usage_count: int = Field(default=0)
    export_usage_count: int = Field(default=0)
    
    # Insights and recommendations
    insights: Dict[str, Any] = Field(default_factory=dict, description="Generated insights")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    class Config:
        collection_name = "chat_analytics"


class ChatAuditLog(BaseDBModel):
    """Audit log for chat actions and data access."""
    user_id: str = Field(..., description="ID of the user who performed the action")
    action_type: str = Field(..., description="Type of action performed")
    resource_type: str = Field(..., description="Type of resource affected")
    resource_id: str = Field(..., description="ID of the resource affected")
    
    # Action details
    action_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed action information")
    ip_address: Optional[str] = Field(None, description="IP address of the user")
    user_agent: Optional[str] = Field(None, description="User agent string")
    
    # Context
    conversation_id: Optional[str] = Field(None, description="Related conversation ID")
    session_id: Optional[str] = Field(None, description="User session ID")
    
    # Compliance
    data_subject_id: Optional[str] = Field(None, description="ID of data subject (for GDPR)")
    legal_basis: Optional[str] = Field(None, description="Legal basis for data processing")
    retention_period: Optional[int] = Field(None, description="Data retention period in days")
    
    class Config:
        collection_name = "chat_audit_logs"


# Pydantic models for API requests/responses (not stored in DB)

class ConversationCreateRequest(BaseModel):
    """Request model for creating a new conversation."""
    title: str = Field(..., description="Conversation title")
    agent_role: str = Field(..., description="Agent role to use")
    initial_context: Optional[Dict[str, Any]] = Field(None, description="Initial conversation context")


class MessageCreateRequest(BaseModel):
    """Request model for creating a new message."""
    content: str = Field(..., description="Message content")
    attachments: Optional[List[str]] = Field(None, description="List of attachment IDs")
    context: Optional[Dict[str, Any]] = Field(None, description="Message context")


class ConversationSearchRequest(BaseModel):
    """Request model for searching conversations."""
    query: Optional[str] = Field(None, description="Search query")
    agent_role: Optional[str] = Field(None, description="Filter by agent role")
    date_from: Optional[datetime] = Field(None, description="Start date filter")
    date_to: Optional[datetime] = Field(None, description="End date filter")
    status: Optional[str] = Field(None, description="Filter by conversation status")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(default=20, description="Maximum number of results")
    offset: int = Field(default=0, description="Offset for pagination")


class ConversationExportRequest(BaseModel):
    """Request model for exporting conversation data."""
    format: str = Field(default="json", description="Export format (json, csv, pdf)")
    include_metadata: bool = Field(default=True, description="Include conversation metadata")
    include_attachments: bool = Field(default=False, description="Include attachment data")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range for export")


class AnalyticsRequest(BaseModel):
    """Request model for analytics data."""
    period_type: str = Field(..., description="Analytics period (daily, weekly, monthly)")
    date_from: datetime = Field(..., description="Start date")
    date_to: datetime = Field(..., description="End date")
    include_insights: bool = Field(default=True, description="Include AI-generated insights")
