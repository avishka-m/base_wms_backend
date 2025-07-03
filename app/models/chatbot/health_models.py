"""
Health check models for the WMS Chatbot API.
"""

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Status of the service")
    version: str = Field(..., description="Version of the API")
    timestamp: str = Field(..., description="Timestamp of the health check")
