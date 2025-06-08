from datetime import datetime
from fastapi import APIRouter

from chatbot.models.schemas import HealthCheckResponse

router = APIRouter(tags=["health"])

@router.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    } 