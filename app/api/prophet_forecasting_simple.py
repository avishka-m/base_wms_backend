#apis need to for frontend

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from ..auth.dependencies import get_current_active_user, has_role
from ..services.prophet_forecasting_service import get_prophet_forecasting_service

router = APIRouter()
logger = logging.getLogger(__name__)

# Simple request models for frontend
class ForecastRequest(BaseModel):

    product_id: str = Field(..., description="Product ID to forecast")
    days: int = Field(30, ge=1, le=365, description="Forecast horizon (1-365 days)")
    confidence: float = Field(0.95, ge=0.80, le=0.99, description="Confidence level")

class RecommendationRequest(BaseModel):
    
    product_ids: List[str] = Field(..., description="List of product IDs")
    days_ahead: int = Field(30, ge=7, le=90, description="Planning horizon (7-90 days)")


@router.get("/health")
async def health_check():
    try:
        service = get_prophet_forecasting_service()
        status = service.get_service_status()
        
        return {
            "status": "healthy" if status["status"] == "ready" else "degraded",
            "service": "prophet-forecasting",
            "details": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "prophet-forecasting", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/products")
async def get_available_products():
    
    try:
        service = get_prophet_forecasting_service()
        result = service.get_available_products()
        
        if result["status"] == "success":
            return {
                "status": "success",
                "products": result["products"],
                "total_count": len(result["products"]),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Failed to get products"))
            
    except Exception as e:
        logger.error(f"Error getting products: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get products: {str(e)}")

@router.post("/forecast")
async def generate_forecast(
    request: ForecastRequest,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Analyst"]))
):

    try:
        service = get_prophet_forecasting_service()
        
        # Generate forecast
        result = service.generate_forecast(
            product_id=request.product_id,
            horizon_days=request.days,
            confidence_interval=request.confidence
        )
        
        if result["status"] == "success":
            return {
                "status": "success",
                "forecast": result["forecast"],
                "metadata": result.get("metadata", {}),
                "requested_by": current_user.get("username", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("message", "Forecast generation failed"))
            
    except Exception as e:
        logger.error(f"Forecast error for {request.product_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

@router.post("/recommendations")
async def get_inventory_recommendations(
    request: RecommendationRequest,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Analyst"]))
):
    
    try:
        service = get_prophet_forecasting_service()
        
        # Get recommendations
        result = service.get_inventory_recommendations(
            product_ids=request.product_ids,
            planning_horizon_days=request.days_ahead
        )
        
        if result["status"] == "success":
            return {
                "status": "success",
                "recommendations": result["recommendations"],
                "summary": result.get("summary", {}),
                "requested_by": current_user.get("username", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("message", "Recommendations failed"))
            
    except Exception as e:
        logger.error(f"Recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")

@router.get("/status")
async def get_service_status():
    
    try:
        service = get_prophet_forecasting_service()
        
        # Get basic status
        service_status = service.get_service_status()
        model_status = service.get_models_status()
        
        return {
            "status": "success",
            "service": service_status,
            "models": {
                "total_trained": model_status.get("total_models", 0),
                "ready_for_forecast": model_status.get("ready_models", 0),
                "last_updated": model_status.get("last_training_time", "unknown")
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


