from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import logging

from ..auth.dependencies import has_role

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    item_id: str
    prediction_horizon_days: int = 30
    confidence_interval: float = 0.95
    include_external_factors: bool = True

class BatchPredictionRequest(BaseModel):
    item_ids: List[str]
    prediction_horizon_days: int = 30
    confidence_interval: float = 0.95
    include_external_factors: bool = True

class ItemAnalysisRequest(BaseModel):
    item_id: str
    comparison_items: Optional[List[str]] = None
    analysis_period_days: int = 90
    include_recommendations: bool = True

class CategoryPredictionRequest(BaseModel):
    category: str
    prediction_horizon_days: int = 30
    confidence_interval: float = 0.95

# Global instances (in production, consider dependency injection)
def get_prediction_service():
    """Get the seasonal prediction service instance"""
    from ..services.simplified_seasonal_prediction_service import get_simplified_seasonal_prediction_service
    return get_simplified_seasonal_prediction_service()

@router.get("/health")
async def prediction_health_check():
    """Health check for prediction services"""
    try:
        service = get_prediction_service()
        status = service.get_service_status()
        
        if status["status"] == "unavailable":
            raise HTTPException(
                status_code=503, 
                detail="Seasonal inventory prediction module not available. Please check installation."
            )
        
        return {
            "status": "healthy",
            "service": "seasonal-inventory-predictions",
            "service_status": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Prediction service unavailable")

@router.post("/item/predict")
async def predict_item_demand(
    request: PredictionRequest,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Analyst"]))
) -> Dict[str, Any]:
    """
    Predict demand for a specific item using Prophet forecasting.
    
    Returns future demand predictions with confidence intervals and trends.
    """
    try:
        logger.info(f"Predicting demand for item {request.item_id}")
        
        service = get_prediction_service()
        result = await service.predict_item_demand(
            item_id=request.item_id,
            horizon_days=request.prediction_horizon_days,
            confidence_interval=request.confidence_interval,
            include_external_factors=request.include_external_factors
        )
        
        if result["status"] != "success":
            raise HTTPException(
                status_code=404 if result["status"] == "no_data" else 500,
                detail=result["message"]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting item demand: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/items/batch-predict")
async def predict_multiple_items(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Analyst"]))
) -> Dict[str, Any]:
    """
    Predict demand for multiple items in batch.
    
    For large batches, processing happens in background.
    """
    try:
        logger.info(f"Batch prediction for {len(request.item_ids)} items")
        
        service = get_prediction_service()
        
        if len(request.item_ids) > 50:
            # For large batches, process in background
            task_id = f"batch_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            background_tasks.add_task(
                _process_batch_predictions,
                request.item_ids,
                request.prediction_horizon_days,
                request.confidence_interval,
                request.include_external_factors,
                task_id
            )
            
            return {
                "status": "processing",
                "task_id": task_id,
                "item_count": len(request.item_ids),
                "estimated_completion": (datetime.now() + timedelta(minutes=len(request.item_ids) * 2)).isoformat(),
                "message": "Large batch processing in background. Use task_id to check status."
            }
        
        # For smaller batches, process immediately
        result = await service.predict_multiple_items(
            item_ids=request.item_ids,
            horizon_days=request.prediction_horizon_days,
            confidence_interval=request.confidence_interval,
            include_external_factors=request.include_external_factors
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.post("/item/analyze")
async def analyze_item_patterns(
    request: ItemAnalysisRequest,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Analyst"]))
) -> Dict[str, Any]:
    """
    Perform detailed analysis of item demand patterns and relationships.
    
    Includes seasonality analysis, trend detection, and item-to-item comparisons.
    """
    try:
        logger.info(f"Analyzing patterns for item {request.item_id}")
        
        service = get_prediction_service()
        result = await service.analyze_item_patterns(
            item_id=request.item_id,
            comparison_items=request.comparison_items,
            analysis_period_days=request.analysis_period_days
        )
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing item patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/category/{category}/predict")
async def predict_category_demand(
    category: str,
    prediction_horizon_days: int = Query(30, description="Number of days to predict"),
    confidence_interval: float = Query(0.95, description="Confidence interval for predictions"),
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Analyst"]))
) -> Dict[str, Any]:
    """
    Predict demand for all items in a category.
    
    Aggregates predictions across all items in the specified category.
    """
    try:
        logger.info(f"Predicting demand for category {category}")
        
        service = get_prediction_service()
        result = await service.get_category_predictions(
            category=category,
            horizon_days=prediction_horizon_days,
            confidence_interval=confidence_interval
        )
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting category demand: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Category prediction failed: {str(e)}")

@router.get("/recommendations/inventory")
async def get_inventory_recommendations(
    days_ahead: int = Query(30, description="Days ahead for recommendations"),
    min_confidence: float = Query(0.8, description="Minimum confidence for recommendations"),
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Analyst"]))
) -> Dict[str, Any]:
    """
    Get inventory management recommendations based on predictions.
    
    Returns suggested actions for restocking, reducing inventory, etc.
    """
    try:
        logger.info(f"Generating inventory recommendations for {days_ahead} days ahead")
        
        service = get_prediction_service()
        result = await service.get_inventory_recommendations(
            days_ahead=days_ahead,
            min_confidence=min_confidence
        )
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")

@router.get("/models/status")
async def get_model_status(
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Analyst"]))
) -> Dict[str, Any]:
    """
    Get status of trained models and their performance metrics.
    """
    try:
        service = get_prediction_service()
        status = service.get_service_status()
        
        return {
            "service_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/models/retrain")
async def retrain_models(
    background_tasks: BackgroundTasks,
    item_ids: Optional[List[str]] = None,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Trigger model retraining for specific items or all items.
    """
    try:
        task_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        background_tasks.add_task(
            _retrain_models_task,
            item_ids,
            task_id
        )
        
        return {
            "status": "started",
            "task_id": task_id,
            "item_count": len(item_ids) if item_ids else "all",
            "estimated_completion": (datetime.now() + timedelta(hours=2)).isoformat(),
            "message": "Model retraining started in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting model retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed to start: {str(e)}")

# Background task functions
async def _process_batch_predictions(item_ids, horizon_days, confidence_interval, include_external, task_id):
    """Background task for processing large batch predictions"""
    logger.info(f"Starting batch prediction task {task_id}")
    # Implementation would store results in database or cache
    # For now, just log the progress
    logger.info(f"Completed batch prediction task {task_id}")

async def _retrain_models_task(item_ids, task_id):
    """Background task for model retraining"""
    logger.info(f"Starting model retraining task {task_id}")
    # Implementation would retrain models and update the forecaster
    logger.info(f"Completed model retraining task {task_id}")
