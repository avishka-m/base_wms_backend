# """
# Prophet Forecasting API Routes - Simplified for Frontend Use Only

# ⚠️  SIMPLIFIED VERSION - Only Essential Endpoints for Frontend
# 🔧 Administrative tasks use command-line tools:
#    - Training: simple_batch_train.py
#    - Evaluation: evaluate_models_with_split.py  
#    - Batch Operations: train_all_models.py
   
# 📊 Only 5 Essential Endpoints Remaining:
#    1. GET  /health             - Service health check
#    2. GET  /products           - Available products  
#    3. POST /forecast           - Single product forecast
#    4. POST /recommendations    - Inventory recommendations
#    5. GET  /status            - Basic model status
# """

# from fastapi import APIRouter, HTTPException, Depends
# from typing import Dict, Any, Optional, List
# from datetime import datetime
# from pydantic import BaseModel, Field
# import logging

# from ..auth.dependencies import get_current_active_user, has_role
# from ..services.prophet_forecasting_service import get_prophet_forecasting_service

# router = APIRouter()
# logger = logging.getLogger(__name__)

# # Simple request models for frontend
# class ForecastRequest(BaseModel):
#     """Simple forecast request"""
#     product_id: str = Field(..., description="Product ID to forecast")
#     days: int = Field(30, ge=1, le=365, description="Forecast horizon (1-365 days)")
#     confidence: float = Field(0.95, ge=0.80, le=0.99, description="Confidence level")

# class RecommendationRequest(BaseModel):
#     """Inventory recommendation request"""
#     product_ids: List[str] = Field(..., description="List of product IDs")
#     days_ahead: int = Field(30, ge=7, le=90, description="Planning horizon (7-90 days)")

# # ============================================================================
# # ESSENTIAL ENDPOINTS FOR FRONTEND USE
# # ============================================================================

# @router.get("/health")
# async def health_check():
#     """🩺 Health check for Prophet forecasting service"""
#     try:
#         service = get_prophet_forecasting_service()
#         status = service.get_service_status()
        
#         return {
#             "status": "healthy" if status["status"] == "ready" else "degraded",
#             "service": "prophet-forecasting",
#             "details": status,
#             "timestamp": datetime.now().isoformat()
#         }
#     except Exception as e:
#         logger.error(f"Health check failed: {str(e)}")
#         return {
#             "status": "unhealthy",
#             "service": "prophet-forecasting", 
#             "error": str(e),
#             "timestamp": datetime.now().isoformat()
#         }

# @router.get("/products")
# async def get_available_products():
#     """📦 Get list of products available for forecasting"""
#     try:
#         service = get_prophet_forecasting_service()
#         result = service.get_available_products()
        
#         if result["status"] == "success":
#             return {
#                 "status": "success",
#                 "products": result["products"],
#                 "total_count": len(result["products"]),
#                 "timestamp": datetime.now().isoformat()
#             }
#         else:
#             raise HTTPException(status_code=500, detail=result.get("message", "Failed to get products"))
            
#     except Exception as e:
#         logger.error(f"Error getting products: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to get products: {str(e)}")

# @router.post("/forecast")
# async def generate_forecast(
#     request: ForecastRequest,
#     current_user: Dict[str, Any] = Depends(get_current_active_user)
# ):
#     """📈 Generate Prophet forecast for a single product"""
#     try:
#         service = get_prophet_forecasting_service()
        
#         # Generate forecast
#         result = service.generate_forecast(
#             product_id=request.product_id,
#             horizon_days=request.days,
#             confidence_interval=request.confidence
#         )
        
#         if result["status"] == "success":
#             return {
#                 "status": "success",
#                 "forecast": result["forecast"],
#                 "metadata": result.get("metadata", {}),
#                 "requested_by": current_user.get("username", "unknown"),
#                 "timestamp": datetime.now().isoformat()
#             }
#         else:
#             raise HTTPException(status_code=400, detail=result.get("message", "Forecast generation failed"))
            
#     except Exception as e:
#         logger.error(f"Forecast error for {request.product_id}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

# @router.post("/recommendations")
# async def get_inventory_recommendations(
#     request: RecommendationRequest,
#     current_user: Dict[str, Any] = Depends(has_role(["Manager", "Analyst"]))
# ):
#     """🎯 Get inventory recommendations for multiple products"""
#     try:
#         service = get_prophet_forecasting_service()
        
#         # Get recommendations
#         result = service.get_inventory_recommendations(
#             product_ids=request.product_ids,
#             planning_horizon_days=request.days_ahead
#         )
        
#         if result["status"] == "success":
#             return {
#                 "status": "success",
#                 "recommendations": result["recommendations"],
#                 "summary": result.get("summary", {}),
#                 "requested_by": current_user.get("username", "unknown"),
#                 "timestamp": datetime.now().isoformat()
#             }
#         else:
#             raise HTTPException(status_code=400, detail=result.get("message", "Recommendations failed"))
            
#     except Exception as e:
#         logger.error(f"Recommendations error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")

# @router.get("/status")
# async def get_service_status():
#     """📊 Get basic Prophet service and model status"""
#     try:
#         service = get_prophet_forecasting_service()
        
#         # Get basic status
#         service_status = service.get_service_status()
#         model_status = service.get_models_status()
        
#         return {
#             "status": "success",
#             "service": service_status,
#             "models": {
#                 "total_trained": model_status.get("total_models", 0),
#                 "ready_for_forecast": model_status.get("ready_models", 0),
#                 "last_updated": model_status.get("last_training_time", "unknown")
#             },
#             "timestamp": datetime.now().isoformat()
#         }
        
#     except Exception as e:
#         logger.error(f"Status check error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# # ============================================================================
# # CATEGORY-BASED MODEL ENDPOINTS (NEW)
# # ============================================================================

# @router.get("/categories/status")
# async def get_category_status():
#     """🎯 Get status of the 6-category model system"""
#     try:
#         service = get_prophet_forecasting_service()
#         status = service.get_category_status()
        
#         return {
#             "status": "success",
#             "category_system": status,
#             "timestamp": datetime.now().isoformat()
#         }
        
#     except Exception as e:
#         logger.error(f"Category status error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Category status failed: {str(e)}")

# @router.post("/categories/train")
# async def train_category_models(
#     current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
# ):
#     """🚀 Train all 6 category models with optimized hyperparameters"""
#     try:
#         service = get_prophet_forecasting_service()
        
#         logger.info(f"Category training initiated by user: {current_user.get('username', 'unknown')}")
        
#         # Train category models
#         result = await service.train_category_models()
        
#         if result["status"] == "success":
#             return {
#                 "status": "success",
#                 "message": "Category models trained successfully",
#                 "training_results": result,
#                 "timestamp": datetime.now().isoformat()
#             }
#         else:
#             raise HTTPException(status_code=500, detail=result.get("message", "Training failed"))
            
#     except Exception as e:
#         logger.error(f"Category training error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Category training failed: {str(e)}")

# @router.get("/categories")
# async def get_categories():
#     """📋 Get list of supported categories with their configurations"""
#     try:
#         service = get_prophet_forecasting_service()
        
#         # Get category configurations
#         categories_info = []
        
#         if hasattr(service, '_category_hyperparams'):
#             for category, hyperparams in service._category_hyperparams.items():
#                 # Count products in category
#                 product_count = 0
#                 if hasattr(service, '_product_category_mapping'):
#                     product_count = sum(1 for cat in service._product_category_mapping.values() if cat == category)
                
#                 categories_info.append({
#                     "category": category,
#                     "product_count": product_count,
#                     "hyperparameters": hyperparams,
#                     "display_name": category.replace('_', ' ').title()
#                 })
        
#         return {
#             "status": "success",
#             "categories": categories_info,
#             "total_categories": len(categories_info),
#             "timestamp": datetime.now().isoformat()
#         }
        
#     except Exception as e:
#         logger.error(f"Categories info error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Categories info failed: {str(e)}")

# @router.get("/categories/{category}/products")
# async def get_category_products(category: str):
#     """🏷️ Get products in a specific category"""
#     try:
#         service = get_prophet_forecasting_service()
        
#         if not hasattr(service, '_product_category_mapping'):
#             raise HTTPException(status_code=500, detail="Product-category mapping not available")
        
#         # Get products for this category
#         category_products = [
#             {"product_id": pid, "category": cat}
#             for pid, cat in service._product_category_mapping.items() 
#             if cat == category
#         ]
        
#         if not category_products:
#             raise HTTPException(status_code=404, detail=f"No products found for category: {category}")
        
#         return {
#             "status": "success",
#             "category": category,
#             "products": category_products,
#             "product_count": len(category_products),
#             "timestamp": datetime.now().isoformat()
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Category products error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Category products failed: {str(e)}")

# # ============================================================================
# # ADMINISTRATIVE NOTE
# # ============================================================================
# """
# 🚨 REMOVED ENDPOINTS (Use Command Line Instead):

# ❌ POST /test/forecast          → Use: python -c "from app.services... import test"
# ❌ POST /forecast/batch         → Use: python simple_batch_train.py --products LIST
# ❌ POST /forecast/custom-range  → Merged into main /forecast endpoint
# ❌ POST /models/train           → Use: python simple_batch_train.py --product ID
# ❌ GET  /models/training-status → Use: python check_products.py --status
# ❌ GET  /products/all          → Duplicate of /products (removed)
# ❌ POST /models/evaluate        → Use: python evaluate_models_with_split.py
# ❌ POST /models/retrain-with-split → Use: python evaluate_models_with_split.py --retrain

# 💡 Why This Is Better:
#    • APIs for user-facing features only
#    • CLI tools for admin/development tasks
#    • Simpler codebase and maintenance
#    • Better separation of concerns
#    • No authentication issues for backend tasks
# """
