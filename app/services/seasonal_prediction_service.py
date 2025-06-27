"""
Seasonal Inventory Prediction Service

This service provides an integration layer between the WMS backend 
and the seasonal inventory prediction AI module.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class SeasonalPredictionService:
    """
    Service class to integrate seasonal inventory predictions with WMS backend.
    
    Provides methods for:
    - Item demand forecasting
    - Batch predictions
    - Inventory recommendations
    - Model management
    """
    
    def __init__(self):
        self._forecaster = None
        self._analysis_service = None
        self._data_orchestrator = None
        self._available = False
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize the seasonal inventory services"""
        try:
            import sys
            import os
            seasonal_path = os.path.join(
                os.path.dirname(__file__), 
                '..', '..', 
                'ai-services', 
                'seasonal-inventory'
            )
            if seasonal_path not in sys.path:
                sys.path.append(seasonal_path)
            
            from src.models.prophet_forecaster import ProphetForecaster
            from src.item_analysis_service import ItemAnalysisService
            from src.data_orchestrator import SeasonalDataOrchestrator
            from config import PROPHET_CONFIG
            
            self._forecaster = ProphetForecaster(config=PROPHET_CONFIG)
            self._analysis_service = ItemAnalysisService()
            self._data_orchestrator = SeasonalDataOrchestrator()
            self._available = True
            
            logger.info("Seasonal inventory prediction services initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Seasonal inventory services not available: {e}")
            self._available = False
        except Exception as e:
            logger.error(f"Error initializing seasonal inventory services: {e}")
            self._available = False
    
    @property
    def is_available(self) -> bool:
        """Check if seasonal inventory services are available"""
        return self._available
    
    def require_available(self):
        """Raise exception if services are not available"""
        if not self._available:
            raise RuntimeError("Seasonal inventory prediction services are not available")
    
    async def predict_item_demand(
        self,
        item_id: str,
        horizon_days: int = 30,
        confidence_interval: float = 0.95,
        include_external_factors: bool = True
    ) -> Dict[str, Any]:
        """
        Predict demand for a specific item.
        
        Args:
            item_id: The item identifier
            horizon_days: Number of days to predict ahead
            confidence_interval: Confidence level for predictions
            include_external_factors: Whether to include external data
            
        Returns:
            Dictionary containing predictions and analysis
        """
        self.require_available()
        
        try:
            # Get historical data
            data = await self._data_orchestrator.create_product_dataset(
                product_id=item_id,
                include_external=include_external_factors
            )
            
            if data is None or data.empty:
                return {
                    "status": "no_data",
                    "message": f"No historical data found for item {item_id}",
                    "item_id": item_id
                }
            
            # Make predictions
            predictions = self._forecaster.predict_product_demand(
                product_id=item_id,
                data=data,
                horizon_days=horizon_days,
                confidence_interval=confidence_interval
            )
            
            # Calculate summary statistics
            summary = self._calculate_prediction_summary(predictions)
            
            return {
                "status": "success",
                "item_id": item_id,
                "horizon_days": horizon_days,
                "confidence_interval": confidence_interval,
                "predictions": predictions.to_dict(orient='records'),
                "summary": summary,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting demand for item {item_id}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "item_id": item_id
            }
    
    async def predict_multiple_items(
        self,
        item_ids: List[str],
        horizon_days: int = 30,
        confidence_interval: float = 0.95,
        include_external_factors: bool = True
    ) -> Dict[str, Any]:
        """
        Predict demand for multiple items.
        
        Args:
            item_ids: List of item identifiers
            horizon_days: Number of days to predict ahead
            confidence_interval: Confidence level for predictions
            include_external_factors: Whether to include external data
            
        Returns:
            Dictionary containing results for all items
        """
        self.require_available()
        
        results = {}
        successful_predictions = 0
        
        # Process each item
        for item_id in item_ids:
            try:
                result = await self.predict_item_demand(
                    item_id=item_id,
                    horizon_days=horizon_days,
                    confidence_interval=confidence_interval,
                    include_external_factors=include_external_factors
                )
                
                results[item_id] = result
                if result["status"] == "success":
                    successful_predictions += 1
                    
            except Exception as e:
                logger.error(f"Error processing item {item_id}: {e}")
                results[item_id] = {
                    "status": "error",
                    "message": str(e),
                    "item_id": item_id
                }
        
        return {
            "status": "completed",
            "total_items": len(item_ids),
            "successful_predictions": successful_predictions,
            "results": results,
            "generated_at": datetime.now().isoformat()
        }
    
    async def analyze_item_patterns(
        self,
        item_id: str,
        comparison_items: Optional[List[str]] = None,
        analysis_period_days: int = 90
    ) -> Dict[str, Any]:
        """
        Perform detailed analysis of item demand patterns.
        
        Args:
            item_id: The item to analyze
            comparison_items: Items to compare against
            analysis_period_days: Period for analysis
            
        Returns:
            Dictionary containing detailed analysis results
        """
        self.require_available()
        
        try:
            analysis_result = await self._analysis_service.analyze_item_patterns(
                item_id=item_id,
                comparison_items=comparison_items,
                analysis_period_days=analysis_period_days,
                include_recommendations=True
            )
            
            return {
                "status": "success",
                "item_id": item_id,
                "analysis_period_days": analysis_period_days,
                "analysis": analysis_result,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing item patterns for {item_id}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "item_id": item_id
            }
    
    async def get_inventory_recommendations(
        self,
        days_ahead: int = 30,
        min_confidence: float = 0.8
    ) -> Dict[str, Any]:
        """
        Get inventory management recommendations.
        
        Args:
            days_ahead: Number of days to look ahead
            min_confidence: Minimum confidence for recommendations
            
        Returns:
            Dictionary containing recommendations
        """
        self.require_available()
        
        try:
            recommendations = await self._analysis_service.generate_inventory_recommendations(
                days_ahead=days_ahead,
                min_confidence=min_confidence
            )
            
            return {
                "status": "success",
                "days_ahead": days_ahead,
                "min_confidence": min_confidence,
                "recommendations": recommendations,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating inventory recommendations: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_category_predictions(
        self,
        category: str,
        horizon_days: int = 30,
        confidence_interval: float = 0.95
    ) -> Dict[str, Any]:
        """
        Get predictions for all items in a category.
        
        Args:
            category: Category to analyze
            horizon_days: Number of days to predict
            confidence_interval: Confidence level
            
        Returns:
            Dictionary containing category predictions
        """
        self.require_available()
        
        try:
            category_predictions = await self._analysis_service.analyze_category_demand(
                category=category,
                prediction_horizon_days=horizon_days,
                confidence_interval=confidence_interval
            )
            
            return {
                "status": "success",
                "category": category,
                "horizon_days": horizon_days,
                "confidence_interval": confidence_interval,
                "predictions": category_predictions,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting category {category} demand: {e}")
            return {
                "status": "error",
                "message": str(e),
                "category": category
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get status of the prediction services.
        
        Returns:
            Dictionary containing service status information
        """
        try:
            if not self._available:
                return {
                    "status": "unavailable",
                    "message": "Seasonal inventory services not initialized",
                    "timestamp": datetime.now().isoformat()
                }
            
            status = {
                "status": "available",
                "services": {
                    "forecaster": bool(self._forecaster),
                    "analysis_service": bool(self._analysis_service),
                    "data_orchestrator": bool(self._data_orchestrator)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Add model status if available
            if self._forecaster:
                try:
                    model_status = self._forecaster.get_model_status()
                    status["model_status"] = model_status
                except Exception as e:
                    status["model_status"] = {"error": str(e)}
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_prediction_summary(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for predictions"""
        try:
            if predictions.empty:
                return {"error": "No predictions available"}
            
            summary = {
                "total_predicted_demand": float(predictions['yhat'].sum()),
                "average_daily_demand": float(predictions['yhat'].mean()),
                "peak_demand_value": float(predictions['yhat'].max()),
                "min_demand_value": float(predictions['yhat'].min()),
                "demand_variance": float(predictions['yhat'].var()),
                "prediction_period": {
                    "start_date": predictions['ds'].min().isoformat(),
                    "end_date": predictions['ds'].max().isoformat(),
                    "total_days": len(predictions)
                }
            }
            
            # Find peak and low demand dates
            peak_idx = predictions['yhat'].idxmax()
            low_idx = predictions['yhat'].idxmin()
            
            summary["peak_demand_date"] = predictions.loc[peak_idx, 'ds'].isoformat()
            summary["low_demand_date"] = predictions.loc[low_idx, 'ds'].isoformat()
            
            # Calculate trend
            first_week_avg = predictions['yhat'].head(7).mean()
            last_week_avg = predictions['yhat'].tail(7).mean()
            trend_change = ((last_week_avg - first_week_avg) / first_week_avg) * 100
            
            summary["trend_analysis"] = {
                "direction": "increasing" if trend_change > 5 else "decreasing" if trend_change < -5 else "stable",
                "change_percentage": float(trend_change),
                "first_week_avg": float(first_week_avg),
                "last_week_avg": float(last_week_avg)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating prediction summary: {e}")
            return {"error": str(e)}

# Global service instance
_seasonal_prediction_service = None

def get_seasonal_prediction_service() -> SeasonalPredictionService:
    """Get the global seasonal prediction service instance"""
    global _seasonal_prediction_service
    if _seasonal_prediction_service is None:
        _seasonal_prediction_service = SeasonalPredictionService()
    return _seasonal_prediction_service
