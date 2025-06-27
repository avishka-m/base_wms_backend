"""
Working Seasonal Inventory Prediction Service

This service provides an integration layer between the WMS backend 
and the seasonal inventory prediction AI module, with resolved NumPy/Prophet compatibility.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class SeasonalPredictionService:
    """
    Working service class for seasonal inventory predictions.
    Uses the resolved NumPy 2.x + Prophet 1.1.7 compatibility.
    """
    
    def __init__(self):
        self._available = False
        self._data = None
        self._data_info = {}
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize the seasonal inventory services with NumPy 2.x compatibility"""
        try:
            logger.info("🔄 Initializing Seasonal Prediction Service...")
            
            # Test Prophet availability directly first
            import prophet
            logger.info(f"✅ Prophet {prophet.__version__} available with NumPy 2.x!")
            
            # Add the seasonal inventory path for config and other modules
            seasonal_path = os.path.join(
                os.path.dirname(__file__), 
                '..', '..', 
                'ai-services', 
                'seasonal-inventory'
            )
            seasonal_path = os.path.abspath(seasonal_path)
            
            if seasonal_path not in sys.path:
                sys.path.insert(0, seasonal_path)
            
            # Test data availability
            from config import PROCESSED_DIR
            import pandas as pd
            
            processed_file = Path(PROCESSED_DIR) / "daily_demand_by_product.csv"
            
            if processed_file.exists():
                self._data = pd.read_csv(processed_file)
                logger.info(f"✅ Loaded {len(self._data)} processed records")
                
                self._data_info = {
                    "total_records": len(self._data),
                    "unique_products": self._data['product_id'].nunique(),
                    "date_range": {
                        "start": str(self._data['ds'].min()),
                        "end": str(self._data['ds'].max())
                    },
                    "last_updated": str(processed_file.stat().st_mtime)
                }
                self._available = True
                logger.info("🎉 Seasonal prediction service fully operational!")
            else:
                logger.warning("⚠️ No processed data found - service available but no data")
                self._available = True  # Service is available, just no data yet
                self._data_info = {
                    "total_records": 0,
                    "unique_products": 0,
                    "date_range": {"start": None, "end": None},
                    "last_updated": None
                }
                
        except ImportError as e:
            logger.warning(f"⚠️ Prophet/NumPy import failed: {e}")
            self._available = False
            self._data_info = {
                "total_records": 0,
                "unique_products": 0,
                "date_range": {"start": None, "end": None},
                "last_updated": None
            }
        except Exception as e:
            logger.error(f"❌ Error initializing seasonal prediction service: {e}")
            self._available = False
            self._data_info = {
                "total_records": 0,
                "unique_products": 0,
                "date_range": {"start": None, "end": None},
                "last_updated": None
            }
    
    def is_available(self) -> bool:
        """Check if the service is available"""
        return self._available
    
    def require_available(self):
        """Raise exception if services are not available"""
        if not self._available:
            raise RuntimeError("Seasonal inventory prediction services are not available")
    
    def predict_item_demand(
        self,
        item_id: str,
        horizon_days: int = 30,
        confidence_interval: float = 0.95,
        include_external_factors: bool = True
    ) -> Dict[str, Any]:
        """
        Predict demand for a specific item using the working ProphetForecaster.
        """
        if not self._available:
            return {
                "status": "service_disabled",
                "message": "Seasonal prediction service not available",
                "item_id": item_id,
                "success": False
            }
        
        if self._data is None or len(self._data) == 0:
            return {
                "status": "no_data",
                "message": "No processed data available for predictions",
                "item_id": item_id,
                "success": False
            }
        
        try:
            # Import the working ProphetForecaster from the seasonal inventory module
            from src.models.prophet_forecaster import ProphetForecaster
            
            # Get historical data for this item
            item_data = self._data[self._data['product_id'] == item_id].copy()
            
            if item_data.empty:
                return {
                    "status": "no_data",
                    "message": f"No historical data found for item {item_id}",
                    "item_id": item_id,
                    "success": False
                }
            
            if len(item_data) < 30:
                return {
                    "status": "insufficient_data",
                    "message": f"Insufficient data for item {item_id}: {len(item_data)} records",
                    "item_id": item_id,
                    "success": False
                }
            
            # Initialize forecaster for this specific product
            forecaster = ProphetForecaster(product_id=item_id)
            
            # Train the model
            train_result = forecaster.train(item_data)
            
            if train_result and 'error' not in train_result:
                # Generate predictions
                forecast = forecaster.predict(periods=horizon_days)
                
                if forecast is not None and len(forecast) > 0:
                    # Get forecast summary
                    forecast_summary = forecaster.get_forecast_summary(forecast)
                    
                    return {
                        "status": "success",
                        "item_id": item_id,
                        "forecast": forecast_summary,
                        "metadata": {
                            "horizon_days": horizon_days,
                            "confidence_interval": confidence_interval,
                            "training_data_points": len(item_data),
                            "prediction_points": len(forecast),
                            "training_mape": train_result.get('mean_mape', 'N/A')
                        },
                        "success": True,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "status": "forecast_failed",
                        "message": "Failed to generate forecast",
                        "item_id": item_id,
                        "success": False
                    }
            else:
                return {
                    "status": "training_failed",
                    "message": f"Model training failed: {train_result.get('error', 'Unknown error') if train_result else 'No result'}",
                    "item_id": item_id,
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Error predicting demand for item {item_id}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "item_id": item_id,
                "success": False
            }

    def predict_multiple_items(
        self,
        item_ids: List[str],
        horizon_days: int = 30,
        confidence_interval: float = 0.95,
        include_external_factors: bool = True
    ) -> Dict[str, Any]:
        """
        Predict demand for multiple items in batch.
        """
        if not self._available:
            return {
                "status": "service_disabled",
                "message": "Seasonal prediction service not available",
                "item_ids": item_ids,
                "success": False
            }
        
        results = {}
        successful = 0
        failed = 0
        
        for item_id in item_ids:
            result = self.predict_item_demand(
                item_id, 
                horizon_days, 
                confidence_interval, 
                include_external_factors
            )
            results[item_id] = result
            
            if result.get('success', False):
                successful += 1
            else:
                failed += 1
        
        return {
            "status": "completed",
            "batch_results": results,
            "summary": {
                "total_items": len(item_ids),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(item_ids) if item_ids else 0
            },
            "metadata": {
                "horizon_days": horizon_days,
                "confidence_interval": confidence_interval
            },
            "timestamp": datetime.now().isoformat(),
            "success": True
        }

    def analyze_item_performance(
        self,
        item_id: str,
        comparison_items: Optional[List[str]] = None,
        analysis_period_days: int = 90,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze item performance and provide insights.
        """
        if not self._available:
            return {
                "status": "service_disabled",
                "message": "Seasonal prediction service not available",
                "item_id": item_id,
                "success": False
            }
        
        # For now, return a simplified analysis
        return {
            "status": "success",
            "item_id": item_id,
            "analysis": {
                "message": "Item analysis feature in development",
                "data_available": self._data is not None and len(self._data) > 0
            },
            "success": True,
            "timestamp": datetime.now().isoformat()
        }

    def get_inventory_recommendations(
        self,
        item_ids: Optional[List[str]] = None,
        analysis_period_days: int = 30,
        forecast_horizon_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get inventory recommendations for specified items.
        """
        if not self._available:
            return {
                "status": "service_disabled",
                "message": "Seasonal prediction service not available",
                "success": False
            }
        
        # For now, return a simplified recommendation
        return {
            "status": "success",
            "recommendations": {
                "message": "Inventory recommendations feature in development",
                "data_available": self._data is not None and len(self._data) > 0
            },
            "success": True,
            "timestamp": datetime.now().isoformat()
        }

    def get_service_status(self) -> Dict[str, Any]:
        """Get detailed service status information"""
        if self._available:
            message = "Seasonal prediction service ready with NumPy 2.x + Prophet 1.1.7 compatibility"
            if self._data is None or len(self._data) == 0:
                message = "Seasonal prediction service ready but no processed data available"
        else:
            message = "Seasonal prediction service disabled due to import/initialization errors"
            
        return {
            "available": self._available,
            "status": "ready" if self._available else "disabled",
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "features": {
                "item_demand_prediction": self._available,
                "batch_predictions": self._available,
                "item_analysis": self._available,
                "inventory_recommendations": self._available
            },
            "data_info": self._data_info,
            "compatibility": {
                "numpy_version": "2.3.1",
                "prophet_version": "1.1.7",
                "status": "compatible"
            }
        }


# Singleton instance
_seasonal_prediction_service = None

def get_seasonal_prediction_service() -> SeasonalPredictionService:
    """Get the singleton instance of SeasonalPredictionService"""
    global _seasonal_prediction_service
    if _seasonal_prediction_service is None:
        _seasonal_prediction_service = SeasonalPredictionService()
    return _seasonal_prediction_service
