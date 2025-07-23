###prophet forcaster and endpoints

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import asyncio
import json

logger = logging.getLogger(__name__)

class ProphetForecastingService:
    async def predict_category_demand(
        self,
        category: str,
        horizon_days: int = 30,
        
    ) -> Dict[str, Any]:
        """Predict demand for a category for a given number of days (7, 30, 90, 365) from tomorrow using the category-level model."""
        try:
            if not self._available:
                return {
                    "status": "error",
                    "message": f"Service unavailable: {self._initialization_error}"
                }
            if category not in self._category_hyperparams:
                return {
                    "status": "error",
                    "message": f"No model available for category '{category}'"
                }
            if horizon_days not in [7, 30, 90, 365]:
                return {
                    "status": "error",
                    "message": "horizon_days must be one of: 7, 30, 90, 365"
                }
            # Start from tomorrow
            start_dt = datetime.now() + timedelta(days=1)
            # Generate future dates
            future_dates = pd.date_range(start=start_dt.date(), periods=horizon_days, freq='D')
            # Use ProphetCategoryPredictor to load and predict
            predictor = self._prophet_category_predictor_class(category)
            if not predictor.load_model():
                return {
                    "status": "error",
                    "message": f"Failed to load Prophet model for category {category}"
                }
            confidence_interval = 0.8    
            forecast_df = predictor.predict(future_dates.tolist())
            if forecast_df is None or forecast_df.empty:
                return {
                    "status": "error",
                    "message": f"No forecast generated for category {category}"
                }
            # Prepare predictions
            predictions = []
            total_demand = 0
            for _, row in forecast_df.iterrows():
                pred_date = row['ds']
                if isinstance(pred_date, str):
                    pred_date = pd.to_datetime(pred_date)
                category_demand = max(0, row['yhat'])
                lower = max(0, row['yhat_lower'])
                upper = max(0, row['yhat_upper'])
                predictions.append({
                    "date": pred_date.strftime('%Y-%m-%d'),
                    "predicted_demand": round(category_demand, 2),
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2),
                     "confidence": confidence_interval,
                    "category": category
                })
                total_demand += category_demand
            avg_daily_demand = total_demand / len(predictions) if predictions else 0
            peak_demand = max([p["predicted_demand"] for p in predictions]) if predictions else 0
            return {
                "status": "success",
                "category": category,
                "forecast_period": {
                    "start_date": start_dt.strftime('%Y-%m-%d'),
                    "end_date": (start_dt + timedelta(days=horizon_days-1)).strftime('%Y-%m-%d'),
                    "days": horizon_days
                },
                "predictions": predictions,
                "summary": {
                    "total_predicted_demand": round(total_demand, 2),
                    "average_daily_demand": round(avg_daily_demand, 2),
                    "peak_daily_demand": round(peak_demand, 2),
                    "confidence_interval": confidence_interval
                },
                "model_info": {
                    "model_type": "Prophet-Category",
                    "category_model": category,
                    "hyperparameters": self._category_hyperparams.get(category, {}),
                    "model_file": str(self._models_path / f"category_{category}_prophet_model.pkl")
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in category prediction for {category}: {e}")
            return {
                "status": "error",
                "message": f"Category prediction failed: {str(e)}"
            }
    #Service class for Prophet-based demand forecasting with API integration
    
    def __init__(self):
        self._forecaster = None
        self._available = False
        self._initialization_error = None
        self._models_path = None
        
        # NEW: Category-based model support
        self._category_models = {}
        self._category_hyperparams = {}
        self._product_category_mapping = {}
        
        self._initialize_service()
    
    def _initialize_service(self):
        # Initialize the Prophet category-level forecasting service
        try:
            logger.info("Initializing Prophet category-level forecasting service...")
            import sys
            from pathlib import Path
            # Add the project root to Python path
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            ai_services_path = project_root / "ai_services"
            seasonal_inventory_path = ai_services_path / "seasonal_inventory"
            sys.path.insert(0, str(project_root))
            sys.path.insert(0, str(ai_services_path))
            sys.path.insert(0, str(seasonal_inventory_path))
            # Import the ProphetCategoryPredictor class directly
            from ai_services.seasonal_inventory.src.models.prophet_forecaster import ProphetCategoryPredictor
            self._prophet_category_predictor_class = ProphetCategoryPredictor
            # Import config values directly
            config_path = project_root / "ai_services" / "seasonal_inventory" / "config.py"
            if config_path.exists():
                spec = sys.modules.get('ai_services.seasonal_inventory.config')
                if spec is None:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    sys.modules['ai_services.seasonal_inventory.config'] = config_module
                else:
                    config_module = spec
                models_dir = getattr(config_module, 'MODELS_DIR', None)
                processed_dir = getattr(config_module, 'PROCESSED_DIR', None)
            else:
                models_dir = str(project_root / "ai_services" / "seasonal_inventory" / "data" / "models")
                processed_dir = str(project_root / "ai_services" / "seasonal_inventory" / "data" / "processed")
            self._models_path = Path(models_dir)
            self._processed_dir = Path(processed_dir)
            self._load_data()
            self._load_category_configurations()
            self._load_product_category_mapping()
            logger.info("Prophet category-level forecaster initialized successfully")
            self._available = True
        except Exception as e:
            logger.error(f"Failed to initialize Prophet category-level forecasting service: {e}")
            self._initialization_error = str(e)
            self._available = False
    
    def _load_data(self):
        """Load the training data"""
        try:
            data_path = self._processed_dir / 'daily_demand_by_product_modern.csv'
            if data_path.exists():
                self._data = pd.read_csv(data_path)
                # Ensure date column is datetime
                if 'date' in self._data.columns:
                    self._data['date'] = pd.to_datetime(self._data['date'])
                    self._data['ds'] = self._data['date']
                elif 'ds' in self._data.columns:
                    self._data['ds'] = pd.to_datetime(self._data['ds'])
                    self._data['date'] = self._data['ds']
                
                logger.info(f"Loaded data with {len(self._data)} records")
            else:
                logger.error(f"Data file not found: {data_path}")
                self._data = None
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self._data = None
    
    def _load_category_configurations(self):
        """Load optimized hyperparameters for each of the 6 categories"""
        logger.info("Loading category-specific hyperparameters...")
        
        #Replace with your best found parameters
        self._category_hyperparams = {
            "books_media": {
                #  OPTIMIZED: Based on hyperparameter tuning results
                "seasonality_mode": "additive",
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "changepoint_prior_scale": 0.5,
                "seasonality_prior_scale": 10.0,
                "holidays_prior_scale": 10.0,
                "changepoint_range": 0.8,
                # Note: optimal_regressors ['weekend', 'month_end', 'payday', 'summer'] 
                # would need to be implemented separately in model training
            },
            "clothing": {
                #  OPTIMIZED: Based on hyperparameter tuning results
                "seasonality_mode": "additive",
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "changepoint_prior_scale": 1.0,
                "seasonality_prior_scale": 10.0,
                "holidays_prior_scale": 10.0,
                "changepoint_range": 0.8,
                # Note: optimal_regressors ['weekend'] 
                # would need to be implemented separately in model training
            },
            "electronics": {
                # OPTIMIZED: Based on hyperparameter tuning results
                "seasonality_mode": "additive",
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "changepoint_prior_scale": 1.0,
                "seasonality_prior_scale": 10.0,
                "holidays_prior_scale": 10.0,
                "changepoint_range": 0.8,
                # Note: optimal_regressors ['weekend'] 
                # would need to be implemented separately in model training
            },
            "health_beauty": {
                # OPTIMIZED: Based on hyperparameter tuning results
                "seasonality_mode": "additive",
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "changepoint_prior_scale": 0.01,
                "seasonality_prior_scale": 10.0,
                "holidays_prior_scale": 10.0,
                "changepoint_range": 0.8,
                # Note: optimal_regressors ['weekend'] 
                # would need to be implemented separately in model training
            },
            "home_garden": {
                # OPTIMIZED: Based on hyperparameter tuning results
                "seasonality_mode": "additive",
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "changepoint_prior_scale": 1.0,
                "seasonality_prior_scale": 10.0,
                "holidays_prior_scale": 10.0,
                "changepoint_range": 0.8,
                # Note: optimal_regressors ['weekend', 'month_end', 'payday', 'summer'] 
                # would need to be implemented separately in model training
            },
            "sports_outdoors": {
                # OPTIMIZED: Based on hyperparameter tuning results
                "seasonality_mode": "additive",
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 0.1,  #  Weak seasonality (unique!)
                "holidays_prior_scale": 10.0,
                "changepoint_range": 0.8,
                # Note: optimal_regressors ['weekend', 'month_end', 'payday'] 
                # would need to be implemented separately in model training
            }
        }
        
        logger.info(f"Loaded hyperparameters for {len(self._category_hyperparams)} categories")
    
    def _load_product_category_mapping(self):
        """Load product-category mapping from the dataset"""
        try:
            if self._data is not None and 'category' in self._data.columns:
                # Create mapping from the dataset
                product_categories = self._data[['product_id', 'category']].drop_duplicates()
                
                for _, row in product_categories.iterrows():
                    self._product_category_mapping[row['product_id']] = row['category']
                
                logger.info(f"Loaded category mapping for {len(self._product_category_mapping)} products")
                
                # Log category distribution
                category_counts = {}
                for category in self._product_category_mapping.values():
                    category_counts[category] = category_counts.get(category, 0) + 1
                
                logger.info(f"Category distribution: {category_counts}")
                    
        except Exception as e:
            logger.error(f"Error loading product-category mapping: {e}")
            self._product_category_mapping = {}
    
    # All product-level and multi-item prediction methods removed. Only category-level prediction remains.

# Global service instance
_prophet_service = None

def get_prophet_forecasting_service() -> ProphetForecastingService:
    """Get singleton instance of Prophet forecasting service"""
    global _prophet_service
    if _prophet_service is None:
        _prophet_service = ProphetForecastingService()
    return _prophet_service


# test if tis script is workiing or not.
# import asyncio

# if __name__ == "__main__":
#     service = ProphetForecastingService()
#     # Test data
#     test_category = "electronics"  # Use a valid category from your config
#     test_horizon = 7

#     # Test predict_category_demand
#     async def test_predict():
#         result = await service.predict_category_demand(test_category, test_horizon)
#         print("Prediction result:", result)

#     asyncio.run(test_predict())

#     # Optionally, test internal loaders (should not raise errors)
#     service._load_data()
#     service._load_category_configurations()
#     service._load_product_category_mapping()
#     print("Data loaded:", hasattr(service, "_data") and service._data is not None)
#     print("Category hyperparams loaded:", bool(service._category_hyperparams))
#     print("Product-category mapping loaded:", bool(service._product_category_mapping))
