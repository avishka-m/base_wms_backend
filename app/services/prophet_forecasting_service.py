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
        #Initialize the Prophet forecasting
        try:
            logger.info("Initializing Prophet forecasting service...")
            
            # Import Prophet forecaster
            try:
                # Import with proper absolute path
                import sys
                from pathlib import Path
                
                # Add the project root to Python path
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent.parent
                ai_services_path = project_root / "ai_services"
                seasonal_inventory_path = ai_services_path / "seasonal_inventory"
                
                # Add paths to sys.path
                sys.path.insert(0, str(project_root))
                sys.path.insert(0, str(ai_services_path))
                sys.path.insert(0, str(seasonal_inventory_path))
                
                # Import the ProphetForecaster class directly
                from ai_services.seasonal_inventory.src.models.prophet_forecaster import ProphetForecaster
                
                # Store ProphetForecaster class reference for later use
                self._prophet_forecaster_class = ProphetForecaster
                
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
                    # Fallback to default paths
                    models_dir = str(project_root / "ai_services" / "seasonal_inventory" / "data" / "models")
                    processed_dir = str(project_root / "ai_services" / "seasonal_inventory" / "data" / "processed")
                
                # Initialize forecaster
                self._forecaster = ProphetForecaster()
                self._models_path = Path(models_dir)
                self._processed_dir = Path(processed_dir)
                
                # Load data for service
                self._load_data()
                
                # NEW: Initialize category configurations
                self._load_category_configurations()
                self._load_product_category_mapping()
                
                logger.info("Prophet forecaster initialized successfully")
                self._available = True
                
            except ImportError as e:
                raise ImportError(f"Prophet forecaster not available: {e}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prophet forecasting service: {e}")
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
        
        # 🎯 OPTIMIZED HYPERPARAMETERS - Replace with your best found parameters
        self._category_hyperparams = {
            "books_media": {
                # ✅ OPTIMIZED: Based on hyperparameter tuning results
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
                # ✅ OPTIMIZED: Based on hyperparameter tuning results
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
                # ✅ OPTIMIZED: Based on hyperparameter tuning results
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
                # ✅ OPTIMIZED: Based on hyperparameter tuning results
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
                # ✅ OPTIMIZED: Based on hyperparameter tuning results
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
                # ✅ OPTIMIZED: Based on hyperparameter tuning results
                "seasonality_mode": "additive",
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 0.1,  # ⭐ Weak seasonality (unique!)
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
    
    def _determine_product_category(self, product_id: str) -> str:
        """Determine category for a product"""
        return self._product_category_mapping.get(product_id, "unknown")

    @property
    def data(self):
        """Get the loaded data"""
        return self._data
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the forecasting service"""
        if not self._available:
            return {
                "status": "unavailable",
                "error": self._initialization_error,
                "models_available": 0
            }
        
        # Check available models
        models_count = 0
        if self._models_path and self._models_path.exists():
            models_count = len(list(self._models_path.glob("*.pkl")))
        
        return {
            "status": "available",
            "models_available": models_count,
            "models_path": str(self._models_path) if self._models_path else None
        }
    
    def get_category_status(self) -> Dict[str, Any]:
        """📊 Get status of 6-category model system"""
        if not self._available:
            return {
                "status": "unavailable",
                "error": self._initialization_error
            }
        
        try:
            # Check which category models exist
            category_model_status = {}
            trained_categories = 0
            
            for category in self._category_hyperparams.keys():
                model_path = self._models_path / f"category_{category}_prophet_model.pkl"
                
                if model_path.exists():
                    # Get model file stats
                    stat = model_path.stat()
                    category_model_status[category] = {
                        "status": "trained",
                        "model_file": str(model_path),
                        "file_size_mb": round(stat.st_size / (1024*1024), 2),
                        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "hyperparameters": self._category_hyperparams[category]
                    }
                    trained_categories += 1
                else:
                    category_model_status[category] = {
                        "status": "not_trained",
                        "hyperparameters": self._category_hyperparams[category]
                    }
            
            # Calculate product distribution
            category_distribution = {}
            if self._product_category_mapping:
                for category in self._category_hyperparams.keys():
                    count = sum(1 for cat in self._product_category_mapping.values() if cat == category)
                    category_distribution[category] = count
            
            return {
                "status": "ready",
                "system_type": "6-category-models",
                "categories": list(self._category_hyperparams.keys()),
                "summary": {
                    "total_categories": len(self._category_hyperparams),
                    "trained_categories": trained_categories,
                    "untrained_categories": len(self._category_hyperparams) - trained_categories,
                    "training_completion": f"{(trained_categories/len(self._category_hyperparams)*100):.1f}%"
                },
                "category_models": category_model_status,
                "product_distribution": category_distribution,
                "total_products_mapped": len(self._product_category_mapping),
                "models_path": str(self._models_path),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting category status: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def predict_item_demand(
        self, 
        item_id: str, 
        horizon_days: int = 30,
        start_date: Optional[str] = None,
        confidence_interval: float = 0.95,
        include_external_factors: bool = True
    ) -> Dict[str, Any]:
        """🔮 Enhanced prediction using category-based models with fallback to individual models"""
        try:
            if not self._available:
                return {
                    "status": "error",
                    "message": f"Service unavailable: {self._initialization_error}"
                }
            
            # Determine product category
            product_category = self._determine_product_category(item_id)
            
            logger.info(f"Generating forecast for {item_id} (category: {product_category}), {horizon_days} days")
            
            # Parse start date or use tomorrow as default
            if start_date:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            else:
                start_dt = datetime.now() + timedelta(days=1)
            
            # Check if category model is available
            if product_category != "unknown" and product_category in self._category_hyperparams:
                result = await self._predict_with_category_model(
                    item_id, product_category, horizon_days, start_dt, confidence_interval
                )
                
                if result["status"] == "success":
                    return result
                else:
                    logger.warning(f"Category prediction failed for {item_id}, falling back to individual model")
            
            # Fallback to individual product model
            return await self._predict_with_individual_model(
                item_id, horizon_days, start_dt, confidence_interval
            )
            
        except Exception as e:
            logger.error(f"Error predicting demand for {item_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Prediction failed: {str(e)}"
            }
    
    async def _predict_with_category_model(
        self, 
        item_id: str, 
        category: str, 
        horizon_days: int, 
        start_dt: datetime, 
        confidence_interval: float
    ) -> Dict[str, Any]:
        """🎯 Predict using category-based model"""
        try:
            # Check if category model exists
            category_model_path = self._models_path / f"category_{category}_prophet_model.pkl"
            
            if not category_model_path.exists():
                return {
                    "status": "error",
                    "message": f"Category model not found for {category}. Train category models first."
                }
            
            logger.info(f"Using category model for {category}")
            
            # Create future dates
            future_dates = pd.date_range(
                start=start_dt.date(),
                periods=horizon_days,
                freq='D'
            )
            
            # Load and use category model
            import joblib
            category_model = joblib.load(category_model_path)
            
            # Create future dataframe
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Generate category-level forecast
            forecast_df = category_model.predict(future_df)
            
            if forecast_df is None or forecast_df.empty:
                return {
                    "status": "error",
                    "message": f"No forecast generated for category {category}"
                }
            
            # Calculate product's share of category demand
            product_share = self._calculate_product_category_share(item_id, category)
            
            # Scale down forecast to product level
            predictions = []
            total_demand = 0
            
            for _, row in forecast_df.iterrows():
                pred_date = row['ds']
                if isinstance(pred_date, str):
                    pred_date = pd.to_datetime(pred_date)
                
                # Scale category forecast to product level
                category_demand = max(0, row['yhat'])
                product_demand = category_demand * product_share
                
                category_lower = max(0, row['yhat_lower'])
                category_upper = max(0, row['yhat_upper'])
                
                product_lower = category_lower * product_share
                product_upper = category_upper * product_share
                
                predictions.append({
                    "date": pred_date.strftime('%Y-%m-%d'),
                    "predicted_demand": round(product_demand, 2),
                    "lower_bound": round(product_lower, 2),
                    "upper_bound": round(product_upper, 2),
                    "confidence": confidence_interval,
                    "category": category,
                    "category_share": round(product_share * 100, 2)
                })
                
                total_demand += product_demand
            
            # Calculate summary statistics
            avg_daily_demand = total_demand / len(predictions) if predictions else 0
            peak_demand = max([p["predicted_demand"] for p in predictions]) if predictions else 0
            
            return {
                "status": "success",
                "item_id": item_id,
                "category": category,
                "category_share": f"{product_share*100:.2f}%",
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
                    "model_file": str(category_model_path),
                    "product_share_method": "historical_proportion"
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in category prediction for {item_id}: {e}")
            return {
                "status": "error",
                "message": f"Category prediction failed: {str(e)}"
            }
    
    async def _predict_with_individual_model(
        self, 
        item_id: str, 
        horizon_days: int, 
        start_dt: datetime, 
        confidence_interval: float
    ) -> Dict[str, Any]:
        """🔄 Fallback prediction using individual product model"""
        try:
            logger.info(f"Using individual model for {item_id}")
            
            # Create date range for prediction
            future_dates = pd.date_range(
                start=start_dt.date(),
                periods=horizon_days,
                freq='D'
            )
            
            # Try to load existing model or train new one
            model_path = self._models_path / f"{item_id}_prophet_model.pkl"
            
            if model_path.exists():
                logger.info(f"Loading existing individual model for {item_id}")
                forecast_df = self._forecaster.load_and_predict(
                    model_path=str(model_path),
                    future_dates=future_dates.tolist()
                )
            else:
                logger.info(f"Training new individual model for {item_id}")
                # Train model for this product
                forecast_df = self._forecaster.train_and_predict(
                    product_filter=item_id,
                    future_dates=future_dates.tolist(),
                    save_model=True
                )
            
            if forecast_df is None or forecast_df.empty:
                return {
                    "status": "no_data",
                    "message": f"No data available for product {item_id}"
                }
            
            # Convert forecast to API response format
            predictions = []
            total_demand = 0
            
            for _, row in forecast_df.iterrows():
                pred_date = row['ds']
                if isinstance(pred_date, str):
                    pred_date = pd.to_datetime(pred_date)
                
                demand = max(0, row['yhat'])  # Ensure non-negative demand
                lower_bound = max(0, row['yhat_lower'])
                upper_bound = max(0, row['yhat_upper'])
                
                predictions.append({
                    "date": pred_date.strftime('%Y-%m-%d'),
                    "predicted_demand": round(demand, 2),
                    "lower_bound": round(lower_bound, 2),
                    "upper_bound": round(upper_bound, 2),
                    "confidence": confidence_interval
                })
                
                total_demand += demand
            
            # Calculate summary statistics
            avg_daily_demand = total_demand / len(predictions) if predictions else 0
            peak_demand = max([p["predicted_demand"] for p in predictions]) if predictions else 0
            
            return {
                "status": "success",
                "item_id": item_id,
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
                    "model_type": "Prophet-Individual",
                    "trained_on": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "model_file": str(model_path) if model_path.exists() else "newly_trained"
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in individual prediction for {item_id}: {e}")
            return {
                "status": "error",
                "message": f"Individual prediction failed: {str(e)}"
            }
    
    def _calculate_product_category_share(self, product_id: str, category: str) -> float:
        """📊 Calculate product's share of total category demand"""
        try:
            if self._data is None:
                return 0.1  # Default 10% share
            
            # Get recent data (last 90 days) for calculation
            recent_date = self._data['ds'].max() - timedelta(days=90)
            recent_data = self._data[self._data['ds'] >= recent_date]
            
            # Get category products
            category_products = [
                pid for pid, cat in self._product_category_mapping.items() 
                if cat == category
            ]
            
            # Calculate shares
            category_total = recent_data[
                recent_data['product_id'].isin(category_products)
            ]['y'].sum()
            
            product_total = recent_data[
                recent_data['product_id'] == product_id
            ]['y'].sum()
            
            if category_total > 0:
                share = min(product_total / category_total, 1.0)
                logger.info(f"Product {product_id} share of {category}: {share*100:.2f}%")
                return share
            else:
                # Equal share if no recent data
                equal_share = 1.0 / len(category_products) if category_products else 0.1
                logger.info(f"Using equal share for {product_id}: {equal_share*100:.2f}%")
                return equal_share
                
        except Exception as e:
            logger.warning(f"Error calculating product share for {product_id}: {e}")
            return 0.1  # Default 10% share
    
    async def predict_multiple_items(
        self,
        item_ids: List[str],
        horizon_days: int = 30,
        start_date: Optional[str] = None,
        confidence_interval: float = 0.95,
        include_external_factors: bool = True
    ) -> Dict[str, Any]:
        """Predict demand for multiple items"""
        try:
            logger.info(f"Batch prediction for {len(item_ids)} items")
            
            results = {}
            successful = 0
            failed = 0
            
            for item_id in item_ids:
                result = await self.predict_item_demand(
                    item_id=item_id,
                    horizon_days=horizon_days,
                    start_date=start_date,
                    confidence_interval=confidence_interval,
                    include_external_factors=include_external_factors
                )
                
                results[item_id] = result
                
                if result["status"] == "success":
                    successful += 1
                else:
                    failed += 1
            
            return {
                "status": "success",
                "batch_summary": {
                    "total_items": len(item_ids),
                    "successful": successful,
                    "failed": failed,
                    "success_rate": f"{(successful/len(item_ids)*100):.1f}%"
                },
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return {
                "status": "error",
                "message": f"Batch prediction failed: {str(e)}"
            }
    
    async def get_available_products(self) -> Dict[str, Any]:
        """Get list of products available for forecasting"""
        try:
            if not self._available:
                return {
                    "status": "error",
                    "message": f"Service unavailable: {self._initialization_error}"
                }
            
            # Get products from data
            if self._data is not None and 'product_id' in self._data.columns:
                products = self._data['product_id'].unique().tolist()
                
                # Check which products have trained models
                trained_models = []
                if self._models_path and self._models_path.exists():
                    model_files = list(self._models_path.glob("*_prophet_model.pkl"))
                    trained_models = [f.stem.replace('_prophet_model', '') for f in model_files]
                
                product_info = []
                for product in products:
                    product_data = self._data[self._data['product_id'] == product]
                    last_date = product_data['ds'].max() if 'ds' in product_data.columns else None
                    
                    product_info.append({
                        "product_id": product,
                        "has_trained_model": product in trained_models,
                        "data_points": len(product_data),
                        "last_data_date": last_date.strftime('%Y-%m-%d') if last_date else None
                    })
                
                return {
                    "status": "success",
                    "total_products": len(products),
                    "products_with_models": len(trained_models),
                    "products": product_info,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "message": "No product data available"
                }
                
        except Exception as e:
            logger.error(f"Error getting available products: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to get products: {str(e)}"
            }
    
    # NOTE: Training methods removed since we use pre-trained models
    # All models are loaded from ai_services/seasonal_inventory/data/models/
    # Training was completed externally with optimized hyperparameters

# Global service instance
_prophet_service = None

def get_prophet_forecasting_service() -> ProphetForecastingService:
    """Get singleton instance of Prophet forecasting service"""
    global _prophet_service
    if _prophet_service is None:
        _prophet_service = ProphetForecastingService()
    return _prophet_service
