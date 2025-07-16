import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path#for handling the file system paths
import pickle #or saving the model to the disk
import json#for reading and writing the model configuration or result
import hashlib
import os

# Prophet imports
# try:
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
    # PROPHET_AVAILABLE = True
# except ImportError:
#     PROPHET_AVAILABLE = False
#     logging.warning("Prophet not installed. Run: pip install prophet")

import matplotlib.pyplot as plt
import seaborn as sns

from ai_services.seasonal_inventory.config import PROPHET_CONFIG, MODELS_DIR, PROCESSED_DIR
try:
    from ai_services.seasonal_inventory.config import MODEL_CACHE_STRATEGY, MODEL_CACHE_HOURS
except ImportError:
    MODEL_CACHE_STRATEGY = "data_hash"  # Default strategy for static data
    MODEL_CACHE_HOURS = 24

# Optional import - only needed if saving to WMS database
try:
    from base_wms_backend.app.services.inventory_service import InventoryService
    INVENTORY_SERVICE_AVAILABLE = True
except ImportError:
    INVENTORY_SERVICE_AVAILABLE = False
    print("Warning: InventoryService not available. Forecasts won't be saved to WMS database.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProphetForecaster:
    
    def __init__(self, model_config: Dict = None, product_id: str = None):
    
            #model_config: Prophet model configuration
            #product_id: Specific product ID for individual forecasting

        self.model_config = model_config or PROPHET_CONFIG
        self.product_id = product_id
        self.model = None
        self.is_trained = False
        self.training_data = None
        self.forecast_results = None
        
        # Model storage
        self.models_dir = Path(MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Data checksum
        self.data_hash = None  # Store hash of training data
        
        logger.info(f"Prophet Forecaster initialized for product: {product_id or 'ALL'}")
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'y') -> pd.DataFrame:
        
        #Prepare data for Prophet training.
        """'''
        Args:
            data: Input DataFrame
            target_column: Name of the target variable column
            
        Returns:
            Prophet-ready DataFrame
        """
        logger.info("Preparing data for Prophet training")
        
        # Ensure required columns exist
        # if 'ds' not in data.columns:
        #     raise ValueError("Data must have 'ds' (date) column")
        
        # if target_column not in data.columns:
        #     raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Create Prophet DataFrame
        prophet_data = pd.DataFrame()
        prophet_data['ds'] = pd.to_datetime(data['ds'])
        prophet_data['y'] = pd.to_numeric(data[target_column])
        
        # Filter for specific product if specified
        if self.product_id and 'product_id' in data.columns:
            product_data = data[data['product_id'] == self.product_id]
            if product_data.empty:
                raise ValueError(f"No data found for product_id: {self.product_id}")
            prophet_data = pd.DataFrame()
            prophet_data['ds'] = pd.to_datetime(product_data['ds'])
            prophet_data['y'] = pd.to_numeric(product_data[target_column])
        
        # Add is_weekend column
        prophet_data['is_weekend'] = prophet_data['ds'].dt.dayofweek >= 5
        prophet_data['is_weekend'] = prophet_data['is_weekend'].astype(int)
        
        # Add is_holiday column (using Sri Lanka holidays as example, can be customized)
        from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
        class SLHolidayCalendar(AbstractHolidayCalendar):
            rules = [
                # Add major Sri Lankan public holidays (example, not exhaustive)
                Holiday('New Year', month=4, day=14),
                Holiday('Independence Day', month=2, day=4),
                Holiday('Valentine Day', month=2, day=14),
                Holiday('Christmas', month=12, day=25),
                # Add more as needed
            ]
        holidays = SLHolidayCalendar().holidays(start=prophet_data['ds'].min(), end=prophet_data['ds'].max())
        prophet_data['is_holiday'] = prophet_data['ds'].dt.normalize().isin(holidays).astype(int)
        
        # Remove missing values
        initial_rows = len(prophet_data)
        prophet_data = prophet_data.dropna(subset=['ds', 'y'])
        final_rows = len(prophet_data)
        
        if initial_rows > final_rows:
            logger.warning(f"Removed {initial_rows - final_rows} rows with missing values")
        
        # Sort by date
        prophet_data = prophet_data.sort_values('ds').reset_index(drop=True)
        
        # Aggregate daily if multiple entries per day
        if prophet_data['ds'].dt.date.duplicated().any():
            logger.info("Aggregating duplicate dates")
            agg_dict = {'y': 'sum', 'is_weekend': 'max', 'is_holiday': 'max'}  # Sum demand, max for flags
            prophet_data = prophet_data.groupby(prophet_data['ds'].dt.date).agg(agg_dict).reset_index()
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        logger.info(f"Data prepared: {len(prophet_data)} records from {prophet_data['ds'].min()} to {prophet_data['ds'].max()}")
        
        return prophet_data
    
    def create_model(self) -> Prophet:
    #     """
    #     Create and configure Prophet model.
        
    #     Returns:
    #         Configured Prophet model
    #     """
        logger.info("Creating Prophet model")
        
    #     # Base model configuration
        base_config = self.model_config.get('base_model', {})
        
        model = Prophet(
            growth=base_config.get('growth', 'linear'),
            seasonality_mode=base_config.get('seasonality_mode', 'multiplicative'),
            yearly_seasonality=base_config.get('yearly_seasonality', True),
            weekly_seasonality=base_config.get('weekly_seasonality', True),
            daily_seasonality=base_config.get('daily_seasonality', False),
            changepoint_prior_scale=base_config.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=base_config.get('seasonality_prior_scale', 10.0),
            holidays_prior_scale=base_config.get('holidays_prior_scale', 10.0),
            interval_width=base_config.get('interval_width', 0.8)
        )

        # Example: Add custom monthly seasonality
        model.add_seasonality(
            name='monthly',
            period=30.5,  # days in a month
            fourier_order=5
        )
        logger.info("Added custom seasonality: monthly (period=30.5, fourier_order=5)")

        # Add custom seasonalities from config if present
        custom_seasonalities = self.model_config.get('custom_seasonalities', [])
        for seasonality in custom_seasonalities:
            model.add_seasonality(
                name=seasonality['name'],
                period=seasonality['period'],
                fourier_order=seasonality['fourier_order'],
                prior_scale=seasonality.get('prior_scale', 10.0)
            )
            logger.info(f"Added custom seasonality: {seasonality['name']}")
        
        # Add holidays
        holidays_config = self.model_config.get('holidays', {})
        if 'countries' in holidays_config:
            # This would require holiday data preparation
            logger.info("Holiday configuration available")
        
        logger.info("Prophet model created")
        return model
    
    def add_external_regressors(self, model: Prophet, data: pd.DataFrame) -> Prophet:
        """
        
        Args:
            model: Prophet model
            data: Training data with external features
            
        Returns:
            Model with external regressors added
        """
        # Use external features from config if available, else fallback to default
        external_regressors = getattr(self.model_config, 'external_features', None)
        if external_regressors is None:
            try:
                from ai_services.seasonal_inventory.config import FEATURE_CONFIG
                external_regressors = FEATURE_CONFIG.get("external_features", [])
            except ImportError:
                external_regressors = []
        
        # If still not found, fallback to hardcoded list
        if not external_regressors:
            external_regressors = [
                'is_holiday', 'is_weekend',
                'is_month_end'
            ]
        
        for regressor in external_regressors:
            if regressor in data.columns:
                model.add_regressor(regressor)
                logger.info(f" Added external regressor: {regressor}")
        
        return model
    
    def train(self, data: pd.DataFrame, target_column: str = 'y') -> Dict:
        """
        Train the Prophet model.
        
        Args:
            data: Training data
            target_column: Target variable column name
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting Prophet model training")
        
        # Calculate data hash before training
        if self.product_id:
            self.data_hash = self._calculate_data_hash(data)
        
        # Prepare data
        self.training_data = self.prepare_data(data, target_column)
        
        if len(self.training_data) < 30:
            raise ValueError("Insufficient data for training. Need at least 30 data points.")
        
        # Create model
        self.model = self.create_model()
        
        # Add external regressors
        self.model = self.add_external_regressors(self.model, self.training_data)
        
        # Train model
        logger.info("Fitting Prophet model...")
        self.model.fit(self.training_data)
        
        # Validation
        logger.info("Running cross-validation...")
        cv_results = self._run_cross_validation()
        
        self.is_trained = True
        logger.info("Model training completed successfully")
        
        return cv_results
    
    def _run_cross_validation(self) -> Dict:
        """
        Run cross-validation on the trained model.
        
        Returns:
            Cross-validation results
        """
        try:
            # Determine CV parameters based on data size
            data_days = (self.training_data['ds'].max() - self.training_data['ds'].min()).days
            
            if data_days < 365:
                initial_training= f"{int(data_days * 0.6)}d"
                period = f"{int(data_days * 0.1)}d"
                horizon = f"{int(data_days * 0.1)}d"
            else:
                initial_training = "365d"
                period = "30d"
                horizon = "90d"
            
            logger.info(f" CV params: initial={initial_training}, period={period}, horizon={horizon}")
            
            cv_df = cross_validation(
                self.model, 
                initial=initial_training,
                period=period, 
                horizon=horizon,
                parallel='processes'
            )
            
            metrics = performance_metrics(cv_df)
            
            logger.info(" Cross-validation completed")
            
            return {
                'cv_results': cv_df,
                'metrics': metrics,
                'mean_mape': metrics['mape'].mean(),
                'mean_rmse': metrics['rmse'].mean(),
                'mean_mae': metrics['mae'].mean()
            }
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {'error': str(e)}
    
    def predict(self, periods: int = 90, include_history: bool = True, 
                future_regressors: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            periods: Number of future periods to forecast
            include_history: Whether to include historical fit
            future_regressors: External regressor values for future periods
            
        Returns:
            Forecast DataFrame
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        logger.info(f"Generating forecast for {periods} periods")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, include_history=include_history)

        # If regressors are used and not provided, auto-generate them for future periods
        regressors_needed = [col for col in ['is_weekend', 'is_holiday'] if col in self.training_data.columns]
        if future_regressors is None and regressors_needed:
            logger.info("Auto-generating is_weekend and is_holiday for future periods")
            future_regressors = pd.DataFrame({'ds': future['ds']})
            future_regressors['is_weekend'] = future_regressors['ds'].dt.dayofweek >= 5
            future_regressors['is_weekend'] = future_regressors['is_weekend'].astype(int)
            from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
            class SLHolidayCalendar(AbstractHolidayCalendar):
                rules = [
                    Holiday('New Year', month=4, day=14),
                    Holiday('Independence Day', month=2, day=4),
                    Holiday('Valentine Day', month=2, day=14),
                    Holiday('Christmas', month=12, day=25),
                ]
            holidays = SLHolidayCalendar().holidays(start=future_regressors['ds'].min(), end=future_regressors['ds'].max())
            future_regressors['is_holiday'] = future_regressors['ds'].dt.normalize().isin(holidays).astype(int)

    #future regressors are created 
        if future_regressors is not None:
            for col in future_regressors.columns:
                if col == 'ds' or col in future.columns:
                    continue  # Skip if already exists or is ds
                future = future.merge(
                    future_regressors[['ds', col]], 
                    on='ds', 
                    how='left'
                )
                future[col] = future[col].fillna(method='ffill')
                future[col] = future[col].fillna(0)

        # Generate forecast
        forecast = self.model.predict(future)
        forecast['product_id'] = self.product_id
        forecast['forecast_date'] = datetime.now()
        self.forecast_results = forecast
        logger.info(f"Forecast generated: {len(forecast)} predictions")
        return forecast
    
    def get_forecast_summary(self, forecast: pd.DataFrame = None) -> Dict:
       
        if forecast is None:
            forecast = self.forecast_results #use the last created forcast.
        
        if forecast is None:
            raise ValueError("No forecast available")
        
        # Future predictions only
        train_end = self.training_data['ds'].max()
        future_forecast = forecast[forecast['ds'] > train_end]
        # Intermittency warning
        zero_days = (self.training_data['y'] == 0).sum()
        total_days = len(self.training_data)
        zero_ratio = zero_days / total_days if total_days > 0 else 0
        summary = {
            'forecast_periods': len(future_forecast),
            'forecast_start': future_forecast['ds'].min(),
            'forecast_end': future_forecast['ds'].max(),
            'mean_prediction': future_forecast['yhat'].mean(),
            'total_predicted_demand': future_forecast['yhat'].sum(),
            'max_prediction': future_forecast['yhat'].max(),
            'min_prediction': future_forecast['yhat'].min(),
            'prediction_std': future_forecast['yhat'].std(),
            'confidence_width': (future_forecast['yhat_upper'] - future_forecast['yhat_lower']).mean(),
            'zero_demand_ratio': zero_ratio,
        }
        if zero_ratio > 0.8:
            summary['warning'] = f"High intermittency: {zero_ratio:.0%} of days have zero demand. Forecasts may be mostly zero."
        return summary
    
    def save_model(self, filename: str = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            filename: Custom filename (auto-generated if None)
            
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            product_suffix = f"_{self.product_id}" if self.product_id else "_global"
            filename = f"prophet_model{product_suffix}_{timestamp}.pkl"
        
        model_path = self.models_dir / filename
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'config': self.model_config,
            'product_id': self.product_id,
            'training_data_shape': self.training_data.shape,
            'trained_date': datetime.now(),
            'is_trained': self.is_trained,
            'data_hash': self.data_hash  # Save data hash
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to: {model_path}")
        return str(model_path)
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of training data to detect changes"""
        # Filter data for this product and sort for consistent hashing
        product_data = data[data['product_id'] == self.product_id].copy()
        product_data = product_data.sort_values(['ds', 'y']).reset_index(drop=True)
        
        # Create hash from relevant columns
        data_string = product_data[['ds', 'y']].to_string()
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def _should_retrain(self, data: pd.DataFrame, model_path: str) -> bool:
        """Check if model should be retrained based on data changes"""
        current_hash = self._calculate_data_hash(data)
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            saved_hash = model_data.get('data_hash', '')
            return current_hash != saved_hash
        except:
            return True  # Retrain if can't read model
    
    def _should_use_cached_model(self, data: pd.DataFrame, model_path: str) -> bool:
        """
        Determine if cached model should be used based on the configured caching strategy.
        
        Strategies:
        - "never": Always retrain (useful for development/testing)
        - "always": Never retrain, always use cached models (best for static data)
        - "time_based": Retrain after specified hours (original behavior)
        - "data_hash": Retrain only when data changes (recommended for static data)
        
        Returns:
            True if cached model should be used, False if retraining is needed
        """
        if MODEL_CACHE_STRATEGY == "never":
            logger.info("Cache strategy: NEVER - Always retraining")
            return False
        elif MODEL_CACHE_STRATEGY == "always":
            logger.info("Cache strategy: ALWAYS - Using cached model without checks")
            return True
        elif MODEL_CACHE_STRATEGY == "time_based":
            # Use time-based caching (original 24-hour logic)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
                is_recent = datetime.now() - mtime < timedelta(hours=MODEL_CACHE_HOURS)
                logger.info(f"Cache strategy: TIME_BASED - Model age: {datetime.now() - mtime}, Recent: {is_recent}")
                return is_recent
            except:
                return False
        elif MODEL_CACHE_STRATEGY == "data_hash":
            # Use data hash-based caching (recommended for static data)
            should_use_cache = not self._should_retrain(data, model_path)
            logger.info(f"Cache strategy: DATA_HASH - Data unchanged: {should_use_cache}")
            return should_use_cache
        else:
            # Default to data hash strategy
            logger.warning(f"Unknown cache strategy '{MODEL_CACHE_STRATEGY}', defaulting to data_hash")
            return not self._should_retrain(data, model_path)
    
    def load_and_predict(self, model_path: str, future_dates: List) -> Optional[pd.DataFrame]:
        """
        Load a saved model and make predictions for specified future dates
        
        Args:
            model_path: Path to the saved Prophet model
            future_dates: List of future dates to predict
            
        Returns:
            DataFrame with predictions or None if failed
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # Load the saved model
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
                
            self.model = saved_data['model']
            self.product_id = saved_data.get('product_id', 'unknown')
            
            # Create future dataframe
            future_df = pd.DataFrame({'ds': pd.to_datetime(future_dates)})
            
            # Make predictions
            forecast = self.model.predict(future_df)
            
            # Return relevant columns
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
        except Exception as e:
            logger.error(f"Error loading model and predicting: {e}")
            return None
    
    def train_and_predict(self, product_filter: str, future_dates: List, save_model: bool = True) -> Optional[pd.DataFrame]:
        """
        Train a new model for a specific product and make predictions
        
        Args:
            product_filter: Product ID to filter data for
            future_dates: List of future dates to predict
            save_model: Whether to save the trained model
            
        Returns:
            DataFrame with predictions or None if failed
        """
        try:
            # Load and filter data
            if not hasattr(self, 'data') or self.data is None:
                self._load_data()
            
            if self.data is None:
                logger.error("No data available for training")
                return None
            
            # Filter data for specific product
            product_data = self.data[self.data['product_id'] == product_filter].copy()
            
            if product_data.empty:
                logger.error(f"No data found for product {product_filter}")
                return None
            
            logger.info(f"Training model for product {product_filter} with {len(product_data)} data points")
            
            # Prepare data for Prophet
            prophet_data = self.prepare_data(product_data, target_column='demand')
            
            # Train model
            training_result = self.train(prophet_data, target_column='y')
            
            if training_result['status'] != 'success':
                logger.error(f"Failed to train model for {product_filter}")
                return None
            
            # Make predictions
            future_df = pd.DataFrame({'ds': pd.to_datetime(future_dates)})
            forecast = self.model.predict(future_df)
            
            # Save model if requested
            if save_model:
                model_filename = f"{product_filter}_prophet_model.pkl"
                self.save_model(model_filename)
                logger.info(f"Model saved for product {product_filter}")
            
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
        except Exception as e:
            logger.error(f"Error training and predicting for {product_filter}: {e}")
            return None
    
    def train_product_model(self, product_id: str, save_model: bool = True) -> Optional[str]:
        """
        Train a model for a specific product without making predictions
        
        Args:
            product_id: Product ID to train model for
            save_model: Whether to save the trained model
            
        Returns:
            Path to saved model file or None if failed
        """
        try:
            # Load and filter data
            if not hasattr(self, 'data') or self.data is None:
                self._load_data()
            
            if self.data is None:
                logger.error("No data available for training")
                return None
            
            # Filter data for specific product
            product_data = self.data[self.data['product_id'] == product_id].copy()
            
            if product_data.empty:
                logger.error(f"No data found for product {product_id}")
                return None
            
            if len(product_data) < 14:  # Need at least 2 weeks of data
                logger.warning(f"Insufficient data for product {product_id}: {len(product_data)} days")
                return None
            
            logger.info(f"Training model for product {product_id} with {len(product_data)} data points")
            
            # Prepare data for Prophet
            prophet_data = self.prepare_data(product_data, target_column='demand')
            
            # Train model
            training_result = self.train(prophet_data, target_column='y')
            
            if training_result['status'] != 'success':
                logger.error(f"Failed to train model for {product_id}")
                return None
            
            # Save model if requested
            if save_model:
                model_filename = f"{product_id}_prophet_model.pkl"
                model_path = self.save_model(model_filename)
                logger.info(f"Model saved for product {product_id}: {model_path}")
                return model_path
            
            return "model_trained_but_not_saved"
            
        except Exception as e:
            logger.error(f"Error training model for {product_id}: {e}")
            return None
    
    def _load_data(self):
        """Load the training data if not already loaded"""
        try:
            data_path = Path(PROCESSED_DIR) / 'daily_demand_by_product_modern.csv'
            if data_path.exists():
                self.data = pd.read_csv(data_path)
                # Ensure date column is datetime
                if 'date' in self.data.columns:
                    self.data['date'] = pd.to_datetime(self.data['date'])
                elif 'ds' in self.data.columns:
                    self.data['ds'] = pd.to_datetime(self.data['ds'])
                    self.data['date'] = self.data['ds']
                
                logger.info(f"Loaded data with {len(self.data)} records")
            else:
                logger.error(f"Data file not found: {data_path}")
                self.data = None
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.data = None
    
    def plot_forecast(self, forecast: pd.DataFrame = None, save_path: str = None) -> None:
        """
        Plot the forecast results with color description (legend).
        Args:
            forecast: Forecast DataFrame
            save_path: Path to save the plot
        """
        if forecast is None:
            forecast = self.forecast_results
        
        if forecast is None:
            raise ValueError("No forecast to plot")
        
        # Create the plot
        fig = self.model.plot(forecast, figsize=(12, 8))
        plt.title(f'Seasonal Inventory Forecast - {self.product_id or "All Products"}')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        # Add legend with color description
        handles, labels = fig.gca().get_legend_handles_labels()
        # Prophet default: yhat (blue), yhat_lower/upper (light blue), actuals (black dots)
        custom_labels = [
            'Forecast (yhat)',
            'Uncertainty Interval',
            'Actuals'
        ]
        if len(labels) >= 3:
            plt.legend(handles[:3], custom_labels, loc='upper left')
        else:
            plt.legend(loc='upper left')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to: {save_path}")
        
        plt.show()
    
    def plot_components(self, forecast: pd.DataFrame = None, save_path: str = None) -> None:
        """
        Plot forecast components (trend, seasonality, etc.).
        
        Args:
            forecast: Forecast DataFrame
            save_path: Path to save the plot
        """
        if forecast is None:
            forecast = self.forecast_results
        
        if forecast is None:
            raise ValueError("No forecast to plot")
        
        # Create components plot
        fig = self.model.plot_components(forecast, figsize=(12, 10))
        plt.suptitle(f'Forecast Components - {self.product_id or "All Products"}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Components plot saved to: {save_path}")
        
        plt.show()
    
    def evaluate_with_train_test_split(self, data: pd.DataFrame, train_ratio: float = 0.8, 
                                     test_ratio: float = 0.2, target_column: str = 'y') -> Dict:
        """
        Evaluate model performance using proper train/test split.
        
        This is different from Prophet's cross-validation - it uses a chronological
        split to simulate real-world deployment where you train on historical data
        and test on truly unseen future data.
        
        Args:
            data: Full dataset for evaluation
            train_ratio: Proportion of data for training (e.g., 0.8 = 80%)
            test_ratio: Proportion of data for testing (e.g., 0.2 = 20%)
            target_column: Target variable column name
            
        Returns:
            Dictionary with train/test split evaluation results
        """
        logger.info(f"Starting train/test split evaluation: {train_ratio:.1%} train, {test_ratio:.1%} test")
        
        try:
            # Prepare the data
            prepared_data = self.prepare_data(data, target_column)
            
            if len(prepared_data) < 50:
                raise ValueError("Insufficient data for train/test split. Need at least 50 data points.")
            
            # Calculate split points (chronological split for time series)
            total_points = len(prepared_data)
            train_size = int(total_points * train_ratio)
            
            # Split data chronologically
            train_data = prepared_data.iloc[:train_size].copy()
            test_data = prepared_data.iloc[train_size:].copy()
            
            logger.info(f"Split data: {len(train_data)} train points, {len(test_data)} test points")
            
            if len(test_data) < 10:
                raise ValueError("Test set too small. Need at least 10 test points.")
            
            # Train model on training data only
            temp_model = self.create_model()
            temp_model = self.add_external_regressors(temp_model, train_data)
            temp_model.fit(train_data)
            
            # Generate predictions for test period
            test_periods = len(test_data)
            future = temp_model.make_future_dataframe(periods=test_periods)
            
            # Add external regressors for future periods if needed
            regressors_needed = [col for col in ['is_weekend', 'is_holiday'] if col in train_data.columns]
            if regressors_needed:
                future = self._add_future_regressors(future, test_data)
            
            forecast = temp_model.predict(future)
            
            # Extract test period predictions
            train_end_date = train_data['ds'].max()
            test_forecast = forecast[forecast['ds'] > train_end_date].copy()
            
            # Align test predictions with actual test data
            test_comparison = pd.merge(
                test_data[['ds', 'y']].rename(columns={'y': 'actual'}),
                test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'yhat': 'predicted'}),
                on='ds',
                how='inner'
            )
            
            if len(test_comparison) == 0:
                raise ValueError("No matching dates between test data and predictions")
            
            # Calculate performance metrics
            actual = test_comparison['actual'].values
            predicted = test_comparison['predicted'].values
            
            # Handle edge cases
            actual_nonzero = actual[actual != 0]
            predicted_nonzero = predicted[actual != 0]
            
            # Calculate metrics
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            
            # MAPE calculation (handle division by zero)
            if len(actual_nonzero) > 0:
                mape = np.mean(np.abs((actual_nonzero - predicted_nonzero) / actual_nonzero)) * 100
            else:
                mape = float('inf')  # All actuals are zero
            
            # Mean Absolute Scaled Error (MASE) - using naive forecast as baseline
            if len(train_data) > 1:
                naive_mae = np.mean(np.abs(np.diff(train_data['y'].values)))
                mase = mae / naive_mae if naive_mae > 0 else float('inf')
            else:
                mase = float('inf')
            
            # Additional metrics
            mean_actual = np.mean(actual)
            mean_predicted = np.mean(predicted)
            bias = mean_predicted - mean_actual
            bias_percent = (bias / mean_actual * 100) if mean_actual != 0 else float('inf')
            
            # Coverage metrics for confidence intervals
            in_upper = (actual <= test_comparison['yhat_upper'].values).sum()
            in_lower = (actual >= test_comparison['yhat_lower'].values).sum()
            in_interval = ((actual >= test_comparison['yhat_lower'].values) & 
                          (actual <= test_comparison['yhat_upper'].values)).sum()
            coverage = in_interval / len(actual) * 100
            
            # Compile results
            evaluation_results = {
                "status": "success",
                "evaluation_timestamp": datetime.now().isoformat(),
                "data_split": {
                    "total_points": total_points,
                    "train_points": len(train_data),
                    "test_points": len(test_data),
                    "train_ratio_actual": len(train_data) / total_points,
                    "test_ratio_actual": len(test_data) / total_points,
                    "train_period": f"{train_data['ds'].min().date()} to {train_data['ds'].max().date()}",
                    "test_period": f"{test_data['ds'].min().date()} to {test_data['ds'].max().date()}"
                },
                "test_metrics": {
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "mape": float(mape) if mape != float('inf') else None,
                    "mase": float(mase) if mase != float('inf') else None,
                    "bias": float(bias),
                    "bias_percent": float(bias_percent) if bias_percent != float('inf') else None,
                    "coverage_percent": float(coverage)
                },
                "data_summary": {
                    "mean_actual": float(mean_actual),
                    "mean_predicted": float(mean_predicted),
                    "actual_std": float(np.std(actual)),
                    "predicted_std": float(np.std(predicted)),
                    "zero_actual_days": int((actual == 0).sum()),
                    "zero_predicted_days": int((predicted == 0).sum())
                },
                "detailed_comparison": test_comparison.to_dict('records')[:10]  # First 10 for space
            }
            
            # Add performance interpretation
            interpretation = []
            if mape is not None and mape != float('inf'):
                if mape < 10:
                    interpretation.append("🟢 Excellent accuracy (MAPE < 10%)")
                elif mape < 20:
                    interpretation.append("🟡 Good accuracy (MAPE 10-20%)")
                elif mape < 50:
                    interpretation.append("🟠 Fair accuracy (MAPE 20-50%)")
                else:
                    interpretation.append("🔴 Poor accuracy (MAPE > 50%)")
            
            if coverage < 80:
                interpretation.append("⚠️ Low confidence interval coverage - model may be overconfident")
            elif coverage > 95:
                interpretation.append("⚠️ High confidence interval coverage - model may be underconfident")
            else:
                interpretation.append("✅ Good confidence interval coverage")
            
            if abs(bias_percent) > 10 and bias_percent != float('inf'):
                bias_direction = "over-predicting" if bias > 0 else "under-predicting"
                interpretation.append(f"⚠️ Significant bias detected: {bias_direction} by {abs(bias_percent):.1f}%")
            
            evaluation_results["interpretation"] = interpretation
            
            logger.info(f"Train/test split evaluation completed - MAPE: {mape:.2f}%, RMSE: {rmse:.2f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in train/test split evaluation: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "evaluation_timestamp": datetime.now().isoformat()
            }
    
    def _add_future_regressors(self, future_df: pd.DataFrame, reference_data: pd.DataFrame) -> pd.DataFrame:
        """Helper method to add external regressors to future dataframe"""
        # Add is_weekend
        if 'is_weekend' in reference_data.columns:
            future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
        
        # Add is_holiday
        if 'is_holiday' in reference_data.columns:
            from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
            class SLHolidayCalendar(AbstractHolidayCalendar):
                rules = [
                    Holiday('New Year', month=4, day=14),
                    Holiday('Independence Day', month=2, day=4),
                    Holiday('Valentine Day', month=2, day=14),
                    Holiday('Christmas', month=12, day=25),
                ]
            holidays = SLHolidayCalendar().holidays(
                start=future_df['ds'].min(), 
                end=future_df['ds'].max()
            )
            future_df['is_holiday'] = future_df['ds'].dt.normalize().isin(holidays).astype(int)
        
        return future_df
    
# ============================================================================
# UNIFIED TRAINING & EVALUATION SYSTEM
# ============================================================================

class ProphetTrainingSystem:
    """Unified system for training and evaluating Prophet models with proper metrics"""
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            # Default path relative to the forecaster location
            self.data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "daily_demand_by_product_modern.csv"
        else:
            self.data_path = Path(data_path)
        
        self.models_path = Path(__file__).parent.parent.parent / "data" / "models"
        
    def load_data(self) -> pd.DataFrame:
        """Load training data"""
        if not self.data_path.exists():
            print(f"❌ Data file not found: {self.data_path}")
            print("💡 Generate data first or check the path")
            return None
        
        df = pd.read_csv(self.data_path)
        df['ds'] = pd.to_datetime(df['ds'])
        print(f"✅ Loaded {len(df):,} records for {df['product_id'].nunique()} products")
        return df
    
    def train_and_evaluate_single_product(self, product_id: str, train_ratio: float = 0.8) -> dict:
        """Train and evaluate a Prophet model for a single product with proper train/test split"""
        print(f"\n🚀 Training & Evaluating {product_id} with {train_ratio:.0%}/{1-train_ratio:.0%} split")
        
        # Load data
        df = self.load_data()
        if df is None:
            return {"status": "error", "message": "Data not available"}
        
        # Filter for this product
        product_data = df[df['product_id'] == product_id].copy()
        if product_data.empty:
            return {"status": "error", "message": f"No data found for {product_id}"}
        
        print(f"📊 Found {len(product_data)} data points from {product_data['ds'].min().date()} to {product_data['ds'].max().date()}")
        
        # Use the existing evaluate_with_train_test_split method
        forecaster = ProphetForecaster(product_id=product_id)
        result = forecaster.evaluate_with_train_test_split(
            data=product_data,
            train_ratio=train_ratio,
            test_ratio=1-train_ratio,
            target_column='y'
        )
        
        if result["status"] == "success":
            metrics = result["test_metrics"]
            print(f"\n📊 EVALUATION RESULTS:")
            print(f"   MAPE: {metrics['mape']:.2f}% (lower is better)")
            print(f"   RMSE: {metrics['rmse']:.2f} (lower is better)")
            print(f"   MAE: {metrics['mae']:.2f} (lower is better)")
            print(f"   R²: {metrics.get('r2_score', 'N/A')} (higher is better)")
            print(f"   Coverage: {metrics['coverage_percent']:.1f}% (should be ~95%)")
            
            # Show interpretation
            for insight in result.get("interpretation", []):
                print(f"   {insight}")
            
            return result
        else:
            print(f"❌ Evaluation failed: {result.get('message', 'Unknown error')}")
            return result
    
    def evaluate_multiple_products(self, product_ids: list, train_ratio: float = 0.8) -> dict:
        """Evaluate multiple products and provide comprehensive summary"""
        print(f"\n🔍 Training & Evaluating {len(product_ids)} products")
        print("=" * 70)
        
        results = {}
        successful = 0
        all_metrics = []
        
        for i, product_id in enumerate(product_ids, 1):
            print(f"\n[{i}/{len(product_ids)}] Processing {product_id}")
            result = self.train_and_evaluate_single_product(product_id, train_ratio)
            
            if result["status"] == "success":
                results[product_id] = result
                all_metrics.append(result["test_metrics"])
                successful += 1
        
        print(f"\n📈 SUMMARY: {successful}/{len(product_ids)} successful evaluations")
        
        if successful > 0:
            self._generate_comprehensive_report(all_metrics, results)
        
        return results
    
    def _generate_comprehensive_report(self, all_metrics: list, results: dict):
        """Generate comprehensive evaluation report with statistics"""
        print("\n" + "="*80)
        print("📊 PROPHET MODEL EVALUATION REPORT")
        print("="*80)
        
        # Calculate aggregate statistics
        mape_values = [m["mape"] for m in all_metrics if m["mape"] is not None]
        rmse_values = [m["rmse"] for m in all_metrics]
        mae_values = [m["mae"] for m in all_metrics]
        coverage_values = [m["coverage_percent"] for m in all_metrics]
        
        print(f"📈 Models Evaluated: {len(all_metrics)}")
        print(f"📊 MAPE: {np.mean(mape_values):.2f}% ± {np.std(mape_values):.2f} (range: {np.min(mape_values):.2f}%-{np.max(mape_values):.2f}%)")
        print(f"📊 RMSE: {np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f} (range: {np.min(rmse_values):.2f}-{np.max(rmse_values):.2f})")
        print(f"📊 MAE: {np.mean(mae_values):.2f} ± {np.std(mae_values):.2f} (range: {np.min(mae_values):.2f}-{np.max(mae_values):.2f})")
        print(f"📊 Coverage: {np.mean(coverage_values):.1f}% ± {np.std(coverage_values):.1f} (range: {np.min(coverage_values):.1f}%-{np.max(coverage_values):.1f}%)")
        
        # Model quality distribution based on MAPE
        print("\n🎯 Model Quality Distribution (based on MAPE):")
        excellent = len([m for m in mape_values if m <= 10])
        good = len([m for m in mape_values if 10 < m <= 20])
        fair = len([m for m in mape_values if 20 < m <= 50])
        poor = len([m for m in mape_values if m > 50])
        
        total = len(mape_values)
        print(f"   🎯 Excellent (≤10%): {excellent}/{total} ({excellent/total*100:.1f}%)")
        print(f"   ✅ Good (10-20%): {good}/{total} ({good/total*100:.1f}%)")
        print(f"   ⚠️ Fair (20-50%): {fair}/{total} ({fair/total*100:.1f}%)")
        print(f"   ❌ Poor (>50%): {poor}/{total} ({poor/total*100:.1f}%)")
        
        # Best and worst performing models
        print("\n🏆 Best Performing Models:")
        sorted_results = sorted(results.items(), key=lambda x: x[1]["test_metrics"]["mape"])
        for i, (product_id, result) in enumerate(sorted_results[:3], 1):
            mape = result["test_metrics"]["mape"]
            rmse = result["test_metrics"]["rmse"]
            print(f"   {i}. {product_id}: MAPE {mape:.2f}%, RMSE {rmse:.2f}")
        
        if len(sorted_results) > 3:
            print("\n⚠️ Worst Performing Models:")
            for i, (product_id, result) in enumerate(sorted_results[-3:], 1):
                mape = result["test_metrics"]["mape"]
                rmse = result["test_metrics"]["rmse"]
                print(f"   {i}. {product_id}: MAPE {mape:.2f}%, RMSE {rmse:.2f}")
        
        # Overall assessment
        print("\n💡 Overall Assessment:")
        avg_mape = np.mean(mape_values)
        avg_coverage = np.mean(coverage_values)
        
        if avg_mape <= 15 and avg_coverage >= 80:
            print("   🎉 EXCELLENT: Models are ready for production deployment!")
            print("   ✅ High accuracy and reliable uncertainty estimation")
        elif avg_mape <= 25 and avg_coverage >= 70:
            print("   ✅ GOOD: Models show solid performance")
            print("   💡 Consider fine-tuning for improved accuracy")
        elif avg_mape <= 40:
            print("   ⚠️ FAIR: Models have acceptable performance")
            print("   🔧 Recommend feature engineering and hyperparameter tuning")
        else:
            print("   ❌ POOR: Models need significant improvement")
            print("   🔧 Consider data quality issues, feature engineering, or alternative approaches")
        
        # Specific recommendations
        print("\n🔧 Specific Recommendations:")
        if avg_mape > 25:
            print("   • High prediction errors detected - review data quality and outliers")
        if avg_coverage < 80:
            print("   • Uncertainty estimation needs improvement - consider wider prediction intervals")
        if np.std(mape_values) > 15:
            print("   • High variance in model performance - investigate product-specific patterns")
        
        print("="*80)
    
    def sample_and_evaluate(self, sample_size: int = 5, min_data_points: int = 100, train_ratio: float = 0.8):
        """Sample random products and evaluate them"""
        df = self.load_data()
        if df is None:
            return None
        
        # Find products with sufficient data
        product_counts = df.groupby('product_id').size()
        suitable_products = product_counts[product_counts >= min_data_points].index.tolist()
        
        if len(suitable_products) < sample_size:
            selected_products = suitable_products
            print(f"⚠️ Only {len(suitable_products)} products have ≥{min_data_points} data points")
        else:
            selected_products = np.random.choice(suitable_products, sample_size, replace=False).tolist()
        
        print(f"🎯 Selected {len(selected_products)} products for evaluation:")
        for i, product in enumerate(selected_products, 1):
            product_count = product_counts[product]
            print(f"   {i}. {product} ({product_count} data points)")
        
        return self.evaluate_multiple_products(selected_products, train_ratio)

# ============================================================================
# COMMAND LINE INTERFACE FOR TRAINING & EVALUATION
# ============================================================================

def main():
    """Command line interface for Prophet training and evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prophet Model Training & Evaluation System")
    
    # Training/evaluation operations
    parser.add_argument("--product", help="Specific product ID to train/evaluate")
    parser.add_argument("--products", help="Comma-separated list of product IDs")
    parser.add_argument("--sample", type=int, default=5, help="Number of random products to sample")
    parser.add_argument("--min-data-points", type=int, default=100, help="Minimum data points required")
    
    # Parameters
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training data ratio (0.8 = 80%)")
    parser.add_argument("--data-path", help="Path to the data file")
    
    # Demo
    parser.add_argument("--demo", action="store_true", help="Run demo with sample products")
    
    args = parser.parse_args()
    
    # Initialize training system
    trainer = ProphetTrainingSystem(data_path=args.data_path)
    
    print("🚀 Prophet Model Training & Evaluation System")
    print("=" * 60)
    
    if args.demo:
        print("🎭 Running demo with sample products...")
        trainer.sample_and_evaluate(sample_size=3, train_ratio=args.train_ratio)
        
    elif args.product:
        # Single product
        trainer.train_and_evaluate_single_product(args.product, args.train_ratio)
        
    elif args.products:
        # Multiple specific products
        product_list = [p.strip() for p in args.products.split(',')]
        trainer.evaluate_multiple_products(product_list, args.train_ratio)
        
    else:
        # Sample random products
        trainer.sample_and_evaluate(
            sample_size=args.sample, 
            min_data_points=args.min_data_points,
            train_ratio=args.train_ratio
        )
    
    print("\n🎉 Evaluation completed!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Demo mode
        print("🎭 Running Prophet evaluation demo...")
        trainer = ProphetTrainingSystem()
        trainer.sample_and_evaluate(sample_size=2, train_ratio=0.8)

