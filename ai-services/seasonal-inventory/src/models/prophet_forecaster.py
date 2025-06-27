"""
Prophet Forecaster - Core ML Model for Seasonal Inventory Prediction

This module implements the Prophet model for time series forecasting
with custom seasonality and external regressors.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json

# Prophet imports
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_cross_validation_metric
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not installed. Run: pip install prophet")

import matplotlib.pyplot as plt
import seaborn as sns

from config import PROPHET_CONFIG, MODELS_DIR, PROCESSED_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeasonalProphetForecaster:
    """
    Prophet-based forecaster for seasonal inventory demand prediction.
    """
    
    def __init__(self, model_config: Dict = None, product_id: str = None):
        """
        Initialize the Prophet forecaster.
        
        Args:
            model_config: Prophet model configuration
            product_id: Specific product ID for individual forecasting
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet package not available. Install with: pip install prophet")
        
        self.model_config = model_config or PROPHET_CONFIG
        self.product_id = product_id
        self.model = None
        self.is_trained = False
        self.training_data = None
        self.forecast_results = None
        
        # Model storage
        self.models_dir = Path(MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔮 Prophet Forecaster initialized for product: {product_id or 'ALL'}")
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'y') -> pd.DataFrame:
        """
        Prepare data for Prophet training.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target variable column
            
        Returns:
            Prophet-ready DataFrame
        """
        logger.info("🔄 Preparing data for Prophet training")
        
        # Ensure required columns exist
        if 'ds' not in data.columns:
            raise ValueError("Data must have 'ds' (date) column")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
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
        
        # Add external regressors if available
        external_cols = ['temperature', 'humidity', 'is_holiday', 'is_weekend', 
                        'is_month_end', 'economic_index']
        
        for col in external_cols:
            if col in data.columns:
                prophet_data[col] = data[col]
                logger.info(f"   📊 Added external regressor: {col}")
        
        # Remove missing values
        initial_rows = len(prophet_data)
        prophet_data = prophet_data.dropna(subset=['ds', 'y'])
        final_rows = len(prophet_data)
        
        if initial_rows > final_rows:
            logger.warning(f"⚠️ Removed {initial_rows - final_rows} rows with missing values")
        
        # Sort by date
        prophet_data = prophet_data.sort_values('ds').reset_index(drop=True)
        
        # Aggregate daily if multiple entries per day
        if prophet_data['ds'].dt.date.duplicated().any():
            logger.info("📅 Aggregating duplicate dates")
            agg_dict = {'y': 'sum'}  # Sum demand for same day
            
            # Aggregate external regressors
            for col in external_cols:
                if col in prophet_data.columns:
                    if col.startswith('is_'):
                        agg_dict[col] = 'max'  # Boolean flags
                    else:
                        agg_dict[col] = 'mean'  # Numeric values
            
            prophet_data = prophet_data.groupby(prophet_data['ds'].dt.date).agg(agg_dict).reset_index()
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        logger.info(f"✅ Data prepared: {len(prophet_data)} records from {prophet_data['ds'].min()} to {prophet_data['ds'].max()}")
        
        return prophet_data
    
    def create_model(self) -> Prophet:
        """
        Create and configure Prophet model.
        
        Returns:
            Configured Prophet model
        """
        logger.info("🏗️ Creating Prophet model")
        
        # Base model configuration
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
        
        # Add custom seasonalities
        custom_seasonalities = self.model_config.get('custom_seasonalities', [])
        for seasonality in custom_seasonalities:
            model.add_seasonality(
                name=seasonality['name'],
                period=seasonality['period'],
                fourier_order=seasonality['fourier_order'],
                prior_scale=seasonality.get('prior_scale', 10.0)
            )
            logger.info(f"   🔄 Added custom seasonality: {seasonality['name']}")
        
        # Add holidays
        holidays_config = self.model_config.get('holidays', {})
        if 'countries' in holidays_config:
            # This would require holiday data preparation
            logger.info("   📅 Holiday configuration available")
        
        logger.info("✅ Prophet model created")
        return model
    
    def add_external_regressors(self, model: Prophet, data: pd.DataFrame) -> Prophet:
        """
        Add external regressors to the model.
        
        Args:
            model: Prophet model
            data: Training data with external features
            
        Returns:
            Model with external regressors added
        """
        external_regressors = [
            'temperature', 'humidity', 'is_holiday', 'is_weekend', 
            'is_month_end', 'economic_index'
        ]
        
        for regressor in external_regressors:
            if regressor in data.columns:
                model.add_regressor(regressor)
                logger.info(f"   📈 Added external regressor: {regressor}")
        
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
        logger.info("🚀 Starting Prophet model training")
        
        # Prepare data
        self.training_data = self.prepare_data(data, target_column)
        
        if len(self.training_data) < 30:
            raise ValueError("Insufficient data for training. Need at least 30 data points.")
        
        # Create model
        self.model = self.create_model()
        
        # Add external regressors
        self.model = self.add_external_regressors(self.model, self.training_data)
        
        # Train model
        logger.info("📊 Fitting Prophet model...")
        self.model.fit(self.training_data)
        
        # Validation
        logger.info("🔍 Running cross-validation...")
        cv_results = self._run_cross_validation()
        
        self.is_trained = True
        logger.info("✅ Model training completed successfully")
        
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
                initial = f"{int(data_days * 0.6)}d"
                period = f"{int(data_days * 0.2)}d"
                horizon = f"{int(data_days * 0.1)}d"
            elif data_days < 730:  # Less than 2 years
                initial = f"{int(data_days * 0.7)}d"
                period = f"{int(data_days * 0.1)}d"
                horizon = f"{int(data_days * 0.08)}d"
            else:
                initial = "365d"
                period = "30d"
                horizon = "90d"
            
            logger.info(f"   📊 CV params: initial={initial}, period={period}, horizon={horizon}")
            
            cv_df = cross_validation(
                self.model, 
                initial=initial,
                period=period, 
                horizon=horizon,
                parallel='processes'
            )
            
            metrics = performance_metrics(cv_df)
            
            logger.info("   ✅ Cross-validation completed")
            
            return {
                'cv_results': cv_df,
                'metrics': metrics,
                'mean_mape': metrics['mape'].mean(),
                'mean_rmse': metrics['rmse'].mean(),
                'mean_mae': metrics['mae'].mean()
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Cross-validation failed: {e}")
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
        
        logger.info(f"🔮 Generating forecast for {periods} periods")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods)
        
        # Add future regressor values if provided
        if future_regressors is not None:
            for col in future_regressors.columns:
                if col in future.columns:
                    continue  # Skip if already exists
                
                # Merge future regressor values
                future = future.merge(
                    future_regressors[['ds', col]], 
                    on='ds', 
                    how='left'
                )
                
                # Forward fill missing values
                future[col] = future[col].fillna(method='ffill')
                future[col] = future[col].fillna(0)  # Fill any remaining NaN with 0
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Add metadata
        forecast['product_id'] = self.product_id
        forecast['forecast_date'] = datetime.now()
        
        # Store results
        self.forecast_results = forecast
        
        logger.info(f"✅ Forecast generated: {len(forecast)} predictions")
        
        return forecast
    
    def get_forecast_summary(self, forecast: pd.DataFrame = None) -> Dict:
        """
        Get summary statistics of the forecast.
        
        Args:
            forecast: Forecast DataFrame (uses stored results if None)
            
        Returns:
            Summary statistics
        """
        if forecast is None:
            forecast = self.forecast_results
        
        if forecast is None:
            raise ValueError("No forecast available")
        
        # Future predictions only
        train_end = self.training_data['ds'].max()
        future_forecast = forecast[forecast['ds'] > train_end]
        
        summary = {
            'forecast_periods': len(future_forecast),
            'forecast_start': future_forecast['ds'].min(),
            'forecast_end': future_forecast['ds'].max(),
            'mean_prediction': future_forecast['yhat'].mean(),
            'total_predicted_demand': future_forecast['yhat'].sum(),
            'max_prediction': future_forecast['yhat'].max(),
            'min_prediction': future_forecast['yhat'].min(),
            'prediction_std': future_forecast['yhat'].std(),
            'confidence_width': (future_forecast['yhat_upper'] - future_forecast['yhat_lower']).mean()
        }
        
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
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Model saved to: {model_path}")
        return str(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_config = model_data['config']
            self.product_id = model_data['product_id']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"📁 Model loaded from: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def plot_forecast(self, forecast: pd.DataFrame = None, save_path: str = None) -> None:
        """
        Plot the forecast results.
        
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
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Forecast plot saved to: {save_path}")
        
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
            logger.info(f"📊 Components plot saved to: {save_path}")
        
        plt.show()


def quick_forecast_from_database() -> Dict:
    """
    Quick function to generate forecasts from WMS database data.
    
    Returns:
        Forecast results
    """
    logger.info("🚀 Quick forecast from database data")
    
    try:
        # Load processed WMS data
        processed_dir = Path(PROCESSED_DIR)
        wms_file = processed_dir / "wms_historical_data.csv"
        
        if not wms_file.exists():
            raise FileNotFoundError(f"WMS data not found at {wms_file}")
        
        # Load data
        data = pd.read_csv(wms_file)
        logger.info(f"📊 Loaded {len(data)} records from WMS database")
        
        # Get top products by volume
        top_products = data.groupby('product_id')['y'].sum().nlargest(5).index
        
        results = {}
        
        for product_id in top_products:
            logger.info(f"🔮 Forecasting for product: {product_id}")
            
            # Create forecaster for this product
            forecaster = SeasonalProphetForecaster(product_id=str(product_id))
            
            # Train model
            cv_results = forecaster.train(data)
            
            # Generate forecast
            forecast = forecaster.predict(periods=90)
            
            # Get summary
            summary = forecaster.get_forecast_summary()
            
            # Save model
            model_path = forecaster.save_model()
            
            results[product_id] = {
                'forecast': forecast,
                'summary': summary,
                'cv_results': cv_results,
                'model_path': model_path
            }
            
            logger.info(f"✅ Forecast completed for {product_id}")
        
        logger.info(f"🎉 Completed forecasts for {len(results)} products")
        return results
        
    except Exception as e:
        logger.error(f"❌ Quick forecast failed: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    # Demo usage
    print("🔮 Prophet Forecaster Demo")
    print("=" * 40)
    
    if not PROPHET_AVAILABLE:
        print("❌ Prophet not installed. Run: pip install prophet")
        exit(1)
    
    # Run quick forecast from database
    results = quick_forecast_from_database()
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"✅ Generated forecasts for {len(results)} products")
        
        for product_id, result in results.items():
            summary = result['summary']
            print(f"\n📊 Product {product_id}:")
            print(f"   • Forecast periods: {summary['forecast_periods']}")
            print(f"   • Total predicted demand: {summary['total_predicted_demand']:.0f}")
            print(f"   • Average daily demand: {summary['mean_prediction']:.1f}")


# Alias for backward compatibility
ProphetForecaster = SeasonalProphetForecaster
