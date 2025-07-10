import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path#for handling the file system paths
import pickle #or saving the model to the disk
import json#for reading and writing the model configuration or result

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

from ..config import PROPHET_CONFIG, MODELS_DIR, PROCESSED_DIR
from base_wms_backend.app.services.inventory_service import InventoryService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

df = pd.read_csv(str(Path(PROCESSED_DIR) / "daily_demand_by_product_modern.csv"))
print(df.columns)

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
        Add external regressors to the model.
        
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
                from ..config import FEATURE_CONFIG
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
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to: {model_path}")
        return str(model_path)
    
# #     def load_model(self, model_path: str) -> bool:
# #         """
# #         Load a trained model from disk.
        
# #         Args:
# #             model_path: Path to saved model
            
# #         Returns:
# #             True if loaded successfully
# #         """
# #         try:
# #             with open(model_path, 'rb') as f:
# #                 model_data = pickle.load(f)
            
# #             self.model = model_data['model']
# #             self.model_config = model_data['config']
# #             self.product_id = model_data['product_id']
# #             self.is_trained = model_data['is_trained']
            
# #             logger.info(f" Model loaded from: {model_path}")
# #             return True
            
# #         except Exception as e:
# #             logger.error(f" Failed to load model: {e}")
# #             return False
    
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
    
def plot_demand_by_month_and_year(data: pd.DataFrame, product_id: str = None):
    """
    Plot demand by month and by year for a given product (or all products).
    """
    import matplotlib.pyplot as plt
    if product_id:
        data = data[data['product_id'] == product_id]
    data = data.copy()
    data['month'] = data['ds'].dt.month
    data['year'] = data['ds'].dt.year
    monthly = data.groupby(['year', 'month'])['y'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    for year in sorted(monthly['year'].unique()):
        plt.plot(
            monthly[monthly['year'] == year]['month'],
            monthly[monthly['year'] == year]['y'],
            marker='o', label=f'Year {year}'
        )
    plt.title(f"Monthly Demand Pattern{' for Product ' + str(product_id) if product_id else ''}")
    plt.xlabel('Month')
    plt.ylabel('Total Demand')
    plt.legend()
    plt.show()
    # Demand by year (total)
    yearly = data.groupby('year')['y'].sum().reset_index()
    plt.figure(figsize=(8, 4))
    plt.bar(yearly['year'], yearly['y'])
    plt.title(f"Yearly Demand Pattern{' for Product ' + str(product_id) if product_id else ''}")
    plt.xlabel('Year')
    plt.ylabel('Total Demand')
    plt.show()

def plot_demand_by_month_year(data: pd.DataFrame, product_id: str, month: int = None, year: int = None):
    """
    Plot demand for a specific product, filtered by month and/or year.
    Args:
        data: DataFrame with columns ['ds', 'y', 'product_id']
        product_id: Product to plot
        month: (Optional) Month (1-12) to filter
        year: (Optional) Year (e.g., 2024) to filter
    """
    df = data[data['product_id'] == product_id].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    if month is not None:
        df = df[df['ds'].dt.month == month]
    if year is not None:
        df = df[df['ds'].dt.year == year]
    if df.empty:
        print(f"No data for product {product_id} in month={month}, year={year}")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], marker='o', linestyle='-', label=f'Demand for {product_id}')
    plt.title(f"Demand for Product {product_id}" + (f" - {year}" if year else "") + (f"/Month {month}" if month else ""))
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def process_and_describe_data(data: pd.DataFrame, min_days: int = 30, max_zero_ratio: float = 0.8) -> pd.DataFrame:
    """
    Clean data, remove outliers, and remove products with insufficient days or too many zero-demand days.
    """
    # Remove missing values
    data = data.dropna(subset=['ds', 'y', 'product_id'])
    data['ds'] = pd.to_datetime(data['ds'])
    # Remove outliers in y using IQR
    Q1 = data['y'].quantile(0.25)
    Q3 = data['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(data)
    data = data[(data['y'] >= lower) & (data['y'] <= upper)]
    after = len(data)
    print(f"Removed {before - after} outlier rows from y column.")
    # Remove products with insufficient days
    counts = data.groupby('product_id')['ds'].nunique()
    valid_products = counts[counts >= min_days].index
    filtered_data = data[data['product_id'].isin(valid_products)]
    # Remove products with too many zero-demand days
    zero_ratio = filtered_data.groupby('product_id')['y'].apply(lambda x: (x == 0).sum() / len(x))
    valid_products = zero_ratio[zero_ratio <= max_zero_ratio].index
    filtered_data = filtered_data[filtered_data['product_id'].isin(valid_products)]
    print(f"Removed products with more than {int(max_zero_ratio*100)}% zero-demand days. Remaining products: {len(valid_products)}")
    print("\nDataset description (filtered, outliers removed):")
    print(filtered_data.describe(include='all'))
    print(f"Number of products with at least {min_days} days and <= {int(max_zero_ratio*100)}% zero days: {len(valid_products)}")
    print("Sample product counts:")
    print(counts[counts >= min_days].head())
    return filtered_data

def quick_forecast_from_database() -> Dict:
  
    logger.info("Quick forecast from database data")
    
    try:
        # Load processed WMS data
        processed_dir = Path(PROCESSED_DIR)
        wms_file = processed_dir / "daily_demand_by_product_modern.csv"

        if not wms_file.exists():
            raise FileNotFoundError(f"WMS data not found at {wms_file}")
        
        # Load data
        data = pd.read_csv(wms_file)
        
        # Process and describe data
        data = process_and_describe_data(data, min_days=30)
        logger.info(f"Loaded {len(data)} records from WMS database after filtering")
        
        # Get top products by volume
        top_products = data.groupby('product_id')['y'].sum().nlargest(5).index
        
        results = {}
        
        for product_id in top_products:
            logger.info(f"Forecasting for product: {product_id}")
            # Create forecaster for this product
            forecaster = ProphetForecaster(product_id=str(product_id))
            # Train model
            cv_results = forecaster.train(data)
            # Save the training dataframe used for this product
            training_df = forecaster.training_data.copy()
            # Generate forecast
            forecast = forecaster.predict(periods=90)
            # Save each day's forecast to the database
            for _, row in forecast.iterrows():
                InventoryService.save_demand_forecast(
                    product_id=str(product_id),
                    date=row['ds'],
                    predicted_demand=row['yhat']
                )
            # Get summary
            summary = forecaster.get_forecast_summary()
            # Save model
            model_path = forecaster.save_model()
            results[product_id] = {
                'forecast': forecast,
                'summary': summary,
                'cv_results': cv_results,
                'model_path': model_path,
                'training_data': training_df  # <-- Added here
            }
            logger.info(f"Forecast completed for {product_id}")
            forecaster.plot_forecast(forecast)
            forecaster.plot_components(forecast)
        
        logger.info(f"Completed forecasts for {len(results)} products")
        return results
        
    except Exception as e:
        logger.error(f" Quick forecast failed: {e}")
        return {'error': str(e)}


def forecast_for_product_and_range(data: pd.DataFrame, product_id: str, start_date: str, end_date: str, model_config: Dict = None) -> pd.DataFrame:
    """
    Forecast demand for a specific product and date range, using a cached model if it's less than 24 hours old.
    Args:
        data: DataFrame with columns ['ds', 'y', 'product_id']
        product_id: Product to forecast
        start_date: Forecast start date (YYYY-MM-DD)
        end_date: Forecast end date (YYYY-MM-DD)
        model_config: Optional Prophet config
    Returns:
        Forecast DataFrame for the requested range
    """
    import os
    from datetime import datetime, timedelta
    from pathlib import Path
    models_dir = Path(MODELS_DIR)
    model_files = list(models_dir.glob(f"prophet_model_{product_id}_*.pkl"))
    forecaster = ProphetForecaster(model_config=model_config, product_id=product_id)
    model_loaded = False
    if model_files:
        # Use the most recent model
        model_files.sort(reverse=True)
        model_path = model_files[0]
        mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
        if datetime.now() - mtime < timedelta(hours=24):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                forecaster.model = model_data['model']
                forecaster.model_config = model_data['config']
                forecaster.product_id = model_data['product_id']
                forecaster.is_trained = model_data['is_trained']
                model_loaded = True
                logger.info(f"Loaded cached model for product {product_id} from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load cached model, will retrain: {e}")
    if not model_loaded:
        logger.info(f"No valid cached model for product {product_id}, training new model.")
        forecaster.train(data)
        forecaster.save_model()
    # Prepare future dataframe for the requested date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    periods = (end_dt - start_dt).days + 1
    forecast = forecaster.predict(periods=periods, include_history=True)
    forecast_range = forecast[(forecast['ds'] >= start_dt) & (forecast['ds'] <= end_dt)]
    return forecast_range

if __name__ == "__main__":

    print("Prophet Forecaster")

    
    # if not PROPHET_AVAILABLE:
    #     print("Prophet not installed. Run: pip install prophet")
    #     exit(1)
    #check for product ids
    
    print(df['product_id'].astype(str).unique())
    print('10023' in df['product_id'].astype(str).unique())
    
    
    # Run quick forecast from database
    results = quick_forecast_from_database()
    # Plot monthly and yearly demand for each top product
    if 'error' not in results:
        for product_id, result in results.items():
            print(f"\nPlotting monthly/yearly demand for product {product_id}...")
            plot_demand_by_month_and_year(result['training_data'], product_id=product_id)
            # Plot demand for each year and each month in that year
            training_data = result['training_data']
            years = training_data['ds'].dt.year.unique()
            for year in years:
                print(f"Plotting demand for product {product_id} in year {year}...")
                plot_demand_by_month_year(training_data, product_id=product_id, year=year)
                months = training_data[training_data['ds'].dt.year == year]['ds'].dt.month.unique()
                for month in months:
                    print(f"Plotting demand for product {product_id} in {year}-{month:02d}...")
                    plot_demand_by_month_year(training_data, product_id=product_id, year=year, month=month)
    else:
        print(f"Error: {results['error']}")
    
    # Predict for a specific product and date range

    example_product_id = 'PROD_2022_BOOK_0000' #one product 

    if example_product_id:
        start_date = '2025-08-01'
        end_date = '2025-08-31'
        print(f"\nForecast for product {example_product_id} from {start_date} to {end_date}:")
        forecast_df = forecast_for_product_and_range(df, product_id=example_product_id, start_date=start_date, end_date=end_date)
        print(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    else:
        print("No product found for example forecast.")

