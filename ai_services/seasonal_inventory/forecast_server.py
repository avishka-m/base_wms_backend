#!/usr/bin/env python3
"""
Forecast Server - Load trained models and serve predictions.
This operates independently of the WMS database.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_services.seasonal_inventory.src.models.prophet_forecaster import ProphetCategoryPredictor
from ai_services.seasonal_inventory.config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastServer:
    """Serves forecasts using pre-trained models."""
    
    def __init__(self):
        self.models_dir = Path(MODELS_DIR)
        self._available_products = None
    
    def get_available_categories(self) -> List[str]:
        """Get list of categories with trained models."""
        if self._available_products is None:
            self._available_products = []
            if not self.models_dir.exists():
                logger.warning(f"Models directory not found: {self.models_dir}")
                return self._available_products
            # Look for .pkl files in models directory
            for model_file in self.models_dir.glob("*.pkl"):
                filename = model_file.stem
                # Support both 'prophet_model_CATEGORY_HASH.pkl' and 'category_CATEGORYNAME_prophet_model.pkl'
                if filename.startswith("prophet_model_"):
                    parts = filename.split("_")
                    if len(parts) >= 3:
                        category = "_".join(parts[2:-1])
                        if category not in self._available_products:
                            self._available_products.append(category)
                elif filename.startswith("category_") and filename.endswith("_prophet_model"):
                    parts = filename.split("_")
                    if len(parts) >= 4:
                        # Join all parts between 'category' and 'prophet_model'
                        category = "_".join(parts[1:-2])
                        if category not in self._available_products:
                            self._available_products.append(category)
            logger.info(f"Found trained models for {len(self._available_products)} categories")
        return self._available_products
    
    def predict(self, category: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get forecast for a specific category and date range.
        Args:
            category: Category identifier
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
        Returns:
            DataFrame with forecast data
        """
        available_categories = self.get_available_categories()
        if category not in available_categories:
            raise ValueError(f"No trained model found for category {category}. "
                           f"Available categories: {available_categories}")
        # Initialize forecaster for this category
        forecaster = ProphetCategoryPredictor(category=category)
        # Load the trained model
        if not forecaster.load_model():
            raise ValueError(f"Failed to load model for category {category}")
        # Convert date strings to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        # Generate list of dates for prediction
        future_dates = pd.date_range(start=start_dt, end=end_dt).strftime('%Y-%m-%d').tolist()
        # Make predictions
        forecast = forecaster.predict(future_dates)
        # Filter forecast to requested date range (should already be filtered, but for safety)
        forecast_filtered = forecast[
            (forecast['ds'] >= start_dt) & 
            (forecast['ds'] <= end_dt)
        ].copy()
        # Add category column for clarity
        forecast_filtered['category'] = category
        return forecast_filtered
    
    def get_model_info(self, category: str) -> Dict:
        """Get information about a trained model."""
        forecaster = ProphetCategoryPredictor(category=category)
        # Use the model_path attribute directly
        model_path = forecaster.model_path
        if not model_path.exists():
            return {"error": f"No model found for category {category}"}
        # Get model file info
        stat = model_path.stat()
        return {
            "category": category,
            "model_path": str(model_path),
            "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_date": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
            "modified_date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def list_all_models(self) -> List[Dict]:
        """Get information about all available models."""
        categories = self.get_available_categories()
        models_info = []
        for category in categories:
            info = self.get_model_info(category)
            models_info.append(info)
        return models_info

def demo_forecast_server():
    """Demonstrate the forecast server functionality."""
    server = ForecastServer()
    
    # List available products
    categories = server.get_available_categories()
    print("\n Available Categories with Trained Models:")
    print("-" * 40)
    for category in categories:
        print(f"  - {category}")
    if not categories:
        print(" No trained models found. Run train_all_products.py first.")
        return
    # Show model information
    print("\n Model Information:")
    print("-" * 40)
    models_info = server.list_all_models()
    for info in models_info:
        print(f"Category: {info['category']}")
        print(f"  Size: {info['file_size_mb']} MB")
        print(f"  Created: {info['created_date']}")
        print()
    # Demo prediction for first category
    if categories:
        demo_category = categories[0]
        print(f"\nDemo Forecast for Category: {demo_category}")
        print("-" * 40)
        try:
            # Get 30-day forecast starting from tomorrow
            start_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            forecast = server.predict(demo_category, start_date, end_date)
            print(f"Forecast for {start_date} to {end_date}:")
            print(f"Total days: {len(forecast)}")
            print("\nFirst 5 days:")
            print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
            print(f"\nForecast Summary:")
            print(f"  Average daily demand: {forecast['yhat'].mean():.2f}")
            print(f"  Min demand: {forecast['yhat'].min():.2f}")
            print(f"  Max demand: {forecast['yhat'].max():.2f}")
        except Exception as e:
            print(f"Failed to generate forecast: {str(e)}")

if __name__ == "__main__":
    demo_forecast_server()
