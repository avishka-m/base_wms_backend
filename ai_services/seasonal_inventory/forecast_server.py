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

from ai_services.seasonal_inventory.src.models.prophet_forecaster import ProphetForecaster
from ai_services.seasonal_inventory.config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastServer:
    """Serves forecasts using pre-trained models."""
    
    def __init__(self):
        self.models_dir = Path(MODELS_DIR)
        self._available_products = None
    
    def get_available_products(self) -> List[str]:
        """Get list of products with trained models."""
        if self._available_products is None:
            self._available_products = []
            
            if not self.models_dir.exists():
                logger.warning(f"Models directory not found: {self.models_dir}")
                return self._available_products
            
            # Look for .pkl files in models directory
            for model_file in self.models_dir.glob("*.pkl"):
                # Extract product_id from filename (assuming format: prophet_model_PRODUCT_ID_HASH.pkl)
                filename = model_file.stem
                if filename.startswith("prophet_model_"):
                    parts = filename.split("_")
                    if len(parts) >= 3:
                        # Join all parts between "model" and the last part (hash)
                        product_id = "_".join(parts[2:-1])
                        if product_id not in self._available_products:
                            self._available_products.append(product_id)
            
            logger.info(f"Found trained models for {len(self._available_products)} products")
        
        return self._available_products
    
    def predict(self, product_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get forecast for a specific product and date range.
        
        Args:
            product_id: Product identifier
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            DataFrame with forecast data
        """
        available_products = self.get_available_products()
        
        if product_id not in available_products:
            raise ValueError(f"No trained model found for product {product_id}. "
                           f"Available products: {available_products}")
        
        # Initialize forecaster for this product
        forecaster = ProphetForecaster(product_id=product_id)
        
        # Load the trained model
        if not forecaster.load_model():
            raise ValueError(f"Failed to load model for product {product_id}")
        
        # Convert date strings to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Calculate number of days
        days = (end_dt - start_dt).days + 1
        
        # Make predictions
        forecast = forecaster.predict(days=days)
        
        # Filter forecast to requested date range
        forecast_filtered = forecast[
            (forecast['ds'] >= start_dt) & 
            (forecast['ds'] <= end_dt)
        ].copy()
        
        # Add product_id column for clarity
        forecast_filtered['product_id'] = product_id
        
        return forecast_filtered
    
    def get_model_info(self, product_id: str) -> Dict:
        """Get information about a trained model."""
        forecaster = ProphetForecaster(product_id=product_id)
        
        # Check if model exists
        model_path = forecaster._get_model_path()
        if not model_path.exists():
            return {"error": f"No model found for product {product_id}"}
        
        # Get model file info
        stat = model_path.stat()
        
        return {
            "product_id": product_id,
            "model_path": str(model_path),
            "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_date": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
            "modified_date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def list_all_models(self) -> List[Dict]:
        """Get information about all available models."""
        products = self.get_available_products()
        models_info = []
        
        for product_id in products:
            info = self.get_model_info(product_id)
            models_info.append(info)
        
        return models_info

def demo_forecast_server():
    """Demonstrate the forecast server functionality."""
    server = ForecastServer()
    
    # List available products
    products = server.get_available_products()
    print("\n🔍 Available Products with Trained Models:")
    print("-" * 40)
    for product in products:
        print(f"  - {product}")
    
    if not products:
        print("❌ No trained models found. Run train_all_products.py first.")
        return
    
    # Show model information
    print("\n📊 Model Information:")
    print("-" * 40)
    models_info = server.list_all_models()
    for info in models_info:
        print(f"Product: {info['product_id']}")
        print(f"  Size: {info['file_size_mb']} MB")
        print(f"  Created: {info['created_date']}")
        print()
    
    # Demo prediction for first product
    if products:
        demo_product = products[0]
        print(f"\n🎯 Demo Forecast for Product: {demo_product}")
        print("-" * 40)
        
        try:
            # Get 30-day forecast starting from today
            start_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            
            forecast = server.predict(demo_product, start_date, end_date)
            
            print(f"Forecast for {start_date} to {end_date}:")
            print(f"Total days: {len(forecast)}")
            print("\nFirst 5 days:")
            print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
            
            print(f"\nForecast Summary:")
            print(f"  Average daily demand: {forecast['yhat'].mean():.2f}")
            print(f"  Min demand: {forecast['yhat'].min():.2f}")
            print(f"  Max demand: {forecast['yhat'].max():.2f}")
            
        except Exception as e:
            print(f"❌ Failed to generate forecast: {str(e)}")

if __name__ == "__main__":
    demo_forecast_server()
