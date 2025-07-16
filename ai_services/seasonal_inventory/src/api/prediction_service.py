"""
Prediction Service for Seasonal Inventory Forecasting

This service provides easy-to-use functions for generating inventory predictions
from your WMS database data.
"""

import pandas as pd
import asyncio
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

# Import our components
from data_collection.wms_data_extractor import WMSDataExtractor
from data_orchestrator import SeasonalDataOrchestrator
from models.prophet_forecaster import SeasonalProphetForecaster
from config import PROCESSED_DIR, MODELS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InventoryPredictionService:
    """
    High-level service for generating inventory predictions from database data.
    """
    
    def __init__(self):
        """Initialize the prediction service."""
        self.models_dir = Path(MODELS_DIR)
        self.processed_dir = Path(PROCESSED_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🎯 Inventory Prediction Service initialized")
    
    async def collect_and_prepare_data(self, days_back: int = 730) -> pd.DataFrame:
        """
        Collect data from WMS database and prepare for forecasting.
        
        Args:
            days_back: Number of days of historical data to collect
            
        Returns:
            Prepared DataFrame ready for Prophet
        """
        logger.info(f"📊 Collecting {days_back} days of WMS data")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Extract WMS data
        extractor = WMSDataExtractor()
        try:
            wms_data = await extractor.create_comprehensive_dataset(start_date, end_date)
            
            if wms_data.empty:
                logger.warning("⚠️ No WMS data found. Using sample data for demo.")
                return self._create_sample_data()
            
            logger.info(f"✅ Collected {len(wms_data)} records from WMS database")
            return wms_data
            
        except Exception as e:
            logger.error(f"❌ Failed to collect WMS data: {e}")
            logger.info("🔄 Generating sample data for demonstration")
            return self._create_sample_data()
            
        finally:
            await extractor.close()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample data for demonstration when database is not available.
        
        Returns:
            Sample DataFrame with realistic seasonal patterns
        """
        logger.info("🎭 Creating sample data with seasonal patterns")
        
        # Generate 2 years of daily data
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        
        sample_data = []
        
        # Sample products
        products = ['PROD_001', 'PROD_002', 'PROD_003', 'PROD_004', 'PROD_005']
        categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports']
        
        for i, product_id in enumerate(products):
            for date in dates:
                # Create realistic seasonal demand patterns
                base_demand = 50 + i * 10  # Different base levels per product
                
                # Yearly seasonality (higher in winter for electronics, summer for sports)
                yearly_factor = 1.0
                if categories[i] == 'Electronics':
                    yearly_factor = 1.2 if date.month in [11, 12] else 0.9
                elif categories[i] == 'Sports':
                    yearly_factor = 1.3 if date.month in [5, 6, 7] else 0.8
                elif categories[i] == 'Clothing':
                    yearly_factor = 1.1 if date.month in [3, 4, 9, 10] else 0.95
                
                # Weekly seasonality (higher on weekends)
                weekly_factor = 1.2 if date.weekday() in [5, 6] else 1.0
                
                # Monthly patterns (higher at month start/end)
                monthly_factor = 1.1 if date.day <= 5 or date.day >= 25 else 1.0
                
                # Random noise
                import random
                noise_factor = random.uniform(0.8, 1.2)
                
                # Calculate final demand
                demand = int(base_demand * yearly_factor * weekly_factor * monthly_factor * noise_factor)
                demand = max(1, demand)  # Ensure positive demand
                
                sample_data.append({
                    'ds': date,
                    'y': demand,
                    'product_id': product_id,
                    'category': categories[i],
                    'warehouse_id': 'WAREHOUSE_001',
                    'unit_price': 10 + i * 5,
                    'total_value': demand * (10 + i * 5)
                })
        
        df = pd.DataFrame(sample_data)
        
        # Save sample data
        sample_file = self.processed_dir / "sample_wms_data.csv"
        df.to_csv(sample_file, index=False)
        logger.info(f"💾 Sample data saved to {sample_file}")
        
        return df
    
    def get_top_products(self, data: pd.DataFrame, top_n: int = 10) -> List[str]:
        """
        Get top products by total demand volume.
        
        Args:
            data: Historical data
            top_n: Number of top products to return
            
        Returns:
            List of top product IDs
        """
        product_volumes = data.groupby('product_id')['y'].sum().nlargest(top_n)
        logger.info(f"📈 Top {top_n} products by volume: {list(product_volumes.index)}")
        return list(product_volumes.index)
    
    def train_product_models(self, data: pd.DataFrame, product_ids: List[str] = None) -> Dict[str, Dict]:
        """
        Train Prophet models for specific products.
        
        Args:
            data: Historical data
            product_ids: List of product IDs (trains top 5 if None)
            
        Returns:
            Dictionary of training results per product
        """
        if product_ids is None:
            product_ids = self.get_top_products(data, top_n=5)
        
        logger.info(f"🚀 Training models for {len(product_ids)} products")
        
        results = {}
        
        for product_id in product_ids:
            try:
                logger.info(f"🔄 Training model for product: {product_id}")
                
                # Create forecaster
                forecaster = SeasonalProphetForecaster(product_id=product_id)
                
                # Train model
                cv_results = forecaster.train(data)
                
                # Save model
                model_path = forecaster.save_model()
                
                results[product_id] = {
                    'forecaster': forecaster,
                    'cv_results': cv_results,
                    'model_path': model_path,
                    'status': 'success'
                }
                
                logger.info(f"✅ Model trained successfully for {product_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to train model for {product_id}: {e}")
                results[product_id] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        successful_models = sum(1 for r in results.values() if r['status'] == 'success')
        logger.info(f"🎯 Training completed: {successful_models}/{len(product_ids)} models successful")
        
        return results
    
    def generate_predictions(self, trained_models: Dict[str, Dict], 
                           forecast_days: int = 90) -> Dict[str, Dict]:
        """
        Generate predictions for all trained models.
        
        Args:
            trained_models: Dictionary of trained models
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary of predictions per product
        """
        logger.info(f"🔮 Generating {forecast_days}-day forecasts")
        
        predictions = {}
        
        for product_id, model_data in trained_models.items():
            if model_data['status'] != 'success':
                continue
            
            try:
                forecaster = model_data['forecaster']
                
                # Generate forecast
                forecast = forecaster.predict(periods=forecast_days)
                
                # Get summary
                summary = forecaster.get_forecast_summary()
                
                # Extract future predictions only
                train_end = forecaster.training_data['ds'].max()
                future_forecast = forecast[forecast['ds'] > train_end].copy()
                
                predictions[product_id] = {
                    'forecast': future_forecast,
                    'summary': summary,
                    'status': 'success'
                }
                
                logger.info(f"✅ Predictions generated for {product_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate predictions for {product_id}: {e}")
                predictions[product_id] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        successful_predictions = sum(1 for p in predictions.values() if p['status'] == 'success')
        logger.info(f"🎯 Predictions completed: {successful_predictions}/{len(trained_models)} successful")
        
        return predictions
    
    async def get_instant_predictions(self, product_ids: List[str] = None, 
                                    forecast_days: int = 90) -> Dict[str, Dict]:
        """
        One-click function to get predictions from your database.
        
        Args:
            product_ids: Specific products to forecast (top 5 if None)
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with predictions and summaries
        """
        logger.info("🚀 Starting instant prediction pipeline")
        
        try:
            # Step 1: Collect data from database
            data = await self.collect_and_prepare_data()
            
            if data.empty:
                return {'error': 'No data available for predictions'}
            
            # Step 2: Train models
            trained_models = self.train_product_models(data, product_ids)
            
            # Step 3: Generate predictions
            predictions = self.generate_predictions(trained_models, forecast_days)
            
            # Step 4: Create summary report
            summary_report = self._create_summary_report(predictions)
            
            logger.info("🎉 Instant predictions completed successfully!")
            
            return {
                'predictions': predictions,
                'summary': summary_report,
                'data_info': {
                    'total_records': len(data),
                    'date_range': f"{data['ds'].min()} to {data['ds'].max()}",
                    'products_analyzed': len(predictions)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Instant predictions failed: {e}")
            return {'error': str(e)}
    
    def _create_summary_report(self, predictions: Dict[str, Dict]) -> Dict:
        """
        Create a summary report of all predictions.
        
        Args:
            predictions: Dictionary of predictions per product
            
        Returns:
            Summary report
        """
        successful_predictions = {k: v for k, v in predictions.items() if v['status'] == 'success'}
        
        if not successful_predictions:
            return {'total_products': 0, 'error': 'No successful predictions'}
        
        # Aggregate statistics
        total_predicted_demand = sum(p['summary']['total_predicted_demand'] 
                                   for p in successful_predictions.values())
        
        avg_daily_demand = sum(p['summary']['mean_prediction'] 
                             for p in successful_predictions.values()) / len(successful_predictions)
        
        # Top products by predicted demand
        product_demands = [(pid, p['summary']['total_predicted_demand']) 
                          for pid, p in successful_predictions.items()]
        product_demands.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_products': len(successful_predictions),
            'total_predicted_demand': total_predicted_demand,
            'average_daily_demand': avg_daily_demand,
            'top_products_by_demand': product_demands[:5],
            'forecast_period_days': successful_predictions[list(successful_predictions.keys())[0]]['summary']['forecast_periods']
        }
    
    def print_prediction_summary(self, results: Dict):
        """
        Print a formatted summary of prediction results.
        
        Args:
            results: Results from get_instant_predictions()
        """
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print("\n🎯 INVENTORY PREDICTION RESULTS")
        print("=" * 50)
        
        # Data info
        data_info = results['data_info']
        print(f"📊 Data Analysis:")
        print(f"   • Total records analyzed: {data_info['total_records']:,}")
        print(f"   • Date range: {data_info['date_range']}")
        print(f"   • Products forecasted: {data_info['products_analyzed']}")
        
        # Summary statistics
        summary = results['summary']
        print(f"\n📈 Forecast Summary:")
        print(f"   • Total predicted demand: {summary['total_predicted_demand']:,.0f}")
        print(f"   • Average daily demand: {summary['average_daily_demand']:.1f}")
        print(f"   • Forecast period: {summary['forecast_period_days']} days")
        
        # Top products
        print(f"\n🏆 Top Products by Predicted Demand:")
        for i, (product_id, demand) in enumerate(summary['top_products_by_demand'], 1):
            print(f"   {i}. {product_id}: {demand:,.0f}")
        
        # Individual product details
        print(f"\n📋 Individual Product Forecasts:")
        predictions = results['predictions']
        for product_id, pred in predictions.items():
            if pred['status'] == 'success':
                s = pred['summary']
                print(f"   • {product_id}:")
                print(f"     - Total demand: {s['total_predicted_demand']:,.0f}")
                print(f"     - Daily average: {s['mean_prediction']:.1f}")
                print(f"     - Peak demand: {s['max_prediction']:.1f}")


async def quick_predict_inventory(product_ids: List[str] = None, 
                                forecast_days: int = 90) -> Dict:
    """
    Quick function to get inventory predictions from your database.
    
    Args:
        product_ids: Specific products to forecast (top 5 if None)
        forecast_days: Number of days to forecast
        
    Returns:
        Prediction results
    """
    service = InventoryPredictionService()
    return await service.get_instant_predictions(product_ids, forecast_days)


async def main():
    """Demo of the prediction service."""
    print("🎯 Inventory Prediction Service - Demo")
    print("=" * 50)
    
    # Get instant predictions
    results = await quick_predict_inventory()
    
    # Print results
    service = InventoryPredictionService()
    service.print_prediction_summary(results)
    
    print("\n🚀 Demo completed! Your database can now generate predictions.")


if __name__ == "__main__":
    asyncio.run(main())
