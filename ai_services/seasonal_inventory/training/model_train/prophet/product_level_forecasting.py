"""
Product-Level Prophet Forecasting System
Handles individual product seasonal patterns and optimized hyperparameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import json
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')

class ProductLevelForecaster:
    """
    Advanced Prophet forecaster for individual products with category-optimized parameters
    """
    
    def __init__(self, config_path: str = 'prophet_cv_optimized_config.py'):
        """
        Initialize product forecaster with category-optimized parameters
        
        Args:
            config_path: Path to configuration file with optimized parameters
        """
        self.models = {}
        self.product_metadata = {}
        self.forecasts = {}
        
        # Load optimized parameters per category
        try:
            exec(open(config_path).read(), globals())
            self.category_params = CATEGORY_HYPERPARAMETERS
            print(f"✅ Loaded optimized parameters for {len(self.category_params)} categories")
        except:
            print("⚠️ Using default parameters (optimized config not found)")
            self.category_params = self._get_default_params()
    
    def _get_default_params(self) -> Dict:
        """Default parameters if optimized config not available"""
        return {
            'books_media': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
            'clothing': {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
            'electronics': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
            'health_beauty': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
            'home_garden': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
            'sports_outdoors': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'}
        }
    
    def analyze_product_patterns(self, df: pd.DataFrame, product_id: str) -> Dict:
        """
        Analyze seasonal and trend patterns for a specific product
        
        Args:
            df: Full dataset
            product_id: Product to analyze
            
        Returns:
            Dictionary with pattern analysis
        """
        product_data = df[df['product_id'] == product_id].copy()
        product_data = product_data.sort_values('ds').reset_index(drop=True)
        
        category = product_data['category'].iloc[0]
        
        # Basic statistics
        stats = {
            'product_id': product_id,
            'category': category,
            'total_demand': product_data['y'].sum(),
            'avg_daily_demand': product_data['y'].mean(),
            'std_demand': product_data['y'].std(),
            'cv': product_data['y'].std() / product_data['y'].mean() if product_data['y'].mean() > 0 else 0,
            'min_demand': product_data['y'].min(),
            'max_demand': product_data['y'].max(),
            'data_points': len(product_data),
            'date_range': (product_data['ds'].min(), product_data['ds'].max())
        }
        
        # Seasonal analysis
        product_data['month'] = product_data['ds'].dt.month
        product_data['dow'] = product_data['ds'].dt.dayofweek
        product_data['quarter'] = product_data['ds'].dt.quarter
        
        monthly_avg = product_data.groupby('month')['y'].mean()
        dow_avg = product_data.groupby('dow')['y'].mean()
        quarterly_avg = product_data.groupby('quarter')['y'].mean()
        
        # Seasonality strength (coefficient of variation of seasonal means)
        stats['monthly_seasonality'] = monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() > 0 else 0
        stats['weekly_seasonality'] = dow_avg.std() / dow_avg.mean() if dow_avg.mean() > 0 else 0
        stats['quarterly_seasonality'] = quarterly_avg.std() / quarterly_avg.mean() if quarterly_avg.mean() > 0 else 0
        
        # Trend analysis (simple linear trend)
        x = np.arange(len(product_data))
        trend_coef = np.polyfit(x, product_data['y'], 1)[0]
        stats['trend_slope'] = trend_coef
        stats['trend_direction'] = 'increasing' if trend_coef > 0 else 'decreasing' if trend_coef < 0 else 'stable'
        
        # Zero-demand analysis
        zero_days = (product_data['y'] == 0).sum()
        stats['zero_demand_ratio'] = zero_days / len(product_data)
        stats['has_intermittent_demand'] = stats['zero_demand_ratio'] > 0.1
        
        return stats
    
    def get_product_parameters(self, product_stats: Dict) -> Dict:
        """
        Get optimized parameters for a product based on its category and characteristics
        
        Args:
            product_stats: Product analysis results
            
        Returns:
            Optimized Prophet parameters
        """
        category = product_stats['category']
        base_params = self.category_params.get(category, self.category_params['books_media']).copy()
        
        # Adjust parameters based on product characteristics
        
        # High variability products - increase flexibility
        if product_stats['cv'] > 0.8:
            base_params['changepoint_prior_scale'] = min(1.0, base_params['changepoint_prior_scale'] * 1.5)
        
        # Low demand products - reduce seasonality sensitivity
        if product_stats['avg_daily_demand'] < 5:
            base_params['seasonality_prior_scale'] = max(0.01, base_params['seasonality_prior_scale'] * 0.5)
        
        # Intermittent demand - special handling
        if product_stats['has_intermittent_demand']:
            base_params['changepoint_prior_scale'] = max(0.001, base_params['changepoint_prior_scale'] * 0.5)
            base_params['seasonality_prior_scale'] = 0.01
        
        # Strong seasonal products - enhance seasonality
        if product_stats['monthly_seasonality'] > 0.3:
            base_params['seasonality_prior_scale'] = min(50.0, base_params['seasonality_prior_scale'] * 2)
        
        # Add fixed parameters
        base_params.update({
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'holidays_prior_scale': 10.0
        })
        
        return base_params
    
    def train_product_model(self, df: pd.DataFrame, product_id: str, 
                           validate: bool = True) -> Tuple[Prophet, Dict]:
        """
        Train Prophet model for a specific product
        
        Args:
            df: Full dataset
            product_id: Product to train
            validate: Whether to perform validation
            
        Returns:
            Tuple of (trained_model, validation_metrics)
        """
        # Get product data
        product_data = df[df['product_id'] == product_id].copy()
        product_data = product_data[['ds', 'y']].sort_values('ds').reset_index(drop=True)
        
        if len(product_data) < 60:  # Need minimum data
            raise ValueError(f"Insufficient data for {product_id}: {len(product_data)} days (minimum 60)")
        
        # Analyze product patterns
        product_stats = self.analyze_product_patterns(df, product_id)
        self.product_metadata[product_id] = product_stats
        
        # Get optimized parameters
        params = self.get_product_parameters(product_stats)
        
        print(f"Training {product_id} ({product_stats['category']}):")
        print(f"  • Data points: {len(product_data)}")
        print(f"  • Avg demand: {product_stats['avg_daily_demand']:.1f}")
        print(f"  • CV: {product_stats['cv']:.2f}")
        print(f"  • Monthly seasonality: {product_stats['monthly_seasonality']:.2f}")
        
        # Train model
        model = Prophet(**params)
        model.fit(product_data)
        
        # Store model
        self.models[product_id] = model
        
        # Validation if requested
        validation_metrics = {}
        if validate and len(product_data) > 180:  # Need enough data for CV
            try:
                # Quick validation with small CV
                cv_results = cross_validation(
                    model, 
                    initial='365 days', 
                    period='90 days', 
                    horizon='30 days',
                    parallel='processes'
                )
                metrics = performance_metrics(cv_results)
                validation_metrics = {
                    'rmse': metrics['rmse'].mean(),
                    'mae': metrics['mae'].mean(),
                    'mape': metrics['mape'].mean(),
                    'cv_folds': len(cv_results['cutoff'].unique())
                }
                print(f"  • CV RMSE: {validation_metrics['rmse']:.2f}")
            except Exception as e:
                print(f"  • CV failed: {str(e)[:50]}...")
                validation_metrics = {'error': str(e)}
        
        return model, validation_metrics
    
    def predict_product(self, product_id: str, periods: int = 30, 
                       include_history: bool = False) -> pd.DataFrame:
        """
        Generate forecast for a specific product
        
        Args:
            product_id: Product to forecast
            periods: Number of periods to forecast
            include_history: Include historical predictions
            
        Returns:
            Forecast DataFrame
        """
        if product_id not in self.models:
            raise ValueError(f"Model for {product_id} not trained")
        
        model = self.models[product_id]
        future = model.make_future_dataframe(periods=periods, include_history=include_history)
        forecast = model.predict(future)
        
        # Add product metadata
        forecast['product_id'] = product_id
        forecast['category'] = self.product_metadata[product_id]['category']
        
        # Store forecast
        self.forecasts[product_id] = forecast
        
        return forecast
    
    def bulk_train_products(self, df: pd.DataFrame, product_list: List[str] = None, 
                           max_products: int = None, min_demand: float = 1.0) -> Dict:
        """
        Train models for multiple products
        
        Args:
            df: Full dataset
            product_list: Specific products to train (if None, use all)
            max_products: Maximum number of products to train
            min_demand: Minimum average daily demand
            
        Returns:
            Training summary
        """
        if product_list is None:
            # Filter products by minimum demand
            product_demand = df.groupby('product_id')['y'].mean()
            product_list = product_demand[product_demand >= min_demand].index.tolist()
        
        if max_products:
            product_list = product_list[:max_products]
        
        print(f"Training models for {len(product_list)} products...")
        
        summary = {
            'total_products': len(product_list),
            'successful': 0,
            'failed': 0,
            'failed_products': [],
            'validation_results': {}
        }
        
        for i, product_id in enumerate(product_list):
            try:
                print(f"\n[{i+1}/{len(product_list)}] {product_id}")
                model, validation = self.train_product_model(df, product_id, validate=True)
                summary['successful'] += 1
                summary['validation_results'][product_id] = validation
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)[:100]}")
                summary['failed'] += 1
                summary['failed_products'].append((product_id, str(e)))
        
        print(f"\n✅ Training complete: {summary['successful']} successful, {summary['failed']} failed")
        return summary
    
    def create_product_comparison(self, df: pd.DataFrame, product_ids: List[str], 
                                 periods: int = 30) -> None:
        """
        Create comparison plots for multiple products
        
        Args:
            df: Full dataset
            product_ids: Products to compare
            periods: Forecast periods
        """
        fig, axes = plt.subplots(len(product_ids), 1, figsize=(15, 4*len(product_ids)))
        if len(product_ids) == 1:
            axes = [axes]
        
        for i, product_id in enumerate(product_ids):
            # Get historical data
            historical = df[df['product_id'] == product_id].copy()
            historical = historical.sort_values('ds')
            
            # Get forecast
            if product_id in self.forecasts:
                forecast = self.forecasts[product_id]
            else:
                forecast = self.predict_product(product_id, periods)
            
            # Plot
            ax = axes[i]
            
            # Historical data
            ax.plot(historical['ds'], historical['y'], 'o-', label='Actual', alpha=0.7, markersize=2)
            
            # Forecast
            future_data = forecast.tail(periods)
            ax.plot(future_data['ds'], future_data['yhat'], 'r-', label='Forecast', linewidth=2)
            ax.fill_between(future_data['ds'], future_data['yhat_lower'], 
                           future_data['yhat_upper'], alpha=0.3, color='red')
            
            # Styling
            stats = self.product_metadata.get(product_id, {})
            title = f"{product_id} ({stats.get('category', 'Unknown')})\n"
            title += f"Avg: {stats.get('avg_daily_demand', 0):.1f}, CV: {stats.get('cv', 0):.2f}"
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('product_forecasts_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def demo_product_forecasting():
    """
    Demonstration of product-level forecasting
    """
    print("="*80)
    print("PRODUCT-LEVEL FORECASTING DEMO")
    print("="*80)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Initialize forecaster
    forecaster = ProductLevelForecaster()
    
    # Select diverse products for demo
    print("\n📊 Selecting diverse products for demonstration...")
    
    # Get high-volume products from different categories
    demo_products = []
    for category in df['category'].unique():
        cat_products = df[df['category'] == category].groupby('product_id')['y'].sum()
        top_product = cat_products.nlargest(1).index[0]
        demo_products.append(top_product)
    
    print(f"Demo products: {demo_products}")
    
    # Train models
    print(f"\n🤖 Training models for {len(demo_products)} products...")
    summary = forecaster.bulk_train_products(df, demo_products)
    
    # Generate forecasts
    print(f"\n📈 Generating 30-day forecasts...")
    for product_id in demo_products:
        if product_id in forecaster.models:
            forecast = forecaster.predict_product(product_id, periods=30)
            future_demand = forecast.tail(30)['yhat'].sum()
            print(f"  {product_id}: {future_demand:.0f} total demand (next 30 days)")
    
    # Create comparison plots
    print(f"\n📊 Creating comparison plots...")
    successful_products = [p for p in demo_products if p in forecaster.models]
    if successful_products:
        forecaster.create_product_comparison(df, successful_products[:4], periods=30)
    
    # Summary statistics
    print(f"\n📋 SUMMARY:")
    print(f"  • Products analyzed: {len(demo_products)}")
    print(f"  • Models trained: {summary['successful']}")
    print(f"  • Training failures: {summary['failed']}")
    
    if summary['failed_products']:
        print(f"  • Failed products: {[p[0] for p in summary['failed_products']]}")
    
    return forecaster, summary

if __name__ == "__main__":
    # Run demonstration
    forecaster, summary = demo_product_forecasting()
    
    print(f"\n🎯 Product-level forecasting system ready!")
    print(f"  • Use forecaster.train_product_model(df, 'PRODUCT_ID') for individual products")
    print(f"  • Use forecaster.bulk_train_products(df, product_list) for batch training")
    print(f"  • Use forecaster.predict_product('PRODUCT_ID', periods=30) for forecasts")
