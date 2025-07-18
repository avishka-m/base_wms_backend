"""
Production Example: Using Optimized Prophet Hyperparameters
This example shows how to use the optimized hyperparameters for forecasting
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import prophet_config
import matplotlib.pyplot as plt

def train_optimized_models():
    """
    Train Prophet models using optimized hyperparameters for each category
    """
    
    print("="*60)
    print("TRAINING OPTIMIZED PROPHET MODELS")
    print("="*60)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Get all categories
    categories = prophet_config.get_all_categories()
    print(f"Categories: {categories}")
    
    models = {}
    performance = {}
    
    for category in categories:
        print(f"\nTraining {category} model...")
        
        # Get optimized parameters for this category
        params = prophet_config.get_category_params(category)
        print(f"Parameters: {params}")
        
        # Prepare category data
        cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
        
        # Train model with optimized parameters
        model = Prophet(**params)
        model.fit(cat_data)
        
        models[category] = model
        
        # Get expected performance
        perf = prophet_config.CATEGORY_PERFORMANCE[category]
        performance[category] = perf
        
        print(f"Expected RMSE: {perf['rmse']:.2f}")
        print(f"Improvement: {perf['improvement']}")
    
    return models, performance

def forecast_with_optimized_models(models, product_id, start_date, end_date):
    """
    Generate forecast for specific product using optimized models
    """
    
    # Load product-category mapping
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    product_category = df[df['product_id'] == product_id]['category'].iloc[0]
    
    print(f"\nForecasting product: {product_id}")
    print(f"Category: {product_category}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Get the trained model for this category
    model = models[product_category]
    
    # Calculate forecast periods
    periods = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    
    # Generate forecast
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # Filter to requested date range
    forecast_filtered = forecast[
        (forecast['ds'] >= start_date) & 
        (forecast['ds'] <= end_date)
    ]
    
    return forecast_filtered

def compare_default_vs_optimized():
    """
    Compare performance of default vs optimized parameters
    """
    
    print("\n" + "="*60)
    print("COMPARING DEFAULT VS OPTIMIZED PARAMETERS")
    print("="*60)
    
    # Load sample data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Test on books_media category
    category = 'books_media'
    cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
    
    # Split data
    split_idx = int(len(cat_data) * 0.8)
    train_data = cat_data[:split_idx]
    test_data = cat_data[split_idx:]
    
    # Test default parameters
    print(f"\nTesting default parameters...")
    default_model = Prophet()
    default_model.fit(train_data)
    
    default_future = default_model.make_future_dataframe(periods=len(test_data))
    default_forecast = default_model.predict(default_future)
    default_pred = default_forecast['yhat'][split_idx:].values
    
    from sklearn.metrics import mean_squared_error
    default_rmse = np.sqrt(mean_squared_error(test_data['y'].values, default_pred))
    
    # Test optimized parameters
    print(f"Testing optimized parameters...")
    optimized_params = prophet_config.get_category_params(category)
    optimized_model = Prophet(**optimized_params)
    optimized_model.fit(train_data)
    
    optimized_future = optimized_model.make_future_dataframe(periods=len(test_data))
    optimized_forecast = optimized_model.predict(optimized_future)
    optimized_pred = optimized_forecast['yhat'][split_idx:].values
    
    optimized_rmse = np.sqrt(mean_squared_error(test_data['y'].values, optimized_pred))
    
    # Compare results
    improvement = ((default_rmse - optimized_rmse) / default_rmse) * 100
    
    print(f"\nRESULTS FOR {category}:")
    print(f"Default RMSE: {default_rmse:.2f}")
    print(f"Optimized RMSE: {optimized_rmse:.2f}")
    print(f"Improvement: {improvement:.1f}%")
    
    return default_rmse, optimized_rmse, improvement

if __name__ == "__main__":
    
    print("PRODUCTION PROPHET WITH OPTIMIZED HYPERPARAMETERS")
    print("="*60)
    
    # Train all optimized models
    models, performance = train_optimized_models()
    
    # Compare default vs optimized
    default_rmse, optimized_rmse, improvement = compare_default_vs_optimized()
    
    # Example forecast
    print(f"\n{'='*60}")
    print("EXAMPLE FORECAST")
    print(f"{'='*60}")
    
    # Get sample product
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    sample_product = df['product_id'].iloc[0]
    
    # Generate forecast for next 30 days
    forecast = forecast_with_optimized_models(
        models, 
        sample_product, 
        '2025-01-01', 
        '2025-01-30'
    )
    
    print(f"\nForecast for {sample_product} (first 5 days):")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head().to_string(index=False))
    
    print(f"\nTotal 30-day demand: {forecast['yhat'].sum():.2f}")
    print(f"Average daily demand: {forecast['yhat'].mean():.2f}")
    
    print(f"\n{'='*60}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*60}")
    
    print("Optimal parameters by category:")
    for category in prophet_config.get_all_categories():
        params = prophet_config.get_category_params(category)
        perf = prophet_config.CATEGORY_PERFORMANCE[category]
        print(f"\n{category.upper()}:")
        print(f"  changepoint_prior_scale: {params['changepoint_prior_scale']}")
        print(f"  seasonality_prior_scale: {params['seasonality_prior_scale']}")
        print(f"  seasonality_mode: {params['seasonality_mode']}")
        print(f"  Expected RMSE: {perf['rmse']:.2f}")
    
    print(f"\n✅ Ready for production!")
    print(f"   Use prophet_config.get_category_params(category) to get optimal parameters")
    print(f"   Average improvement: ~17% better than default parameters")
