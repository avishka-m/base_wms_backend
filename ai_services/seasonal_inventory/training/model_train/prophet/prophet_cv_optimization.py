"""
Prophet Hyperparameter Optimization with Cross-Validation
Using Prophet's built-in cross_validation for robust parameter tuning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
fro        python_cv_config += f'        "validation_method": "cross_validation"\n'
        python_cv_config += f'    }},\n'
    
    python_cv_config += '}'learn.metrics import mean_absolute_error, mean_squared_error
import json
import yaml
import warnings
from itertools import product
import time

warnings.filterwarnings('ignore')

def prophet_cross_validation_optimization():
    """
    Find optimal hyperparameters using Prophet's cross-validation
    This is more robust than simple train-validation split
    """
    
    print("="*80)
    print("PROPHET CROSS-VALIDATION HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    categories = df['category'].unique()
    print(f"Categories to optimize: {list(categories)}")
    
    # Cross-validation parameters
    # For 2+ years of data, we can do multiple folds
    initial = '730 days'    # 2 years initial training
    period = '180 days'     # 6 months between cutoffs  
    horizon = '90 days'     # 3 months forecast horizon
    
    print(f"\nCross-validation setup:")
    print(f"  Initial training period: {initial}")
    print(f"  Period between cutoffs: {period}")
    print(f"  Forecast horizon: {horizon}")
    
    # Parameter grid - more focused based on your previous results
    param_combinations = [
        # Best performers from previous optimization
        {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
        
        # Multiplicative variants
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'multiplicative'},
    ]
    
    optimal_params = {}
    category_results = {}
    
    for category in categories:
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATING: {category.upper()}")
        print(f"{'='*60}")
        
        # Prepare category data
        cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
        cat_data = cat_data.sort_values('ds').reset_index(drop=True)
        
        print(f"Data points: {len(cat_data)}")
        print(f"Date range: {cat_data['ds'].min()} to {cat_data['ds'].max()}")
        
        best_score = float('inf')
        best_params = None
        cv_results = []
        
        for i, params in enumerate(param_combinations):
            try:
                print(f"\n  [{i+1:2d}/{len(param_combinations)}] Testing: {params}")
                
                # Add default parameters
                full_params = {
                    'yearly_seasonality': True,
                    'weekly_seasonality': True,
                    'daily_seasonality': False,
                    'holidays_prior_scale': 10.0,
                    **params
                }
                
                # Train model
                model = Prophet(**full_params)
                model.fit(cat_data)
                
                # Perform cross-validation
                df_cv = cross_validation(
                    model, 
                    initial=initial, 
                    period=period, 
                    horizon=horizon,
                    parallel="processes"  # Use parallel processing
                )
                
                # Calculate performance metrics
                df_performance = performance_metrics(df_cv)
                
                # Get average RMSE across all CV folds
                avg_rmse = df_performance['rmse'].mean()
                avg_mae = df_performance['mae'].mean()
                avg_mape = df_performance['mape'].mean()
                
                cv_results.append({
                    'params': full_params,
                    'rmse': avg_rmse,
                    'mae': avg_mae, 
                    'mape': avg_mape,
                    'rmse_std': df_performance['rmse'].std(),
                    'n_folds': len(df_performance)
                })
                
                if avg_rmse < best_score:
                    best_score = avg_rmse
                    best_params = full_params.copy()
                
                print(f"      RMSE: {avg_rmse:7.2f} ± {df_performance['rmse'].std():5.2f} ({len(df_performance)} folds)")
                
            except Exception as e:
                print(f"      ERROR: {str(e)[:60]}...")
                continue
        
        optimal_params[category] = best_params
        category_results[category] = {
            'best_params': best_params,
            'best_rmse': best_score,
            'cv_results': cv_results
        }
        
        print(f"\n🏆 BEST CV PARAMETERS FOR {category}:")
        print(f"   Cross-validated RMSE: {best_score:.2f}")
        for key, value in best_params.items():
            print(f"   {key}: {value}")
    
    return optimal_params, category_results

def compare_cv_vs_simple_split():
    """
    Compare cross-validation results with simple train-validation split
    """
    
    print(f"\n{'='*80}")
    print("COMPARING: CROSS-VALIDATION vs SIMPLE SPLIT")
    print(f"{'='*80}")
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Test on one category
    category = 'books_media'
    cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
    
    # Test parameters
    test_params = {
        'changepoint_prior_scale': 0.5,
        'seasonality_prior_scale': 1.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'holidays_prior_scale': 10.0
    }
    
    print(f"Testing category: {category}")
    print(f"Test parameters: {test_params}")
    
    # Method 1: Simple train-validation split
    print(f"\n1. SIMPLE TRAIN-VALIDATION SPLIT (80/20):")
    split_idx = int(len(cat_data) * 0.8)
    train_data = cat_data[:split_idx]
    val_data = cat_data[split_idx:]
    
    model_simple = Prophet(**test_params)
    model_simple.fit(train_data)
    
    future_simple = model_simple.make_future_dataframe(periods=len(val_data))
    forecast_simple = model_simple.predict(future_simple)
    
    val_pred = forecast_simple['yhat'][split_idx:].values
    val_actual = val_data['y'].values
    simple_rmse = np.sqrt(mean_squared_error(val_actual, val_pred))
    
    print(f"   Train periods: {len(train_data)}")
    print(f"   Validation periods: {len(val_data)}")
    print(f"   RMSE: {simple_rmse:.2f}")
    
    # Method 2: Cross-validation
    print(f"\n2. CROSS-VALIDATION:")
    model_cv = Prophet(**test_params)
    model_cv.fit(cat_data)
    
    df_cv = cross_validation(
        model_cv,
        initial='730 days',
        period='180 days', 
        horizon='90 days'
    )
    
    df_performance = performance_metrics(df_cv)
    cv_rmse = df_performance['rmse'].mean()
    cv_rmse_std = df_performance['rmse'].std()
    
    print(f"   Number of folds: {len(df_performance)}")
    print(f"   RMSE: {cv_rmse:.2f} ± {cv_rmse_std:.2f}")
    print(f"   RMSE range: {df_performance['rmse'].min():.2f} - {df_performance['rmse'].max():.2f}")
    
    # Comparison
    print(f"\n📊 COMPARISON:")
    print(f"   Simple split RMSE: {simple_rmse:.2f}")
    print(f"   Cross-validation RMSE: {cv_rmse:.2f} ± {cv_rmse_std:.2f}")
    
    if abs(simple_rmse - cv_rmse) / cv_rmse > 0.1:
        print(f"   ⚠️  Significant difference! CV is more reliable.")
    else:
        print(f"   ✅ Results are consistent.")
    
    return simple_rmse, cv_rmse, cv_rmse_std

def create_cv_config_files(optimal_params, category_results):
    """
    Create configuration files with cross-validation results
    """
    
    print(f"\n{'='*60}")
    print("CREATING CV-OPTIMIZED CONFIGURATION FILES")
    print(f"{'='*60}")
    
    # Enhanced configuration with CV statistics
    cv_config = {
        "model_config": {
            "prophet_version": "1.1.0",
            "optimization_method": "cross_validation",
            "cv_setup": {
                "initial": "730 days",
                "period": "180 days", 
                "horizon": "90 days"
            },
            "created_date": "2025-01-15",
            "description": "Cross-validation optimized Prophet hyperparameters",
            "validation_method": "Time series cross-validation with multiple folds",
            "optimization_metric": "Average RMSE across CV folds"
        },
        "categories": {}
    }
    
    # Add performance statistics for each category
    for category, params in optimal_params.items():
        cv_results = category_results[category]['cv_results']
        best_result = min(cv_results, key=lambda x: x['rmse'])
        
        cv_config["categories"][category] = {
            "hyperparameters": params,
            "performance": {
                "best_rmse": category_results[category]['best_rmse'],
                "rmse_std": best_result['rmse_std'],
                "n_cv_folds": best_result['n_folds'],
                "mae": best_result['mae'],
                "mape": best_result['mape']
            },
            "notes": f"Cross-validation optimized for {category} category"
        }
    
    # Save enhanced config
    with open('prophet_cv_config.json', 'w') as f:
        json.dump(cv_config, f, indent=4)
    
    # Python module with CV results
    python_cv_config = f'''"""
Cross-Validation Optimized Prophet Configuration
Generated using Prophet's built-in cross-validation
Generated on: 2025-01-15
"""

# Cross-validation optimized hyperparameters
CATEGORY_HYPERPARAMETERS = {{
'''
    
    for category, params in optimal_params.items():
        python_cv_config += f'    "{category}": {{\n'
        for key, value in params.items():
            if isinstance(value, str):
                python_cv_config += f'        "{key}": "{value}",\n'
            else:
                python_cv_config += f'        "{key}": {value},\n'
        python_cv_config += f'    }},\n'
    
    python_cv_config += '''}

# Cross-validation performance metrics
CATEGORY_PERFORMANCE = {
'''
    
    for category in optimal_params.keys():
        results = category_results[category]
        best_result = min(results['cv_results'], key=lambda x: x['rmse'])
        
        python_cv_config += f'    "{category}": {{\n'
        python_cv_config += f'        "rmse": {best_result["rmse"]:.2f},\n'
        python_cv_config += f'        "rmse_std": {best_result["rmse_std"]:.2f},\n'
        python_cv_config += f'        "mae": {best_result["mae"]:.2f},\n'
        python_cv_config += f'        "mape": {best_result["mape"]:.2f},\n'
        python_cv_config += f'        "cv_folds": {best_result["n_folds"]},\n'
        python_cv_config += f'        "validation_method": "cross_validation"\n'
        python_cv_config += f'    }},\n'
    
    python_cv_config += '''}

# Cross-validation setup used
CV_CONFIG = {
    "initial": "730 days",
    "period": "180 days", 
    "horizon": "90 days",
    "parallel": "processes"
}

def get_category_params(category):
    """Get cross-validation optimized parameters for a specific category"""
    return CATEGORY_HYPERPARAMETERS.get(category, DEFAULT_PARAMETERS)

def get_category_performance(category):
    """Get cross-validation performance metrics for a category"""
    return CATEGORY_PERFORMANCE.get(category, {})

def get_all_categories():
    """Get list of all available categories"""
    return list(CATEGORY_HYPERPARAMETERS.keys())

DEFAULT_PARAMETERS = {
    "changepoint_prior_scale": 0.5,
    "seasonality_prior_scale": 10.0,
    "seasonality_mode": "additive",
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "holidays_prior_scale": 10.0
}
'''
    
    with open('prophet_cv_config.py', 'w') as f:
        f.write(python_cv_config)
    
    print("✅ Cross-validation configuration files created:")
    print("   • prophet_cv_config.json (JSON with CV metrics)")
    print("   • prophet_cv_config.py (Python module with CV results)")

if __name__ == "__main__":
    
    print("🔄 PROPHET CROSS-VALIDATION OPTIMIZATION")
    print("="*80)
    
    # First, compare methods
    print("Step 1: Comparing validation methods...")
    simple_rmse, cv_rmse, cv_std = compare_cv_vs_simple_split()
    
    # Run cross-validation optimization
    print(f"\nStep 2: Running cross-validation optimization...")
    start_time = time.time()
    
    optimal_params, category_results = prophet_cross_validation_optimization()
    
    end_time = time.time()
    print(f"\nCross-validation optimization completed in {end_time - start_time:.1f} seconds")
    
    # Create configuration files
    create_cv_config_files(optimal_params, category_results)
    
    # Summary
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    
    print("\n📊 CROSS-VALIDATION OPTIMIZED PARAMETERS:")
    for category, params in optimal_params.items():
        results = category_results[category]
        best_result = min(results['cv_results'], key=lambda x: x['rmse'])
        
        print(f"\n{category.upper()}:")
        print(f"   CV RMSE: {best_result['rmse']:.2f} ± {best_result['rmse_std']:.2f}")
        print(f"   CV Folds: {best_result['n_folds']}")
        print(f"   Parameters:")
        for key, value in params.items():
            if key not in ['yearly_seasonality', 'weekly_seasonality', 'daily_seasonality']:
                print(f"     • {key}: {value}")
    
    print(f"\n✅ CROSS-VALIDATION BENEFITS:")
    print("   • More robust parameter estimates")
    print("   • Multiple validation folds reduce overfitting")
    print("   • Better estimates of model uncertainty")
    print("   • Accounts for different time periods")
    
    print(f"\n🚀 READY FOR PRODUCTION!")
    print("   Use prophet_cv_config.py for cross-validation optimized parameters")
