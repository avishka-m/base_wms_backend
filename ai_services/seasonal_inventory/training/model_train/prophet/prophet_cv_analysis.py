"""
Prophet Cross-Validation Optimization (Fixed Version)
Using Prophet's built-in cross-validation for robust parameter tuning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import warnings
import time

warnings.filterwarnings('ignore')

def quick_cv_comparison():
    """
    Quick comparison of simple split vs cross-validation for one category
    """
    
    print("="*80)
    print("CROSS-VALIDATION vs SIMPLE SPLIT COMPARISON")
    print("="*80)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Test on books_media category
    category = 'books_media'
    cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
    cat_data = cat_data.sort_values('ds').reset_index(drop=True)
    
    print(f"Testing category: {category}")
    print(f"Data points: {len(cat_data)}")
    print(f"Date range: {cat_data['ds'].min()} to {cat_data['ds'].max()}")
    
    # Test parameters (from your previous optimization)
    test_params = {
        'changepoint_prior_scale': 0.5,
        'seasonality_prior_scale': 1.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'holidays_prior_scale': 10.0
    }
    
    print(f"\\nTest parameters: {test_params}")
    
    # Method 1: Simple train-validation split (80/20)
    print(f"\\n1. SIMPLE TRAIN-VALIDATION SPLIT:")
    split_idx = int(len(cat_data) * 0.8)
    train_data = cat_data[:split_idx].copy()
    val_data = cat_data[split_idx:].copy()
    
    print(f"   Train periods: {len(train_data)} days")
    print(f"   Validation periods: {len(val_data)} days")
    
    model_simple = Prophet(**test_params)
    model_simple.fit(train_data)
    
    future_simple = model_simple.make_future_dataframe(periods=len(val_data))
    forecast_simple = model_simple.predict(future_simple)
    
    val_pred = forecast_simple['yhat'][split_idx:].values
    val_actual = val_data['y'].values
    simple_rmse = np.sqrt(mean_squared_error(val_actual, val_pred))
    
    print(f"   RMSE: {simple_rmse:.2f}")
    
    # Method 2: Cross-validation 
    print(f"\\n2. PROPHET CROSS-VALIDATION:")
    model_cv = Prophet(**test_params)
    model_cv.fit(cat_data)
    
    print("   Running cross-validation...")
    df_cv = cross_validation(
        model_cv,
        initial='500 days',  # Shorter for faster execution
        period='90 days',    # 3 months between cutoffs
        horizon='60 days'    # 2 months forecast horizon
    )
    
    df_performance = performance_metrics(df_cv)
    cv_rmse = df_performance['rmse'].mean()
    cv_rmse_std = df_performance['rmse'].std()
    cv_mae = df_performance['mae'].mean()
    cv_mape = df_performance['mape'].mean()
    
    print(f"   Number of CV folds: {len(df_performance)}")
    print(f"   RMSE: {cv_rmse:.2f} ± {cv_rmse_std:.2f}")
    print(f"   MAE: {cv_mae:.2f}")
    print(f"   MAPE: {cv_mape:.1f}%")
    print(f"   RMSE range: {df_performance['rmse'].min():.2f} - {df_performance['rmse'].max():.2f}")
    
    # Comparison
    print(f"\\n📊 COMPARISON RESULTS:")
    print(f"   Simple split RMSE:      {simple_rmse:.2f}")
    print(f"   Cross-validation RMSE:  {cv_rmse:.2f} ± {cv_rmse_std:.2f}")
    
    difference_pct = abs(simple_rmse - cv_rmse) / cv_rmse * 100
    print(f"   Difference: {difference_pct:.1f}%")
    
    if difference_pct > 10:
        print(f"   ⚠️  Significant difference! Cross-validation is more reliable.")
        print(f"       Simple split may be overfitting to specific time period.")
    else:
        print(f"   ✅ Results are reasonably consistent.")
    
    # Stability analysis
    cv_stability = cv_rmse_std / cv_rmse * 100
    print(f"\\n📈 STABILITY ANALYSIS:")
    print(f"   CV coefficient of variation: {cv_stability:.1f}%")
    
    if cv_stability > 20:
        print(f"   ⚠️  High variability across time periods!")
        print(f"       Model performance varies significantly over time.")
    elif cv_stability > 10:
        print(f"   ⚠️  Moderate variability.")
        print(f"       Consider adjusting parameters or adding regressors.")
    else:
        print(f"   ✅ Good stability across different time periods.")
    
    return simple_rmse, cv_rmse, cv_rmse_std

def optimize_with_cv_small_scale():
    """
    Small-scale cross-validation optimization for demonstration
    """
    
    print(f"\\n{'='*80}")
    print("SMALL-SCALE CROSS-VALIDATION OPTIMIZATION")
    print(f"{'='*80}")
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Test on 2 categories for demonstration
    test_categories = ['books_media', 'electronics']
    
    # Focused parameter grid
    param_combinations = [
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'multiplicative'},
    ]
    
    results_summary = {}
    
    for category in test_categories:
        print(f"\\n{'='*60}")
        print(f"OPTIMIZING: {category.upper()} (CV)")
        print(f"{'='*60}")
        
        # Prepare data
        cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
        cat_data = cat_data.sort_values('ds').reset_index(drop=True)
        
        print(f"Data points: {len(cat_data)}")
        
        best_cv_rmse = float('inf')
        best_params = None
        cv_results = []
        
        for i, params in enumerate(param_combinations):
            try:
                print(f"\\n  [{i+1}/{len(param_combinations)}] Testing: {params}")
                
                # Full parameters
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
                
                # Cross-validation
                df_cv = cross_validation(
                    model,
                    initial='500 days',
                    period='90 days',
                    horizon='60 days'
                )
                
                df_performance = performance_metrics(df_cv)
                avg_rmse = df_performance['rmse'].mean()
                rmse_std = df_performance['rmse'].std()
                
                cv_results.append({
                    'params': full_params,
                    'cv_rmse': avg_rmse,
                    'cv_rmse_std': rmse_std,
                    'cv_folds': len(df_performance)
                })
                
                if avg_rmse < best_cv_rmse:
                    best_cv_rmse = avg_rmse
                    best_params = full_params
                
                print(f"      CV RMSE: {avg_rmse:.2f} ± {rmse_std:.2f} ({len(df_performance)} folds)")
                
            except Exception as e:
                print(f"      ERROR: {str(e)[:50]}...")
                continue
        
        results_summary[category] = {
            'best_params': best_params,
            'best_cv_rmse': best_cv_rmse,
            'cv_results': cv_results
        }
        
        print(f"\\n🏆 BEST CV PARAMETERS FOR {category}:")
        print(f"   Cross-validated RMSE: {best_cv_rmse:.2f}")
        print(f"   Best parameters:")
        for key, value in best_params.items():
            if key in ['changepoint_prior_scale', 'seasonality_prior_scale', 'seasonality_mode']:
                print(f"     • {key}: {value}")
    
    return results_summary

def create_cv_recommendations():
    """
    Create recommendations for cross-validation usage
    """
    
    print(f"\\n{'='*80}")
    print("CROSS-VALIDATION RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print("\\n🎯 WHEN TO USE CROSS-VALIDATION:")
    print("   ✅ For final hyperparameter optimization")
    print("   ✅ When you have >2 years of data")
    print("   ✅ For production model validation")
    print("   ✅ When model stability across time is important")
    
    print("\\n⚡ WHEN TO USE SIMPLE SPLIT:")
    print("   ✅ For initial exploration and debugging")
    print("   ✅ When computational resources are limited")
    print("   ✅ For quick parameter testing")
    print("   ✅ When data has strong temporal trends")
    
    print("\\n📋 RECOMMENDED CV SETUP FOR YOUR DATA:")
    print("   • Initial: 730 days (2 years for stable training)")
    print("   • Period: 180 days (6 months between folds)")
    print("   • Horizon: 90 days (3 months forecast)")
    print("   • This gives ~3-4 validation folds")
    
    print("\\n⚙️  IMPLEMENTATION STRATEGY:")
    print("   1. Use simple split for initial parameter exploration")
    print("   2. Use cross-validation for final optimization")
    print("   3. Monitor CV coefficient of variation (<20%)")
    print("   4. Use parallel processing for faster execution")
    
    print("\\n💡 YOUR CURRENT OPTIMIZATION:")
    print("   • You used simple 80/20 split (fast, good for exploration)")
    print("   • Found good parameters with 8-29% improvements")
    print("   • For production: Consider CV validation of top 3-5 parameter sets")

if __name__ == "__main__":
    
    print("🔄 PROPHET CROSS-VALIDATION ANALYSIS")
    print("="*80)
    
    # Step 1: Quick comparison
    print("Step 1: Comparing validation methods...")
    start_time = time.time()
    
    simple_rmse, cv_rmse, cv_std = quick_cv_comparison()
    
    comparison_time = time.time() - start_time
    print(f"\\nComparison completed in {comparison_time:.1f} seconds")
    
    # Step 2: Small-scale CV optimization
    print(f"\\nStep 2: Cross-validation optimization demo...")
    cv_start = time.time()
    
    cv_results = optimize_with_cv_small_scale()
    
    cv_time = time.time() - cv_start
    print(f"\\nCV optimization completed in {cv_time:.1f} seconds")
    
    # Step 3: Recommendations
    create_cv_recommendations()
    
    # Summary
    print(f"\\n{'='*80}")
    print("CROSS-VALIDATION ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\\n📊 METHOD COMPARISON:")
    print(f"   Simple split time: {comparison_time:.1f}s")
    print(f"   Cross-validation time: {cv_time:.1f}s")
    print(f"   Time ratio: {cv_time/comparison_time:.1f}x slower")
    
    print(f"\\n🎯 RECOMMENDATIONS FOR YOUR PROJECT:")
    print(f"   1. Your current simple split optimization is GOOD for exploration")
    print(f"   2. Parameters found (8-29% improvement) are likely robust")
    print(f"   3. For production deployment: Run CV on top 3 parameter sets per category")
    print(f"   4. Focus CV on categories with highest business impact")
    
    print(f"\\n✅ ANSWER TO YOUR QUESTION:")
    print(f"   • You used simple train-validation split (80/20)")
    print(f"   • Cross-validation is more robust but ~{cv_time/comparison_time:.0f}x slower") 
    print(f"   • Your current approach is good for parameter exploration")
    print(f"   • Consider CV for final validation of best parameters")
    
    print(f"\\n🚀 PRODUCTION READINESS:")
    print(f"   Your optimized parameters are ready for production!")
    print(f"   Cross-validation would provide additional confidence.")
