"""
Comprehensive Cross-Validation Hyperparameter Optimization
Using Prophet's cross_validation for robust parameter tuning across all categories
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
from itertools import product

warnings.filterwarnings('ignore')

def comprehensive_cv_optimization():
    """
    Comprehensive cross-validation optimization for all categories
    """
    
    print("="*80)
    print("COMPREHENSIVE CROSS-VALIDATION HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    categories = df['category'].unique()
    print(f"Categories to optimize: {list(categories)}")
    
    # Cross-validation setup
    cv_config = {
        'initial': '600 days',    # ~1.6 years initial training
        'period': '120 days',     # 4 months between cutoffs
        'horizon': '90 days'      # 3 months forecast horizon
    }
    
    print(f"\nCross-validation configuration:")
    print(f"  Initial training: {cv_config['initial']}")
    print(f"  Period between cutoffs: {cv_config['period']}")
    print(f"  Forecast horizon: {cv_config['horizon']}")
    
    # Comprehensive parameter grid
    param_combinations = [
        # Conservative changepoint approaches
        {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 0.01, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
        
        {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 0.01, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'additive'},
        
        {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'additive'},
        
        # Balanced approaches
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 0.01, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'additive'},
        
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.01, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'additive'},
        
        # High flexibility approaches
        {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 0.01, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'additive'},
        
        # Multiplicative seasonality variants
        {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'multiplicative'},
        
        {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'multiplicative'},
        
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'multiplicative'},
        
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'multiplicative'},
        
        {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'multiplicative'},
    ]
    
    print(f"Total parameter combinations to test: {len(param_combinations)}")
    print(f"Estimated time: ~{len(param_combinations) * len(categories) * 2:.0f} minutes")
    
    optimal_params = {}
    cv_results = {}
    
    for category in categories:
        print(f"\n{'='*70}")
        print(f"CROSS-VALIDATING: {category.upper()}")
        print(f"{'='*70}")
        
        # Prepare category data
        cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
        cat_data = cat_data.sort_values('ds').reset_index(drop=True)
        
        print(f"Data points: {len(cat_data)}")
        print(f"Date range: {cat_data['ds'].min()} to {cat_data['ds'].max()}")
        
        best_cv_rmse = float('inf')
        best_params = None
        category_cv_results = []
        
        for i, params in enumerate(param_combinations):
            try:
                print(f"\\n  [{i+1:2d}/{len(param_combinations)}] Testing: {params}")
                
                # Full parameters with defaults
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
                    initial=cv_config['initial'],
                    period=cv_config['period'],
                    horizon=cv_config['horizon'],
                    parallel="processes"
                )
                
                # Calculate performance metrics
                df_performance = performance_metrics(df_cv)
                
                # Aggregate metrics
                avg_rmse = df_performance['rmse'].mean()
                rmse_std = df_performance['rmse'].std()
                avg_mae = df_performance['mae'].mean()
                avg_mape = df_performance['mape'].mean()
                n_folds = len(df_performance)
                
                # Calculate stability metric (coefficient of variation)
                cv_stability = rmse_std / avg_rmse * 100
                
                category_cv_results.append({
                    'params': full_params,
                    'rmse': avg_rmse,
                    'rmse_std': rmse_std,
                    'mae': avg_mae,
                    'mape': avg_mape,
                    'cv_stability': cv_stability,
                    'n_folds': n_folds,
                    'rmse_min': df_performance['rmse'].min(),
                    'rmse_max': df_performance['rmse'].max()
                })
                
                if avg_rmse < best_cv_rmse:
                    best_cv_rmse = avg_rmse
                    best_params = full_params.copy()
                
                print(f"      RMSE: {avg_rmse:7.2f} ± {rmse_std:5.2f} | Stability: {cv_stability:4.1f}% | Folds: {n_folds}")
                
            except Exception as e:
                print(f"      ERROR: {str(e)[:60]}...")
                continue
        
        optimal_params[category] = best_params
        cv_results[category] = {
            'best_params': best_params,
            'best_rmse': best_cv_rmse,
            'all_results': category_cv_results
        }
        
        # Find best result for detailed reporting
        best_result = min(category_cv_results, key=lambda x: x['rmse'])
        
        print(f"\\n🏆 BEST CV PARAMETERS FOR {category}:")
        print(f"   CV RMSE: {best_result['rmse']:.2f} ± {best_result['rmse_std']:.2f}")
        print(f"   Stability: {best_result['cv_stability']:.1f}% (lower is better)")
        print(f"   CV Folds: {best_result['n_folds']}")
        print(f"   RMSE Range: {best_result['rmse_min']:.2f} - {best_result['rmse_max']:.2f}")
        print(f"   Best parameters:")
        for key, value in best_params.items():
            if key in ['changepoint_prior_scale', 'seasonality_prior_scale', 'seasonality_mode']:
                print(f"     • {key}: {value}")
    
    return optimal_params, cv_results, cv_config

def analyze_cv_results(cv_results):
    """
    Analyze cross-validation results and provide insights
    """
    
    print(f"\\n{'='*80}")
    print("CROSS-VALIDATION RESULTS ANALYSIS")
    print(f"{'='*80}")
    
    for category, results in cv_results.items():
        best_result = min(results['all_results'], key=lambda x: x['rmse'])
        
        print(f"\\n📊 {category.upper()} ANALYSIS:")
        print(f"   Best RMSE: {best_result['rmse']:.2f} ± {best_result['rmse_std']:.2f}")
        print(f"   Stability: {best_result['cv_stability']:.1f}%")
        
        # Stability assessment
        if best_result['cv_stability'] < 15:
            print(f"   ✅ Excellent stability across time periods")
        elif best_result['cv_stability'] < 25:
            print(f"   ⚠️  Moderate stability - acceptable for production")
        else:
            print(f"   ❌ High variability - model performance varies significantly")
        
        # Parameter insights
        params = best_result['params']
        print(f"   Key parameters:")
        print(f"     • Changepoint flexibility: {params['changepoint_prior_scale']}")
        print(f"     • Seasonality strength: {params['seasonality_prior_scale']}")
        print(f"     • Seasonality type: {params['seasonality_mode']}")
        
        # Compare top 3 results
        sorted_results = sorted(results['all_results'], key=lambda x: x['rmse'])[:3]
        print(f"   Top 3 parameter sets:")
        for j, result in enumerate(sorted_results):
            p = result['params']
            print(f"     {j+1}. RMSE: {result['rmse']:.2f} | CP: {p['changepoint_prior_scale']} | SP: {p['seasonality_prior_scale']} | Mode: {p['seasonality_mode']}")

def create_cv_optimized_config(optimal_params, cv_results, cv_config):
    """
    Create production configuration files with cross-validation results
    """
    
    print(f"\\n{'='*80}")
    print("CREATING CV-OPTIMIZED CONFIGURATION FILES")
    print(f"{'='*80}")
    
    # Calculate improvement over defaults for each category
    default_params = {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'holidays_prior_scale': 10.0
    }
    
    # Enhanced configuration with comprehensive CV results
    cv_config_dict = {
        "model_config": {
            "prophet_version": "1.1.0",
            "optimization_method": "cross_validation",
            "cv_setup": cv_config,
            "created_date": "2025-01-15",
            "description": "Cross-validation optimized Prophet hyperparameters with comprehensive grid search",
            "validation_method": "Time series cross-validation with multiple folds",
            "optimization_metric": "Average RMSE across CV folds",
            "total_combinations_tested": len(cv_results[list(cv_results.keys())[0]]['all_results']),
            "default_baseline": default_params
        },
        "categories": {}
    }
    
    # Process each category
    for category, params in optimal_params.items():
        results = cv_results[category]
        best_result = min(results['all_results'], key=lambda x: x['rmse'])
        
        # Calculate top 3 parameter sets
        sorted_results = sorted(results['all_results'], key=lambda x: x['rmse'])[:3]
        top_3_params = []
        for result in sorted_results:
            top_3_params.append({
                'parameters': {k: v for k, v in result['params'].items() 
                             if k in ['changepoint_prior_scale', 'seasonality_prior_scale', 'seasonality_mode']},
                'rmse': result['rmse'],
                'rmse_std': result['rmse_std'],
                'stability': result['cv_stability']
            })
        
        cv_config_dict["categories"][category] = {
            "best_hyperparameters": params,
            "performance": {
                "cv_rmse": best_result['rmse'],
                "cv_rmse_std": best_result['rmse_std'],
                "cv_mae": best_result['mae'],
                "cv_mape": best_result['mape'],
                "cv_stability_pct": best_result['cv_stability'],
                "cv_folds": best_result['n_folds'],
                "rmse_range": {
                    "min": best_result['rmse_min'],
                    "max": best_result['rmse_max']
                }
            },
            "top_3_alternatives": top_3_params,
            "optimization_notes": f"Cross-validation optimized across {len(results['all_results'])} parameter combinations",
            "stability_assessment": "excellent" if best_result['cv_stability'] < 15 else 
                                   "good" if best_result['cv_stability'] < 25 else "variable"
        }
    
    # Save comprehensive JSON config
    with open('prophet_cv_optimized_config.json', 'w') as f:
        json.dump(cv_config_dict, f, indent=4)
    
    # Create Python configuration module
    python_config = '''"""
Cross-Validation Optimized Prophet Configuration
Comprehensive hyperparameter optimization using Prophet's cross-validation
Generated on: 2025-01-15
"""

# Cross-validation optimized hyperparameters (best performing)
CATEGORY_HYPERPARAMETERS = {
'''
    
    for category, params in optimal_params.items():
        python_config += f'    "{category}": {{\n'
        for key, value in params.items():
            if isinstance(value, str):
                python_config += f'        "{key}": "{value}",\n'
            else:
                python_config += f'        "{key}": {value},\n'
        python_config += f'    }},\n'
    
    python_config += '}\n\n'
    
    # Add performance metrics
    python_config += '# Cross-validation performance metrics\n'
    python_config += 'CATEGORY_CV_PERFORMANCE = {\n'
    
    for category in optimal_params.keys():
        results = cv_results[category]
        best_result = min(results['all_results'], key=lambda x: x['rmse'])
        
        python_config += f'    "{category}": {{\n'
        python_config += f'        "cv_rmse": {best_result["rmse"]:.2f},\n'
        python_config += f'        "cv_rmse_std": {best_result["rmse_std"]:.2f},\n'
        python_config += f'        "cv_mae": {best_result["mae"]:.2f},\n'
        python_config += f'        "cv_mape": {best_result["mape"]:.2f},\n'
        python_config += f'        "stability_pct": {best_result["cv_stability"]:.1f},\n'
        python_config += f'        "cv_folds": {best_result["n_folds"]},\n'
        python_config += f'        "rmse_min": {best_result["rmse_min"]:.2f},\n'
        python_config += f'        "rmse_max": {best_result["rmse_max"]:.2f},\n'
        python_config += f'        "validation_method": "cross_validation",\n'
        stability = "excellent" if best_result['cv_stability'] < 15 else "good" if best_result['cv_stability'] < 25 else "variable"
        python_config += f'        "stability_rating": "{stability}"\n'
        python_config += f'    }},\n'
    
    python_config += '}\n\n'
    
    # Add alternative parameter sets
    python_config += '# Top 3 alternative parameter sets per category\n'
    python_config += 'CATEGORY_ALTERNATIVES = {\n'
    
    for category in optimal_params.keys():
        results = cv_results[category]
        sorted_results = sorted(results['all_results'], key=lambda x: x['rmse'])[:3]
        
        python_config += f'    "{category}": [\n'
        for result in sorted_results:
            params_subset = {k: v for k, v in result['params'].items() 
                           if k in ['changepoint_prior_scale', 'seasonality_prior_scale', 'seasonality_mode']}
            python_config += f'        {{\n'
            python_config += f'            "params": {params_subset},\n'
            python_config += f'            "rmse": {result["rmse"]:.2f},\n'
            python_config += f'            "stability": {result["cv_stability"]:.1f}\n'
            python_config += f'        }},\n'
        python_config += f'    ],\n'
    
    python_config += '}\n\n'
    
    # Add utility functions
    python_config += '''# Cross-validation configuration used
CV_CONFIG = ''' + str(cv_config) + '''

def get_category_params(category):
    """Get cross-validation optimized parameters for a specific category"""
    if category not in CATEGORY_HYPERPARAMETERS:
        raise ValueError(f"Category '{category}' not found. Available: {list(CATEGORY_HYPERPARAMETERS.keys())}")
    return CATEGORY_HYPERPARAMETERS[category]

def get_category_performance(category):
    """Get cross-validation performance metrics for a category"""
    return CATEGORY_CV_PERFORMANCE.get(category, {})

def get_alternative_params(category, rank=1):
    """Get alternative parameter sets (rank 1=best, 2=second best, etc.)"""
    if category not in CATEGORY_ALTERNATIVES:
        raise ValueError(f"Category '{category}' not found")
    if rank < 1 or rank > len(CATEGORY_ALTERNATIVES[category]):
        raise ValueError(f"Rank must be between 1 and {len(CATEGORY_ALTERNATIVES[category])}")
    
    alt_params = CATEGORY_ALTERNATIVES[category][rank-1]['params']
    # Add default values
    full_params = {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'holidays_prior_scale': 10.0,
        **alt_params
    }
    return full_params

def get_all_categories():
    """Get list of all available categories"""
    return list(CATEGORY_HYPERPARAMETERS.keys())

def get_stability_rating(category):
    """Get stability rating for a category (excellent/good/variable)"""
    return CATEGORY_CV_PERFORMANCE.get(category, {}).get('stability_rating', 'unknown')

# Default fallback parameters
DEFAULT_PARAMETERS = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "seasonality_mode": "additive",
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "holidays_prior_scale": 10.0
}
'''
    
    with open('prophet_cv_optimized_config.py', 'w') as f:
        f.write(python_config)
    
    print("✅ Cross-validation optimized configuration files created:")
    print("   • prophet_cv_optimized_config.json (Comprehensive CV results)")
    print("   • prophet_cv_optimized_config.py (Python module with alternatives)")

if __name__ == "__main__":
    
    print("🔄 COMPREHENSIVE CROSS-VALIDATION OPTIMIZATION")
    print("="*80)
    
    # Run comprehensive cross-validation optimization
    start_time = time.time()
    
    optimal_params, cv_results, cv_config = comprehensive_cv_optimization()
    
    end_time = time.time()
    optimization_time = end_time - start_time
    print(f"\\nComprehensive CV optimization completed in {optimization_time/60:.1f} minutes")
    
    # Analyze results
    analyze_cv_results(cv_results)
    
    # Create configuration files
    create_cv_optimized_config(optimal_params, cv_results, cv_config)
    
    # Final summary
    print(f"\\n{'='*80}")
    print("CROSS-VALIDATION OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    
    print(f"\\n⏱️  PERFORMANCE SUMMARY:")
    print(f"   Total optimization time: {optimization_time/60:.1f} minutes")
    print(f"   Categories optimized: {len(optimal_params)}")
    
    total_combinations = len(cv_results[list(cv_results.keys())[0]]['all_results'])
    print(f"   Parameter combinations tested: {total_combinations} per category")
    print(f"   Total CV evaluations: {total_combinations * len(optimal_params)}")
    
    print(f"\\n📊 BEST PARAMETERS BY CATEGORY:")
    for category, params in optimal_params.items():
        results = cv_results[category]
        best_result = min(results['all_results'], key=lambda x: x['rmse'])
        
        print(f"\\n{category.upper()}:")
        print(f"   CV RMSE: {best_result['rmse']:.2f} ± {best_result['rmse_std']:.2f}")
        print(f"   Stability: {best_result['cv_stability']:.1f}%")
        print(f"   Best parameters:")
        for key, value in params.items():
            if key in ['changepoint_prior_scale', 'seasonality_prior_scale', 'seasonality_mode']:
                print(f"     • {key}: {value}")
    
    print(f"\\n🚀 PRODUCTION READY!")
    print("   Use prophet_cv_optimized_config.py for cross-validation optimized parameters")
    print("   Each category has been optimized with comprehensive grid search")
    print("   Alternative parameter sets available for A/B testing")
