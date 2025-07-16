"""
Compare: ONE model with category regressor vs 6 separate category models
This tests whether we can get the benefits of category-specific patterns
while maintaining the simplicity of a single model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def compare_unified_regressor_vs_category_models():
    """
    Compare three approaches:
    1. Six separate category models
    2. One unified model with category as regressor
    3. One unified model with category dummy variables
    """
    
    print("="*80)
    print("UNIFIED MODEL WITH CATEGORY REGRESSOR vs 6 CATEGORY MODELS")
    print("="*80)
    print("Testing: Can ONE model with category regressor beat 6 separate models?")
    print("="*80)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    categories = df['category'].unique()
    print(f"Categories: {list(categories)}")
    print(f"Total data points: {len(df):,}")
    
    # Prepare unified dataset
    unified_data = df.groupby(['ds', 'category'])['y'].sum().reset_index()
    print(f"Unified data shape: {unified_data.shape}")
    
    # Split data (80% train, 20% test)
    unique_dates = unified_data['ds'].unique()
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    
    train_data = unified_data[unified_data['ds'] <= split_date].copy()
    test_data = unified_data[unified_data['ds'] > split_date].copy()
    
    print(f"Train data: {len(train_data)} rows")
    print(f"Test data: {len(test_data)} rows")
    print(f"Split date: {split_date}")
    
    results = {}
    
    # APPROACH 1: SIX SEPARATE CATEGORY MODELS
    print(f"\n{'='*60}")
    print("APPROACH 1: SIX SEPARATE CATEGORY MODELS")
    print(f"{'='*60}")
    
    category_models = {}
    category_forecasts = {}
    category_metrics = {}
    
    for category in categories:
        print(f"\nTraining {category} model...")
        
        # Prepare category data
        cat_train = train_data[train_data['category'] == category][['ds', 'y']].copy()
        cat_test = test_data[test_data['category'] == category][['ds', 'y']].copy()
        
        if len(cat_train) == 0 or len(cat_test) == 0:
            print(f"  ⚠️ Insufficient data for {category}")
            continue
        
        # Train Prophet model with optimal parameters
        model = Prophet(changepoint_prior_scale=0.5)  # From our previous analysis
        model.fit(cat_train)
        
        # Create future dataframe for test period
        future = model.make_future_dataframe(periods=len(cat_test))
        forecast = model.predict(future)
        
        # Extract test predictions
        test_predictions = forecast['yhat'][-len(cat_test):].values
        actual = cat_test['y'].values
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, test_predictions))
        mae = mean_absolute_error(actual, test_predictions)
        r2 = r2_score(actual, test_predictions)
        mape = np.mean(np.abs((actual - test_predictions) / actual)) * 100
        
        category_models[category] = model
        category_forecasts[category] = forecast
        category_metrics[category] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'data_points': len(actual)
        }
        
        print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
    
    # Calculate weighted average metrics for category models
    total_points = sum([m['data_points'] for m in category_metrics.values()])
    weighted_rmse = sum([m['rmse'] * m['data_points'] for m in category_metrics.values()]) / total_points
    weighted_mae = sum([m['mae'] * m['data_points'] for m in category_metrics.values()]) / total_points
    weighted_r2 = sum([m['r2'] * m['data_points'] for m in category_metrics.values()]) / total_points
    
    results['6_separate_models'] = {
        'rmse': weighted_rmse,
        'mae': weighted_mae,
        'r2': weighted_r2,
        'complexity': 'High (6 models)',
        'maintenance': 'Complex'
    }
    
    print(f"\n📊 CATEGORY MODELS SUMMARY:")
    print(f"  Weighted Average RMSE: {weighted_rmse:.2f}")
    print(f"  Weighted Average MAE: {weighted_mae:.2f}")
    print(f"  Weighted Average R²: {weighted_r2:.4f}")
    
    # APPROACH 2: ONE MODEL WITH CATEGORY AS NUMERICAL REGRESSOR
    print(f"\n{'='*60}")
    print("APPROACH 2: ONE MODEL WITH CATEGORY AS NUMERICAL REGRESSOR")
    print(f"{'='*60}")
    
    # Encode categories as numbers
    le = LabelEncoder()
    train_encoded = train_data.copy()
    test_encoded = test_data.copy()
    
    train_encoded['category_encoded'] = le.fit_transform(train_encoded['category'])
    test_encoded['category_encoded'] = le.transform(test_encoded['category'])
    
    print(f"Category encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Train unified model with category regressor
    unified_model_numeric = Prophet(changepoint_prior_scale=0.5)
    unified_model_numeric.add_regressor('category_encoded')
    unified_model_numeric.fit(train_encoded[['ds', 'y', 'category_encoded']])
    
    # Predict on test set
    future_numeric = test_encoded[['ds', 'category_encoded']].copy()
    forecast_numeric = unified_model_numeric.predict(future_numeric)
    
    # Calculate metrics
    actual_unified = test_encoded['y'].values
    pred_unified_numeric = forecast_numeric['yhat'].values
    
    rmse_numeric = np.sqrt(mean_squared_error(actual_unified, pred_unified_numeric))
    mae_numeric = mean_absolute_error(actual_unified, pred_unified_numeric)
    r2_numeric = r2_score(actual_unified, pred_unified_numeric)
    mape_numeric = np.mean(np.abs((actual_unified - pred_unified_numeric) / actual_unified)) * 100
    
    results['unified_numeric_regressor'] = {
        'rmse': rmse_numeric,
        'mae': mae_numeric,
        'r2': r2_numeric,
        'mape': mape_numeric,
        'complexity': 'Low (1 model)',
        'maintenance': 'Simple'
    }
    
    print(f"  RMSE: {rmse_numeric:.2f}")
    print(f"  MAE: {mae_numeric:.2f}")
    print(f"  R²: {r2_numeric:.4f}")
    print(f"  MAPE: {mape_numeric:.2f}%")
    
    # APPROACH 3: ONE MODEL WITH CATEGORY DUMMY VARIABLES
    print(f"\n{'='*60}")
    print("APPROACH 3: ONE MODEL WITH CATEGORY DUMMY VARIABLES")
    print(f"{'='*60}")
    
    # Create dummy variables for categories
    train_dummies = train_data.copy()
    test_dummies = test_data.copy()
    
    # Create dummy variables (excluding one category to avoid multicollinearity)
    dummy_categories = categories[1:]  # Exclude first category as reference
    
    for cat in dummy_categories:
        train_dummies[f'is_{cat}'] = (train_dummies['category'] == cat).astype(int)
        test_dummies[f'is_{cat}'] = (test_dummies['category'] == cat).astype(int)
    
    print(f"Created dummy variables for: {dummy_categories}")
    print(f"Reference category: {categories[0]}")
    
    # Train unified model with dummy regressors
    unified_model_dummy = Prophet(changepoint_prior_scale=0.5)
    
    for cat in dummy_categories:
        unified_model_dummy.add_regressor(f'is_{cat}')
    
    # Prepare training data with dummies
    train_cols = ['ds', 'y'] + [f'is_{cat}' for cat in dummy_categories]
    unified_model_dummy.fit(train_dummies[train_cols])
    
    # Predict on test set
    test_cols = ['ds'] + [f'is_{cat}' for cat in dummy_categories]
    future_dummy = test_dummies[test_cols].copy()
    forecast_dummy = unified_model_dummy.predict(future_dummy)
    
    # Calculate metrics
    pred_unified_dummy = forecast_dummy['yhat'].values
    
    rmse_dummy = np.sqrt(mean_squared_error(actual_unified, pred_unified_dummy))
    mae_dummy = mean_absolute_error(actual_unified, pred_unified_dummy)
    r2_dummy = r2_score(actual_unified, pred_unified_dummy)
    mape_dummy = np.mean(np.abs((actual_unified - pred_unified_dummy) / actual_unified)) * 100
    
    results['unified_dummy_regressors'] = {
        'rmse': rmse_dummy,
        'mae': mae_dummy,
        'r2': r2_dummy,
        'mape': mape_dummy,
        'complexity': 'Medium (1 model, 5 regressors)',
        'maintenance': 'Moderate'
    }
    
    print(f"  RMSE: {rmse_dummy:.2f}")
    print(f"  MAE: {mae_dummy:.2f}")
    print(f"  R²: {r2_dummy:.4f}")
    print(f"  MAPE: {mape_dummy:.2f}%")
    
    # APPROACH 4: SIMPLE UNIFIED MODEL (NO CATEGORY INFO)
    print(f"\n{'='*60}")
    print("APPROACH 4: SIMPLE UNIFIED MODEL (BASELINE)")
    print(f"{'='*60}")
    
    # Aggregate all categories together
    simple_train = train_data.groupby('ds')['y'].sum().reset_index()
    simple_test = test_data.groupby('ds')['y'].sum().reset_index()
    
    # Train simple unified model
    simple_model = Prophet(changepoint_prior_scale=0.5)
    simple_model.fit(simple_train)
    
    # Predict
    future_simple = simple_model.make_future_dataframe(periods=len(simple_test))
    forecast_simple = simple_model.predict(future_simple)
    
    # For comparison, we need to distribute predictions back to categories
    # Using historical proportions
    category_proportions = train_data.groupby('category')['y'].sum() / train_data['y'].sum()
    
    # Get total predictions for test period
    total_pred = forecast_simple['yhat'][-len(simple_test):].values
    
    # Distribute to categories and flatten for comparison
    distributed_pred = []
    actual_simple = []
    
    for i, date in enumerate(simple_test['ds']):
        total_pred_day = total_pred[i]
        actual_day = test_data[test_data['ds'] == date]
        
        for category in categories:
            pred_category = total_pred_day * category_proportions[category]
            distributed_pred.append(pred_category)
            
            actual_category = actual_day[actual_day['category'] == category]['y'].sum()
            actual_simple.append(actual_category)
    
    distributed_pred = np.array(distributed_pred)
    actual_simple = np.array(actual_simple)
    
    rmse_simple = np.sqrt(mean_squared_error(actual_simple, distributed_pred))
    mae_simple = mean_absolute_error(actual_simple, distributed_pred)
    r2_simple = r2_score(actual_simple, distributed_pred)
    mape_simple = np.mean(np.abs((actual_simple - distributed_pred) / actual_simple)) * 100
    
    results['simple_unified'] = {
        'rmse': rmse_simple,
        'mae': mae_simple,
        'r2': r2_simple,
        'mape': mape_simple,
        'complexity': 'Very Low (1 model)',
        'maintenance': 'Very Simple'
    }
    
    print(f"  RMSE: {rmse_simple:.2f}")
    print(f"  MAE: {mae_simple:.2f}")
    print(f"  R²: {r2_simple:.4f}")
    print(f"  MAPE: {mape_simple:.2f}%")
    
    # COMPARISON AND RECOMMENDATION
    print(f"\n{'='*80}")
    print("COMPREHENSIVE COMPARISON")
    print(f"{'='*80}")
    
    # Create comparison table
    comparison_data = []
    for approach, metrics in results.items():
        comparison_data.append({
            'Approach': approach.replace('_', ' ').title(),
            'RMSE': f"{metrics['rmse']:.2f}",
            'MAE': f"{metrics['mae']:.2f}",
            'R²': f"{metrics['r2']:.4f}",
            'MAPE': f"{metrics.get('mape', 0):.2f}%",
            'Complexity': metrics['complexity'],
            'Maintenance': metrics['maintenance']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Find best performing approach
    best_approach = min(results.keys(), key=lambda x: results[x]['rmse'])
    best_rmse = results[best_approach]['rmse']
    
    print(f"\n🏆 BEST PERFORMING APPROACH: {best_approach.replace('_', ' ').title()}")
    print(f"🏆 BEST RMSE: {best_rmse:.2f}")
    
    # Performance differences
    print(f"\n📊 PERFORMANCE DIFFERENCES FROM BEST:")
    for approach, metrics in results.items():
        diff = ((metrics['rmse'] - best_rmse) / best_rmse) * 100
        print(f"  {approach.replace('_', ' ').title():30s}: +{diff:5.1f}% worse" if diff > 0 else f"  {approach.replace('_', ' ').title():30s}: {diff:6.1f}% (BEST)")
    
    # Detailed analysis per category for best unified approach
    if 'unified' in best_approach:
        print(f"\n{'='*60}")
        print("CATEGORY-LEVEL PERFORMANCE ANALYSIS")
        print(f"{'='*60}")
        
        # Analyze performance per category for best unified model
        if best_approach == 'unified_dummy_regressors':
            predictions = pred_unified_dummy
        else:
            predictions = pred_unified_numeric
        
        category_analysis = []
        
        for category in categories:
            cat_mask = test_data['category'] == category
            cat_actual = test_data[cat_mask]['y'].values
            cat_pred = predictions[cat_mask]
            
            cat_rmse = np.sqrt(mean_squared_error(cat_actual, cat_pred))
            cat_mae = mean_absolute_error(cat_actual, cat_pred)
            cat_r2 = r2_score(cat_actual, cat_pred)
            
            category_analysis.append({
                'Category': category,
                'RMSE': f"{cat_rmse:.2f}",
                'MAE': f"{cat_mae:.2f}",
                'R²': f"{cat_r2:.4f}",
                'Data Points': len(cat_actual)
            })
        
        category_df = pd.DataFrame(category_analysis)
        print(category_df.to_string(index=False))
    
    # FINAL RECOMMENDATION
    print(f"\n{'='*80}")
    print("FINAL RECOMMENDATION")
    print(f"{'='*80}")
    
    # Calculate improvement over separate models
    separate_rmse = results['6_separate_models']['rmse']
    best_unified_rmse = results[best_approach]['rmse']
    improvement = ((separate_rmse - best_unified_rmse) / separate_rmse) * 100
    
    print(f"Performance comparison:")
    print(f"  6 Separate Models RMSE: {separate_rmse:.2f}")
    print(f"  Best Unified Approach RMSE: {best_unified_rmse:.2f}")
    print(f"  Performance difference: {improvement:+.1f}%")
    
    if abs(improvement) < 5:  # Less than 5% difference
        print(f"\n🎯 RECOMMENDATION: USE UNIFIED MODEL WITH CATEGORY REGRESSOR")
        print(f"Reasons:")
        print(f"  ✅ Performance is comparable ({abs(improvement):.1f}% difference)")
        print(f"  ✅ Much simpler to maintain (1 model vs 6)")
        print(f"  ✅ Easier deployment and monitoring")
        print(f"  ✅ Consistent training and retraining process")
        print(f"  ✅ Better for scaling to new categories")
        
        if best_approach == 'unified_dummy_regressors':
            print(f"\n📋 IMPLEMENTATION: Use dummy variables approach")
            print(f"  • Add dummy variables for categories (except reference)")
            print(f"  • Train one Prophet model with category regressors")
            print(f"  • Easier to interpret category effects")
        else:
            print(f"\n📋 IMPLEMENTATION: Use numerical encoding approach")
            print(f"  • Encode categories as numbers")
            print(f"  • Train one Prophet model with category regressor")
            print(f"  • Simpler implementation")
    
    else:
        if improvement > 0:
            print(f"\n🎯 RECOMMENDATION: USE UNIFIED MODEL")
            print(f"Unified model is {improvement:.1f}% better!")
        else:
            print(f"\n🎯 RECOMMENDATION: USE 6 SEPARATE MODELS")
            print(f"Separate models are {abs(improvement):.1f}% better")
            print(f"The performance gain justifies the added complexity")
    
    return results, best_approach

if __name__ == "__main__":
    results, best_approach = compare_unified_regressor_vs_category_models()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Answer: The best approach is {best_approach.replace('_', ' ').title()}")
    print(f"This analysis shows whether adding category regressors can match")
    print(f"the performance of separate models while maintaining simplicity.")
