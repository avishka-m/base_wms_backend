"""
Analyze if seasonality patterns are consistent across categories
This will determine if we can use ONE model for ALL products instead of 6 category models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

def analyze_unified_vs_category_models():
    """Compare unified model vs category-specific models"""
    
    print("="*80)
    print("UNIFIED MODEL vs CATEGORY MODELS ANALYSIS")
    print("="*80)
    print("Question: Can we train ONE model for ALL products instead of 6 category models?")
    print("="*80)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    categories = df['category'].unique()
    print(f"Categories: {list(categories)}")
    print(f"Total data points: {len(df):,}")
    
    # 1. TEST SEASONALITY MODE CONSISTENCY
    print(f"\n{'='*60}")
    print("1. SEASONALITY MODE ANALYSIS (Additive vs Multiplicative)")
    print(f"{'='*60}")
    
    seasonality_results = {}
    
    for category in categories:
        print(f"\nTesting {category}...")
        
        # Prepare category data
        cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
        cat_data = cat_data.sort_values('ds').reset_index(drop=True)
        
        # Split data
        split_idx = int(len(cat_data) * 0.8)
        train_data = cat_data[:split_idx]
        test_data = cat_data[split_idx:]
        actual = test_data['y'].values
        
        # Test additive vs multiplicative
        results = {}
        
        # Additive seasonality
        try:
            model_add = Prophet(seasonality_mode='additive')
            model_add.fit(train_data)
            future = model_add.make_future_dataframe(periods=len(test_data))
            forecast = model_add.predict(future)
            pred = forecast['yhat'][split_idx:].values
            rmse_add = np.sqrt(mean_squared_error(actual, pred))
            results['additive'] = rmse_add
        except:
            results['additive'] = float('inf')
        
        # Multiplicative seasonality
        try:
            model_mult = Prophet(seasonality_mode='multiplicative')
            model_mult.fit(train_data)
            future = model_mult.make_future_dataframe(periods=len(test_data))
            forecast = model_mult.predict(future)
            pred = forecast['yhat'][split_idx:].values
            rmse_mult = np.sqrt(mean_squared_error(actual, pred))
            results['multiplicative'] = rmse_mult
        except:
            results['multiplicative'] = float('inf')
        
        # Determine best
        best_mode = min(results, key=results.get)
        seasonality_results[category] = {
            'best_mode': best_mode,
            'additive_rmse': results['additive'],
            'multiplicative_rmse': results['multiplicative'],
            'improvement': abs(results['additive'] - results['multiplicative']) / min(results.values()) * 100
        }
        
        print(f"  Additive RMSE: {results['additive']:.2f}")
        print(f"  Multiplicative RMSE: {results['multiplicative']:.2f}")
        print(f"  Best: {best_mode} (improvement: {seasonality_results[category]['improvement']:.1f}%)")
    
    # Check consistency
    best_modes = [seasonality_results[cat]['best_mode'] for cat in categories]
    mode_consistency = len(set(best_modes)) == 1
    most_common_mode = max(set(best_modes), key=best_modes.count)
    
    print(f"\n📊 SEASONALITY MODE CONSISTENCY:")
    print(f"All categories prefer same mode: {mode_consistency}")
    print(f"Most common mode: {most_common_mode} ({best_modes.count(most_common_mode)}/{len(categories)} categories)")
    
    # 2. TEST UNIFIED MODEL PERFORMANCE
    print(f"\n{'='*60}")
    print("2. UNIFIED MODEL vs CATEGORY MODELS PERFORMANCE")
    print(f"{'='*60}")
    
    # Prepare unified dataset (all categories combined)
    unified_data = df.groupby('ds')['y'].sum().reset_index()
    unified_data = unified_data.sort_values('ds').reset_index(drop=True)
    
    print(f"Unified dataset shape: {unified_data.shape}")
    print(f"Total daily demand range: {unified_data['y'].min():.0f} to {unified_data['y'].max():.0f}")
    
    # Split unified data
    split_idx = int(len(unified_data) * 0.8)
    unified_train = unified_data[:split_idx]
    unified_test = unified_data[split_idx:]
    
    # Train unified model with best mode
    print(f"Training unified model with {most_common_mode} seasonality...")
    unified_model = Prophet(seasonality_mode=most_common_mode)
    unified_model.fit(unified_train)
    
    unified_future = unified_model.make_future_dataframe(periods=len(unified_test))
    unified_forecast = unified_model.predict(unified_future)
    
    unified_pred = unified_forecast['yhat'][split_idx:].values
    unified_actual = unified_test['y'].values
    
    unified_metrics = {
        'rmse': np.sqrt(mean_squared_error(unified_actual, unified_pred)),
        'mae': mean_absolute_error(unified_actual, unified_pred),
        'r2': r2_score(unified_actual, unified_pred),
        'mape': np.mean(np.abs((unified_actual - unified_pred) / unified_actual)) * 100
    }
    
    print(f"Unified model performance:")
    print(f"  RMSE: {unified_metrics['rmse']:.2f}")
    print(f"  MAE:  {unified_metrics['mae']:.2f}")
    print(f"  R²:   {unified_metrics['r2']:.4f}")
    print(f"  MAPE: {unified_metrics['mape']:.2f}%")
    
    # 3. COMPARE WITH CATEGORY-SPECIFIC MODELS
    print(f"\n{'='*60}")
    print("3. CATEGORY-SPECIFIC MODEL PERFORMANCE")
    print(f"{'='*60}")
    
    category_metrics = {}
    total_category_rmse = 0
    total_data_points = 0
    
    for category in categories:
        print(f"\nTraining {category} model...")
        
        # Prepare category data
        cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
        cat_data = cat_data.sort_values('ds').reset_index(drop=True)
        
        # Split data
        split_idx = int(len(cat_data) * 0.8)
        train_data = cat_data[:split_idx]
        test_data = cat_data[split_idx:]
        actual = test_data['y'].values
        
        # Train with best seasonality mode for this category
        best_mode = seasonality_results[category]['best_mode']
        cat_model = Prophet(seasonality_mode=best_mode)
        cat_model.fit(train_data)
        
        future = cat_model.make_future_dataframe(periods=len(test_data))
        forecast = cat_model.predict(future)
        pred = forecast['yhat'][split_idx:].values
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(actual, pred)),
            'mae': mean_absolute_error(actual, pred),
            'r2': r2_score(actual, pred),
            'mape': np.mean(np.abs((actual - pred) / actual)) * 100,
            'data_points': len(actual)
        }
        
        category_metrics[category] = metrics
        
        # Accumulate for weighted average
        total_category_rmse += metrics['rmse'] * metrics['data_points']
        total_data_points += metrics['data_points']
        
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE:  {metrics['mae']:.2f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
    
    # Calculate weighted average RMSE for category models
    weighted_avg_rmse = total_category_rmse / total_data_points
    
    # 4. COMPARISON SUMMARY
    print(f"\n{'='*80}")
    print("4. UNIFIED vs CATEGORY MODELS COMPARISON")
    print(f"{'='*80}")
    
    print(f"Unified Model:")
    print(f"  RMSE: {unified_metrics['rmse']:.2f}")
    print(f"  R²:   {unified_metrics['r2']:.4f}")
    print(f"  MAPE: {unified_metrics['mape']:.2f}%")
    
    print(f"\nCategory Models (weighted average):")
    print(f"  RMSE: {weighted_avg_rmse:.2f}")
    
    improvement = (unified_metrics['rmse'] - weighted_avg_rmse) / weighted_avg_rmse * 100
    
    print(f"\nPerformance difference:")
    if improvement > 0:
        print(f"  Category models are {improvement:.1f}% better (lower RMSE)")
        recommendation = "Use category-specific models"
    else:
        print(f"  Unified model is {abs(improvement):.1f}% better (lower RMSE)")
        recommendation = "Use unified model"
    
    # 5. DETAILED CATEGORY BREAKDOWN
    print(f"\n{'='*60}")
    print("5. INDIVIDUAL CATEGORY PERFORMANCE")
    print(f"{'='*60}")
    
    summary_data = []
    for category in categories:
        metrics = category_metrics[category]
        seasonality = seasonality_results[category]
        
        summary_data.append({
            'Category': category,
            'RMSE': metrics['rmse'],
            'R²': metrics['r2'],
            'MAPE': metrics['mape'],
            'Best_Mode': seasonality['best_mode'],
            'Data_Points': metrics['data_points']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # 6. SEASONAL PATTERN VISUALIZATION
    print(f"\n{'='*60}")
    print("6. CREATING SEASONALITY COMPARISON PLOTS")
    print(f"{'='*60}")
    
    create_seasonality_comparison_plots(df, categories, most_common_mode)
    
    # 7. FINAL RECOMMENDATIONS
    print(f"\n{'='*80}")
    print("FINAL RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print(f"✅ Seasonality mode consistency: {mode_consistency}")
    print(f"✅ Most effective mode: {most_common_mode}")
    print(f"✅ Unified model RMSE: {unified_metrics['rmse']:.2f}")
    print(f"✅ Category models RMSE: {weighted_avg_rmse:.2f}")
    print(f"✅ Performance difference: {abs(improvement):.1f}%")
    
    if abs(improvement) < 5:  # Less than 5% difference
        print(f"\n🎯 RECOMMENDATION: USE UNIFIED MODEL")
        print(f"Reasons:")
        print(f"  • Performance difference is minimal ({abs(improvement):.1f}%)")
        print(f"  • Much simpler to maintain and deploy")
        print(f"  • Consistent seasonality patterns across categories")
        print(f"  • Single model to retrain and monitor")
        
        print(f"\n📋 IMPLEMENTATION STRATEGY:")
        print(f"  1. Train one Prophet model with {most_common_mode} seasonality")
        print(f"  2. Use total demand across all categories")
        print(f"  3. For category-specific forecasts, apply historical proportions")
        print(f"  4. Monitor model performance monthly")
        
    else:
        print(f"\n🎯 RECOMMENDATION: USE CATEGORY-SPECIFIC MODELS")
        print(f"Reasons:")
        print(f"  • Significant performance difference ({abs(improvement):.1f}%)")
        print(f"  • Categories have distinct patterns")
        print(f"  • Better accuracy justifies complexity")
        
        print(f"\n📋 IMPLEMENTATION STRATEGY:")
        print(f"  1. Train 6 separate Prophet models")
        print(f"  2. Use category-specific seasonality modes")
        print(f"  3. Aggregate forecasts for total demand")
        print(f"  4. Monitor each model individually")
    
    return {
        'unified_metrics': unified_metrics,
        'category_metrics': category_metrics,
        'seasonality_results': seasonality_results,
        'recommendation': recommendation,
        'performance_difference': improvement
    }

def create_seasonality_comparison_plots(df, categories, best_mode):
    """Create plots comparing seasonality patterns across categories"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Seasonality Pattern Comparison Across Categories', fontsize=16)
    
    # Weekly patterns
    for i, category in enumerate(categories):
        ax = axes[i//2, i%2]
        
        cat_data = df[df['category'] == category].copy()
        cat_data['day_of_week'] = cat_data['ds'].dt.dayofweek
        weekly_pattern = cat_data.groupby('day_of_week')['y'].mean()
        
        ax.plot(weekly_pattern.index, weekly_pattern.values, 'o-', linewidth=2, markersize=6)
        ax.set_title(f'{category.title()} - Weekly Pattern')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Average Demand')
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weekly_patterns_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Monthly patterns
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Monthly Seasonality Comparison Across Categories', fontsize=16)
    
    for i, category in enumerate(categories):
        ax = axes[i//2, i%2]
        
        cat_data = df[df['category'] == category].copy()
        cat_data['month'] = cat_data['ds'].dt.month
        monthly_pattern = cat_data.groupby('month')['y'].mean()
        
        ax.plot(monthly_pattern.index, monthly_pattern.values, 'o-', linewidth=2, markersize=6)
        ax.set_title(f'{category.title()} - Monthly Pattern')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Demand')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('monthly_patterns_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Seasonality comparison plots saved:")
    print("- weekly_patterns_comparison.png")
    print("- monthly_patterns_comparison.png")

if __name__ == "__main__":
    results = analyze_unified_vs_category_models()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("Answer to your question:")
    if abs(results['performance_difference']) < 5:
        print("✅ YES - Use ONE unified model for all products!")
        print("The seasonality patterns are consistent enough across categories.")
    else:
        print("❌ NO - Use separate category models for better accuracy.")
        print("Categories have significantly different patterns.")
