"""
Analyze if different categories have different seasonalities
This will help determine if we need category-specific parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def analyze_category_seasonalities():
    """Analyze seasonality patterns across all 6 categories"""
    
    print("="*80)
    print("ANALYZING CATEGORY-SPECIFIC SEASONALITIES")
    print("="*80)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    categories = df['category'].unique()
    print(f"Categories to analyze: {list(categories)}")
    
    # Prepare aggregated data for each category
    category_data = {}
    for category in categories:
        cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
        cat_data = cat_data.sort_values('ds').reset_index(drop=True)
        category_data[category] = cat_data
        print(f"{category}: {len(cat_data)} days of data")
    
    # 1. WEEKLY SEASONALITY ANALYSIS
    print(f"\n{'='*60}")
    print("1. WEEKLY SEASONALITY PATTERNS")
    print(f"{'='*60}")
    
    plt.figure(figsize=(15, 10))
    
    weekly_patterns = {}
    for i, category in enumerate(categories):
        data = category_data[category].copy()
        data['day_of_week'] = data['ds'].dt.day_name()
        data['dow_num'] = data['ds'].dt.dayofweek
        
        # Calculate average demand by day of week
        weekly_avg = data.groupby(['day_of_week', 'dow_num'])['y'].mean().reset_index()
        weekly_avg = weekly_avg.sort_values('dow_num')
        
        weekly_patterns[category] = {
            'days': weekly_avg['day_of_week'].tolist(),
            'values': weekly_avg['y'].tolist(),
            'peak_day': weekly_avg.loc[weekly_avg['y'].idxmax(), 'day_of_week'],
            'low_day': weekly_avg.loc[weekly_avg['y'].idxmin(), 'day_of_week'],
            'variation': weekly_avg['y'].std() / weekly_avg['y'].mean()  # Coefficient of variation
        }
        
        # Plot
        plt.subplot(2, 3, i+1)
        plt.plot(weekly_avg['dow_num'], weekly_avg['y'], 'o-', linewidth=2, markersize=6)
        plt.title(f'{category.title()}\nPeak: {weekly_patterns[category]["peak_day"]}\nCV: {weekly_patterns[category]["variation"]:.3f}')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Demand')
        plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Weekly Seasonality Patterns by Category', fontsize=16)
    plt.tight_layout()
    plt.savefig('weekly_seasonality_by_category.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print weekly analysis
    print("Weekly seasonality analysis:")
    for category, pattern in weekly_patterns.items():
        print(f"{category:15s}: Peak={pattern['peak_day']:9s}, Low={pattern['low_day']:9s}, Variation={pattern['variation']:.3f}")
    
    # 2. MONTHLY/YEARLY SEASONALITY ANALYSIS
    print(f"\n{'='*60}")
    print("2. MONTHLY/YEARLY SEASONALITY PATTERNS")
    print(f"{'='*60}")
    
    plt.figure(figsize=(15, 10))
    
    monthly_patterns = {}
    for i, category in enumerate(categories):
        data = category_data[category].copy()
        data['month'] = data['ds'].dt.month
        data['month_name'] = data['ds'].dt.month_name()
        
        # Calculate average demand by month
        monthly_avg = data.groupby(['month', 'month_name'])['y'].mean().reset_index()
        monthly_avg = monthly_avg.sort_values('month')
        
        monthly_patterns[category] = {
            'months': monthly_avg['month_name'].tolist(),
            'values': monthly_avg['y'].tolist(),
            'peak_month': monthly_avg.loc[monthly_avg['y'].idxmax(), 'month_name'],
            'low_month': monthly_avg.loc[monthly_avg['y'].idxmin(), 'month_name'],
            'variation': monthly_avg['y'].std() / monthly_avg['y'].mean()
        }
        
        # Plot
        plt.subplot(2, 3, i+1)
        plt.plot(monthly_avg['month'], monthly_avg['y'], 'o-', linewidth=2, markersize=6)
        plt.title(f'{category.title()}\nPeak: {monthly_patterns[category]["peak_month"]}\nCV: {monthly_patterns[category]["variation"]:.3f}')
        plt.xlabel('Month')
        plt.ylabel('Average Demand')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Monthly Seasonality Patterns by Category', fontsize=16)
    plt.tight_layout()
    plt.savefig('monthly_seasonality_by_category.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print monthly analysis
    print("Monthly seasonality analysis:")
    for category, pattern in monthly_patterns.items():
        print(f"{category:15s}: Peak={pattern['peak_month']:9s}, Low={pattern['low_month']:9s}, Variation={pattern['variation']:.3f}")
    
    # 3. PROPHET'S AUTOMATIC SEASONALITY DETECTION
    print(f"\n{'='*60}")
    print("3. PROPHET'S SEASONALITY DETECTION BY CATEGORY")
    print(f"{'='*60}")
    
    prophet_seasonalities = {}
    
    for category in categories:
        print(f"\nAnalyzing {category}...")
        data = category_data[category].copy()
        
        # Split data for training
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        
        # Train Prophet model
        model = Prophet()
        model.fit(train_data)
        
        # Generate forecast to get seasonality components
        future = model.make_future_dataframe(periods=100)  # Extend a bit for full seasonality
        forecast = model.predict(future)
        
        # Extract seasonality information
        seasonality_info = {
            'weekly_range': 0,
            'yearly_range': 0,
            'trend_strength': 0
        }
        
        if 'weekly' in forecast.columns:
            seasonality_info['weekly_range'] = forecast['weekly'].max() - forecast['weekly'].min()
        
        if 'yearly' in forecast.columns:
            seasonality_info['yearly_range'] = forecast['yearly'].max() - forecast['yearly'].min()
        
        if 'trend' in forecast.columns:
            # Measure trend strength as the total change relative to average
            trend_change = abs(forecast['trend'].iloc[-1] - forecast['trend'].iloc[0])
            avg_level = forecast['yhat'].mean()
            seasonality_info['trend_strength'] = trend_change / avg_level if avg_level != 0 else 0
        
        prophet_seasonalities[category] = seasonality_info
        
        print(f"  Weekly seasonality range: {seasonality_info['weekly_range']:.1f}")
        print(f"  Yearly seasonality range: {seasonality_info['yearly_range']:.1f}")
        print(f"  Trend strength: {seasonality_info['trend_strength']:.3f}")
    
    # 4. CATEGORY SIMILARITY ANALYSIS
    print(f"\n{'='*60}")
    print("4. CATEGORY SIMILARITY ANALYSIS")
    print(f"{'='*60}")
    
    # Create comparison matrix
    similarity_metrics = []
    
    for category in categories:
        weekly_cv = weekly_patterns[category]['variation']
        monthly_cv = monthly_patterns[category]['variation']
        weekly_range = prophet_seasonalities[category]['weekly_range']
        yearly_range = prophet_seasonalities[category]['yearly_range']
        trend_strength = prophet_seasonalities[category]['trend_strength']
        
        similarity_metrics.append({
            'category': category,
            'weekly_cv': weekly_cv,
            'monthly_cv': monthly_cv,
            'weekly_range': weekly_range,
            'yearly_range': yearly_range,
            'trend_strength': trend_strength
        })
    
    similarity_df = pd.DataFrame(similarity_metrics)
    print(similarity_df.to_string(index=False))
    
    # 5. PARAMETER TESTING BY CATEGORY
    print(f"\n{'='*60}")
    print("5. OPTIMAL PARAMETERS BY CATEGORY")
    print(f"{'='*60}")
    
    category_optimal_params = {}
    
    param_configs = [
        {'name': 'Default', 'params': {}},
        {'name': 'Flexible', 'params': {'changepoint_prior_scale': 0.5}},
        {'name': 'Conservative', 'params': {'changepoint_prior_scale': 0.01}},
        {'name': 'Strong Seasonality', 'params': {'seasonality_prior_scale': 50}},
        {'name': 'Multiplicative', 'params': {'seasonality_mode': 'multiplicative'}},
    ]
    
    for category in categories:
        print(f"\nTesting parameters for {category}...")
        data = category_data[category].copy()
        
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        actual = test_data['y'].values
        
        results = {}
        
        for config in param_configs:
            try:
                model = Prophet(**config['params'])
                model.fit(train_data)
                
                future = model.make_future_dataframe(periods=len(test_data))
                forecast = model.predict(future)
                
                pred = forecast['yhat'][split_idx:].values
                rmse = np.sqrt(mean_squared_error(actual, pred))
                results[config['name']] = rmse
                
            except:
                results[config['name']] = float('inf')
        
        best_config = min(results, key=results.get)
        category_optimal_params[category] = {
            'best_config': best_config,
            'best_rmse': results[best_config],
            'all_results': results
        }
        
        print(f"  Best config: {best_config} (RMSE: {results[best_config]:.2f})")
    
    # 6. FINAL RECOMMENDATIONS
    print(f"\n{'='*80}")
    print("FINAL RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Check if categories have similar patterns
    weekly_cvs = [weekly_patterns[cat]['variation'] for cat in categories]
    monthly_cvs = [monthly_patterns[cat]['variation'] for cat in categories]
    
    weekly_cv_std = np.std(weekly_cvs)
    monthly_cv_std = np.std(monthly_cvs)
    
    print(f"Weekly variation across categories (std): {weekly_cv_std:.4f}")
    print(f"Monthly variation across categories (std): {monthly_cv_std:.4f}")
    
    if weekly_cv_std < 0.1 and monthly_cv_std < 0.1:
        print("\n✅ CATEGORIES HAVE SIMILAR SEASONALITY PATTERNS")
        print("Recommendation: Use same parameters for all categories")
        print("Strategy: Single optimized parameter set")
    else:
        print("\n🎯 CATEGORIES HAVE DIFFERENT SEASONALITY PATTERNS")
        print("Recommendation: Use category-specific parameters")
        print("Strategy: Individual parameter optimization per category")
    
    # Best performing config overall
    all_best_configs = [category_optimal_params[cat]['best_config'] for cat in categories]
    most_common_config = max(set(all_best_configs), key=all_best_configs.count)
    
    print(f"\nMost successful config across categories: {most_common_config}")
    print(f"Used by {all_best_configs.count(most_common_config)}/{len(categories)} categories")
    
    return {
        'weekly_patterns': weekly_patterns,
        'monthly_patterns': monthly_patterns,
        'prophet_seasonalities': prophet_seasonalities,
        'optimal_params': category_optimal_params,
        'similarity_metrics': similarity_df
    }

if __name__ == "__main__":
    results = analyze_category_seasonalities()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("Check the generated plots and recommendations above.")
    print("Files created:")
    print("- weekly_seasonality_by_category.png")
    print("- monthly_seasonality_by_category.png")
