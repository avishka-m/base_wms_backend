#!/usr/bin/env python3
"""
DETAILED MISMATCH ANALYSIS
==========================
Comprehensive analysis of prediction vs actual mismatches
Identifies specific patterns, outliers, and improvement opportunities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load data and prepare for analysis"""
    print("Loading data for mismatch analysis...")
    
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Get category-level aggregated data
    category_data = df.groupby(['category', 'ds'])['y'].sum().reset_index()
    
    categories = df['category'].unique()
    print(f"Categories: {list(categories)}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    
    return category_data, categories

def create_simple_baseline_model(train_data, test_data):
    """Create simple baseline Prophet model for comparison"""
    
    # Simple Prophet model without enhancements
    model = Prophet()
    model.fit(train_data[['ds', 'y']])
    
    forecast = model.predict(test_data[['ds']])
    
    return model, forecast

def create_holiday_calendar():
    """Create holiday calendar"""
    major_holidays = pd.DataFrame([
        # Black Friday
        {'holiday': 'black_friday', 'ds': '2022-11-25', 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'black_friday', 'ds': '2023-11-24', 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'black_friday', 'ds': '2024-11-29', 'lower_window': -1, 'upper_window': 1},
        
        # Christmas season
        {'holiday': 'christmas_season', 'ds': '2022-12-24', 'lower_window': -3, 'upper_window': 2},
        {'holiday': 'christmas_season', 'ds': '2023-12-24', 'lower_window': -3, 'upper_window': 2},
        {'holiday': 'christmas_season', 'ds': '2024-12-24', 'lower_window': -3, 'upper_window': 2},
        
        # Valentine's Day
        {'holiday': 'valentine_day', 'ds': '2022-02-14', 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'valentine_day', 'ds': '2023-02-14', 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'valentine_day', 'ds': '2024-02-14', 'lower_window': -1, 'upper_window': 1},
    ])
    
    major_holidays['ds'] = pd.to_datetime(major_holidays['ds'])
    return major_holidays

def create_enhanced_model(train_data, test_data, holidays_df):
    """Create enhanced model with holidays and regressors"""
    
    # Enhanced Prophet model
    model = Prophet(
        changepoint_prior_scale=0.5,
        holidays=holidays_df
    )
    model.add_regressor('is_weekend', prior_scale=10, standardize=False)
    
    # Add regressors to training data
    train_enhanced = train_data.copy()
    train_enhanced['is_weekend'] = train_enhanced['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    
    model.fit(train_enhanced[['ds', 'y', 'is_weekend']])
    
    # Add regressors to test data
    test_enhanced = test_data.copy()
    test_enhanced['is_weekend'] = test_enhanced['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    
    forecast = model.predict(test_enhanced[['ds', 'is_weekend']])
    
    return model, forecast

def analyze_prediction_errors(actual, predicted, dates, category):
    """Detailed analysis of prediction errors"""
    
    errors = actual - predicted
    abs_errors = np.abs(errors)
    pct_errors = (errors / actual) * 100
    
    # Basic statistics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs(pct_errors))
    
    # Error distribution analysis
    error_stats = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_error': np.max(abs_errors),
        'min_error': np.min(abs_errors),
        'q75_error': np.percentile(abs_errors, 75),
        'q95_error': np.percentile(abs_errors, 95),
    }
    
    # Identify problematic periods
    high_error_threshold = np.percentile(abs_errors, 90)
    high_error_mask = abs_errors > high_error_threshold
    high_error_dates = dates[high_error_mask]
    high_error_actual = actual[high_error_mask]
    high_error_predicted = predicted[high_error_mask]
    high_error_values = abs_errors[high_error_mask]
    
    # Pattern analysis
    dates_series = pd.to_datetime(dates)
    weekend_mask = dates_series.dayofweek.isin([5, 6])
    weekend_errors = abs_errors[weekend_mask]
    weekday_errors = abs_errors[~weekend_mask]
    
    month_end_mask = dates_series.day >= 25
    month_end_errors = abs_errors[month_end_mask]
    
    pattern_analysis = {
        'weekend_mae': np.mean(weekend_errors) if len(weekend_errors) > 0 else 0,
        'weekday_mae': np.mean(weekday_errors) if len(weekday_errors) > 0 else 0,
        'month_end_mae': np.mean(month_end_errors) if len(month_end_errors) > 0 else 0,
        'weekend_vs_weekday_ratio': np.mean(weekend_errors) / np.mean(weekday_errors) if len(weekend_errors) > 0 and len(weekday_errors) > 0 else 1,
    }
    
    return {
        'error_stats': error_stats,
        'pattern_analysis': pattern_analysis,
        'high_error_dates': high_error_dates,
        'high_error_actual': high_error_actual,
        'high_error_predicted': high_error_predicted,
        'high_error_values': high_error_values,
        'errors': errors,
        'abs_errors': abs_errors,
        'pct_errors': pct_errors
    }

def create_mismatch_visualization(category_data, category, baseline_forecast, enhanced_forecast, split_date):
    """Create comprehensive mismatch visualization"""
    
    # Prepare data
    test_data = category_data[category_data['ds'] >= split_date].copy()
    actual = test_data['y'].values
    baseline_pred = baseline_forecast['yhat'].values
    enhanced_pred = enhanced_forecast['yhat'].values
    dates = test_data['ds'].values
    
    # Analyze errors
    baseline_analysis = analyze_prediction_errors(actual, baseline_pred, dates, category)
    enhanced_analysis = analyze_prediction_errors(actual, enhanced_pred, dates, category)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'{category.title()} - Prediction Mismatch Analysis', fontsize=16, fontweight='bold')
    
    # 1. Time series comparison
    ax1 = axes[0, 0]
    ax1.plot(dates, actual, 'k.-', label='Actual', alpha=0.8, markersize=4)
    ax1.plot(dates, baseline_pred, 'r-', label='Baseline Forecast', alpha=0.7, linewidth=2)
    ax1.plot(dates, enhanced_pred, 'b-', label='Enhanced Forecast', alpha=0.7, linewidth=2)
    
    # Highlight high error periods
    high_error_dates = enhanced_analysis['high_error_dates']
    if len(high_error_dates) > 0:
        for date in high_error_dates:
            ax1.axvline(x=date, color='orange', alpha=0.3, linestyle='--')
    
    ax1.set_title('Actual vs Predicted Values')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Demand')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Error distribution
    ax2 = axes[0, 1]
    ax2.hist(baseline_analysis['abs_errors'], bins=30, alpha=0.5, label='Baseline Errors', color='red', density=True)
    ax2.hist(enhanced_analysis['abs_errors'], bins=30, alpha=0.5, label='Enhanced Errors', color='blue', density=True)
    ax2.set_title('Error Distribution')
    ax2.set_xlabel('Absolute Error')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residual analysis
    ax3 = axes[1, 0]
    ax3.scatter(enhanced_pred, enhanced_analysis['errors'], alpha=0.6, s=20)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax3.set_title('Residual Plot (Enhanced Model)')
    ax3.set_xlabel('Predicted Values')
    ax3.set_ylabel('Residuals (Actual - Predicted)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Weekly pattern analysis
    ax4 = axes[1, 1]
    
    # Group errors by day of week
    test_df = pd.DataFrame({
        'ds': dates,
        'actual': actual,
        'predicted': enhanced_pred,
        'error': enhanced_analysis['abs_errors']
    })
    test_df['dayofweek'] = pd.to_datetime(test_df['ds']).dt.dayofweek
    test_df['day_name'] = pd.to_datetime(test_df['ds']).dt.day_name()
    
    day_errors = test_df.groupby('day_name')['error'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    ax4.bar(range(7), day_errors.values, color=['lightblue' if day in ['Saturday', 'Sunday'] else 'lightcoral' for day in day_errors.index])
    ax4.set_title('Average Error by Day of Week')
    ax4.set_xlabel('Day of Week')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{category}_mismatch_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return baseline_analysis, enhanced_analysis

def print_detailed_mismatch_report(category, baseline_analysis, enhanced_analysis):
    """Print detailed mismatch analysis report"""
    
    print(f"\n{'='*80}")
    print(f"DETAILED MISMATCH ANALYSIS: {category.upper()}")
    print(f"{'='*80}")
    
    # Performance comparison
    print("\n📊 MODEL PERFORMANCE COMPARISON:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Baseline':<12} {'Enhanced':<12} {'Improvement':<12}")
    print("-" * 50)
    
    baseline_stats = baseline_analysis['error_stats']
    enhanced_stats = enhanced_analysis['error_stats']
    
    rmse_improvement = (baseline_stats['rmse'] - enhanced_stats['rmse']) / baseline_stats['rmse'] * 100
    mae_improvement = (baseline_stats['mae'] - enhanced_stats['mae']) / baseline_stats['mae'] * 100
    r2_improvement = (enhanced_stats['r2'] - baseline_stats['r2']) / baseline_stats['r2'] * 100
    
    print(f"{'RMSE':<20} {baseline_stats['rmse']:<12.1f} {enhanced_stats['rmse']:<12.1f} {rmse_improvement:<11.1f}%")
    print(f"{'MAE':<20} {baseline_stats['mae']:<12.1f} {enhanced_stats['mae']:<12.1f} {mae_improvement:<11.1f}%")
    print(f"{'R²':<20} {baseline_stats['r2']:<12.3f} {enhanced_stats['r2']:<12.3f} {r2_improvement:<11.1f}%")
    print(f"{'MAPE':<20} {baseline_stats['mape']:<12.1f} {enhanced_stats['mape']:<12.1f} {'N/A':<12}")
    
    # Error characteristics
    print(f"\n🎯 ERROR CHARACTERISTICS (Enhanced Model):")
    print("-" * 40)
    print(f"Mean Error (Bias):      {enhanced_stats['mean_error']:8.1f}")
    print(f"Error Std Dev:          {enhanced_stats['std_error']:8.1f}")
    print(f"Max Absolute Error:     {enhanced_stats['max_error']:8.1f}")
    print(f"95th Percentile Error:  {enhanced_stats['q95_error']:8.1f}")
    print(f"75th Percentile Error:  {enhanced_stats['q75_error']:8.1f}")
    
    # Pattern analysis
    print(f"\n📈 PATTERN-SPECIFIC ERROR ANALYSIS:")
    print("-" * 40)
    baseline_patterns = baseline_analysis['pattern_analysis']
    enhanced_patterns = enhanced_analysis['pattern_analysis']
    
    print(f"Weekend MAE:            {enhanced_patterns['weekend_mae']:8.1f} (Baseline: {baseline_patterns['weekend_mae']:6.1f})")
    print(f"Weekday MAE:            {enhanced_patterns['weekday_mae']:8.1f} (Baseline: {baseline_patterns['weekday_mae']:6.1f})")
    print(f"Month-end MAE:          {enhanced_patterns['month_end_mae']:8.1f} (Baseline: {baseline_patterns['month_end_mae']:6.1f})")
    print(f"Weekend/Weekday Ratio:  {enhanced_patterns['weekend_vs_weekday_ratio']:8.2f} (Baseline: {baseline_patterns['weekend_vs_weekday_ratio']:6.2f})")
    
    # High error periods
    print(f"\n🚨 HIGH ERROR PERIODS (Top 10% worst predictions):")
    print("-" * 60)
    high_error_dates = enhanced_analysis['high_error_dates']
    high_error_actual = enhanced_analysis['high_error_actual']
    high_error_predicted = enhanced_analysis['high_error_predicted']
    high_error_values = enhanced_analysis['high_error_values']
    
    if len(high_error_dates) > 0:
        print(f"{'Date':<12} {'Actual':<8} {'Predicted':<10} {'Error':<8} {'Error %':<8}")
        print("-" * 60)
        
        for i in range(min(10, len(high_error_dates))):
            date_str = pd.to_datetime(high_error_dates[i]).strftime('%Y-%m-%d')
            actual_val = high_error_actual[i]
            pred_val = high_error_predicted[i]
            error_val = high_error_values[i]
            error_pct = (error_val / actual_val) * 100
            
            print(f"{date_str:<12} {actual_val:<8.0f} {pred_val:<10.0f} {error_val:<8.0f} {error_pct:<7.1f}%")
    
    # Improvement opportunities
    print(f"\n💡 IMPROVEMENT OPPORTUNITIES:")
    print("-" * 40)
    
    if enhanced_patterns['weekend_vs_weekday_ratio'] > 1.5:
        print("• Weekend patterns need better modeling (consider stronger weekend regressors)")
    
    if enhanced_patterns['month_end_mae'] > enhanced_patterns['weekday_mae'] * 1.3:
        print("• Month-end patterns not fully captured (add month-end regressors)")
    
    if enhanced_stats['mape'] > 20:
        print("• High percentage errors suggest need for multiplicative seasonality")
    
    if enhanced_stats['mean_error'] > enhanced_stats['std_error'] * 0.1:
        print("• Systematic bias detected (model consistently over/under predicts)")
    
    if len(high_error_dates) > len(enhanced_analysis['errors']) * 0.15:
        print("• Too many outliers - consider outlier detection and handling")
    
    print("• Consider adding more specific regressors (promotions, weather, economic indicators)")
    print("• Implement ensemble methods combining multiple models")
    print("• Add product-level modeling for high-variance categories")

def main():
    """Main execution function"""
    
    print("="*80)
    print("COMPREHENSIVE MISMATCH ANALYSIS")
    print("="*80)
    
    # Load data
    category_data, categories = load_and_prepare_data()
    holidays_df = create_holiday_calendar()
    
    # Analyze each category
    for category in categories:
        print(f"\nAnalyzing {category}...")
        
        # Get category data
        cat_data = category_data[category_data['category'] == category].copy()
        cat_data = cat_data.sort_values('ds').reset_index(drop=True)
        
        # Split data (80/20)
        split_idx = int(len(cat_data) * 0.8)
        train_data = cat_data[:split_idx].copy()
        test_data = cat_data[split_idx:].copy()
        split_date = test_data['ds'].iloc[0]
        
        # Create models
        baseline_model, baseline_forecast = create_simple_baseline_model(train_data, test_data)
        enhanced_model, enhanced_forecast = create_enhanced_model(train_data, test_data, holidays_df)
        
        # Analyze mismatches
        baseline_analysis, enhanced_analysis = create_mismatch_visualization(
            cat_data, category, baseline_forecast, enhanced_forecast, split_date
        )
        
        # Print detailed report
        print_detailed_mismatch_report(category, baseline_analysis, enhanced_analysis)
        
        # Break after first category for detailed analysis
        break
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("✅ Detailed mismatch analysis completed")
    print("✅ Visualization saved")
    print("✅ Improvement recommendations provided")

if __name__ == "__main__":
    main()
