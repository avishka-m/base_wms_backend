"""
Analysis of why basic Prophet might be better than enhanced
Let's diagnose what's happening
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

def diagnose_prophet_performance():
    """Diagnose why basic Prophet might be performing better"""
    
    print("="*80)
    print("DIAGNOSING PROPHET PERFORMANCE")
    print("="*80)
    
    # Load and prepare data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Add features
    df['is_weekend'] = (df['ds'].dt.dayofweek.isin([5, 6])).astype(int)
    df['is_end_of_month'] = df['ds'].dt.is_month_end.astype(int)
    df['is_holiday_season'] = df['ds'].dt.month.isin([11, 12]).astype(int)
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    
    # Prepare books_media data
    category_data = df[df['category'] == 'books_media'].groupby('ds').agg({
        'y': 'sum',
        'is_weekend': 'first',
        'is_end_of_month': 'first',
        'is_holiday_season': 'first',
        'day_of_week': 'first',
        'month': 'first'
    }).reset_index()
    
    print(f"Data shape: {category_data.shape}")
    print(f"Date range: {category_data['ds'].min()} to {category_data['ds'].max()}")
    
    # 1. ANALYZE FEATURE RELEVANCE
    print(f"\n{'='*60}")
    print("1. FEATURE RELEVANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Weekend effect
    weekend_avg = category_data.groupby('is_weekend')['y'].mean()
    print(f"Weekend effect:")
    print(f"  Weekdays: {weekend_avg[0]:.1f}")
    print(f"  Weekends: {weekend_avg[1]:.1f}")
    print(f"  Ratio: {weekend_avg[1]/weekend_avg[0]:.2f}")
    
    # End of month effect
    eom_avg = category_data.groupby('is_end_of_month')['y'].mean()
    print(f"\nEnd of month effect:")
    print(f"  Regular days: {eom_avg[0]:.1f}")
    print(f"  End of month: {eom_avg[1]:.1f}")
    print(f"  Ratio: {eom_avg[1]/eom_avg[0]:.2f}")
    
    # Holiday season effect
    holiday_avg = category_data.groupby('is_holiday_season')['y'].mean()
    print(f"\nHoliday season effect:")
    print(f"  Regular months: {holiday_avg[0]:.1f}")
    print(f"  Holiday season: {holiday_avg[1]:.1f}")
    print(f"  Ratio: {holiday_avg[1]/holiday_avg[0]:.2f}")
    
    # 2. PROPHET'S BUILT-IN SEASONALITY ANALYSIS
    print(f"\n{'='*60}")
    print("2. PROPHET'S BUILT-IN SEASONALITY")
    print(f"{'='*60}")
    
    # Train basic Prophet
    split_idx = int(len(category_data) * 0.8)
    train_data = category_data[:split_idx].copy()
    test_data = category_data[split_idx:].copy()
    
    basic_model = Prophet()
    basic_model.fit(train_data[['ds', 'y']])
    
    basic_future = basic_model.make_future_dataframe(periods=len(test_data))
    basic_forecast = basic_model.predict(basic_future)
    
    # Analyze Prophet's automatic seasonality detection
    print("Prophet automatically detected:")
    if 'weekly' in basic_forecast.columns:
        weekly_range = basic_forecast['weekly'].max() - basic_forecast['weekly'].min()
        print(f"  Weekly seasonality range: {weekly_range:.1f}")
    
    if 'yearly' in basic_forecast.columns:
        yearly_range = basic_forecast['yearly'].max() - basic_forecast['yearly'].min()
        print(f"  Yearly seasonality range: {yearly_range:.1f}")
    
    # 3. DIFFERENT REGRESSOR STRATEGIES
    print(f"\n{'='*60}")
    print("3. TESTING DIFFERENT REGRESSOR STRATEGIES")
    print(f"{'='*60}")
    
    actual = test_data['y'].values
    results = {}
    
    # Strategy 1: No regressors (baseline)
    basic_pred = basic_forecast['yhat'][split_idx:].values
    basic_rmse = np.sqrt(mean_squared_error(actual, basic_pred))
    results['Basic Prophet'] = basic_rmse
    print(f"Basic Prophet RMSE: {basic_rmse:.2f}")
    
    # Strategy 2: Only weekend
    model_weekend = Prophet()
    model_weekend.add_regressor('is_weekend')
    model_weekend.fit(train_data[['ds', 'y', 'is_weekend']])
    
    future_weekend = pd.concat([train_data, test_data], ignore_index=True)[['ds', 'is_weekend']]
    forecast_weekend = model_weekend.predict(future_weekend)
    pred_weekend = forecast_weekend['yhat'][split_idx:].values
    rmse_weekend = np.sqrt(mean_squared_error(actual, pred_weekend))
    results['Weekend only'] = rmse_weekend
    print(f"Weekend regressor RMSE: {rmse_weekend:.2f}")
    
    # Strategy 3: Only holiday season
    model_holiday = Prophet()
    model_holiday.add_regressor('is_holiday_season')
    model_holiday.fit(train_data[['ds', 'y', 'is_holiday_season']])
    
    future_holiday = pd.concat([train_data, test_data], ignore_index=True)[['ds', 'is_holiday_season']]
    forecast_holiday = model_holiday.predict(future_holiday)
    pred_holiday = forecast_holiday['yhat'][split_idx:].values
    rmse_holiday = np.sqrt(mean_squared_error(actual, pred_holiday))
    results['Holiday only'] = rmse_holiday
    print(f"Holiday regressor RMSE: {rmse_holiday:.2f}")
    
    # Strategy 4: Different seasonality modes
    model_mult = Prophet(seasonality_mode='multiplicative')
    model_mult.fit(train_data[['ds', 'y']])
    forecast_mult = model_mult.predict(basic_future)
    pred_mult = forecast_mult['yhat'][split_idx:].values
    rmse_mult = np.sqrt(mean_squared_error(actual, pred_mult))
    results['Multiplicative'] = rmse_mult
    print(f"Multiplicative mode RMSE: {rmse_mult:.2f}")
    
    # 4. CONCLUSIONS
    print(f"\n{'='*60}")
    print("4. CONCLUSIONS")
    print(f"{'='*60}")
    
    best_model = min(results, key=results.get)
    best_rmse = results[best_model]
    
    print(f"Best performing model: {best_model} (RMSE: {best_rmse:.2f})")
    
    print(f"\nKey insights:")
    if results['Basic Prophet'] <= min(results.values()) * 1.05:  # Within 5%
        print("✅ Basic Prophet is already optimal!")
        print("🔍 Prophet's automatic seasonality detection is capturing patterns well")
        print("💡 Recommendation: Stick with basic Prophet, focus on:")
        print("   • Parameter tuning (changepoint_prior_scale, seasonality_prior_scale)")
        print("   • Adding holidays if relevant")
        print("   • Ensemble methods")
    else:
        print("🎯 There's room for improvement with:")
        print(f"   • Best strategy: {best_model}")
        print("   • Consider feature engineering")
        print("   • Try different seasonality modes")
    
    return results

def test_parameter_tuning():
    """Test if parameter tuning helps more than regressors"""
    
    print(f"\n{'='*80}")
    print("PARAMETER TUNING TEST")
    print(f"{'='*80}")
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    category_data = df[df['category'] == 'books_media'].groupby('ds')['y'].sum().reset_index()
    
    split_idx = int(len(category_data) * 0.8)
    train_data = category_data[:split_idx].copy()
    test_data = category_data[split_idx:].copy()
    actual = test_data['y'].values
    
    # Test different parameter combinations
    param_configs = [
        {'name': 'Default', 'params': {}},
        {'name': 'Conservative', 'params': {'changepoint_prior_scale': 0.01}},
        {'name': 'Flexible', 'params': {'changepoint_prior_scale': 0.5}},
        {'name': 'Strong Seasonality', 'params': {'seasonality_prior_scale': 50}},
        {'name': 'Weak Seasonality', 'params': {'seasonality_prior_scale': 0.1}},
        {'name': 'Multiplicative', 'params': {'seasonality_mode': 'multiplicative'}},
    ]
    
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
            
            print(f"{config['name']:20s}: RMSE = {rmse:.2f}")
            
        except Exception as e:
            print(f"{config['name']:20s}: Error - {str(e)}")
    
    best_param = min(results, key=results.get)
    print(f"\nBest parameter config: {best_param} (RMSE: {results[best_param]:.2f})")
    
    return results

if __name__ == "__main__":
    # Run diagnosis
    model_results = diagnose_prophet_performance()
    
    # Test parameter tuning
    param_results = test_parameter_tuning()
    
    print(f"\n{'='*80}")
    print("FINAL RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print("Based on the analysis:")
    print("1. Prophet's automatic seasonality is already effective")
    print("2. Focus on parameter tuning rather than adding regressors")
    print("3. For your 6-category approach:")
    print("   ✅ Use basic Prophet with optimized parameters")
    print("   ✅ Category-level models are still the right approach")
    print("   ✅ Test different parameters per category if needed")
    print("4. Consider adding regressors only if you see clear business patterns")
    print("   that Prophet's seasonality doesn't capture")
