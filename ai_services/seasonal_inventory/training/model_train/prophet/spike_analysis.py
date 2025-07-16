"""
Advanced Spike Pattern Analysis and Handling
Identify and handle remaining demand spikes not captured by holiday regressors
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

warnings.filterwarnings('ignore')

def analyze_remaining_spikes():
    """Analyze spikes not captured by existing holiday patterns"""
    
    print("🔍 ANALYZING REMAINING SPIKE PATTERNS")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Create daily aggregated demand
    daily_demand = df.groupby('ds')['y'].sum().reset_index()
    daily_demand = daily_demand.sort_values('ds').reset_index(drop=True)
    
    # Add date features
    daily_demand['year'] = daily_demand['ds'].dt.year
    daily_demand['month'] = daily_demand['ds'].dt.month
    daily_demand['day'] = daily_demand['ds'].dt.day
    daily_demand['day_of_week'] = daily_demand['ds'].dt.day_of_week
    daily_demand['day_of_year'] = daily_demand['ds'].dt.day_of_year
    daily_demand['week_of_year'] = daily_demand['ds'].dt.isocalendar().week
    
    # Calculate rolling statistics for spike detection
    daily_demand['rolling_7'] = daily_demand['y'].rolling(window=7, center=True).mean()
    daily_demand['rolling_14'] = daily_demand['y'].rolling(window=14, center=True).mean()
    daily_demand['rolling_30'] = daily_demand['y'].rolling(window=30, center=True).mean()
    daily_demand['rolling_std_7'] = daily_demand['y'].rolling(window=7, center=True).std()
    
    # Identify known holidays (from previous analysis)
    known_holidays = [
        '2022-11-25', '2023-11-24', '2024-11-29',  # Black Friday
        '2022-11-28', '2023-11-27', '2024-12-02',  # Cyber Monday
        '2022-12-24', '2023-12-24', '2024-12-24',  # Christmas Eve
        '2022-12-25', '2023-12-25', '2024-12-25',  # Christmas
        '2022-12-31', '2023-12-31', '2024-12-31',  # New Year's Eve
        '2022-02-14', '2023-02-14', '2024-02-14',  # Valentine's Day
    ]
    
    daily_demand['is_known_holiday'] = daily_demand['ds'].dt.strftime('%Y-%m-%d').isin(known_holidays)
    
    # Calculate spike ratios and z-scores
    daily_demand['spike_ratio_7'] = daily_demand['y'] / daily_demand['rolling_7']
    daily_demand['spike_ratio_14'] = daily_demand['y'] / daily_demand['rolling_14']
    daily_demand['z_score'] = stats.zscore(daily_demand['y'])
    
    # Identify significant spikes (statistical thresholds)
    spike_threshold_ratio = 1.3  # 30% above rolling average
    spike_threshold_zscore = 2.5  # 2.5 standard deviations
    
    daily_demand['is_spike_ratio'] = daily_demand['spike_ratio_7'] >= spike_threshold_ratio
    daily_demand['is_spike_zscore'] = np.abs(daily_demand['z_score']) >= spike_threshold_zscore
    daily_demand['is_spike'] = daily_demand['is_spike_ratio'] | daily_demand['is_spike_zscore']
    
    # Filter unexplained spikes (spikes that are NOT known holidays)
    unexplained_spikes = daily_demand[daily_demand['is_spike'] & ~daily_demand['is_known_holiday']].copy()
    
    print(f"📊 Spike Analysis Results:")
    print(f"   Total days analyzed: {len(daily_demand)}")
    print(f"   Total spikes detected: {daily_demand['is_spike'].sum()}")
    print(f"   Known holiday spikes: {daily_demand[daily_demand['is_spike'] & daily_demand['is_known_holiday']]['is_spike'].sum()}")
    print(f"   Unexplained spikes: {len(unexplained_spikes)}")
    print(f"   Unexplained spike rate: {len(unexplained_spikes)/len(daily_demand)*100:.1f}%")
    
    return daily_demand, unexplained_spikes

def categorize_spike_patterns(daily_demand, unexplained_spikes):
    """Categorize different types of spike patterns"""
    
    print(f"\n🔍 SPIKE PATTERN CATEGORIZATION:")
    print("-" * 40)
    
    # Pattern 1: Weekend spikes
    weekend_spikes = unexplained_spikes[unexplained_spikes['day_of_week'].isin([5, 6])]
    
    # Pattern 2: Month-end spikes (last 3 days of month)
    unexplained_spikes['days_to_month_end'] = (unexplained_spikes['ds'] + pd.offsets.MonthEnd(0) - unexplained_spikes['ds']).dt.days
    month_end_spikes = unexplained_spikes[unexplained_spikes['days_to_month_end'] <= 3]
    
    # Pattern 3: Seasonal spikes (July summer pattern observed)
    summer_spikes = unexplained_spikes[unexplained_spikes['month'].isin([6, 7, 8])]
    
    # Pattern 4: Consecutive day patterns
    unexplained_spikes['prev_spike'] = unexplained_spikes['ds'].isin(unexplained_spikes['ds'] - timedelta(days=1))
    unexplained_spikes['next_spike'] = unexplained_spikes['ds'].isin(unexplained_spikes['ds'] + timedelta(days=1))
    consecutive_spikes = unexplained_spikes[unexplained_spikes['prev_spike'] | unexplained_spikes['next_spike']]
    
    # Pattern 5: Outlier spikes (isolated, extreme values)
    outlier_threshold = daily_demand['y'].quantile(0.99)
    outlier_spikes = unexplained_spikes[unexplained_spikes['y'] >= outlier_threshold]
    
    print(f"📈 Spike Pattern Categories:")
    print(f"   Weekend spikes: {len(weekend_spikes)} ({len(weekend_spikes)/len(unexplained_spikes)*100:.1f}%)")
    print(f"   Month-end spikes: {len(month_end_spikes)} ({len(month_end_spikes)/len(unexplained_spikes)*100:.1f}%)")
    print(f"   Summer spikes: {len(summer_spikes)} ({len(summer_spikes)/len(unexplained_spikes)*100:.1f}%)")
    print(f"   Consecutive spikes: {len(consecutive_spikes)} ({len(consecutive_spikes)/len(unexplained_spikes)*100:.1f}%)")
    print(f"   Extreme outliers: {len(outlier_spikes)} ({len(outlier_spikes)/len(unexplained_spikes)*100:.1f}%)")
    
    # Show top unexplained spikes
    print(f"\n🔥 Top 10 Unexplained Spikes:")
    top_spikes = unexplained_spikes.nlargest(10, 'y')[['ds', 'y', 'spike_ratio_7', 'day_of_week', 'month']]
    for _, row in top_spikes.iterrows():
        day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][row['day_of_week']]
        print(f"   {row['ds'].strftime('%Y-%m-%d')} ({day_name}): {row['y']:6.0f} (x{row['spike_ratio_7']:.2f})")
    
    return {
        'weekend': weekend_spikes,
        'month_end': month_end_spikes,
        'summer': summer_spikes,
        'consecutive': consecutive_spikes,
        'outliers': outlier_spikes
    }

def create_enhanced_spike_regressors(daily_demand):
    """Create additional regressors to handle spike patterns"""
    
    print(f"\n🛠️ CREATING ENHANCED SPIKE REGRESSORS:")
    print("-" * 45)
    
    enhanced_df = daily_demand.copy()
    
    # 1. Weekend intensity regressor (not just binary)
    enhanced_df['weekend_intensity'] = 0
    enhanced_df.loc[enhanced_df['day_of_week'] == 5, 'weekend_intensity'] = 1  # Friday
    enhanced_df.loc[enhanced_df['day_of_week'] == 6, 'weekend_intensity'] = 2  # Saturday (highest)
    enhanced_df.loc[enhanced_df['day_of_week'] == 0, 'weekend_intensity'] = 1.5  # Sunday
    
    # 2. Month-end effect regressor
    enhanced_df['month_end_effect'] = 0
    for _, row in enhanced_df.iterrows():
        days_to_end = (row['ds'] + pd.offsets.MonthEnd(0) - row['ds']).days
        if days_to_end <= 3:
            enhanced_df.loc[enhanced_df['ds'] == row['ds'], 'month_end_effect'] = 4 - days_to_end
    
    # 3. Summer surge regressor (June-August with peak in July)
    enhanced_df['summer_surge'] = 0
    enhanced_df.loc[enhanced_df['month'] == 6, 'summer_surge'] = 1  # June
    enhanced_df.loc[enhanced_df['month'] == 7, 'summer_surge'] = 2  # July (peak)
    enhanced_df.loc[enhanced_df['month'] == 8, 'summer_surge'] = 1  # August
    
    # 4. Payday effect regressor (1st, 15th of month)
    enhanced_df['payday_effect'] = 0
    enhanced_df.loc[enhanced_df['day'].isin([1, 15]), 'payday_effect'] = 1
    enhanced_df.loc[enhanced_df['day'].isin([2, 16]), 'payday_effect'] = 0.5  # Next day effect
    
    # 5. Promotional period regressor (first week of month)
    enhanced_df['promo_period'] = 0
    enhanced_df.loc[enhanced_df['day'] <= 7, 'promo_period'] = 1
    
    # 6. Anomaly detection regressor (using isolation forest approach)
    from sklearn.ensemble import IsolationForest
    
    features_for_anomaly = enhanced_df[['y', 'day_of_week', 'month', 'day_of_year']].fillna(0)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    enhanced_df['anomaly_score'] = iso_forest.fit_predict(features_for_anomaly)
    enhanced_df['is_anomaly'] = (enhanced_df['anomaly_score'] == -1).astype(int)
    
    print(f"✅ Enhanced regressors created:")
    print(f"   • weekend_intensity: Peak Saturday effect")
    print(f"   • month_end_effect: End-of-month shopping")
    print(f"   • summer_surge: July peak pattern")
    print(f"   • payday_effect: 1st/15th salary days")
    print(f"   • promo_period: First week promotions")
    print(f"   • is_anomaly: Statistical anomaly detection")
    
    return enhanced_df

def test_enhanced_model_with_spike_handling():
    """Test Prophet model with enhanced spike handling"""
    
    print(f"\n🧪 TESTING ENHANCED MODEL WITH SPIKE HANDLING:")
    print("-" * 55)
    
    # Get enhanced data
    daily_demand, unexplained_spikes = analyze_remaining_spikes()
    enhanced_data = create_enhanced_spike_regressors(daily_demand)
    
    # Test on one category (electronics - most volatile)
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Get electronics category data
    electronics_data = df[df['category'] == 'electronics'].groupby('ds')['y'].sum().reset_index()
    electronics_data = electronics_data.sort_values('ds').reset_index(drop=True)
    
    # Merge with enhanced regressors
    electronics_enhanced = electronics_data.merge(
        enhanced_data[['ds', 'weekend_intensity', 'month_end_effect', 'summer_surge', 
                      'payday_effect', 'promo_period', 'is_anomaly']], 
        on='ds', how='left'
    ).fillna(0)
    
    # Split data for testing
    split_idx = int(len(electronics_enhanced) * 0.8)
    train_data = electronics_enhanced[:split_idx]
    test_data = electronics_enhanced[split_idx:]
    
    print(f"📊 Testing on Electronics Category:")
    print(f"   Train period: {train_data['ds'].min()} to {train_data['ds'].max()}")
    print(f"   Test period: {test_data['ds'].min()} to {test_data['ds'].max()}")
    
    # Test different model configurations
    models_to_test = [
        {
            'name': 'Baseline (No Spike Handling)',
            'regressors': []
        },
        {
            'name': 'Weekend Enhanced',
            'regressors': ['weekend_intensity']
        },
        {
            'name': 'Full Spike Handling',
            'regressors': ['weekend_intensity', 'month_end_effect', 'summer_surge', 
                          'payday_effect', 'promo_period']
        },
        {
            'name': 'Anomaly Detection Enhanced',
            'regressors': ['weekend_intensity', 'month_end_effect', 'summer_surge', 
                          'payday_effect', 'promo_period', 'is_anomaly']
        }
    ]
    
    results = {}
    
    for model_config in models_to_test:
        try:
            # Create holiday calendar
            holidays_df = create_enhanced_holiday_calendar()
            
            # Initialize model
            model = Prophet(
                holidays=holidays_df,
                changepoint_prior_scale=0.5,
                seasonality_mode='multiplicative'
            )
            
            # Add regressors
            for regressor in model_config['regressors']:
                model.add_regressor(regressor, prior_scale=10, standardize=True)
            
            # Prepare training data
            train_cols = ['ds', 'y'] + model_config['regressors']
            model.fit(train_data[train_cols])
            
            # Make predictions
            test_cols = ['ds'] + model_config['regressors']
            forecast = model.predict(test_data[test_cols])
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(test_data['y'], forecast['yhat']))
            mae = mean_absolute_error(test_data['y'], forecast['yhat'])
            r2 = r2_score(test_data['y'], forecast['yhat'])
            
            results[model_config['name']] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'regressors': model_config['regressors']
            }
            
            print(f"\n📈 {model_config['name']}:")
            print(f"   RMSE: {rmse:.1f}")
            print(f"   MAE: {mae:.1f}")
            print(f"   R²: {r2:.3f}")
            print(f"   Regressors: {', '.join(model_config['regressors']) if model_config['regressors'] else 'None'}")
            
        except Exception as e:
            print(f"❌ Error with {model_config['name']}: {str(e)}")
    
    return results

def create_enhanced_holiday_calendar():
    """Create enhanced holiday calendar including seasonal patterns"""
    
    holidays_df = pd.DataFrame([
        # Original holidays
        {'holiday': 'black_friday', 'ds': '2022-11-25', 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'black_friday', 'ds': '2023-11-24', 'lower_window': -1, 'upper_window': 1},
        {'holiday': 'black_friday', 'ds': '2024-11-29', 'lower_window': -1, 'upper_window': 1},
        
        {'holiday': 'christmas_season', 'ds': '2022-12-24', 'lower_window': -3, 'upper_window': 2},
        {'holiday': 'christmas_season', 'ds': '2023-12-24', 'lower_window': -3, 'upper_window': 2},
        {'holiday': 'christmas_season', 'ds': '2024-12-24', 'lower_window': -3, 'upper_window': 2},
        
        # Enhanced: Summer surge events (identified from analysis)
        {'holiday': 'summer_peak', 'ds': '2022-07-16', 'lower_window': -3, 'upper_window': 3},
        {'holiday': 'summer_peak', 'ds': '2023-07-15', 'lower_window': -3, 'upper_window': 3},
        {'holiday': 'summer_peak', 'ds': '2024-07-15', 'lower_window': -3, 'upper_window': 3},
        
        # Enhanced: Month-end shopping surges
        {'holiday': 'month_end_surge', 'ds': '2022-01-31', 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'month_end_surge', 'ds': '2022-02-28', 'lower_window': 0, 'upper_window': 0},
        {'holiday': 'month_end_surge', 'ds': '2022-03-31', 'lower_window': 0, 'upper_window': 0},
        # ... (would continue for all months, abbreviated for space)
    ])
    
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
    return holidays_df

def visualize_spike_patterns(daily_demand, unexplained_spikes):
    """Create comprehensive spike pattern visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Spike Pattern Analysis', fontsize=16, fontweight='bold')
    
    # 1. Time series with spike highlighting
    ax1 = axes[0, 0]
    ax1.plot(daily_demand['ds'], daily_demand['y'], 'b-', alpha=0.6, linewidth=0.8)
    ax1.plot(daily_demand['ds'], daily_demand['rolling_30'], 'r-', alpha=0.8, linewidth=1.5, label='30-day average')
    
    # Highlight unexplained spikes
    spike_dates = unexplained_spikes['ds']
    spike_demands = unexplained_spikes['y']
    ax1.scatter(spike_dates, spike_demands, color='red', s=30, alpha=0.8, zorder=5, label='Unexplained spikes')
    
    ax1.set_title('Daily Demand with Unexplained Spikes')
    ax1.set_ylabel('Total Daily Demand')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Spike intensity by day of week
    ax2 = axes[0, 1]
    spike_by_dow = unexplained_spikes.groupby('day_of_week').size()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    bars = ax2.bar(range(7), [spike_by_dow.get(i, 0) for i in range(7)], color='orange', alpha=0.7)
    ax2.set_title('Unexplained Spikes by Day of Week')
    ax2.set_ylabel('Number of Spikes')
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(days)
    
    # 3. Monthly spike distribution
    ax3 = axes[1, 0]
    spike_by_month = unexplained_spikes.groupby('month').size()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    bars = ax3.bar(range(1, 13), [spike_by_month.get(i, 0) for i in range(1, 13)], color='green', alpha=0.7)
    ax3.set_title('Unexplained Spikes by Month')
    ax3.set_ylabel('Number of Spikes')
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(months, rotation=45)
    
    # 4. Spike magnitude distribution
    ax4 = axes[1, 1]
    spike_ratios = unexplained_spikes['spike_ratio_7'].dropna()
    ax4.hist(spike_ratios, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(spike_ratios.mean(), color='red', linestyle='--', label=f'Mean: {spike_ratios.mean():.2f}')
    ax4.set_title('Distribution of Spike Intensities')
    ax4.set_xlabel('Spike Ratio (vs 7-day avg)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spike_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Spike pattern visualization saved as 'spike_pattern_analysis.png'")

def main():
    """Main execution for spike analysis"""
    
    print("🔍 ADVANCED SPIKE PATTERN ANALYSIS & HANDLING")
    print("=" * 70)
    
    # Analyze remaining spikes
    daily_demand, unexplained_spikes = analyze_remaining_spikes()
    
    # Categorize spike patterns
    spike_categories = categorize_spike_patterns(daily_demand, unexplained_spikes)
    
    # Create enhanced regressors
    enhanced_data = create_enhanced_spike_regressors(daily_demand)
    
    # Test enhanced model
    model_results = test_enhanced_model_with_spike_handling()
    
    # Create visualizations
    visualize_spike_patterns(daily_demand, unexplained_spikes)
    
    # Summary recommendations
    print(f"\n{'='*70}")
    print("🎯 SPIKE HANDLING RECOMMENDATIONS")
    print("=" * 70)
    
    print("📊 Key Findings:")
    if model_results:
        best_model = min(model_results.items(), key=lambda x: x[1]['rmse'])
        print(f"   • Best performing model: {best_model[0]}")
        print(f"   • RMSE improvement: {model_results.get('Baseline (No Spike Handling)', {}).get('rmse', 0) - best_model[1]['rmse']:.1f}")
    
    print(f"\n💡 Implementation Strategy:")
    print("1. ✅ Add weekend_intensity regressor (non-binary weekend effect)")
    print("2. ✅ Include month_end_effect regressor (end-of-month shopping)")
    print("3. ✅ Add summer_surge regressor (July peak pattern)")
    print("4. ✅ Include payday_effect regressor (1st/15th salary impact)")
    print("5. ✅ Add promotional period detection")
    print("6. ⚡ Implement anomaly detection for extreme outliers")
    print("7. 🔄 Use ensemble approach for remaining unexplained variance")
    
    print(f"\n🚀 Next Steps:")
    print("1. Integrate enhanced regressors into final_optimized_model.py")
    print("2. Test on all categories with cross-validation")
    print("3. Implement outlier capping for extreme spikes")
    print("4. Add external data sources (weather, events) if available")
    print("5. Consider ensemble methods for remaining unexplained patterns")
    
    return enhanced_data, model_results

if __name__ == "__main__":
    enhanced_data, results = main()
