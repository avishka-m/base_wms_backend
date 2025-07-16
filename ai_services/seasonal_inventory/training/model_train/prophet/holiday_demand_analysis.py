"""
Holiday and High Demand Analysis
Identify holidays and events that cause demand spikes in the dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_holiday_demand_patterns():
    """Analyze demand patterns around holidays and special events"""
    
    print("🎄 HOLIDAY AND HIGH DEMAND ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Create daily aggregated demand
    daily_demand = df.groupby('ds')['y'].sum().reset_index()
    daily_demand = daily_demand.sort_values('ds').reset_index(drop=True)
    
    print(f"📊 Dataset Overview:")
    print(f"   Date range: {daily_demand['ds'].min()} to {daily_demand['ds'].max()}")
    print(f"   Total days: {len(daily_demand):,}")
    print(f"   Average daily demand: {daily_demand['y'].mean():.0f}")
    print(f"   Max daily demand: {daily_demand['y'].max():.0f}")
    
    # Add date features
    daily_demand['year'] = daily_demand['ds'].dt.year
    daily_demand['month'] = daily_demand['ds'].dt.month
    daily_demand['day'] = daily_demand['ds'].dt.day
    daily_demand['day_of_week'] = daily_demand['ds'].dt.day_of_week
    daily_demand['day_name'] = daily_demand['ds'].dt.day_name()
    daily_demand['month_name'] = daily_demand['ds'].dt.month_name()
    
    # Calculate rolling average for baseline comparison
    daily_demand['rolling_7'] = daily_demand['y'].rolling(window=7, center=True).mean()
    daily_demand['rolling_30'] = daily_demand['y'].rolling(window=30, center=True).mean()
    
    # Identify high demand days (above 95th percentile)
    threshold_95 = daily_demand['y'].quantile(0.95)
    threshold_99 = daily_demand['y'].quantile(0.99)
    
    high_demand_days = daily_demand[daily_demand['y'] >= threshold_95].copy()
    extreme_demand_days = daily_demand[daily_demand['y'] >= threshold_99].copy()
    
    print(f"\n📈 High Demand Analysis:")
    print(f"   95th percentile threshold: {threshold_95:.0f}")
    print(f"   99th percentile threshold: {threshold_99:.0f}")
    print(f"   High demand days (>95th): {len(high_demand_days)}")
    print(f"   Extreme demand days (>99th): {len(extreme_demand_days)}")
    
    return daily_demand, high_demand_days, extreme_demand_days

def identify_holiday_patterns(daily_demand, high_demand_days):
    """Identify specific holidays and their demand impact"""
    
    print(f"\n🎯 HOLIDAY PATTERN IDENTIFICATION:")
    print("-" * 40)
    
    # Define major holidays (adjust dates as needed for your region)
    holidays = {
        'New Year': [(1, 1)],
        'Valentine Day': [(2, 14)],
        'Easter': [(4, 9), (4, 17), (4, 1), (3, 31)],  # Approximate dates for different years
        'Mother Day': [(5, 8), (5, 14), (5, 13)],  # Second Sunday of May (approximate)
        'Father Day': [(6, 19), (6, 18), (6, 16)],  # Third Sunday of June (approximate)
        'Independence Day': [(7, 4)],
        'Halloween': [(10, 31)],
        'Thanksgiving': [(11, 24), (11, 23), (11, 28)],  # Fourth Thursday of November (approximate)
        'Black Friday': [(11, 25), (11, 24), (11, 29)],  # Day after Thanksgiving
        'Cyber Monday': [(11, 28), (11, 27), (12, 2)],  # Monday after Thanksgiving
        'Christmas': [(12, 25)],
        'Boxing Day': [(12, 26)],
        'New Year Eve': [(12, 31)]
    }
    
    # Add holiday flags
    daily_demand['is_holiday'] = False
    daily_demand['holiday_name'] = ''
    
    for holiday_name, dates in holidays.items():
        for month, day in dates:
            mask = (daily_demand['month'] == month) & (daily_demand['day'] == day)
            daily_demand.loc[mask, 'is_holiday'] = True
            daily_demand.loc[mask, 'holiday_name'] += f"{holiday_name}; "
    
    # Analyze demand around holidays
    holiday_demand = daily_demand[daily_demand['is_holiday']].copy()
    
    if len(holiday_demand) > 0:
        print(f"📅 Holiday Demand Analysis:")
        print(f"   Average holiday demand: {holiday_demand['y'].mean():.0f}")
        print(f"   Average non-holiday demand: {daily_demand[~daily_demand['is_holiday']]['y'].mean():.0f}")
        print(f"   Holiday demand uplift: {(holiday_demand['y'].mean() / daily_demand[~daily_demand['is_holiday']]['y'].mean() - 1) * 100:.1f}%")
        
        # Top holiday demand days
        print(f"\n🔥 Top Holiday Demand Days:")
        top_holidays = holiday_demand.nlargest(10, 'y')[['ds', 'y', 'holiday_name', 'day_name']]
        for _, row in top_holidays.iterrows():
            print(f"   {row['ds'].strftime('%Y-%m-%d')} ({row['day_name']}): {row['y']:.0f} - {row['holiday_name'].strip(';')}")
    
    return daily_demand

def analyze_seasonal_patterns(daily_demand):
    """Analyze seasonal and weekly patterns"""
    
    print(f"\n📊 SEASONAL PATTERN ANALYSIS:")
    print("-" * 40)
    
    # Monthly patterns
    monthly_avg = daily_demand.groupby('month_name')['y'].agg(['mean', 'std', 'count']).round(0)
    monthly_avg = monthly_avg.reindex(['January', 'February', 'March', 'April', 'May', 'June',
                                     'July', 'August', 'September', 'October', 'November', 'December'])
    
    print("📅 Monthly Demand Patterns:")
    for month, row in monthly_avg.iterrows():
        print(f"   {month:12s}: Avg={row['mean']:6.0f}, Std={row['std']:6.0f}, Days={row['count']:3.0f}")
    
    # Weekly patterns
    weekly_avg = daily_demand.groupby('day_name')['y'].agg(['mean', 'std']).round(0)
    weekly_avg = weekly_avg.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    print(f"\n📅 Weekly Demand Patterns:")
    for day, row in weekly_avg.iterrows():
        print(f"   {day:12s}: Avg={row['mean']:6.0f}, Std={row['std']:6.0f}")
    
    # Find peak months and days
    peak_month = monthly_avg['mean'].idxmax()
    peak_day = weekly_avg['mean'].idxmax()
    
    print(f"\n🎯 Peak Patterns:")
    print(f"   Peak month: {peak_month} (Avg: {monthly_avg.loc[peak_month, 'mean']:.0f})")
    print(f"   Peak day: {peak_day} (Avg: {weekly_avg.loc[peak_day, 'mean']:.0f})")
    
    return monthly_avg, weekly_avg

def find_demand_spikes(daily_demand, high_demand_days):
    """Find and analyze significant demand spikes"""
    
    print(f"\n⚡ DEMAND SPIKE ANALYSIS:")
    print("-" * 40)
    
    # Calculate demand spikes (compared to 7-day rolling average)
    daily_demand['spike_ratio'] = daily_demand['y'] / daily_demand['rolling_7']
    daily_demand['spike_diff'] = daily_demand['y'] - daily_demand['rolling_7']
    
    # Find top spikes
    top_spikes = daily_demand.nlargest(20, 'spike_ratio')[['ds', 'y', 'rolling_7', 'spike_ratio', 'spike_diff', 'day_name', 'month_name', 'holiday_name']]
    
    print("🚀 Top 20 Demand Spikes (vs 7-day average):")
    for _, row in top_spikes.iterrows():
        holiday_info = row['holiday_name'] if row['holiday_name'] else 'Regular day'
        print(f"   {row['ds'].strftime('%Y-%m-%d')} ({row['day_name']:9s}): {row['y']:6.0f} vs {row['rolling_7']:6.0f} (x{row['spike_ratio']:.2f}) - {holiday_info}")
    
    # Analyze spike patterns
    spike_threshold = 1.5  # 50% above rolling average
    significant_spikes = daily_demand[daily_demand['spike_ratio'] >= spike_threshold]
    
    print(f"\n📊 Spike Pattern Analysis (>50% above 7-day avg):")
    print(f"   Total significant spikes: {len(significant_spikes)}")
    
    if len(significant_spikes) > 0:
        spike_by_month = significant_spikes.groupby('month_name').size()
        spike_by_day = significant_spikes.groupby('day_name').size()
        
        print(f"   Spikes by month:")
        for month, count in spike_by_month.sort_values(ascending=False).head(6).items():
            print(f"     {month}: {count} spikes")
        
        print(f"   Spikes by day of week:")
        for day, count in spike_by_day.sort_values(ascending=False).items():
            print(f"     {day}: {count} spikes")
    
    return top_spikes

def analyze_category_holiday_patterns(df):
    """Analyze holiday patterns by category"""
    
    print(f"\n🏪 CATEGORY-SPECIFIC HOLIDAY PATTERNS:")
    print("-" * 50)
    
    # Create category-level daily data
    category_daily = df.groupby(['category', 'ds'])['y'].sum().reset_index()
    category_daily['ds'] = pd.to_datetime(category_daily['ds'])
    
    # Add date features
    category_daily['month'] = category_daily['ds'].dt.month
    category_daily['day'] = category_daily['ds'].dt.day
    
    # Define key shopping periods
    shopping_periods = {
        'Black Friday/Cyber Monday': [(11, 24), (11, 25), (11, 26), (11, 27), (11, 28), (11, 29)],
        'Christmas Season': [(12, 20), (12, 21), (12, 22), (12, 23), (12, 24), (12, 25), (12, 26)],
        'Valentine Day': [(2, 13), (2, 14), (2, 15)],
        'Back to School': [(8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (8, 21)],
        'Halloween': [(10, 30), (10, 31), (11, 1)]
    }
    
    for period_name, dates in shopping_periods.items():
        print(f"\n📦 {period_name} Impact by Category:")
        
        # Create period mask
        period_mask = False
        for month, day in dates:
            period_mask |= ((category_daily['month'] == month) & (category_daily['day'] == day))
        
        if period_mask.sum() > 0:
            period_demand = category_daily[period_mask].groupby('category')['y'].mean()
            normal_demand = category_daily[~period_mask].groupby('category')['y'].mean()
            
            for category in period_demand.index:
                uplift = (period_demand[category] / normal_demand[category] - 1) * 100
                print(f"   {category:15s}: {period_demand[category]:6.0f} vs {normal_demand[category]:6.0f} ({uplift:+5.1f}%)")

def create_holiday_visualizations(daily_demand, top_spikes):
    """Create visualizations for holiday demand patterns"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Holiday and High Demand Pattern Analysis', fontsize=16, fontweight='bold')
    
    # 1. Time series with spikes highlighted
    ax1 = axes[0, 0]
    ax1.plot(daily_demand['ds'], daily_demand['y'], 'b-', alpha=0.6, linewidth=0.8)
    ax1.plot(daily_demand['ds'], daily_demand['rolling_30'], 'r-', alpha=0.8, linewidth=1.5, label='30-day average')
    
    # Highlight top spikes
    spike_dates = top_spikes['ds'].head(10)
    spike_demands = top_spikes['y'].head(10)
    ax1.scatter(spike_dates, spike_demands, color='red', s=50, alpha=0.8, zorder=5)
    
    ax1.set_title('Daily Demand with Major Spikes Highlighted')
    ax1.set_ylabel('Total Daily Demand')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Monthly patterns
    ax2 = axes[0, 1]
    monthly_data = daily_demand.groupby('month_name')['y'].mean()
    months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_data = monthly_data.reindex(months_order)
    
    bars = ax2.bar(range(12), monthly_data.values, color='skyblue', alpha=0.7, edgecolor='navy')
    ax2.set_title('Average Demand by Month')
    ax2.set_ylabel('Average Daily Demand')
    ax2.set_xticks(range(12))
    ax2.set_xticklabels([m[:3] for m in months_order], rotation=45)
    
    # Highlight peak months
    max_idx = monthly_data.values.argmax()
    bars[max_idx].set_color('red')
    bars[max_idx].set_alpha(0.8)
    
    # 3. Weekly patterns
    ax3 = axes[1, 0]
    weekly_data = daily_demand.groupby('day_name')['y'].mean()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_data = weekly_data.reindex(days_order)
    
    bars = ax3.bar(range(7), weekly_data.values, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    ax3.set_title('Average Demand by Day of Week')
    ax3.set_ylabel('Average Daily Demand')
    ax3.set_xticks(range(7))
    ax3.set_xticklabels([d[:3] for d in days_order], rotation=45)
    
    # Highlight peak day
    max_idx = weekly_data.values.argmax()
    bars[max_idx].set_color('red')
    bars[max_idx].set_alpha(0.8)
    
    # 4. Spike distribution
    ax4 = axes[1, 1]
    spike_ratios = daily_demand['spike_ratio'].dropna()
    ax4.hist(spike_ratios, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax4.axvline(spike_ratios.mean(), color='red', linestyle='--', label=f'Mean: {spike_ratios.mean():.2f}')
    ax4.axvline(spike_ratios.quantile(0.95), color='green', linestyle='--', label=f'95th %ile: {spike_ratios.quantile(0.95):.2f}')
    ax4.set_title('Distribution of Demand Spike Ratios')
    ax4.set_xlabel('Spike Ratio (vs 7-day avg)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('holiday_demand_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Visualization saved as 'holiday_demand_analysis.png'")

def main():
    """Main analysis function"""
    
    # Load and analyze data
    daily_demand, high_demand_days, extreme_demand_days = analyze_holiday_demand_patterns()
    
    # Identify holiday patterns
    daily_demand = identify_holiday_patterns(daily_demand, high_demand_days)
    
    # Analyze seasonal patterns
    monthly_avg, weekly_avg = analyze_seasonal_patterns(daily_demand)
    
    # Find demand spikes
    top_spikes = find_demand_spikes(daily_demand, high_demand_days)
    
    # Analyze category-specific patterns
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    analyze_category_holiday_patterns(df)
    
    # Create visualizations
    create_holiday_visualizations(daily_demand, top_spikes)
    
    # Summary insights
    print(f"\n{'='*60}")
    print("🎯 KEY INSIGHTS & RECOMMENDATIONS")
    print(f"{'='*60}")
    
    print("📈 High Demand Drivers Identified:")
    print("1. 🎄 Seasonal patterns - November/December peak")
    print("2. 🛍️  Shopping holidays - Black Friday, Christmas")
    print("3. 📅 Weekly patterns - Weekend vs weekday differences")
    print("4. ⚡ Irregular spikes - Need further investigation")
    
    print(f"\n💡 Forecasting Recommendations:")
    print("1. ✅ Add holiday regressors to Prophet models")
    print("2. 📊 Include seasonal multipliers for peak months")
    print("3. 🔄 Monitor and update holiday calendar annually")
    print("4. 📈 Use category-specific holiday impacts")
    print("5. ⚡ Implement spike detection for unusual events")
    
    return daily_demand, top_spikes

if __name__ == "__main__":
    daily_demand, top_spikes = main()
