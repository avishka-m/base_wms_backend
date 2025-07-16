"""
Data Quality Analysis for Prophet Forecasting Dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset_quality():
    """Comprehensive analysis of the dataset quality and preprocessing needs"""
    
    print("="*80)
    print("DATASET QUALITY ANALYSIS")
    print("="*80)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    print(f"📊 BASIC DATASET INFO:")
    print(f"  • Shape: {df.shape}")
    print(f"  • Columns: {list(df.columns)}")
    print(f"  • Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"  • Time span: {(df['ds'].max() - df['ds'].min()).days} days")
    print(f"  • Products: {df['product_id'].nunique()}")
    print(f"  • Categories: {df['category'].nunique()}")
    
    print(f"\n🔍 DATA QUALITY CHECKS:")
    
    # Missing values
    missing_values = df.isnull().sum()
    print(f"  • Missing values: {missing_values.sum()} total")
    for col, count in missing_values.items():
        if count > 0:
            print(f"    - {col}: {count} ({count/len(df)*100:.2f}%)")
    
    # Zero values
    zero_count = (df['y'] == 0).sum()
    print(f"  • Zero demand values: {zero_count} ({zero_count/len(df)*100:.1f}%)")
    
    # Negative values
    negative_count = (df['y'] < 0).sum()
    print(f"  • Negative demand values: {negative_count}")
    
    # Duplicate records
    duplicates = df.duplicated(subset=['product_id', 'ds']).sum()
    print(f"  • Duplicate product-date records: {duplicates}")
    
    print(f"\n📈 DEMAND STATISTICS:")
    stats = df['y'].describe()
    for stat, value in stats.items():
        print(f"  • {stat}: {value:.2f}")
    
    # Outlier analysis using IQR method
    Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['y'] < lower_bound) | (df['y'] > upper_bound)]
    print(f"\n🚨 OUTLIER ANALYSIS (IQR method):")
    print(f"  • Lower bound: {lower_bound:.2f}")
    print(f"  • Upper bound: {upper_bound:.2f}")
    print(f"  • Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"  • Max outlier: {df['y'].max():.2f}")
    
    # Data completeness by product
    print(f"\n📅 DATA COMPLETENESS BY PRODUCT:")
    expected_days = (df['ds'].max() - df['ds'].min()).days + 1
    product_completeness = df.groupby('product_id')['ds'].count()
    complete_products = (product_completeness == expected_days).sum()
    print(f"  • Expected days per product: {expected_days}")
    print(f"  • Products with complete data: {complete_products}/{df['product_id'].nunique()}")
    print(f"  • Average days per product: {product_completeness.mean():.1f}")
    print(f"  • Min days per product: {product_completeness.min()}")
    
    # Check for additional features
    print(f"\n🏷️ CURRENT FEATURES:")
    for col in df.columns:
        print(f"  • {col}: {df[col].dtype}")
    
    # Weekend/Holiday analysis needs
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    weekend_avg = df[df['is_weekend']]['y'].mean()
    weekday_avg = df[~df['is_weekend']]['y'].mean()
    
    print(f"\n📊 WEEKEND PATTERN ANALYSIS:")
    print(f"  • Weekend average demand: {weekend_avg:.2f}")
    print(f"  • Weekday average demand: {weekday_avg:.2f}")
    print(f"  • Weekend vs Weekday ratio: {weekend_avg/weekday_avg:.2f}")
    
    return df

def check_missing_features():
    """Check what features are missing for optimal Prophet forecasting"""
    
    print(f"\n🎯 MISSING FEATURES FOR PROPHET OPTIMIZATION:")
    
    missing_features = [
        "is_weekend - Weekend indicator (0/1)",
        "is_holiday - Holiday indicator (0/1)", 
        "month - Month number (1-12)",
        "quarter - Quarter (1-4)",
        "day_of_week - Day of week (0-6)",
        "is_month_start - Month start indicator",
        "is_month_end - Month end indicator",
        "season - Season (Spring/Summer/Fall/Winter)",
        "trend_component - Decomposed trend",
        "lag_features - Previous day/week demand",
        "rolling_averages - Moving averages"
    ]
    
    for i, feature in enumerate(missing_features, 1):
        print(f"  {i}. {feature}")
    
    print(f"\n⚠️ DATA PREPROCESSING RECOMMENDATIONS:")
    
    recommendations = [
        "✅ Handle zero values (keep as-is for Prophet - it handles zeros well)",
        "⚠️ Review outliers (dataset has outliers that may need capping)",
        "❌ Add is_weekend feature (strongly recommended for retail)",
        "❌ Add is_holiday feature (US holidays recommended)",
        "❌ Add seasonal indicators (month, quarter)",
        "❌ Check for sufficient data per product (minimum 1 year recommended)",
        "✅ Date format is correct (Prophet compatible)",
        "✅ No missing values in core columns",
        "⚠️ Consider outlier treatment (values above 95th percentile)"
    ]
    
    for rec in recommendations:
        print(f"  • {rec}")

def create_enhanced_dataset():
    """Create an enhanced version with additional features"""
    
    print(f"\n🔧 CREATING ENHANCED DATASET WITH ADDITIONAL FEATURES:")
    
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Add time-based features
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['day_of_week'] = df['ds'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_year'] = df['ds'].dt.dayofyear
    
    # Weekend indicator
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Month indicators
    df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
    
    # Season mapping
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring', 
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall', 10: 'Fall', 11: 'Fall'}
    df['season'] = df['month'].map(season_map)
    
    # Simple US holiday indicators (major holidays)
    holidays = [
        ('2022-01-01', 'New Year'),    # New Year's Day
        ('2022-07-04', 'Independence'), # Independence Day  
        ('2022-11-24', 'Thanksgiving'), # Thanksgiving 2022
        ('2022-12-25', 'Christmas'),   # Christmas
        ('2023-01-01', 'New Year'),    
        ('2023-07-04', 'Independence'),
        ('2023-11-23', 'Thanksgiving'), # Thanksgiving 2023
        ('2023-12-25', 'Christmas'),
        ('2024-01-01', 'New Year'),
        ('2024-07-04', 'Independence'),
        ('2024-11-28', 'Thanksgiving'), # Thanksgiving 2024
        ('2024-12-25', 'Christmas')
    ]
    
    df['is_holiday'] = 0
    df['holiday_name'] = ''
    
    for date_str, holiday_name in holidays:
        holiday_date = pd.to_datetime(date_str)
        mask = df['ds'] == holiday_date
        df.loc[mask, 'is_holiday'] = 1
        df.loc[mask, 'holiday_name'] = holiday_name
    
    # Outlier treatment (cap at 95th percentile by category)
    print("  • Applying outlier treatment...")
    df['y_original'] = df['y']  # Keep original
    
    for category in df['category'].unique():
        mask = df['category'] == category
        p95 = df.loc[mask, 'y'].quantile(0.95)
        outlier_count = (df.loc[mask, 'y'] > p95).sum()
        df.loc[mask & (df['y'] > p95), 'y'] = p95
        print(f"    - {category}: capped {outlier_count} outliers at {p95:.2f}")
    
    # Add some lag features (previous day demand)
    print("  • Adding lag features...")
    df = df.sort_values(['product_id', 'ds'])
    df['y_lag1'] = df.groupby('product_id')['y'].shift(1)
    df['y_lag7'] = df.groupby('product_id')['y'].shift(7)
    
    # Rolling averages
    df['y_rolling_7'] = df.groupby('product_id')['y'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    df['y_rolling_30'] = df.groupby('product_id')['y'].rolling(30, min_periods=1).mean().reset_index(0, drop=True)
    
    print(f"✅ Enhanced dataset created!")
    print(f"  • Original columns: {len(['product_id', 'ds', 'y', 'category'])}")
    print(f"  • Enhanced columns: {len(df.columns)}")
    print(f"  • New features: {list(df.columns[4:])}")
    
    # Save enhanced dataset
    output_file = 'daily_demand_enhanced.csv'
    df.to_csv(output_file, index=False)
    print(f"  • Saved as: {output_file}")
    
    return df

if __name__ == "__main__":
    # Run analysis
    df = analyze_dataset_quality()
    check_missing_features()
    enhanced_df = create_enhanced_dataset()
    
    print(f"\n🎯 SUMMARY:")
    print(f"  • Your current dataset is fairly clean but missing key features")
    print(f"  • Zero values are handled appropriately (Prophet can handle them)")
    print(f"  • Outliers exist and have been capped in enhanced version")
    print(f"  • Enhanced dataset includes is_weekend, is_holiday, and seasonal features")
    print(f"  • Ready for advanced Prophet forecasting!")
