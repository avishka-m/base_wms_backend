"""
Clean Enhanced Dataset for NeuralProphet
Removes US holidays and lag features, keeps essential time features
"""

import pandas as pd
import numpy as np
from datetime import datetime

def create_clean_enhanced_dataset():
    """Create enhanced dataset without US holidays and lag features"""
    
    print("="*80)
    print("CREATING CLEAN ENHANCED DATASET FOR NEURALPROPHET")
    print("="*80)
    
    # Load original data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    print(f"📊 Original dataset: {df.shape}")
    
    # Add essential time-based features only
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['day_of_week'] = df['ds'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_year'] = df['ds'].dt.dayofyear
    
    # Weekend indicator (essential for retail)
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
    
    # Outlier treatment (cap at 95th percentile by category)
    print("🔧 Applying outlier treatment...")
    df['y_original'] = df['y']  # Keep original
    
    for category in df['category'].unique():
        mask = df['category'] == category
        p95 = df.loc[mask, 'y'].quantile(0.95)
        outlier_count = (df.loc[mask, 'y'] > p95).sum()
        df.loc[mask & (df['y'] > p95), 'y'] = p95
        print(f"  • {category}: capped {outlier_count} outliers at {p95:.2f}")
    
    print(f"✅ Clean enhanced dataset created!")
    print(f"  • Original columns: 4")
    print(f"  • Enhanced columns: {len(df.columns)}")
    print(f"  • Features added: {list(df.columns[4:])}")
    
    # Save clean enhanced dataset
    output_file = 'daily_demand_clean_enhanced.csv'
    df.to_csv(output_file, index=False)
    print(f"  • Saved as: {output_file}")
    
    # Show sample
    print(f"\n📋 Sample data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    df = create_clean_enhanced_dataset()
    
    print(f"\n🎯 CLEAN DATASET SUMMARY:")
    print(f"  • No US holidays included")
    print(f"  • No lag features or rolling averages")
    print(f"  • Essential time features: weekend, season, month boundaries")
    print(f"  • Outliers capped at 95th percentile")
    print(f"  • Ready for NeuralProphet modeling!")
