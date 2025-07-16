#!/usr/bin/env python3
"""
Process Kaggle datasets for seasonal forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def process_ecommerce_data():
    """Process the main e-commerce dataset"""
    print('🔄 PROCESSING ECOMMERCE DATASET')
    print('=' * 50)
    
    # Load the main dataset with proper encoding
    try:
        df = pd.read_csv('data/datasets/data.csv', encoding='utf-8')
    except UnicodeDecodeError:
        print('⚠️ UTF-8 encoding failed, trying ISO-8859-1...')
        df = pd.read_csv('data/datasets/data.csv', encoding='ISO-8859-1')
    print(f'📊 Loaded dataset: {df.shape[0]:,} records, {df.shape[1]} columns')
    
    # Check data quality
    print(f'📅 Date range: {df["InvoiceDate"].min()} to {df["InvoiceDate"].max()}')
    print(f'🏷️ Unique products: {df["StockCode"].nunique():,}')
    print(f'🛒 Unique customers: {df["CustomerID"].nunique():,}')
    print(f'🌍 Countries: {df["Country"].nunique()}')
    
    # Process for Prophet format
    print(f'\n🔮 Converting to Prophet format...')
    
    # Convert date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Filter positive quantities only
    df = df[df['Quantity'] > 0]
    
    # Create daily aggregations by product
    daily_demand = df.groupby(['StockCode', df['InvoiceDate'].dt.date]).agg({
        'Quantity': 'sum',
        'UnitPrice': 'mean',
        'Description': 'first',
        'Country': 'first'
    }).reset_index()
    
    daily_demand.rename(columns={
        'InvoiceDate': 'ds',
        'Quantity': 'y',
        'StockCode': 'product_id'
    }, inplace=True)
    
    # Convert ds to datetime
    daily_demand['ds'] = pd.to_datetime(daily_demand['ds'])
    
    # Ensure processed directory exists
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    daily_demand.to_csv('data/processed/daily_demand_by_product.csv', index=False)
    print(f'✅ Saved processed data: {len(daily_demand):,} records')
    
    # Get top products for testing
    top_products = daily_demand.groupby('product_id')['y'].sum().nlargest(10)
    print(f'\n📈 TOP 10 PRODUCTS BY DEMAND:')
    for product, total_demand in top_products.items():
        try:
            product_name = daily_demand[daily_demand['product_id'] == product]['Description'].iloc[0]
            print(f'   {product}: {total_demand:,.0f} units - {product_name[:50]}...')
        except:
            print(f'   {product}: {total_demand:,.0f} units')
    
    # Create summary stats
    summary = {
        'total_records': len(daily_demand),
        'unique_products': daily_demand['product_id'].nunique(),
        'date_range': {
            'start': daily_demand['ds'].min().strftime('%Y-%m-%d'),
            'end': daily_demand['ds'].max().strftime('%Y-%m-%d')
        },
        'total_demand': daily_demand['y'].sum(),
        'avg_daily_demand': daily_demand['y'].mean()
    }
    
    print(f'\n📊 DATASET SUMMARY:')
    print(f'   Total records: {summary["total_records"]:,}')
    print(f'   Unique products: {summary["unique_products"]:,}')
    print(f'   Date range: {summary["date_range"]["start"]} to {summary["date_range"]["end"]}')
    print(f'   Total demand: {summary["total_demand"]:,.0f} units')
    print(f'   Average daily demand per product: {summary["avg_daily_demand"]:.1f} units')
    
    print(f'\n🎯 Dataset ready for forecasting!')
    print(f'   ✅ Processed data saved to: data/processed/daily_demand_by_product.csv')
    print(f'   📊 Ready for training Prophet models')
    
    return daily_demand, summary

if __name__ == "__main__":
    try:
        data, summary = process_ecommerce_data()
        print("\n🎉 Data processing completed successfully!")
    except Exception as e:
        print(f"\n❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
