#!/usr/bin/env python3
#Check what products are available in the dataset directly

import pandas as pd
from pathlib import Path

def main():
    try:
        print("Checking available products in dataset...\n")
        
        # Find the data file
        data_path = Path("ai_services/seasonal_inventory/data/processed/daily_demand_by_product_modern.csv")
        
        if not data_path.exists():
            print(f"Data file not found at: {data_path}")
            return 1
        
        # Load data
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for product column
        if 'product_id' in df.columns:
            products = df['product_id'].unique()
            print(f"\nTotal unique products: {len(products)}")
            
            # Show sample products
            print(f"\nSample products (first 20):")
            for i, product in enumerate(products[:20]):
                product_data = df[df['product_id'] == product]
                print(f"  {i+1:2d}. {product} - {len(product_data)} data points")
            
            if len(products) > 20:
                print(f"  ... and {len(products) - 20} more products")
            
            # Analyze data distribution
            data_points_per_product = df.groupby('product_id').size()
            
            print(f"\nData points distribution:")
            print(f"  Minimum: {data_points_per_product.min()} data points")
            print(f"  Maximum: {data_points_per_product.max()} data points")
            print(f"  Average: {data_points_per_product.mean():.1f} data points")
            print(f"  Median: {data_points_per_product.median():.1f} data points")
            
            # Products with sufficient data for training (30+ points)
            sufficient_data = data_points_per_product[data_points_per_product >= 30]
            print(f"\nProducts suitable for training (30+ data points): {len(sufficient_data)}")
            print(f"Training coverage: {len(sufficient_data)/len(products)*100:.1f}%")
            
            # Check date range
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
            elif 'ds' in df.columns:
                df['ds'] = pd.to_datetime(df['ds'])
                print(f"\nDate range: {df['ds'].min()} to {df['ds'].max()}")
            
            # Check existing models
            models_path = Path("ai_services/seasonal_inventory/data/models")
            if models_path.exists():
                existing_models = list(models_path.glob("*_prophet_model.pkl"))
                print(f"\nExisting trained models: {len(existing_models)}")
                
                if existing_models:
                    print("Sample existing models:")
                    for i, model_file in enumerate(existing_models[:10]):
                        product_id = model_file.stem.replace('_prophet_model', '')
                        print(f"  - {product_id}")
                    if len(existing_models) > 10:
                        print(f"  ... and {len(existing_models) - 10} more models")
            else:
                print(f"\nModels directory not found: {models_path}")
                print("No existing models")
            
            return 0
            
        else:
            print("No 'product_id' column found in dataset")
            print(f"Available columns: {list(df.columns)}")
            return 1
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
