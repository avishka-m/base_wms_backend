#!/usr/bin/env python3
"""
Simple script to evaluate Prophet models with proper 80/20 train/test split

This replaces the complex API endpoints with a simple command-line tool.
Run this to evaluate your models for overfitting after removing the old ones.
"""

import pandas as pd
import sys
from pathlib import Path
import logging
import argparse
from datetime import datetime

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_services.seasonal_inventory.src.models.prophet_forecaster import ProphetForecaster

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_single_product(product_id, train_ratio=0.8, test_ratio=0.2):
    """Evaluate a single product with train/test split"""
    print(f"\n🧪 Evaluating {product_id} with {train_ratio:.0%}/{test_ratio:.0%} split")
    
    # Load data
    data_path = Path("ai_services/seasonal_inventory/data/processed/daily_demand_by_product_modern.csv")
    df = pd.read_csv(data_path)
    
    # Filter for this product
    product_data = df[df['product_id'] == product_id].copy()
    if product_data.empty:
        print(f"❌ No data found for {product_id}")
        return None
    
    print(f"📊 Found {len(product_data)} data points from {product_data['ds'].min()} to {product_data['ds'].max()}")
    
    # Create forecaster and evaluate
    forecaster = ProphetForecaster(product_id=product_id)
    result = forecaster.evaluate_with_train_test_split(
        data=product_data,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        target_column='y'
    )
    
    if result["status"] == "success":
        metrics = result["test_metrics"]
        print(f"✅ MAPE: {metrics['mape']:.2f}%")
        print(f"✅ RMSE: {metrics['rmse']:.2f}")
        print(f"✅ Coverage: {metrics['coverage_percent']:.1f}%")
        
        # Show interpretation
        for insight in result.get("interpretation", []):
            print(f"   {insight}")
        
        return result
    else:
        print(f"❌ Evaluation failed: {result.get('message', 'Unknown error')}")
        return None

def evaluate_multiple_products(product_ids, train_ratio=0.8, test_ratio=0.2):
    """Evaluate multiple products"""
    print(f"\n🔍 Evaluating {len(product_ids)} products with {train_ratio:.0%}/{test_ratio:.0%} split")
    
    results = {}
    successful = 0
    
    for i, product_id in enumerate(product_ids, 1):
        print(f"\n[{i}/{len(product_ids)}] Processing {product_id}")
        result = evaluate_single_product(product_id, train_ratio, test_ratio)
        
        if result and result["status"] == "success":
            results[product_id] = result
            successful += 1
        
    print(f"\n📈 Summary: {successful}/{len(product_ids)} successful evaluations")
    
    if successful > 0:
        # Calculate average metrics
        all_mape = [r["test_metrics"]["mape"] for r in results.values() if r["test_metrics"]["mape"] is not None]
        all_rmse = [r["test_metrics"]["rmse"] for r in results.values()]
        
        if all_mape:
            print(f"📊 Average MAPE: {sum(all_mape)/len(all_mape):.2f}%")
        if all_rmse:
            print(f"📊 Average RMSE: {sum(all_rmse)/len(all_rmse):.2f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Prophet models with train/test split")
    parser.add_argument("--product", help="Single product ID to evaluate")
    parser.add_argument("--products", help="Comma-separated list of product IDs")
    parser.add_argument("--sample", type=int, default=5, help="Number of random products to sample")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Testing data ratio")
    
    args = parser.parse_args()
    
    print("🚀 Prophet Model Evaluation with Train/Test Split")
    print("=" * 60)
    print("🎯 This script evaluates models for overfitting using proper data splitting")
    print("🔄 Unlike Prophet's CV, this simulates real-world deployment scenarios")
    print()
    
    if args.product:
        # Single product
        evaluate_single_product(args.product, args.train_ratio, args.test_ratio)
    
    elif args.products:
        # Multiple specific products
        product_list = [p.strip() for p in args.products.split(',')]
        evaluate_multiple_products(product_list, args.train_ratio, args.test_ratio)
    
    else:
        # Sample random products
        print(f"📊 Loading data to find {args.sample} sample products...")
        data_path = Path("ai_services/seasonal_inventory/data/processed/daily_demand_by_product_modern.csv")
        df = pd.read_csv(data_path)
        
        # Get products with sufficient data
        product_counts = df.groupby('product_id').size()
        sufficient_products = product_counts[product_counts >= 50].index.tolist()
        
        if len(sufficient_products) < args.sample:
            sample_products = sufficient_products
        else:
            import random
            sample_products = random.sample(sufficient_products, args.sample)
        
        print(f"✅ Selected {len(sample_products)} products: {sample_products}")
        evaluate_multiple_products(sample_products, args.train_ratio, args.test_ratio)
    
    print("\n💡 Next Steps:")
    print("1. Compare these metrics with your old cross-validation results")
    print("2. If MAPE is much higher, your old models were likely overfitted")
    print("3. Use simple_batch_train.py to retrain with proper splits")

if __name__ == "__main__":
    main()
