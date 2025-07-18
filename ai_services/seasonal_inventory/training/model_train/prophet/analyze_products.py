import pandas as pd
import numpy as np

# Load and examine product-level data
df = pd.read_csv('daily_demand_by_product_modern.csv')
df['ds'] = pd.to_datetime(df['ds'])

print('=== PRODUCT-LEVEL ANALYSIS ===')
print(f'Total products: {df["product_id"].nunique()}')
print(f'Categories: {df["category"].nunique()}')
print(f'Date range: {df["ds"].min()} to {df["ds"].max()}')

print('\n=== PRODUCTS PER CATEGORY ===')
products_per_cat = df.groupby('category')['product_id'].nunique()
print(products_per_cat)

print('\n=== SAMPLE PRODUCT PATTERNS ===')
# Show sample products from each category
for category in df['category'].unique()[:3]:
    sample_products = df[df['category'] == category]['product_id'].unique()[:2]
    print(f'\n{category.upper()}:')
    for product in sample_products:
        product_data = df[df['product_id'] == product]
        avg_demand = product_data['y'].mean()
        std_demand = product_data['y'].std()
        cv = std_demand / avg_demand if avg_demand > 0 else 0
        print(f'  Product {product}: avg={avg_demand:.1f}, std={std_demand:.1f}, CV={cv:.2f}')

# Check for zero-demand products
print('\n=== DEMAND DISTRIBUTION ===')
zero_demand_products = df.groupby('product_id')['y'].sum() == 0
print(f'Products with zero total demand: {zero_demand_products.sum()}')

# Get some high-volume products for detailed analysis
print('\n=== TOP PRODUCTS BY VOLUME ===')
top_products = df.groupby('product_id')['y'].sum().nlargest(10)
print(top_products)
