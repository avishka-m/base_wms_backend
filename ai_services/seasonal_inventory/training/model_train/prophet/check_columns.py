import pandas as pd

# Load the dataset and check its structure
df = pd.read_csv('daily_demand_by_product_modern.csv')

print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nLast 5 rows:\n{df.tail()}")
