import pandas as pd 


dataset_path="base_wms_backend/ai-services/seasonal-inventory/data/processed/daily_demand_by_product_modern.csv"
df=pd.read_csv(dataset_path)
# unique_item=df['product_id'].nunique()
# print(unique_item)

# unique_item_name=df[['product_id','category']].drop_duplicates()
# print(unique_item_name)

# unique_item_name=df['category'].unique()
# print(unique_item_name)
# no_of_items_per_category=df['category'].value_counts()
# print(no_of_items_per_category)

# Get the time range of the dataset
# Ensure 'ds' column is datetime
if 'ds' in df.columns:
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    start_date = df['ds'].min()
    end_date = df['ds'].max()
    print(f"Time range: {start_date} to {end_date}")
else:
    print("No 'ds' (date) column found in the dataset.")