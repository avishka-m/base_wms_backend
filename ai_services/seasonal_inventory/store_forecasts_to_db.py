

from dotenv import find_dotenv, load_dotenv
import os

print("Current working directory:", os.getcwd())
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
print("Dotenv path forced to:", env_path)
load_dotenv(env_path)

"""
Batch script to generate and store demand forecasts in MongoDB after model retraining.
Run this script after retraining models, not on every user request.
"""

# import os
import pandas as pd
from pymongo import MongoClient, UpdateOne
from datetime import datetime
from ai_services.seasonal_inventory.forecast_server import ForecastServer
# MongoDB setup
def get_db():
    from base_wms_backend.config import base as config_base
    # print("MONGODB_URL being used:", os.getenv("MONGODB_URL"))
    client = MongoClient(config_base.MONGODB_URL)
    return client[config_base.DATABASE_NAME]

def save_forecasts_to_db(forecast_df: pd.DataFrame, category: str, db):
    """
    Save forecast DataFrame to MongoDB (demand_forecasts collection).
    Upserts by category and date (ds).
    """
    # print(f"\nSaving forecasts for category: {category}")
    # print("First 5 rows of forecast_df:")
    # print(forecast_df.head())
    # print("Database name:", db.name)
    # print("Collection name: demand_forecasts")
    
    
    
    collection = db['demand_forecasts']
    operations = []
    for _, row in forecast_df.iterrows():
        operations.append(UpdateOne(
            {'category': category, 'date': row['ds'].strftime('%Y-%m-%d')},
            {'$set': {
                'category': category,
                'date': row['ds'].strftime('%Y-%m-%d'),
                'yhat': float(row['yhat']),
                'yhat_lower': float(row.get('yhat_lower', 0)),
                'yhat_upper': float(row.get('yhat_upper', 0)),
                'updated_at': datetime.utcnow()
            }},
            upsert=True
        ))
    if operations:
        collection.bulk_write(operations)
        # print(f"Saved {len(operations)} forecasts for {category}")
    else:
        pass

def main():
    db = get_db()
    server = ForecastServer()
    categories = server.get_available_categories()
    if not categories:
        print("No trained models found.")
        return
    # Set forecast range to 365 days (1 year) from today
    start_date = datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.now() + pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    for category in categories:
        try:
            forecast_df = server.predict(category, start_date, end_date)
            # Debug print: show forecast date range and number of rows
            print(f"Forecast date range for {category}: {forecast_df['ds'].min()} to {forecast_df['ds'].max()}")
            print(f"Number of rows: {len(forecast_df)}")
            save_forecasts_to_db(forecast_df, category, db)
        except Exception as e:
            print(f"Error for {category}: {e}")

###########################################
# test funtions for this script
# def test_mongo_connection():
#     try:
#         db = get_db()
#         print("MongoDB connection successful.")
#         print("Collections:", db.list_collection_names())
#     except Exception as e:
#         print("MongoDB connection failed:", e)

# def test_forecast_server():
#     try:
#         server = ForecastServer()
#         categories = server.get_available_categories()
#         print("Available categories:", categories)
#         if categories:
#             print("ForecastServer is working.")
#         else:
#             print("No categories found. Check your models.")
#     except Exception as e:
#         print("ForecastServer test failed:", e)

# def test_save_forecasts_to_db():
#     try:
#         db = get_db()
#         # Create a dummy DataFrame
#         data = {
#             'ds': [datetime.now()],
#             'yhat': [100.0],
#             'yhat_lower': [90.0],
#             'yhat_upper': [110.0]
#         }
#         df = pd.DataFrame(data)
#         save_forecasts_to_db(df, "TestCategory", db)
#         print("save_forecasts_to_db ran successfully.")
#     except Exception as e:
#         print("save_forecasts_to_db test failed:", e)








if __name__ == "__main__":
    main()
    # test_mongo_connection()
    # test_forecast_server()
    # test_save_forecasts_to_db()
