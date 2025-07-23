"""
Fetch forecasts for a given category and date range from MongoDB.
Use this in your API or backend service to serve frontend/dashboard requests.
"""

from pymongo import MongoClient
from config.base import MONGODB_URL, DATABASE_NAME


def get_forecast_for_category(category: str, start_date: str, end_date: str):
    """
    Fetch forecasted demand for a category and date range from MongoDB.
    Args:
        category: Category name (str)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    Returns:
        List of forecast dicts sorted by date.
    """
    client = MongoClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    collection = db['demand_forecasts']
    results = list(collection.find({
        'category': category,
        'date': {'$gte': start_date, '$lte': end_date}
    }, {'_id': 0}).sort('date', 1))
    return results

# Example usage:
# forecasts = get_forecast_for_category('electronics', '2025-07-20', '2025-08-20')
# print(forecasts)
