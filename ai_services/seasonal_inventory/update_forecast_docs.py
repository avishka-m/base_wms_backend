# Script to update all documents in demand_forecasts to only keep the required fields
# and to add a placeholder category if missing (update as needed for your real categories)

import os
from pymongo import MongoClient
from dotenv import load_dotenv


# Always load .env from the project root
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env'))
load_dotenv(dotenv_path=dotenv_path)

# Get MongoDB connection info from environment
mongo_url = os.getenv("MONGODB_URL")
db_name = os.getenv("DATABASE_NAME", "warehouse_management")

client = MongoClient(mongo_url)
db = client[db_name]
collection = db["demand_forecasts"]

# Set a default category if missing (customize this logic as needed)
default_category = "UnknownCategory"

for doc in collection.find():
    update_fields = {
        "category": doc.get("category", default_category),
        "date": doc.get("date"),
        "predicted_demand": doc.get("predicted_demand", doc.get("yhat")),
        "lower_bound": doc.get("lower_bound", doc.get("yhat_lower", doc.get("predicted_demand", doc.get("yhat")))),
        "upper_bound": doc.get("upper_bound", doc.get("yhat_upper", doc.get("predicted_demand", doc.get("yhat")))),
    }
    # Remove any fields not in update_fields (except _id)
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": update_fields, "$unset": {k: "" for k in doc if k not in update_fields and k != "_id"}}
    )

print("All documents updated to required format.")
