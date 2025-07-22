from pymongo import MongoClient

# Replace with your MongoDB connection string and database name
MONGO_URI = "mongodb+srv://wms:3cVnhHuj5caki0Ve@cluster0.99chyus.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "warehouse_management"
COLLECTION_NAME = "category_stock"  # New collection name

client = MongoClient(MONGO_URI)
db = client[DB_NAME]


# Example documents for all categories
category_docs = [
    {
        "category": "electronics",
        "stock": 1000,
        "stock_level": 1000,
        "total_stock": 1000,
        "last_updated": "2025-07-21"
    },
    {
        "category": "books_media",
        "stock": 800,
        "stock_level": 800,
        "total_stock": 800,
        "last_updated": "2025-07-21"
    },
    {
        "category": "clothing",
        "stock": 1200,
        "stock_level": 1200,
        "total_stock": 1200,
        "last_updated": "2025-07-21"
    },
    {
        "category": "home_garden",
        "stock": 600,
        "stock_level": 600,
        "total_stock": 600,
        "last_updated": "2025-07-21"
    },
    {
        "category": "sports_outdoors",
        "stock": 400,
        "stock_level": 400,
        "total_stock": 400,
        "last_updated": "2025-07-21"
    },
    {
        "category": "health_beauty",
        "stock": 700,
        "stock_level": 700,
        "total_stock": 700,
        "last_updated": "2025-07-21"
    }
]

# Insert all documents at once
collection = db[COLLECTION_NAME]
collection.insert_many(category_docs)

print(f"Inserted {len(category_docs)} documents into '{COLLECTION_NAME}'.")
