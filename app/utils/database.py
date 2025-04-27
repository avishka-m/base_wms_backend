from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from ..config import MONGODB_URL, DATABASE_NAME

# Synchronous MongoDB client
def get_database():
    client = MongoClient(MONGODB_URL)
    return client[DATABASE_NAME]

# Async MongoDB client
async def get_async_database():
    client = AsyncIOMotorClient(MONGODB_URL)
    return client[DATABASE_NAME]

# Database collections
def get_collection(collection_name: str):
    db = get_database()
    return db[collection_name]

async def get_async_collection(collection_name: str):
    db = await get_async_database()
    return db[collection_name]
