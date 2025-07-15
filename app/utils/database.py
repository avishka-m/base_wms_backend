from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from typing import Any, Dict, List, Union
from ..config import MONGODB_URL, DATABASE_NAME

# Import the optimized MongoDB client
try:
    from .chatbot.mongodb_client import chatbot_mongodb_client
    OPTIMIZED_CLIENT_AVAILABLE = True
except ImportError:
    OPTIMIZED_CLIENT_AVAILABLE = False
    print("⚠️  Optimized MongoDB client not available, using basic client")

# Synchronous MongoDB client (legacy support)
def get_database():
    client = MongoClient(MONGODB_URL)
    return client[DATABASE_NAME]

# OPTIMIZED: Async MongoDB client with connection pooling and caching
async def get_async_database():
    if OPTIMIZED_CLIENT_AVAILABLE:
        # Use optimized client with connection pooling
        return await chatbot_mongodb_client.get_async_database()
    else:
        # Fallback to basic client
        client = AsyncIOMotorClient(MONGODB_URL)
        return client[DATABASE_NAME]

# Database collections (legacy support)
def get_collection(collection_name: str):
    db = get_database()
    return db[collection_name]

# OPTIMIZED: Async collection with caching support
async def get_async_collection(collection_name: str):
    db = await get_async_database()
    return db[collection_name]

# OPTIMIZED: Collection getter with automatic async/sync detection
async def get_collection_optimized(collection_name: str):
    """
    Get collection with optimized connection pooling and caching.
    This is the recommended method for new code.
    """
    if OPTIMIZED_CLIENT_AVAILABLE:
        # Use optimized client with connection pooling and caching
        db = await chatbot_mongodb_client.get_async_database()
        return db[collection_name]
    else:
        # Fallback to basic async client
        return await get_async_collection(collection_name)

def serialize_doc(doc: Union[Dict[str, Any], List[Dict[str, Any]], Any]) -> Union[Dict[str, Any], List[Dict[str, Any]], Any]:
    """
    Convert MongoDB documents to JSON-serializable format by converting ObjectId to string
    """
    if isinstance(doc, list):
        return [serialize_doc(item) for item in doc]
    elif isinstance(doc, dict):
        serialized = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                serialized[key] = str(value)
            elif isinstance(value, dict):
                serialized[key] = serialize_doc(value)
            elif isinstance(value, list):
                serialized[key] = serialize_doc(value)
            else:
                serialized[key] = value
        return serialized
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc
