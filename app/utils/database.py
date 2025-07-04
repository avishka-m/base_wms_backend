from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from typing import Any, Dict, List, Union
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
