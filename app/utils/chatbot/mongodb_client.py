"""
MongoDB Database Client for Chatbot Tools

This module provides direct MongoDB database access for chatbot tools,
following the LangChain MongoDB document loader pattern.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from bson.errors import InvalidId

# Use absolute imports for config
from app.config import MONGODB_URL, DATABASE_NAME

logger = logging.getLogger("wms_chatbot.mongodb_client")

class ChatbotMongoDBClient:
    """
    MongoDB client specifically designed for chatbot tools.
    Provides direct database access following LangChain MongoDB pattern.
    """
    
    def __init__(self):
        """Initialize the MongoDB client."""
        self.connection_string = MONGODB_URL
        self.db_name = DATABASE_NAME
        self._client = None
        self._async_client = None
        logger.info(f"ChatbotMongoDBClient initialized with DB: {self.db_name}")
    
    def get_client(self) -> MongoClient:
        """Get or create synchronous MongoDB client."""
        if self._client is None:
            self._client = MongoClient(self.connection_string)
        return self._client
    
    def get_async_client(self) -> AsyncIOMotorClient:
        """Get or create asynchronous MongoDB client."""
        if self._async_client is None:
            self._async_client = AsyncIOMotorClient(self.connection_string)
        return self._async_client
    
    def get_database(self):
        """Get the database instance."""
        client = self.get_client()
        return client[self.db_name]
    
    async def get_async_database(self):
        """Get the async database instance."""
        client = self.get_async_client()
        return client[self.db_name]
    
    def close(self):
        """Close database connections."""
        if self._client:
            self._client.close()
        if self._async_client:
            self._async_client.close()
    
    def serialize_document(self, doc: Union[Dict[str, Any], List[Dict[str, Any]], Any]) -> Union[Dict[str, Any], List[Dict[str, Any]], Any]:
        """
        Convert MongoDB documents to JSON-serializable format.
        Handles ObjectId conversion and nested documents.
        """
        if isinstance(doc, list):
            return [self.serialize_document(item) for item in doc]
        elif isinstance(doc, dict):
            serialized = {}
            for key, value in doc.items():
                if isinstance(value, ObjectId):
                    serialized[key] = str(value)
                elif isinstance(value, datetime):
                    serialized[key] = value.isoformat()
                elif isinstance(value, dict):
                    serialized[key] = self.serialize_document(value)
                elif isinstance(value, list):
                    serialized[key] = self.serialize_document(value)
                else:
                    serialized[key] = value
            return serialized
        elif isinstance(doc, ObjectId):
            return str(doc)
        elif isinstance(doc, datetime):
            return doc.isoformat()
        else:
            return doc
    
    # Inventory Operations
    async def get_inventory_items(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get inventory items from MongoDB.
        
        Args:
            filter_criteria: MongoDB filter criteria
            
        Returns:
            List of inventory items
        """
        try:
            db = await self.get_async_database()
            collection = db["inventory"]
            
            if filter_criteria is None:
                filter_criteria = {}
            
            cursor = collection.find(filter_criteria)
            items = await cursor.to_list(length=100)  # Limit to 100 items
            
            return [self.serialize_document(item) for item in items]
        except Exception as e:
            logger.error(f"Error getting inventory items: {e}")
            return []
    
    async def get_inventory_item_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get a single inventory item by itemID."""
        try:
            db = await self.get_async_database()
            collection = db["inventory"]
            
            item = await collection.find_one({"itemID": item_id})
            
            if item:
                return self.serialize_document(item)
            return None
        except Exception as e:
            logger.error(f"Error getting inventory item {item_id}: {e}")
            return None
    
    async def search_inventory_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Search inventory items by name (case-insensitive partial match)."""
        try:
            db = await self.get_async_database()
            collection = db["inventory"]
            
            # Use regex for case-insensitive partial matching
            filter_criteria = {"name": {"$regex": name, "$options": "i"}}
            cursor = collection.find(filter_criteria)
            items = await cursor.to_list(length=50)
            
            return [self.serialize_document(item) for item in items]
        except Exception as e:
            logger.error(f"Error searching inventory by name '{name}': {e}")
            return []
    
    async def get_inventory_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get inventory items by category."""
        try:
            db = await self.get_async_database()
            collection = db["inventory"]
            
            filter_criteria = {"category": {"$regex": category, "$options": "i"}}
            cursor = collection.find(filter_criteria)
            items = await cursor.to_list(length=50)
            
            return [self.serialize_document(item) for item in items]
        except Exception as e:
            logger.error(f"Error getting inventory by category '{category}': {e}")
            return []
    
    async def get_low_stock_items(self, threshold: int = 10) -> List[Dict[str, Any]]:
        """Get inventory items with stock below threshold."""
        try:
            db = await self.get_async_database()
            collection = db["inventory"]
            
            filter_criteria = {"stock_level": {"$lt": threshold}}
            cursor = collection.find(filter_criteria)
            items = await cursor.to_list(length=100)
            
            return [self.serialize_document(item) for item in items]
        except Exception as e:
            logger.error(f"Error getting low stock items: {e}")
            return []
    
    async def update_inventory_item(self, item_id: int, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing inventory item.
        
        Args:
            item_id: ID of the inventory item to update
            update_data: Dictionary of fields to update
            
        Returns:
            Result of the update operation
        """
        try:
            db = await self.get_async_database()
            collection = db["inventory"]
            
            # Remove None values from update data
            clean_update_data = {k: v for k, v in update_data.items() if v is not None}
            
            if not clean_update_data:
                return {"success": False, "message": "No valid update data provided"}
            
            # Add updated_at timestamp
            clean_update_data["updated_at"] = datetime.utcnow()
            
            # Perform the update
            result = await collection.update_one(
                {"itemID": item_id},
                {"$set": clean_update_data}
            )
            
            if result.matched_count == 0:
                return {"success": False, "message": f"Item with ID {item_id} not found"}
            
            if result.modified_count > 0:
                # Get the updated item
                updated_item = await self.get_inventory_item_by_id(item_id)
                return {
                    "success": True, 
                    "message": f"Item {item_id} updated successfully",
                    "item": updated_item
                }
            else:
                return {"success": True, "message": "No changes made (data was identical)"}
                
        except Exception as e:
            logger.error(f"Error updating inventory item {item_id}: {e}")
            return {"success": False, "message": f"Database error: {str(e)}"}
    
    async def add_inventory_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new inventory item.
        
        Args:
            item_data: Dictionary containing the new item data
            
        Returns:
            Result of the add operation
        """
        try:
            db = await self.get_async_database()
            collection = db["inventory"]
            
            # Get the next itemID
            max_item = await collection.find_one(sort=[("itemID", -1)])
            next_item_id = (max_item.get("itemID", 0) + 1) if max_item else 1
            
            # Prepare the item document
            new_item = {
                "itemID": next_item_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                **item_data
            }
            
            # Insert the new item
            result = await collection.insert_one(new_item)
            
            if result.inserted_id:
                # Get the inserted item
                inserted_item = await self.get_inventory_item_by_id(next_item_id)
                return {
                    "success": True,
                    "message": f"Item added successfully with ID {next_item_id}",
                    "item": inserted_item,
                    "item_id": next_item_id
                }
            else:
                return {"success": False, "message": "Failed to insert item"}
                
        except Exception as e:
            logger.error(f"Error adding inventory item: {e}")
            return {"success": False, "message": f"Database error: {str(e)}"}
    
    async def get_inventory_analytics(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Get inventory analytics and summary statistics.
        
        Args:
            category: Optional category to filter analytics
            
        Returns:
            Analytics data including totals, low stock, category breakdown
        """
        try:
            db = await self.get_async_database()
            collection = db["inventory"]
            
            # Build filter for category if specified
            match_filter = {}
            if category:
                match_filter = {"category": {"$regex": category, "$options": "i"}}
            
            # Aggregation pipeline for analytics
            pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": None,
                        "total_items": {"$sum": 1},
                        "total_stock": {"$sum": "$stock_level"},
                        "avg_stock": {"$avg": "$stock_level"},
                        "max_stock": {"$max": "$stock_level"},
                        "min_stock": {"$min": "$stock_level"},
                        "categories": {"$addToSet": "$category"},
                        "low_stock_count": {
                            "$sum": {
                                "$cond": [{"$lt": ["$stock_level", 10]}, 1, 0]
                            }
                        },
                        "zero_stock_count": {
                            "$sum": {
                                "$cond": [{"$eq": ["$stock_level", 0]}, 1, 0]
                            }
                        }
                    }
                }
            ]
            
            # Execute aggregation
            cursor = collection.aggregate(pipeline)
            analytics = await cursor.to_list(length=1)
            
            if not analytics:
                return {"total_items": 0, "message": "No inventory data found"}
            
            result = analytics[0]
            result.pop("_id", None)  # Remove the _id field
            
            # Get category breakdown
            category_pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": "$category",
                        "count": {"$sum": 1},
                        "total_stock": {"$sum": "$stock_level"},
                        "avg_stock": {"$avg": "$stock_level"}
                    }
                },
                {"$sort": {"count": -1}}
            ]
            
            category_cursor = collection.aggregate(category_pipeline)
            category_breakdown = await category_cursor.to_list(length=50)
            
            result["category_breakdown"] = [
                {
                    "category": cat["_id"],
                    "count": cat["count"],
                    "total_stock": cat["total_stock"],
                    "avg_stock": round(cat["avg_stock"], 2)
                }
                for cat in category_breakdown
            ]
            
            return self.serialize_document(result)
            
        except Exception as e:
            logger.error(f"Error getting inventory analytics: {e}")
            return {"success": False, "message": f"Database error: {str(e)}"}
    
    # Order Operations
    async def get_orders(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get orders from MongoDB.
        
        Args:
            filter_criteria: MongoDB filter criteria
            
        Returns:
            List of orders
        """
        try:
            db = await self.get_async_database()
            collection = db["orders"]
            
            if filter_criteria is None:
                filter_criteria = {}
            
            cursor = collection.find(filter_criteria).sort("created_at", -1)  # Most recent first
            orders = await cursor.to_list(length=100)  # Limit to 100 orders
            
            return [self.serialize_document(order) for order in orders]
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    async def get_order_by_id(self, order_id: int) -> Optional[Dict[str, Any]]:
        """Get a single order by orderID."""
        try:
            db = await self.get_async_database()
            collection = db["orders"]
            
            order = await collection.find_one({"orderID": order_id})
            
            if order:
                return self.serialize_document(order)
            return None
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return None
    
    async def get_orders_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get orders by status."""
        try:
            db = await self.get_async_database()
            collection = db["orders"]
            
            filter_criteria = {"order_status": status}
            cursor = collection.find(filter_criteria).sort("created_at", -1)
            orders = await cursor.to_list(length=50)
            
            return [self.serialize_document(order) for order in orders]
        except Exception as e:
            logger.error(f"Error getting orders by status '{status}': {e}")
            return []
    
    async def get_orders_by_customer(self, customer_id: int) -> List[Dict[str, Any]]:
        """Get orders by customer ID."""
        try:
            db = await self.get_async_database()
            collection = db["orders"]
            
            filter_criteria = {"customerID": customer_id}
            cursor = collection.find(filter_criteria).sort("created_at", -1)
            orders = await cursor.to_list(length=50)
            
            return [self.serialize_document(order) for order in orders]
        except Exception as e:
            logger.error(f"Error getting orders for customer {customer_id}: {e}")
            return []
    
    # Location Operations
    async def get_locations(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get storage locations."""
        try:
            db = await self.get_async_database()
            collection = db["locations"]
            
            if filter_criteria is None:
                filter_criteria = {}
            
            cursor = collection.find(filter_criteria)
            locations = await cursor.to_list(length=100)
            
            return [self.serialize_document(location) for location in locations]
        except Exception as e:
            logger.error(f"Error getting locations: {e}")
            return []
    
    async def get_location_by_id(self, location_id: int) -> Optional[Dict[str, Any]]:
        """Get a single location by locationID."""
        try:
            db = await self.get_async_database()
            collection = db["locations"]
            
            location = await collection.find_one({"locationID": location_id})
            
            if location:
                return self.serialize_document(location)
            return None
        except Exception as e:
            logger.error(f"Error getting location {location_id}: {e}")
            return None
    
    # Customer Operations
    async def get_customers(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get customers."""
        try:
            db = await self.get_async_database()
            collection = db["customers"]
            
            if filter_criteria is None:
                filter_criteria = {}
            
            cursor = collection.find(filter_criteria)
            customers = await cursor.to_list(length=100)
            
            return [self.serialize_document(customer) for customer in customers]
        except Exception as e:
            logger.error(f"Error getting customers: {e}")
            return []
    
    async def get_customer_by_id(self, customer_id: int) -> Optional[Dict[str, Any]]:
        """Get a single customer by customerID."""
        try:
            db = await self.get_async_database()
            collection = db["customers"]
            
            customer = await collection.find_one({"customerID": customer_id})
            
            if customer:
                return self.serialize_document(customer)
            return None
        except Exception as e:
            logger.error(f"Error getting customer {customer_id}: {e}")
            return None

# Global instance
chatbot_mongodb_client = ChatbotMongoDBClient() 