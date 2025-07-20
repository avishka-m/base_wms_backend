"""
Database Indexing Strategy for MongoDB Atlas Performance Optimization

This module defines and creates database indexes to significantly improve
query performance in MongoDB Atlas. Proper indexing can reduce query
times from seconds to milliseconds.
"""

import logging
from typing import Dict, List, Any
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

logger = logging.getLogger(__name__)

class DatabaseIndexManager:
    """
    Manages database indexes for optimal MongoDB Atlas performance.
    Creates and maintains indexes for all collections based on common query patterns.
    """
    
    def __init__(self, client: MongoClient, database_name: str):
        self.client = client
        self.db = client[database_name]
        self.database_name = database_name
    
    def get_index_definitions(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Define indexes for all collections based on common query patterns.
        
        Returns:
            Dictionary mapping collection names to their index definitions
        """
        return {
            # Inventory Collection - Most frequently queried
            "inventory": [
                # Single field indexes for exact matches
                {"keys": [("itemID", ASCENDING)], "options": {"unique": True}},
                {"keys": [("name", ASCENDING)], "options": {}},
                {"keys": [("category", ASCENDING)], "options": {}},
                {"keys": [("supplierID", ASCENDING)], "options": {}},
                {"keys": [("locationID", ASCENDING)], "options": {}},
                {"keys": [("stock_level", ASCENDING)], "options": {}},
                
                # Compound indexes for complex queries
                {"keys": [("category", ASCENDING), ("stock_level", ASCENDING)], "options": {}},
                {"keys": [("locationID", ASCENDING), ("category", ASCENDING)], "options": {}},
                {"keys": [("stock_level", ASCENDING), ("min_stock_level", ASCENDING)], "options": {}},
                
                # Text search index for name and category
                {"keys": [("name", TEXT), ("category", TEXT)], "options": {}},
                
                # Timestamp indexes for reporting
                {"keys": [("created_at", DESCENDING)], "options": {}},
                {"keys": [("updated_at", DESCENDING)], "options": {}},
            ],
            
            # Orders Collection - Second most queried
            "orders": [
                # Primary key and foreign keys
                {"keys": [("orderID", ASCENDING)], "options": {"unique": True}},
                {"keys": [("customerID", ASCENDING)], "options": {}},
                {"keys": [("assigned_worker", ASCENDING)], "options": {}},
                
                # Status and priority queries
                {"keys": [("order_status", ASCENDING)], "options": {}},
                {"keys": [("priority", ASCENDING)], "options": {}},
                
                # Compound indexes for dashboard queries
                {"keys": [("order_status", ASCENDING), ("priority", ASCENDING)], "options": {}},
                {"keys": [("customerID", ASCENDING), ("order_status", ASCENDING)], "options": {}},
                {"keys": [("assigned_worker", ASCENDING), ("order_status", ASCENDING)], "options": {}},
                
                # Date range queries
                {"keys": [("order_date", DESCENDING)], "options": {}},
                {"keys": [("order_date", DESCENDING), ("order_status", ASCENDING)], "options": {}},
                
                # Address search
                {"keys": [("shipping_address", TEXT)], "options": {}},
            ],
            
            # Workers Collection
            "workers": [
                {"keys": [("workerID", ASCENDING)], "options": {"unique": True}},
                {"keys": [("username", ASCENDING)], "options": {"unique": True}},
                {"keys": [("email", ASCENDING)], "options": {"unique": True}},
                {"keys": [("role", ASCENDING)], "options": {}},
                {"keys": [("department", ASCENDING)], "options": {}},
                {"keys": [("is_active", ASCENDING)], "options": {}},
                
                # Compound indexes for role-based queries
                {"keys": [("role", ASCENDING), ("is_active", ASCENDING)], "options": {}},
                {"keys": [("department", ASCENDING), ("role", ASCENDING)], "options": {}},
                
                # Name search
                {"keys": [("name", TEXT), ("username", TEXT)], "options": {}},
            ],
            
            # Customers Collection
            "customers": [
                {"keys": [("customerID", ASCENDING)], "options": {"unique": True}},
                {"keys": [("email", ASCENDING)], "options": {"unique": True}},
                {"keys": [("phone", ASCENDING)], "options": {}},
                
                # Address and location search
                {"keys": [("address", TEXT), ("name", TEXT)], "options": {}},
                
                # Customer status
                {"keys": [("created_at", DESCENDING)], "options": {}},
            ],
            
            # Locations Collection
            "locations": [
                {"keys": [("locationID", ASCENDING)], "options": {"unique": True}},
                {"keys": [("warehouseID", ASCENDING)], "options": {}},
                {"keys": [("zone", ASCENDING)], "options": {}},
                {"keys": [("is_occupied", ASCENDING)], "options": {}},
                
                # Compound indexes for location management
                {"keys": [("warehouseID", ASCENDING), ("zone", ASCENDING)], "options": {}},
                {"keys": [("warehouseID", ASCENDING), ("is_occupied", ASCENDING)], "options": {}},
                {"keys": [("zone", ASCENDING), ("is_occupied", ASCENDING)], "options": {}},
                
                # Coordinates for spatial queries
                {"keys": [("coordinates.x", ASCENDING), ("coordinates.y", ASCENDING)], "options": {}},
            ],
            
            # Receiving Collection
            "receiving": [
                {"keys": [("receivingID", ASCENDING)], "options": {"unique": True}},
                {"keys": [("supplierID", ASCENDING)], "options": {}},
                {"keys": [("workerID", ASCENDING)], "options": {}},
                {"keys": [("status", ASCENDING)], "options": {}},
                {"keys": [("reference_number", ASCENDING)], "options": {}},
                
                # Date-based queries
                {"keys": [("received_date", DESCENDING)], "options": {}},
                {"keys": [("status", ASCENDING), ("received_date", DESCENDING)], "options": {}},
            ],
            
            # Picking Collection
            "picking": [
                {"keys": [("pickingID", ASCENDING)], "options": {"unique": True}},
                {"keys": [("orderID", ASCENDING)], "options": {}},
                {"keys": [("workerID", ASCENDING)], "options": {}},
                {"keys": [("status", ASCENDING)], "options": {}},
                {"keys": [("priority", ASCENDING)], "options": {}},
                
                # Compound indexes for picking optimization
                {"keys": [("status", ASCENDING), ("priority", ASCENDING)], "options": {}},
                {"keys": [("workerID", ASCENDING), ("status", ASCENDING)], "options": {}},
                
                # Date-based queries
                {"keys": [("pick_date", DESCENDING)], "options": {}},
            ],
            
            # Packing Collection
            "packing": [
                {"keys": [("packingID", ASCENDING)], "options": {"unique": True}},
                {"keys": [("pickingID", ASCENDING)], "options": {}},
                {"keys": [("workerID", ASCENDING)], "options": {}},
                {"keys": [("status", ASCENDING)], "options": {}},
                
                # Compound indexes
                {"keys": [("status", ASCENDING), ("pack_date", DESCENDING)], "options": {}},
                {"keys": [("workerID", ASCENDING), ("status", ASCENDING)], "options": {}},
            ],
            
            # Shipping Collection
            "shipping": [
                {"keys": [("shippingID", ASCENDING)], "options": {"unique": True}},
                {"keys": [("vehicleID", ASCENDING)], "options": {}},
                {"keys": [("status", ASCENDING)], "options": {}},
                
                # Date and delivery tracking
                {"keys": [("ship_date", DESCENDING)], "options": {}},
                {"keys": [("estimated_delivery", ASCENDING)], "options": {}},
                {"keys": [("status", ASCENDING), ("estimated_delivery", ASCENDING)], "options": {}},
                
                # Address search
                {"keys": [("delivery_address", TEXT)], "options": {}},
            ],
            
            # Returns Collection
            "returns": [
                {"keys": [("returnID", ASCENDING)], "options": {"unique": True}},
                {"keys": [("orderID", ASCENDING)], "options": {}},
                {"keys": [("customerID", ASCENDING)], "options": {}},
                {"keys": [("status", ASCENDING)], "options": {}},
                
                # Date-based queries
                {"keys": [("return_date", DESCENDING)], "options": {}},
                {"keys": [("status", ASCENDING), ("return_date", DESCENDING)], "options": {}},
            ],
            
            # Vehicles Collection
            "vehicles": [
                {"keys": [("vehicleID", ASCENDING)], "options": {"unique": True}},
                {"keys": [("vehicle_type", ASCENDING)], "options": {}},
                {"keys": [("status", ASCENDING)], "options": {}},
                {"keys": [("capacity", ASCENDING)], "options": {}},
                
                # Compound indexes for vehicle assignment
                {"keys": [("status", ASCENDING), ("vehicle_type", ASCENDING)], "options": {}},
                {"keys": [("status", ASCENDING), ("capacity", DESCENDING)], "options": {}},
            ],
            
            # Suppliers Collection
            "suppliers": [
                {"keys": [("supplierID", ASCENDING)], "options": {"unique": True}},
                {"keys": [("name", ASCENDING)], "options": {}},
                {"keys": [("email", ASCENDING)], "options": {}},
                
                # Text search
                {"keys": [("name", TEXT), ("contact_person", TEXT)], "options": {}},
            ],
            
            # Warehouses Collection
            "warehouses": [
                {"keys": [("warehouseID", ASCENDING)], "options": {"unique": True}},
                {"keys": [("name", ASCENDING)], "options": {}},
                {"keys": [("location", ASCENDING)], "options": {}},
                
                # Text search
                {"keys": [("name", TEXT), ("location", TEXT)], "options": {}},
            ],
            
            # Chat Collections for Enhanced Chatbot
            "chat_conversations": [
                {"keys": [("conversation_id", ASCENDING)], "options": {"unique": True}},
                {"keys": [("user_id", ASCENDING)], "options": {}},
                {"keys": [("agent_role", ASCENDING)], "options": {}},
                {"keys": [("status", ASCENDING)], "options": {}},
                
                # Date-based queries
                {"keys": [("created_at", DESCENDING)], "options": {}},
                {"keys": [("last_message_at", DESCENDING)], "options": {}},
                
                # Compound indexes
                {"keys": [("user_id", ASCENDING), ("status", ASCENDING)], "options": {}},
                {"keys": [("user_id", ASCENDING), ("last_message_at", DESCENDING)], "options": {}},
                
                # Text search
                {"keys": [("title", TEXT)], "options": {}},
            ],
            
            "chat_messages": [
                {"keys": [("conversation_id", ASCENDING)], "options": {}},
                {"keys": [("user_id", ASCENDING)], "options": {}},
                {"keys": [("message_type", ASCENDING)], "options": {}},
                {"keys": [("timestamp", DESCENDING)], "options": {}},
                
                # Compound indexes for message retrieval
                {"keys": [("conversation_id", ASCENDING), ("timestamp", ASCENDING)], "options": {}},
                {"keys": [("user_id", ASCENDING), ("timestamp", DESCENDING)], "options": {}},
                
                # Text search on message content
                {"keys": [("content", TEXT)], "options": {}},
            ],
            
            # Analytics and Audit Collections
            "chat_analytics": [
                {"keys": [("user_id", ASCENDING)], "options": {}},
                {"keys": [("period_type", ASCENDING)], "options": {}},
                {"keys": [("period_start", DESCENDING)], "options": {}},
                
                # Compound indexes
                {"keys": [("user_id", ASCENDING), ("period_type", ASCENDING)], "options": {}},
                {"keys": [("period_type", ASCENDING), ("period_start", DESCENDING)], "options": {}},
            ],
            
            "chat_audit_logs": [
                {"keys": [("user_id", ASCENDING)], "options": {}},
                {"keys": [("action_type", ASCENDING)], "options": {}},
                {"keys": [("resource_type", ASCENDING)], "options": {}},
                {"keys": [("timestamp", DESCENDING)], "options": {}},
                
                # Compound indexes
                {"keys": [("user_id", ASCENDING), ("timestamp", DESCENDING)], "options": {}},
                {"keys": [("action_type", ASCENDING), ("timestamp", DESCENDING)], "options": {}},
            ],
        }
    
    def create_indexes(self, collections: List[str] = None) -> Dict[str, List[str]]:
        """
        Create indexes for specified collections or all collections.
        
        Args:
            collections: List of collection names to index. If None, indexes all collections.
            
        Returns:
            Dictionary mapping collection names to lists of created index names
        """
        index_definitions = self.get_index_definitions()
        results = {}
        
        # If specific collections not specified, index all
        if collections is None:
            collections = list(index_definitions.keys())
        
        for collection_name in collections:
            if collection_name not in index_definitions:
                logger.warning(f"No index definitions found for collection: {collection_name}")
                continue
            
            collection = self.db[collection_name]
            created_indexes = []
            
            logger.info(f"Creating indexes for collection: {collection_name}")
            
            for index_def in index_definitions[collection_name]:
                try:
                    index_name = collection.create_index(
                        index_def["keys"],
                        **index_def["options"]
                    )
                    created_indexes.append(index_name)
                    logger.info(f"  Created index: {index_name}")
                    
                except OperationFailure as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"  Index already exists: {index_def['keys']}")
                    else:
                        logger.error(f"  Failed to create index {index_def['keys']}: {e}")
                
                except Exception as e:
                    logger.error(f"  Unexpected error creating index {index_def['keys']}: {e}")
            
            results[collection_name] = created_indexes
        
        return results
    
    def drop_indexes(self, collection_name: str, index_names: List[str] = None) -> bool:
        """
        Drop indexes from a collection.
        
        Args:
            collection_name: Name of the collection
            index_names: List of index names to drop. If None, drops all except _id_
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.db[collection_name]
            
            if index_names is None:
                # Drop all indexes except _id_
                collection.drop_indexes()
                logger.info(f"Dropped all indexes for collection: {collection_name}")
            else:
                for index_name in index_names:
                    collection.drop_index(index_name)
                    logger.info(f"Dropped index {index_name} from collection: {collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop indexes from {collection_name}: {e}")
            return False
    
    def list_indexes(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        List all indexes for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            List of index information dictionaries
        """
        try:
            collection = self.db[collection_name]
            return list(collection.list_indexes())
        except Exception as e:
            logger.error(f"Failed to list indexes for {collection_name}: {e}")
            return []
    
    def analyze_index_usage(self, collection_name: str) -> Dict[str, Any]:
        """
        Analyze index usage statistics for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Index usage statistics
        """
        try:
            collection = self.db[collection_name]
            stats = collection.aggregate([
                {"$indexStats": {}}
            ])
            return list(stats)
        except Exception as e:
            logger.error(f"Failed to analyze index usage for {collection_name}: {e}")
            return []

def create_all_indexes(mongodb_url: str, database_name: str) -> Dict[str, List[str]]:
    """
    Convenience function to create all indexes for the warehouse management system.
    
    Args:
        mongodb_url: MongoDB connection string
        database_name: Name of the database
        
    Returns:
        Dictionary mapping collection names to lists of created index names
    """
    client = MongoClient(mongodb_url)
    index_manager = DatabaseIndexManager(client, database_name)
    
    try:
        results = index_manager.create_indexes()
        logger.info(f"Successfully created indexes for {len(results)} collections")
        return results
    finally:
        client.close()

# Atlas Performance Monitoring Queries
PERFORMANCE_MONITORING_QUERIES = {
    "slow_queries": {
        "description": "Monitor slow queries (>100ms)",
        "query": {"millis": {"$gt": 100}},
        "collection": "system.profile"
    },
    "index_usage": {
        "description": "Check index usage statistics",
        "pipeline": [{"$indexStats": {}}]
    },
    "collection_stats": {
        "description": "Get collection statistics",
        "command": "collStats"
    }
} 