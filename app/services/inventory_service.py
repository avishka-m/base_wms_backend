from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.database import get_collection
from ..models.inventory import InventoryCreate, InventoryUpdate

class InventoryService:
    """
    Service for inventory management operations.
    
    This service handles business logic for inventory operations such as
    stock level updates, inventory transfers, and stock verification.
    """
    
    @staticmethod
    async def get_inventory_items(skip: int = 0, limit: int = 100, 
                                 category: Optional[str] = None, 
                                 low_stock: bool = False) -> List[Dict[str, Any]]:
        """
        Get inventory items with optional filtering.
        """
        inventory_collection = get_collection("inventory")
        
        # Build query
        query = {}
        if category:
            query["category"] = category
        if low_stock:
            # Fix: Use aggregation to compare stock_level with min_stock_level
            inventory_items = list(inventory_collection.find({
                "$expr": {"$lte": ["$stock_level", "$min_stock_level"]}
            }).skip(skip).limit(limit))
            return inventory_items
        
        # Execute query
        inventory_items = list(inventory_collection.find(query).skip(skip).limit(limit))
        return inventory_items
    
    @staticmethod
    async def get_inventory_item(item_id: int) -> Dict[str, Any]:
        """
        Get a specific inventory item by ID.
        """
        inventory_collection = get_collection("inventory")
        item = inventory_collection.find_one({"itemID": item_id})
        return item
    
    @staticmethod
    async def create_inventory_item(item: InventoryCreate) -> Dict[str, Any]:
        """
        Create a new inventory item.
        """
        inventory_collection = get_collection("inventory")
        
        # Find the next available itemID
        last_item = inventory_collection.find_one(
            sort=[("itemID", -1)]
        )
        next_id = 1
        if last_item:
            next_id = last_item.get("itemID", 0) + 1
        
        # Prepare item document
        item_data = item.model_dump()
        item_data.update({
            "itemID": next_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        
        # Insert item to database
        result = inventory_collection.insert_one(item_data)
        
        # Return the created item
        created_item = inventory_collection.find_one({"_id": result.inserted_id})
        return created_item
    
    @staticmethod
    async def update_inventory_item(item_id: int, item_update: InventoryUpdate) -> Dict[str, Any]:
        """
        Update an inventory item.
        """
        inventory_collection = get_collection("inventory")
        
        # Prepare update data
        update_data = item_update.model_dump(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        # Update item
        inventory_collection.update_one(
            {"itemID": item_id},
            {"$set": update_data}
        )
        
        # Return updated item
        updated_item = inventory_collection.find_one({"itemID": item_id})
        return updated_item
    
    @staticmethod
    async def delete_inventory_item(item_id: int) -> Dict[str, Any]:
        """
        Delete an inventory item.
        """
        inventory_collection = get_collection("inventory")
        
        # Delete item
        inventory_collection.delete_one({"itemID": item_id})
        
        return {"message": f"Item with ID {item_id} has been deleted"}
    
    @staticmethod
    async def update_stock_level(item_id: int, quantity_change: int, reason: str) -> Dict[str, Any]:
        """
        Update the stock level of an inventory item and log the transaction.
        
        Args:
            item_id: ID of the inventory item
            quantity_change: Amount to change (positive for increase, negative for decrease)
            reason: Reason for the stock change
            
        Returns:
            Updated inventory item
        """
        inventory_collection = get_collection("inventory")
        stock_log_collection = get_collection("stock_log")
        
        # Get current item
        item = inventory_collection.find_one({"itemID": item_id})
        if not item:
            return {"error": f"Item with ID {item_id} not found"}
        
        # Calculate new stock level
        current_stock = item.get("stock_level", 0)
        new_stock = current_stock + quantity_change
        
        # Prevent negative stock
        if (new_stock < 0):
            return {"error": f"Cannot reduce stock below zero. Current stock: {current_stock}"}
        
        # Update stock level
        inventory_collection.update_one(
            {"itemID": item_id},
            {"$set": {"stock_level": new_stock, "updated_at": datetime.utcnow()}}
        )
        
        # Log the stock change
        log_entry = {
            "itemID": item_id,
            "previous_level": current_stock,
            "new_level": new_stock,
            "change": quantity_change,
            "reason": reason,
            "timestamp": datetime.utcnow()
        }
        stock_log_collection.insert_one(log_entry)
        
        # Return updated item
        updated_item = inventory_collection.find_one({"itemID": item_id})
        return updated_item
    
    @staticmethod
    async def get_low_stock_items() -> List[Dict[str, Any]]:
        """
        Get items with stock levels at or below their minimum stock level.
        """
        inventory_collection = get_collection("inventory")
        
        # Find items where stock_level <= min_stock_level
        query = {
            "$expr": {"$lte": ["$stock_level", "$min_stock_level"]}
        }
        
        low_stock_items = list(inventory_collection.find(query))
        return low_stock_items
    
    @staticmethod
    async def transfer_inventory(item_id: int, source_location_id: int, 
                             destination_location_id: int, quantity: int) -> Dict[str, Any]:
        """
        Transfer inventory from one location to another.
        """
        inventory_collection = get_collection("inventory")
        location_collection = get_collection("locations")
        
        # Check item exists
        item = inventory_collection.find_one({"itemID": item_id})
        if not item:
            return {"error": f"Item with ID {item_id} not found"}
        
        # Check source location exists and contains the item
        source_location = location_collection.find_one({"locationID": source_location_id})
        if not source_location:
            return {"error": f"Source location with ID {source_location_id} not found"}
        
        # Check destination location exists
        destination_location = location_collection.find_one({"locationID": destination_location_id})
        if not destination_location:
            return {"error": f"Destination location with ID {destination_location_id} not found"}
        
        # Check destination location is not occupied
        if destination_location.get("is_occupied"):
            return {"error": f"Destination location with ID {destination_location_id} is already occupied"}
        
        # Create a new inventory record for the transferred items
        # (This is a simplified approach - in a real system you might have location-item mappings)
        transferred_item = item.copy()
        transferred_item.pop("_id", None)
        transferred_item["locationID"] = destination_location_id
        transferred_item["stock_level"] = quantity
        transferred_item["created_at"] = datetime.utcnow()
        transferred_item["updated_at"] = datetime.utcnow()
        
        # Reduce stock at source location
        inventory_collection.update_one(
            {"itemID": item_id, "locationID": source_location_id},
            {"$inc": {"stock_level": -quantity}, "$set": {"updated_at": datetime.utcnow()}}
        )
        
        # Insert or update stock at destination location
        existing_item = inventory_collection.find_one({
            "itemID": item_id, 
            "locationID": destination_location_id
        })
        
        if existing_item:
            # Update existing item at destination
            inventory_collection.update_one(
                {"itemID": item_id, "locationID": destination_location_id},
                {"$inc": {"stock_level": quantity}, "$set": {"updated_at": datetime.utcnow()}}
            )
        else:
            # Insert new item at destination
            inventory_collection.insert_one(transferred_item)
        
        # Update location status
        location_collection.update_one(
            {"locationID": destination_location_id},
            {"$set": {"is_occupied": True, "updated_at": datetime.utcnow()}}
        )
        
        return {"message": f"Successfully transferred {quantity} units of item {item_id} to location {destination_location_id}"}
    
    @staticmethod
    async def check_inventory_anomalies() -> List[Dict[str, Any]]:
        """
        Check for inventory anomalies such as negative stock, 
        stock above maximum level, or discrepancies.
        """
        inventory_collection = get_collection("inventory")
        
        anomalies = []
        
        # Check for negative stock
        negative_stock = list(inventory_collection.find({"stock_level": {"$lt": 0}}))
        for item in negative_stock:
            anomalies.append({
                "itemID": item.get("itemID"),
                "name": item.get("name"),
                "anomaly_type": "negative_stock",
                "current_level": item.get("stock_level"),
                "detected_at": datetime.utcnow()
            })
        
        # Check for stock above maximum level
        over_stock = list(inventory_collection.find({
            "$expr": {"$gt": ["$stock_level", "$max_stock_level"]}
        }))
        for item in over_stock:
            anomalies.append({
                "itemID": item.get("itemID"),
                "name": item.get("name"),
                "anomaly_type": "over_stock",
                "current_level": item.get("stock_level"),
                "max_level": item.get("max_stock_level"),
                "detected_at": datetime.utcnow()
            })
        
        # Store anomalies in the database
        if anomalies:
            anomaly_collection = get_collection("inventory_anomalies")
            anomaly_collection.insert_many(anomalies)
        
        return anomalies