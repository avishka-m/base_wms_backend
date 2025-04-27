from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import math

from ..utils.database import get_collection
from ..models.location import LocationCreate, LocationUpdate
from ..models.warehouse import WarehouseCreate, WarehouseUpdate

class WarehouseService:
    """
    Service for warehouse and storage location management.
    
    This service handles business logic for warehouse operations such as
    location assignment, space utilization, and path optimization.
    """
    
    @staticmethod
    async def get_warehouses(skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all warehouses.
        """
        warehouse_collection = get_collection("warehouses")
        warehouses = list(warehouse_collection.find().skip(skip).limit(limit))
        return warehouses
    
    @staticmethod
    async def get_warehouse(warehouse_id: int) -> Dict[str, Any]:
        """
        Get a specific warehouse by ID.
        """
        warehouse_collection = get_collection("warehouses")
        warehouse = warehouse_collection.find_one({"warehouseID": warehouse_id})
        return warehouse
    
    @staticmethod
    async def create_warehouse(warehouse: WarehouseCreate) -> Dict[str, Any]:
        """
        Create a new warehouse.
        """
        warehouse_collection = get_collection("warehouses")
        
        # Find the next available warehouseID
        last_warehouse = warehouse_collection.find_one(
            sort=[("warehouseID", -1)]
        )
        next_id = 1
        if last_warehouse:
            next_id = last_warehouse.get("warehouseID", 0) + 1
        
        # Prepare warehouse document
        warehouse_data = warehouse.model_dump()
        warehouse_data.update({
            "warehouseID": next_id,
            "available_storage": warehouse_data.get("capacity", 0),  # Initially, all storage is available
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        
        # Insert warehouse to database
        result = warehouse_collection.insert_one(warehouse_data)
        
        # Return the created warehouse
        created_warehouse = warehouse_collection.find_one({"_id": result.inserted_id})
        return created_warehouse
    
    @staticmethod
    async def update_warehouse(warehouse_id: int, warehouse_update: WarehouseUpdate) -> Dict[str, Any]:
        """
        Update a warehouse.
        """
        warehouse_collection = get_collection("warehouses")
        
        # Prepare update data
        update_data = warehouse_update.model_dump(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        # Update warehouse
        warehouse_collection.update_one(
            {"warehouseID": warehouse_id},
            {"$set": update_data}
        )
        
        # Return updated warehouse
        updated_warehouse = warehouse_collection.find_one({"warehouseID": warehouse_id})
        return updated_warehouse
    
    @staticmethod
    async def delete_warehouse(warehouse_id: int) -> Dict[str, Any]:
        """
        Delete a warehouse.
        """
        warehouse_collection = get_collection("warehouses")
        location_collection = get_collection("locations")
        
        # Check if warehouse has any locations
        locations = list(location_collection.find({"warehouseID": warehouse_id}))
        if locations:
            return {"error": f"Cannot delete warehouse with ID {warehouse_id} because it has {len(locations)} locations assigned to it"}
        
        # Delete warehouse
        warehouse_collection.delete_one({"warehouseID": warehouse_id})
        
        return {"message": f"Warehouse with ID {warehouse_id} has been deleted"}
    
    @staticmethod
    async def get_locations(warehouse_id: Optional[int] = None, 
                            is_occupied: Optional[bool] = None, 
                            skip: int = 0, 
                            limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get storage locations with optional filtering.
        """
        location_collection = get_collection("locations")
        
        # Build query
        query = {}
        if warehouse_id is not None:
            query["warehouseID"] = warehouse_id
        if is_occupied is not None:
            query["is_occupied"] = is_occupied
        
        # Execute query
        locations = list(location_collection.find(query).skip(skip).limit(limit))
        return locations
    
    @staticmethod
    async def get_location(location_id: int) -> Dict[str, Any]:
        """
        Get a specific storage location by ID.
        """
        location_collection = get_collection("locations")
        location = location_collection.find_one({"locationID": location_id})
        return location
    
    @staticmethod
    async def create_location(location: LocationCreate) -> Dict[str, Any]:
        """
        Create a new storage location.
        """
        location_collection = get_collection("locations")
        warehouse_collection = get_collection("warehouses")
        
        # Check if warehouse exists
        warehouse = warehouse_collection.find_one({"warehouseID": location.warehouseID})
        if not warehouse:
            return {"error": f"Warehouse with ID {location.warehouseID} not found"}
        
        # Find the next available locationID
        last_location = location_collection.find_one(
            sort=[("locationID", -1)]
        )
        next_id = 1
        if last_location:
            next_id = last_location.get("locationID", 0) + 1
        
        # Prepare location document
        location_data = location.model_dump()
        location_data.update({
            "locationID": next_id,
            "is_occupied": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        
        # Insert location to database
        result = location_collection.insert_one(location_data)
        
        # Return the created location
        created_location = location_collection.find_one({"_id": result.inserted_id})
        return created_location
    
    @staticmethod
    async def update_location(location_id: int, location_update: LocationUpdate) -> Dict[str, Any]:
        """
        Update a storage location.
        """
        location_collection = get_collection("locations")
        
        # Prepare update data
        update_data = location_update.model_dump(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        # Update location
        location_collection.update_one(
            {"locationID": location_id},
            {"$set": update_data}
        )
        
        # Return updated location
        updated_location = location_collection.find_one({"locationID": location_id})
        return updated_location
    
    @staticmethod
    async def delete_location(location_id: int) -> Dict[str, Any]:
        """
        Delete a storage location.
        """
        location_collection = get_collection("locations")
        inventory_collection = get_collection("inventory")
        
        # Check if location has any inventory
        inventory = list(inventory_collection.find({"locationID": location_id}))
        if inventory:
            return {"error": f"Cannot delete location with ID {location_id} because it has inventory items assigned to it"}
        
        # Delete location
        location_collection.delete_one({"locationID": location_id})
        
        return {"message": f"Location with ID {location_id} has been deleted"}
    
    @staticmethod
    async def find_optimal_location(item_id: int, warehouse_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Find the optimal storage location for an item based on item properties
        and available space.
        
        This is a simple implementation that finds an empty location. In a real system,
        you would implement a more sophisticated algorithm that considers factors like
        item category, size, access frequency, and proximity to other related items.
        """
        inventory_collection = get_collection("inventory")
        location_collection = get_collection("locations")
        
        # Get item
        item = inventory_collection.find_one({"itemID": item_id})
        if not item:
            return {"error": f"Item with ID {item_id} not found"}
        
        # Build query for available locations
        query = {"is_occupied": False}
        if warehouse_id:
            query["warehouseID"] = warehouse_id
        
        # Find all available locations
        available_locations = list(location_collection.find(query))
        if not available_locations:
            return {"error": "No available locations found", "status": "error"}
        
        # For a simple implementation, just return the first available location
        # In a real system, you would implement a scoring algorithm based on
        # various factors to determine the optimal location
        optimal_location = available_locations[0]
        
        return {
            "status": "success",
            "location": optimal_location,
            "message": f"Found optimal location: Section {optimal_location.get('section')}, Row {optimal_location.get('row')}, Shelf {optimal_location.get('shelf')}, Bin {optimal_location.get('bin')}"
        }
    
    @staticmethod
    async def calculate_warehouse_utilization(warehouse_id: int) -> Dict[str, Any]:
        """
        Calculate the storage utilization of a warehouse.
        """
        warehouse_collection = get_collection("warehouses")
        location_collection = get_collection("locations")
        
        # Get warehouse
        warehouse = warehouse_collection.find_one({"warehouseID": warehouse_id})
        if not warehouse:
            return {"error": f"Warehouse with ID {warehouse_id} not found"}
        
        # Get all locations in the warehouse
        locations = list(location_collection.find({"warehouseID": warehouse_id}))
        total_locations = len(locations)
        occupied_locations = sum(1 for loc in locations if loc.get("is_occupied", False))
        
        # Calculate utilization
        utilization_percentage = 0
        if total_locations > 0:
            utilization_percentage = (occupied_locations / total_locations) * 100
        
        return {
            "warehouseID": warehouse_id,
            "name": warehouse.get("name"),
            "total_locations": total_locations,
            "occupied_locations": occupied_locations,
            "available_locations": total_locations - occupied_locations,
            "utilization_percentage": utilization_percentage,
            "calculated_at": datetime.utcnow()
        }
    
    @staticmethod
    async def optimize_picking_path(picking_locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize the path for picking items from multiple locations.
        
        This is a simple implementation that sorts locations by section, row, shelf, and bin.
        In a real system, you would implement a more sophisticated algorithm like
        Traveling Salesman Problem (TSP) solver or A* path finding.
        """
        # Simple sort by location coordinates
        sorted_locations = sorted(picking_locations, key=lambda x: (
            x.get("section", ""),
            x.get("row", ""),
            x.get("shelf", ""),
            x.get("bin", "")
        ))
        
        return sorted_locations
    
    @staticmethod
    async def calculate_distance(location1: Dict[str, Any], location2: Dict[str, Any]) -> float:
        """
        Calculate the Euclidean distance between two locations.
        
        This is a simplified calculation assuming each section/row/shelf/bin
        can be mapped to numerical coordinates. In a real system, you would
        use actual physical distances or a warehouse map.
        """
        # Convert location coordinates to numerical values
        # This is a simplification - in reality, you would use actual coordinates
        def get_coord(loc: Dict[str, Any]) -> Tuple[float, float, float, float]:
            return (
                ord(loc.get("section", "A")[0]) - ord("A"),
                int(loc.get("row", "1")),
                int(loc.get("shelf", "1")),
                int(loc.get("bin", "1"))
            )
        
        coord1 = get_coord(location1)
        coord2 = get_coord(location2)
        
        # Calculate Euclidean distance
        distance = math.sqrt(
            (coord2[0] - coord1[0]) ** 2 +
            (coord2[1] - coord1[1]) ** 2 +
            (coord2[2] - coord1[2]) ** 2 +
            (coord2[3] - coord1[3]) ** 2
        )
        
        return distance