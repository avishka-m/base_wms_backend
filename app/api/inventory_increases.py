# app/api/inventory_increases.py - FIXED VERSION

from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..auth.dependencies import get_current_active_user, has_role
from ..utils.database import get_collection

# Import the ML path optimization services
import importlib.util
import sys
import os

def load_ml_services():
    """Load ML services using importlib"""
    try:
        # Get the correct path step by step
        current_file = __file__  # app/api/inventory_increases.py
        api_dir = os.path.dirname(current_file)      # app/api/
        app_dir = os.path.dirname(api_dir)           # app/
        backend_dir = os.path.dirname(app_dir)       # base_wms_backend/
        
        ai_services_path = os.path.join(backend_dir, 'ai_services', 'path_optimization')
        
        warehouse_mapper_path = os.path.join(ai_services_path, 'warehouse_mapper.py')
        location_predictor_path = os.path.join(ai_services_path, 'location_predictor.py')
        allocation_service_path = os.path.join(ai_services_path, 'allocation_service.py')
        
        # Check if all files exist
        required_files = {
            'warehouse_mapper': warehouse_mapper_path,
            'location_predictor': location_predictor_path,
            'allocation_service': allocation_service_path
        }
        
        for name, path in required_files.items():
            if not os.path.exists(path):
                print(f"❌ {name}.py not found at {path}")
                return None, None
        
        # Load modules in correct dependency order
        # 1. Load warehouse_mapper first
        warehouse_spec = importlib.util.spec_from_file_location("warehouse_mapper", warehouse_mapper_path)
        warehouse_module = importlib.util.module_from_spec(warehouse_spec)
        sys.modules["warehouse_mapper"] = warehouse_module
        warehouse_spec.loader.exec_module(warehouse_module)
        
        # 2. Load location_predictor
        location_spec = importlib.util.spec_from_file_location("location_predictor", location_predictor_path)
        location_module = importlib.util.module_from_spec(location_spec)
        sys.modules["location_predictor"] = location_module
        location_spec.loader.exec_module(location_module)
        
        # 3. Load allocation_service last
        allocation_spec = importlib.util.spec_from_file_location("allocation_service", allocation_service_path)
        allocation_module = importlib.util.module_from_spec(allocation_spec)
        sys.modules["allocation_service"] = allocation_module
        allocation_spec.loader.exec_module(allocation_module)
        
        return (
            getattr(allocation_module, 'allocation_service', None),
            getattr(warehouse_module, 'warehouse_mapper', None)
        )
        
    except Exception as e:
        print(f"❌ Error loading ML services: {e}")
        return None, None

# Load the services
allocation_service, warehouse_mapper = load_ml_services()

if allocation_service and warehouse_mapper:
    print("✅ ML services loaded successfully in inventory_increases")
else:
    print("⚠️ ML services not available in inventory_increases - using fallback")

router = APIRouter()

@router.post("/")
async def create_inventory_increase(
    request_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Create a new inventory increase record.
    This endpoint is called when receiver updates inventory through the frontend.
    Now includes ML location predictions!
    """
    try:
        receiving_collection = get_collection("receiving")
        inventory_collection = get_collection("inventory")
        seasonal_collection = get_collection("seasonal_demand")
        storage_collection = get_collection("storage_history")
        
        # ✨ DEBUG: Log received data
        print(f"🔍 DEBUG - Received request_data: {request_data}")
        print(f"🔍 DEBUG - Keys in request: {list(request_data.keys())}")
        
        # Extract data from request (accept multiple field name formats)
        item_id = request_data.get("itemID") or request_data.get("itemId")
        item_name = request_data.get("item_name") or request_data.get("itemName")
        quantity = request_data.get("quantity")
        reason = request_data.get("reason", "stock_arrival")
        notes = request_data.get("notes", "")
        
        print(f"🔍 DEBUG - Extracted: itemID={item_id}, quantity={quantity}, item_name={item_name}")
        
        if not item_id or not quantity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"itemID and quantity are required. Received: itemID={item_id}, quantity={quantity}"
            )
        
        # Get item details from inventory
        inventory_item = inventory_collection.find_one({"itemID": item_id})
        if not inventory_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item {item_id} not found in inventory"
            )
        
        # Create receiving record for this inventory increase
        receiving_records = list(receiving_collection.find({}).sort([("receivingID", -1)]).limit(1))
        next_receiving_id = 1
        if receiving_records:
            next_receiving_id = receiving_records[0].get("receivingID", 0) + 1
        
        # ✨ ML PREDICTION: Get optimal location for the item
        predicted_location = None
        predicted_coordinates = None
        prediction_confidence = None
        allocation_reason = "Manual inventory update"
        
        if allocation_service:
            try:
                allocation_result = await allocation_service.allocate_location_for_item(
                    item_id=item_id,
                    category=inventory_item.get("category", "General"),
                    item_size=inventory_item.get("size", "M"),
                    quantity=quantity,
                    db_collection_seasonal=seasonal_collection,
                    db_collection_storage=storage_collection
                )
                
                if allocation_result['success']:
                    predicted_location = allocation_result['allocated_location']
                    predicted_coordinates = allocation_result['coordinates']
                    prediction_confidence = allocation_result['confidence']
                    allocation_reason = allocation_result['allocation_reason']
                    
                    print(f"✅ ML prediction for item {item_id}: {predicted_location} (confidence: {prediction_confidence})")
                    
            except Exception as e:
                print(f"⚠️ ML prediction failed for item {item_id}: {str(e)}")
                # Use fallback values
                predicted_location = "B1.1"
                predicted_coordinates = {'x': 1, 'y': 2, 'floor': 1}
                prediction_confidence = 0.5
                allocation_reason = "Fallback location - ML prediction failed"
        else:
            # Use fallback when ML service is not available
            predicted_location = "B1.1"
            predicted_coordinates = {'x': 1, 'y': 2, 'floor': 1}
            prediction_confidence = 0.5
            allocation_reason = "Fallback location - ML service unavailable"
        
        # Create receiving record with ML prediction
        receiving_data = {
            "receivingID": next_receiving_id,
            "status": "processing",  # Set to processing so it appears in picker dashboard
            "supplierID": 1,  # Default supplier for manual updates
            "received_date": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "received_by": current_user.get("username", "system"),
            "items": [
                {
                    "itemID": item_id,
                    "quantity": quantity,
                    "condition": "good",
                    "processed": False,
                    "notes": notes,
                    "created_by": current_user.get("username", "system"),
                    "created_at": datetime.utcnow().isoformat(),
                    # ✨ Store ML prediction data
                    "predicted_location": predicted_location,
                    "predicted_coordinates": predicted_coordinates,
                    "prediction_confidence": prediction_confidence,
                    "allocation_reason": allocation_reason,
                    "predicted_at": datetime.utcnow().isoformat(),
                    "predicted_by": current_user.get("username", "system")
                }
            ]
        }
        
        # Insert the receiving record
        result = receiving_collection.insert_one(receiving_data)
        
        return {
            "message": "Inventory increase recorded successfully with ML prediction",
            "receiving_id": next_receiving_id,
            "item_id": item_id,
            "item_name": item_name,
            "quantity": quantity,
            "predicted_location": predicted_location,
            "prediction_confidence": prediction_confidence,
            "allocation_reason": allocation_reason,
            "status": "Item is now available for storing in picker dashboard"
        }
        
    except Exception as e:
        print(f"❌ Error creating inventory increase: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create inventory increase: {str(e)}"
        )

@router.get("/")
async def get_inventory_increases(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    limit: int = Query(100, description="Maximum number of items to return"),
    itemID: Optional[int] = Query(None, description="Filter by item ID"),
    start_date: Optional[str] = Query(None, description="Start date filter"),
    end_date: Optional[str] = Query(None, description="End date filter"),
    reason: Optional[str] = Query(None, description="Filter by reason"),
    skip: int = Query(0, description="Number of items to skip")
) -> List[Dict[str, Any]]:
    """
    Get inventory increases (items available for storing).
    Uses ML predictions stored by the receiver workflow!
    """
    try:
        receiving_collection = get_collection("receiving")
        inventory_collection = get_collection("inventory")
        
        # Build query for receiving records
        query = {"status": {"$in": ["pending", "processing", "completed"]}}
        
        # Get receiving records
        receiving_records = list(receiving_collection.find(query).skip(skip).limit(limit * 2))
        
        inventory_increases = []
        
        for record in receiving_records:
            for item in record.get("items", []):
                # Only include unprocessed items (available for storing)
                if not item.get("processed", False) and item.get("locationID") is None:
                    # Get item details from inventory
                    inventory_item = inventory_collection.find_one({"itemID": item["itemID"]})
                    
                    if inventory_item:
                        # ✨ Use STORED ML predictions from the POST endpoint
                        predicted_location = item.get("predicted_location")
                        predicted_coordinates = item.get("predicted_coordinates")
                        prediction_confidence = item.get("prediction_confidence")
                        allocation_reason = item.get("allocation_reason", "ML model prediction")
                        
                        # If no prediction exists, use fallback
                        if not predicted_location:
                            predicted_location = "B1.1"  # Default location
                            predicted_coordinates = {'x': 1, 'y': 2, 'floor': 1}
                            prediction_confidence = 0.5
                            allocation_reason = "Fallback allocation - no ML prediction available"
                            
                        increase_record = {
                            "id": f"{record['receivingID']}-{item['itemID']}",
                            "itemID": item["itemID"],
                            "item_name": inventory_item.get("name", "Unknown"),
                            "size": inventory_item.get("size", "Medium"),
                            "quantity_increased": item["quantity"],
                            "quantity_processed": 0,
                            "reason": "stock_arrival",
                            "source": f"Receiving #{record['receivingID']}",
                            "reference_id": record["receivingID"],
                            "performed_by": record.get("received_by", "system"),
                            "notes": item.get("notes", ""),
                            "timestamp": record.get("received_date"),
                            "processed": False,
                            "partially_processed": False,
                            
                            # ✨ ML prediction data
                            "predicted_location": predicted_location,
                            "predicted_coordinates": predicted_coordinates,
                            "prediction_confidence": prediction_confidence,
                            "allocation_reason": allocation_reason,
                            
                            # For frontend compatibility
                            "suggested_location": {
                                'locationCode': predicted_location,
                                'x': (predicted_coordinates.get('x', 1) - 1) if predicted_coordinates else 0,  # Convert to 0-indexed
                                'y': (predicted_coordinates.get('y', 2) - 1) if predicted_coordinates else 1,  # Convert to 0-indexed
                                'floor': predicted_coordinates.get('floor', 1) if predicted_coordinates else 1,
                                'confidence': prediction_confidence or 0.5
                            },
                            "receiving_location": {
                                'x': 0,  # Receiving point coordinates
                                'y': 0,
                                'floor': 1
                            }
                        }
                        
                        # Apply filters
                        if itemID and item["itemID"] != itemID:
                            continue
                        if reason and increase_record["reason"] != reason:
                            continue
                            
                        inventory_increases.append(increase_record)
                        
                        # Limit results
                        if len(inventory_increases) >= limit:
                            break
            
            if len(inventory_increases) >= limit:
                break
        
        return inventory_increases
        
    except Exception as e:
        print(f"❌ Error in get_inventory_increases: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch inventory increases"
        )

@router.post("/mark-as-stored")
async def mark_inventory_increases_as_stored(
    request_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Picker", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Mark inventory increases as stored.
    Updates with the actual allocated location and records in storage_history!
    """
    try:
        receiving_collection = get_collection("receiving")
        storage_collection = get_collection("storage_history")  # ✨ ADD: Get storage collection
        
        item_id = request_data.get("itemID")
        item_name = request_data.get("item_name")
        quantity_stored = request_data.get("quantity_stored")
        actual_location = request_data.get("actual_location")  # Actual location used
        
        if not item_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="itemID is required"
            )
        
        if not actual_location:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="actual_location is required"
            )
        
        # Find receiving records with this item that's not processed
        receiving_records = list(receiving_collection.find({
            "items.itemID": item_id,
            "items.processed": {"$ne": True}
        }))
        
        if not receiving_records:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No unprocessed receiving record found for item {item_id}"
            )
        
        # Update the first matching record
        record = receiving_records[0]
        stored_item = None
        
        for idx, item in enumerate(record.get("items", [])):
            if item["itemID"] == item_id and not item.get("processed", False):
                
                # Store both predicted and actual location data
                update_data = {
                    f"items.{idx}.processed": True,
                    f"items.{idx}.locationID": actual_location,  # Store actual location
                    f"items.{idx}.quantity_processed": quantity_stored,
                    f"items.{idx}.processed_by": current_user.get("username", "Unknown"),
                    f"items.{idx}.processed_date": datetime.utcnow().isoformat(),
                    f"items.{idx}.actual_location": actual_location,  # Store where it was actually placed
                }
                
                receiving_collection.update_one(
                    {"receivingID": record["receivingID"]},
                    {"$set": update_data}
                )
                
                stored_item = item
                break
        
        if not stored_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item {item_id} not found in receiving record"
            )
        
        # ✨ NEW: Record in storage_history for ML model to track occupied locations
        storage_entry = {
            "locationID": actual_location,
            "action": "stored",
            "itemID": item_id,
            "item_name": item_name or f"Item {item_id}",
            "quantity": quantity_stored or stored_item.get("quantity", 1),
            "category": stored_item.get("category", "General"),
            "stored_by": current_user.get("username", "Unknown"),
            "stored_at": datetime.utcnow().isoformat(),
            "receiving_id": record["receivingID"],
            "predicted_location": stored_item.get("predicted_location"),
            "location_match": stored_item.get("predicted_location") == actual_location,
            "coordinates": {
                "x": 0,  # Will be updated by warehouse_mapper if needed
                "y": 0,
                "floor": 1
            }
        }
        
        # ✨ ADD: Insert storage history record
        storage_result = storage_collection.insert_one(storage_entry)
        
        print(f"✅ Item {item_id} marked as stored in location {actual_location}")
        print(f"✅ Storage history record created: {storage_result.inserted_id}")
        
        # ✨ ADD: Check if ML prediction was accurate
        prediction_accuracy = "Unknown"
        if stored_item.get("predicted_location"):
            if stored_item["predicted_location"] == actual_location:
                prediction_accuracy = "Accurate - Used ML prediction"
            else:
                prediction_accuracy = f"Different - ML predicted {stored_item['predicted_location']}, used {actual_location}"
        
        return {
            "message": f"Inventory increase for item {item_id} marked as stored",
            "itemID": item_id,
            "quantity_stored": quantity_stored,
            "actual_location": actual_location,
            "receiving_id": record["receivingID"],
            "storage_history_id": str(storage_result.inserted_id),
            "prediction_accuracy": prediction_accuracy,
            "ml_prediction": stored_item.get("predicted_location"),
            "confidence": stored_item.get("prediction_confidence")
        }
        
    except Exception as e:
        print(f"❌ Error in mark_inventory_increases_as_stored: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark inventory increase as stored"
        )
    
@router.get("/predict-location/{item_id}")
async def predict_location_for_item(
    item_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get real-time location prediction for a specific item
    """
    try:
        if not allocation_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML prediction service is not available"
            )
            
        inventory_collection = get_collection("inventory")
        seasonal_collection = get_collection("seasonal_demand")
        storage_collection = get_collection("storage_history")
        
        # Get item details
        inventory_item = inventory_collection.find_one({"itemID": item_id})
        if not inventory_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item {item_id} not found"
            )
        
        # Get ML prediction
        allocation_result = await allocation_service.allocate_location_for_item(
            item_id=item_id,
            category=inventory_item.get("category", "General"),
            item_size=inventory_item.get("size", "M"),
            quantity=1,  # Default quantity for prediction
            db_collection_seasonal=seasonal_collection,
            db_collection_storage=storage_collection
        )
        
        return allocation_result
        
    except Exception as e:
        print(f"❌ Error predicting location: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to predict location"
        )

@router.get("/rack-utilization")
async def get_rack_utilization(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get utilization statistics for all rack groups
    """
    try:
        if not warehouse_mapper:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Warehouse mapper service is not available"
            )
            
        storage_collection = get_collection("storage_history")
        
        # Get all occupied locations
        occupied_docs = list(storage_collection.find({"action": "stored"}))
        occupied_sublocations = [doc.get('locationID') for doc in occupied_docs if doc.get('locationID')]
        
        # Calculate utilization for each rack group
        rack_groups = ['B Rack 1', 'B Rack 2', 'B Rack 3', 'P Rack 1', 'P Rack 2', 'D Rack 1', 'D Rack 2']
        utilization_stats = {}
        
        for rack_group in rack_groups:
            base_locations = warehouse_mapper.get_rack_locations(rack_group)
            total_sublocations = len(base_locations) * 4  # 4 floors per location
            
            # Count occupied sublocations in this rack group
            occupied_in_rack = 0
            for sublocation in occupied_sublocations:
                if warehouse_mapper.get_rack_group_from_location(sublocation) == rack_group:
                    occupied_in_rack += 1
            
            utilization_percentage = (occupied_in_rack / total_sublocations) * 100 if total_sublocations > 0 else 0
            
            utilization_stats[rack_group] = {
                'total_capacity': total_sublocations,
                'occupied': occupied_in_rack,
                'free': total_sublocations - occupied_in_rack,
                'utilization_percentage': round(utilization_percentage, 2)
            }
        
        return {
            'rack_utilization': utilization_stats,
            'total_warehouse_capacity': sum(stats['total_capacity'] for stats in utilization_stats.values()),
            'total_occupied': sum(stats['occupied'] for stats in utilization_stats.values()),
            'overall_utilization': round(
                (sum(stats['occupied'] for stats in utilization_stats.values()) / 
                 sum(stats['total_capacity'] for stats in utilization_stats.values())) * 100, 2
            )
        }
        
    except Exception as e:
        print(f"❌ Error getting rack utilization: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get rack utilization"
        )

# Debug endpoint for testing
@router.get("/debug")
async def debug_endpoint():
    return {
        "message": "Inventory increases endpoint is working",
        "timestamp": datetime.utcnow().isoformat(),
        "ml_services_available": {
            "allocation_service": allocation_service is not None,
            "warehouse_mapper": warehouse_mapper is not None
        }
    }