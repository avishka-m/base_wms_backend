from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from typing import List, Dict, Any, Optional

from ..auth.dependencies import get_current_active_user, has_role
from ..models.receiving import ReceivingCreate, ReceivingUpdate, ReceivingResponse
from ..services.workflow_service import WorkflowService
from ..utils.database import get_collection

import importlib.util
import sys
import os
from datetime import datetime

def load_ml_services():
    """Load ML services using importlib"""
    try:
        # Get the correct path step by step
        current_file = __file__  # app/api/receiving.py
        api_dir = os.path.dirname(current_file)      # app/api/
        app_dir = os.path.dirname(api_dir)           # app/
        backend_dir = os.path.dirname(app_dir)       # base_wms_backend/
        
        # ✅ FIXED: ai_services should be in base_wms_backend
        ai_services_path = os.path.join(backend_dir, 'ai_services', 'path_optimization')
        
        warehouse_mapper_path = os.path.join(ai_services_path, 'warehouse_mapper.py')
        location_predictor_path = os.path.join(ai_services_path, 'location_predictor.py')
        allocation_service_path = os.path.join(ai_services_path, 'allocation_service.py')
        
        # Debug output
        print(f"🔍 Current file: {current_file}")
        print(f"🔍 Backend dir: {backend_dir}")
        print(f"🔍 AI services path: {ai_services_path}")
        print(f"🔍 Looking for warehouse_mapper at: {warehouse_mapper_path}")
        print(f"🔍 Looking for location_predictor at: {location_predictor_path}")
        print(f"🔍 Looking for allocation_service at: {allocation_service_path}")
        
        # Check if directory exists
        if not os.path.exists(ai_services_path):
            print(f"❌ Directory {ai_services_path} does not exist")
            return None
        
        # Check if all files exist
        required_files = {
            'warehouse_mapper': warehouse_mapper_path,
            'location_predictor': location_predictor_path,
            'allocation_service': allocation_service_path
        }
        
        for name, path in required_files.items():
            if not os.path.exists(path):
                print(f"❌ {name}.py not found at {path}")
                if os.path.exists(ai_services_path):
                    print(f"📁 Contents of {ai_services_path}: {os.listdir(ai_services_path)}")
                return None
        
        # ✅ Load modules in correct dependency order:
        # 1. warehouse_mapper (no dependencies)
        # 2. location_predictor (depends on warehouse_mapper)
        # 3. allocation_service (depends on both)
        
        # 1. Load warehouse_mapper first
        warehouse_spec = importlib.util.spec_from_file_location("warehouse_mapper", warehouse_mapper_path)
        warehouse_module = importlib.util.module_from_spec(warehouse_spec)
        sys.modules["warehouse_mapper"] = warehouse_module
        warehouse_spec.loader.exec_module(warehouse_module)
        print("✅ warehouse_mapper loaded")
        
        # 2. Load location_predictor
        location_spec = importlib.util.spec_from_file_location("location_predictor", location_predictor_path)
        location_module = importlib.util.module_from_spec(location_spec)
        sys.modules["location_predictor"] = location_module
        location_spec.loader.exec_module(location_module)
        print("✅ location_predictor loaded")
        
        # 3. Load allocation_service last
        allocation_spec = importlib.util.spec_from_file_location("allocation_service", allocation_service_path)
        allocation_module = importlib.util.module_from_spec(allocation_spec)
        sys.modules["allocation_service"] = allocation_module
        allocation_spec.loader.exec_module(allocation_module)
        print("✅ allocation_service loaded")
        
        # Return the allocation_service object
        return getattr(allocation_module, 'allocation_service', None)
        
    except Exception as e:
        print(f"❌ Error loading ML services: {e}")
        import traceback
        traceback.print_exc()
        return None

# Load the services
allocation_service = load_ml_services()

if allocation_service:
    print("✅ ML services loaded successfully via importlib")
else:
    print("⚠️ ML services not available - using fallback")

router = APIRouter()

# Get items by status endpoint
@router.get("/items/by-status")
async def get_items_by_status(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get items grouped by their status for picker dashboard.
    Returns items available for storing and items available for picking.
    """
    try:
        receiving_collection = get_collection("receiving_items")
        inventory_collection = get_collection("inventory")
        
        # Get items available for storing (status = 'received')
        storing_items = list(receiving_collection.find({"status": "received"}))
        
        # Get items available for picking (from inventory with status = 'stored')
        picking_items = list(inventory_collection.find({"status": "stored"}))
        
        # Convert ObjectId to string for JSON serialization
        for item in storing_items:
            if "_id" in item:
                item["_id"] = str(item["_id"])
        
        for item in picking_items:
            if "_id" in item:
                item["_id"] = str(item["_id"])
        
        return {
            "available_for_storing": storing_items,
            "available_for_picking": picking_items
        }
    except Exception as e:
        print(f"Error in get_items_by_status: {str(e)}")
        return {
            "available_for_storing": [],
            "available_for_picking": []
        }

# Get all receiving records
@router.get("/", response_model=List[ReceivingResponse])
async def get_receiving_records(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    supplier_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get all receiving records with optional filtering.
    
    You can filter by status and supplier ID.
    """
    receiving_collection = get_collection("receiving")
    
    # Build query
    query = {}
    if status:
        query["status"] = status
    if supplier_id:
        query["supplierID"] = supplier_id
    
    # Execute query
    receiving_records = list(receiving_collection.find(query).skip(skip).limit(limit))
    
    # Transform records to ensure they match the expected schema
    transformed_records = []
    for record in receiving_records:
        # Ensure workerID is present (fallback for legacy data)
        if "workerID" not in record:
            record["workerID"] = 1  # Default workerID for legacy records
        
        # Transform items to ensure expected_quantity and proper locationID
        if "items" in record:
            for item in record["items"]:
                # Ensure expected_quantity is present
                if "expected_quantity" not in item:
                    item["expected_quantity"] = item.get("quantity", 0)  # Default to actual quantity
                
                # Ensure locationID is integer if present
                if "locationID" in item and isinstance(item["locationID"], str):
                    try:
                        # Try to extract numeric part from location strings like "B01.1"
                        item["locationID"] = 1  # Default to location 1 for string locations
                    except (ValueError, TypeError):
                        item["locationID"] = None
        
        transformed_records.append(record)
    
    return transformed_records

# Get receiving record by ID
@router.get("/{receiving_id}", response_model=ReceivingResponse)
async def get_receiving_record(
    receiving_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific receiving record by ID.
    """
    receiving_collection = get_collection("receiving")
    receiving = receiving_collection.find_one({"receivingID": receiving_id})
    
    if not receiving:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Receiving record with ID {receiving_id} not found"
        )
    
    # Transform record to ensure it matches the expected schema
    # Ensure workerID is present (fallback for legacy data)
    if "workerID" not in receiving:
        receiving["workerID"] = 1  # Default workerID for legacy records
    
    # Transform items to ensure expected_quantity and proper locationID
    if "items" in receiving:
        for item in receiving["items"]:
            # Ensure expected_quantity is present
            if "expected_quantity" not in item:
                item["expected_quantity"] = item.get("quantity", 0)  # Default to actual quantity
            
            # Ensure locationID is integer if present
            if "locationID" in item and isinstance(item["locationID"], str):
                try:
                    # Try to extract numeric part from location strings like "B01.1"
                    item["locationID"] = 1  # Default to location 1 for string locations
                except (ValueError, TypeError):
                    item["locationID"] = None
    
    return receiving

# Create new receiving record
@router.post("/", response_model=ReceivingResponse, status_code=status.HTTP_201_CREATED)
async def create_receiving_record(
    receiving: ReceivingCreate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Create a new receiving record.
    
    Only managers and receiving clerks can create receiving records.
    """
    receiving_collection = get_collection("receiving")
    
    # Find the next available receivingID
    last_receiving = receiving_collection.find_one(
        sort=[("receivingID", -1)]
    )
    next_id = 1
    if last_receiving:
        next_id = last_receiving.get("receivingID", 0) + 1
    
    # Prepare receiving document
    receiving_data = receiving.model_dump()
    receiving_data.update({
        "receivingID": next_id,
        "created_at": receiving_data.get("received_date"),
        "updated_at": receiving_data.get("received_date")
    })
    
    # Initialize items with processed=False
    for item in receiving_data.get("items", []):
        item["processed"] = False
    
    # Insert receiving to database
    result = receiving_collection.insert_one(receiving_data)
    
    # Return the created receiving record
    created_receiving = receiving_collection.find_one({"_id": result.inserted_id})
    return created_receiving

# Update receiving record
@router.put("/{receiving_id}", response_model=ReceivingResponse)
async def update_receiving_record(
    receiving_id: int,
    receiving_update: ReceivingUpdate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Update a receiving record.
    Now automatically predicts locations when status changes to 'processing'
    """
    receiving_collection = get_collection("receiving")
    
    # Check if receiving record exists
    receiving = receiving_collection.find_one({"receivingID": receiving_id})
    if not receiving:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Receiving record with ID {receiving_id} not found"
        )
    
    # Check if already completed
    if receiving.get("status") == "completed" and receiving_update.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update completed receiving record"
        )
    
    # Prepare update data
    update_data = receiving_update.model_dump(exclude_unset=True)
    update_data["updated_at"] = receiving_update.model_dump().get("updated_at")
    
    # Update receiving record
    receiving_collection.update_one(
        {"receivingID": receiving_id},
        {"$set": update_data}
    )
    
    # ✨ NEW: Auto-predict locations when status changes to 'processing'
    if (receiving_update.status == "processing" and 
        receiving.get("status") != "processing"):
        
        try:
            # Trigger location predictions
            await predict_locations_for_receiving_items(receiving_id, current_user)
            print(f"✅ Auto-predicted locations for receiving {receiving_id}")
        except Exception as e:
            print(f"⚠️ Auto-prediction failed for receiving {receiving_id}: {str(e)}")
            # Don't fail the update if prediction fails
    
    # Return updated receiving record
    updated_receiving = receiving_collection.find_one({"receivingID": receiving_id})
    return updated_receiving

# Process receiving
@router.post("/{receiving_id}/process", response_model=ReceivingResponse)
async def process_receiving(
    receiving_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Process a receiving record and update inventory.
    
    This endpoint handles the complete receiving workflow:
    1. Validate the receiving request
    2. Update inventory with received items
    3. Complete the receiving process
    """
    result = await WorkflowService.process_receiving(
        receiving_id=receiving_id,
        worker_id=current_user.get("workerID")
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Get items by storage status - UPDATED WITH ML PREDICTIONS
@router.get("/items/by-status-updated", response_model=Dict[str, List[Dict[str, Any]]])
async def get_items_by_storage_status(
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Picker", "ReceivingClerk"]))
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get items grouped by their storage status.
    NOW INCLUDES ML PREDICTION DATA!
    
    Returns:
    - available_for_storing: Items that have been received but not yet stored (processed=False, locationID=None)
    - available_for_picking: Items that have been stored and are ready for picking (processed=True, locationID exists)
    """
    receiving_collection = get_collection("receiving")
    inventory_collection = get_collection("inventory")
    
    # Get all receiving records with pending or processing status
    receiving_records = list(receiving_collection.find({
        "status": {"$in": ["pending", "processing"]}
    }))
    
    available_for_storing = []
    available_for_picking = []
    
    # ✨ UPDATED: Collect items available for storing WITH ML PREDICTIONS
    for record in receiving_records:
        for item in record.get("items", []):
            if not item.get("processed", False) and item.get("locationID") is None:
                # Get item details from inventory
                inventory_item = inventory_collection.find_one({"itemID": item["itemID"]})
                if inventory_item:
                    # ✨ NEW: Include ML prediction data
                    storing_item = {
                        "receivingID": record["receivingID"],
                        "itemID": item["itemID"],
                        "itemName": inventory_item.get("name", "Unknown"),
                        "quantity": item["quantity"],
                        "condition": item.get("condition", "good"),
                        "receivedDate": record.get("received_date"),
                        "supplierID": record.get("supplierID"),
                        "notes": item.get("notes", ""),
                        # ✨ ADD ML prediction data
                        "predicted_location": item.get("predicted_location"),
                        "predicted_coordinates": item.get("predicted_coordinates"),
                        "prediction_confidence": item.get("prediction_confidence"),
                        "allocation_reason": item.get("allocation_reason", "ML model prediction"),
                        "predicted_at": item.get("predicted_at"),
                        "predicted_by": item.get("predicted_by"),
                        # Add category and size for frontend
                        "category": inventory_item.get("category", "General"),
                        "size": inventory_item.get("size", "M")
                    }
                    
                    # ✨ Add suggested_location for frontend compatibility
                    if item.get("predicted_location") and item.get("predicted_coordinates"):
                        storing_item["suggested_location"] = {
                            'locationCode': item["predicted_location"],
                            'x': item["predicted_coordinates"]['x'] - 1,  # Convert to 0-indexed
                            'y': item["predicted_coordinates"]['y'] - 1,  # Convert to 0-indexed  
                            'floor': item["predicted_coordinates"]['floor'],
                            'confidence': item.get("prediction_confidence", 0.8)
                        }
                    
                    available_for_storing.append(storing_item)
    
    # Get completed receiving records for items available for picking
    completed_records = list(receiving_collection.find({
        "status": "completed"
    }))
    
    for record in completed_records:
        for item in record.get("items", []):
            if item.get("processed", False) and item.get("locationID") is not None:
                # Get item details from inventory
                inventory_item = inventory_collection.find_one({"itemID": item["itemID"]})
                if inventory_item and inventory_item.get("stock_level", 0) > 0:
                    available_for_picking.append({
                        "itemID": item["itemID"],
                        "itemName": inventory_item.get("name", "Unknown"),
                        "stockLevel": inventory_item.get("stock_level", 0),
                        "locationID": item["locationID"],
                        "category": inventory_item.get("category", ""),
                        "size": inventory_item.get("size", "")
                    })
    
    # Also add items from inventory that have stock and location
    inventory_items = list(inventory_collection.find({
        "stock_level": {"$gt": 0},
        "locationID": {"$ne": None}
    }))
    
    for item in inventory_items:
        # Check if not already in the list
        if not any(p["itemID"] == item["itemID"] for p in available_for_picking):
            available_for_picking.append({
                "itemID": item["itemID"],
                "itemName": item.get("name", "Unknown"),
                "stockLevel": item.get("stock_level", 0),
                "locationID": item.get("locationID"),
                "category": item.get("category", ""),
                "size": item.get("size", "")
            })
    
    return {
        "available_for_storing": available_for_storing,
        "available_for_picking": available_for_picking
    }

# Mark item as stored
@router.post("/{receiving_id}/items/{item_id}/store")
async def mark_item_as_stored(
    receiving_id: int,
    item_id: int,
    location_id: str = Body(..., embed=True),
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "Picker", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Mark a received item as stored by assigning it a location.
    """
    receiving_collection = get_collection("receiving")
    
    # Find the receiving record
    receiving = receiving_collection.find_one({"receivingID": receiving_id})
    if not receiving:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Receiving record with ID {receiving_id} not found"
        )
    
    # Find and update the specific item
    item_found = False
    for idx, item in enumerate(receiving.get("items", [])):
        if item["itemID"] == item_id:
            item_found = True
            # Update the item
            receiving_collection.update_one(
                {"receivingID": receiving_id},
                {
                    "$set": {
                        f"items.{idx}.processed": True,
                        f"items.{idx}.locationID": location_id,
                        f"items.{idx}.stored_by": current_user.get("username", "Unknown"),
                        f"items.{idx}.stored_at": datetime.utcnow().isoformat()
                    }
                }
            )
            break
    
    if not item_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with ID {item_id} not found in receiving record {receiving_id}"
        )
    
    # Check if all items are processed and update status if needed
    updated_receiving = receiving_collection.find_one({"receivingID": receiving_id})
    all_processed = all(item.get("processed", False) for item in updated_receiving.get("items", []))
    
    if all_processed and updated_receiving.get("status") != "completed":
        receiving_collection.update_one(
            {"receivingID": receiving_id},
            {"$set": {"status": "completed"}}
        )
    
    return {"message": f"Item {item_id} marked as stored in location {location_id}"}

# ✨ NEW: Predict locations for receiving items
@router.post("/{receiving_id}/predict-locations")
async def predict_locations_for_receiving_items(
    receiving_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    NEW: Predict optimal locations for all items in a receiving record
    Call this when receiver finishes updating/processing items
    """
    try:
        # Check if ML service is available
        if not allocation_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML prediction service is not available"
            )
            
        receiving_collection = get_collection("receiving")
        inventory_collection = get_collection("inventory")
        seasonal_collection = get_collection("seasonal_demand")
        storage_collection = get_collection("storage_history")
        
        # Get receiving record
        receiving = receiving_collection.find_one({"receivingID": receiving_id})
        if not receiving:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Receiving record {receiving_id} not found"
            )
        
        predictions = []
        
        # Process each unprocessed item
        for idx, item in enumerate(receiving.get("items", [])):
            if not item.get("processed", False):
                # Get item details
                inventory_item = inventory_collection.find_one({"itemID": item["itemID"]})
                if inventory_item:
                    try:
                        # Get ML prediction
                        allocation_result = await allocation_service.allocate_location_for_item(
                            item_id=item["itemID"],
                            category=inventory_item.get("category", "General"),
                            item_size=inventory_item.get("size", "M"),
                            quantity=item["quantity"],
                            db_collection_seasonal=seasonal_collection,
                            db_collection_storage=storage_collection,
                            db_collection_location_inventory=get_collection("location_inventory")  
                        )
                        
                        if allocation_result['success']:
                            # Update the receiving record with prediction
                            receiving_collection.update_one(
                                {"receivingID": receiving_id},
                                {
                                    "$set": {
                                        f"items.{idx}.predicted_location": allocation_result['allocated_location'],
                                        f"items.{idx}.predicted_coordinates": allocation_result['coordinates'],
                                        f"items.{idx}.prediction_confidence": allocation_result['confidence'],
                                        f"items.{idx}.allocation_reason": allocation_result['allocation_reason'],
                                        f"items.{idx}.predicted_at": datetime.utcnow().isoformat(),
                                        f"items.{idx}.predicted_by": current_user.get("username", "system")
                                    }
                                }
                            )
                            
                            predictions.append({
                                "itemID": item["itemID"],
                                "item_name": inventory_item.get("name", "Unknown"),
                                "predicted_location": allocation_result['allocated_location'],
                                "coordinates": allocation_result['coordinates'],
                                "confidence": allocation_result['confidence'],
                                "reason": allocation_result['allocation_reason']
                            })
                            
                            print(f"✅ Predicted location for item {item['itemID']}: {allocation_result['allocated_location']}")
                    
                    except Exception as e:
                        print(f"⚠️ Prediction failed for item {item['itemID']}: {str(e)}")
                        predictions.append({
                            "itemID": item["itemID"],
                            "item_name": inventory_item.get("name", "Unknown"),
                            "error": f"Prediction failed: {str(e)}"
                        })
        
        return {
            "receiving_id": receiving_id,
            "predictions": predictions,
            "total_items_predicted": len([p for p in predictions if 'error' not in p]),
            "total_errors": len([p for p in predictions if 'error' in p])
        }
        
    except Exception as e:
        print(f"Error predicting locations for receiving {receiving_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to predict locations"
        )

# ✨ NEW: Test endpoint to create sample data and trigger predictions
@router.post("/test-prediction")
async def test_prediction(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Test endpoint to create sample data and trigger predictions"""
    
    receiving_collection = get_collection("receiving")
    inventory_collection = get_collection("inventory")
    
    # Check if test item exists in inventory
    test_item = inventory_collection.find_one({"itemID": 1})
    if not test_item:
        # Create a test inventory item
        inventory_collection.insert_one({
            "itemID": 1,
            "name": "Test Product",
            "category": "electronics",
            "size": "M",
            "stock_level": 0
        })
        print("✅ Created test inventory item")
    
    # Create a test receiving record
    test_receiving_id = 999
    
    # Remove existing test record if it exists
    receiving_collection.delete_one({"receivingID": test_receiving_id})
    
    test_receiving = {
        "receivingID": test_receiving_id,
        "status": "processing",
        "supplierID": 1,
        "received_date": datetime.utcnow().isoformat(),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "items": [
            {
                "itemID": 1,
                "quantity": 10,
                "condition": "good",
                "processed": False,
                "notes": "Test item for ML prediction"
            }
        ]
    }
    
    # Insert test data
    receiving_collection.insert_one(test_receiving)
    print("✅ Created test receiving record")
    
    # Trigger prediction
    try:
        result = await predict_locations_for_receiving_items(test_receiving_id, current_user)
        return {
            "message": "Test receiving record created and predictions triggered",
            "prediction_result": result,
            "instructions": "Check the picker dashboard to see the predicted item!"
        }
    except Exception as e:
        return {
            "message": "Test receiving record created but prediction failed",
            "error": str(e),
            "instructions": "Check the ML service status"
        }