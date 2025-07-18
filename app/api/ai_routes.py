
from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..auth.dependencies import get_current_active_user, has_role
from ..utils.database import get_collection

# Import the allocation service
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_services'))

try:
    from ai_services.path_optimization.allocation_service import allocation_service
except ImportError:
    print("⚠️ Warning: allocation_service not available")
    allocation_service = None

router = APIRouter()

@router.post("/predict-location")
async def predict_optimal_location(
    request_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Real-time ML prediction for optimal storage location.
    Called when picker clicks "Store" button.
    """
    try:
        # Get required collections
        seasonal_collection = get_collection("seasonal_demand")
        storage_collection = get_collection("storage_history")
        location_inventory_collection = get_collection("location_inventory")
        inventory_collection = get_collection("inventory")
        
        # Extract request data
        item_id = request_data.get("itemID")
        item_name = request_data.get("item_name")
        category = request_data.get("category", "General")
        size = request_data.get("size", "M")
        quantity = request_data.get("quantity", 1)
        
        print(f"🤖 Real-time ML prediction request for item {item_id}")
        
        if not item_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="itemID is required"
            )
        
        # Get item details from inventory if not provided
        if not item_name or category == "General":
            inventory_item = inventory_collection.find_one({"itemID": item_id})
            if inventory_item:
                item_name = item_name or inventory_item.get("item_name", f"Item {item_id}")
                category = inventory_item.get("category", category)
                size = inventory_item.get("size", size)
        
        # Check if allocation service is available
        if not allocation_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML allocation service is not available"
            )
        
        
        allocation_result = await allocation_service.allocate_location_for_item(
            item_id=item_id,
            category=category,
            item_size=size,
            quantity=quantity,
            db_collection_seasonal=seasonal_collection,
            db_collection_storage=storage_collection,
            db_collection_location_inventory=location_inventory_collection  # Uses current availability
        )
        print("🔍 Allocation Result:", allocation_result)

        if not allocation_result['success']:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"No available locations: {allocation_result.get('error', 'Unknown error')}"
            )
        
        # Log successful prediction
        print(f"✅ ML predicted {allocation_result['allocated_location']} for item {item_id}")
        print(f"   Confidence: {allocation_result['confidence']:.2f}")
        print(f"   Reason: {allocation_result['allocation_reason']}")
        print(f"   Available locations checked: {allocation_result.get('total_available_locations', 'unknown')}")
        
        return {
            "success": True,
            "allocated_location": allocation_result['allocated_location'],
            "coordinates": allocation_result['coordinates'],
            "confidence": allocation_result['confidence'],
            "allocation_reason": allocation_result['allocation_reason'],
            "predicted_rack_group": allocation_result['predicted_rack_group'],
            "total_available_locations": allocation_result.get('total_available_locations', 0),
            "item_id": item_id,
            "item_name": item_name,
            "category": category,
            "size": size,
            "quantity": quantity,
            "predicted_at": datetime.utcnow().isoformat(),
            "predicted_by": current_user.get("username", "system")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in ML prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ML prediction failed: {str(e)}"
        )

@router.get("/location-status")
async def get_location_availability_status(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get current warehouse location availability statistics.
    """
    try:
        location_inventory_collection = get_collection("location_inventory")
        
        # Get availability statistics
        total_locations = location_inventory_collection.count_documents({})
        available_locations = location_inventory_collection.count_documents({"available": True})
        occupied_locations = location_inventory_collection.count_documents({"available": False})
        
        # Get statistics by type
        stats_by_type = {}
        for location_type in ['B', 'P', 'D']:
            type_total = location_inventory_collection.count_documents({"type": location_type})
            type_available = location_inventory_collection.count_documents({
                "type": location_type, 
                "available": True
            })
            type_occupied = type_total - type_available
            
            stats_by_type[location_type] = {
                "total": type_total,
                "available": type_available,
                "occupied": type_occupied,
                "utilization_percentage": round((type_occupied / type_total * 100) if type_total > 0 else 0, 1)
            }
        
        return {
            "total_locations": total_locations,
            "available_locations": available_locations,
            "occupied_locations": occupied_locations,
            "utilization_percentage": round((occupied_locations / total_locations * 100) if total_locations > 0 else 0, 1),
            "by_type": stats_by_type,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error getting location status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get location status: {str(e)}"
        )