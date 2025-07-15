from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import List, Dict, Any, Optional

from ..auth.dependencies import get_current_active_user, has_role
from ..models.inventory import InventoryCreate, InventoryUpdate, InventoryResponse
from ..services.inventory_service import InventoryService

router = APIRouter()

# OPTIMIZED: Get all inventory items with proper pagination
@router.get("/", response_model=Dict[str, Any])
async def get_inventory_items(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = Query(0, description="Number of items to skip", ge=0),
    limit: int = Query(50, description="Number of items to return (max 100)", ge=1, le=100),
    category: Optional[str] = Query(None, description="Filter by category"),
    low_stock: bool = Query(False, description="Filter low stock items"),
    search: Optional[str] = Query(None, description="Search term for items")
) -> Dict[str, Any]:
    """
    Get inventory items with pagination and filtering.
    
    OPTIMIZED FEATURES:
    - Pagination with metadata (current page, total pages, etc.)
    - Reduced default limit (50 instead of 100) for better performance
    - Search functionality
    - Proper error handling
    """
    try:
        # Check if we have optimized inventory service available
        try:
            from optimized_inventory_service import OptimizedInventoryService
            # Use optimized service with full pagination metadata
            result = await OptimizedInventoryService.get_inventory_items_paginated(
                skip=skip,
                limit=limit,
                category=category,
                low_stock=low_stock,
                search=search
            )
            return result
        except ImportError:
            # Fallback to standard service
            items = await InventoryService.get_inventory_items(
                skip=skip,
                limit=limit,
                category=category,
                low_stock=low_stock
            )
            
            # Add basic pagination metadata
            total_items = len(items) if len(items) < limit else skip + len(items) + 1  # Approximate
            return {
                "items": items,
                "pagination": {
                    "current_page": (skip // limit) + 1,
                    "items_per_page": limit,
                    "total_items": total_items,
                    "has_next": len(items) == limit,
                    "has_prev": skip > 0
                }
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching inventory items: {str(e)}"
        )

# OPTIMIZED: Get low stock items with pagination
@router.get("/low-stock", response_model=Dict[str, Any])
async def get_low_stock_items(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = Query(0, description="Number of items to skip", ge=0),
    limit: int = Query(50, description="Number of items to return", ge=1, le=100)
) -> Dict[str, Any]:
    """
    Get low stock items with pagination.
    
    OPTIMIZED: Now includes pagination metadata.
    """
    try:
        # Try optimized service first
        try:
            from optimized_inventory_service import OptimizedInventoryService
            result = await OptimizedInventoryService.get_low_stock_items_optimized(
                limit=limit, 
                skip=skip
            )
            return {
                "items": result["items"],
                "pagination": {
                    "current_page": (skip // limit) + 1,
                    "items_per_page": limit,
                    "total_items": result["total_count"],
                    "showing": result["showing"]
                }
            }
        except ImportError:
            # Fallback to standard service
            items = await InventoryService.get_low_stock_items()
            
            # Apply pagination manually for fallback
            paginated_items = items[skip:skip+limit]
            return {
                "items": paginated_items,
                "pagination": {
                    "current_page": (skip // limit) + 1,
                    "items_per_page": limit,
                    "total_items": len(items),
                    "showing": len(paginated_items)
                }
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching low stock items: {str(e)}"
        )

# Get inventory item by ID
@router.get("/{item_id}", response_model=InventoryResponse)
async def get_inventory_item(
    item_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific inventory item by ID.
    """
    item = await InventoryService.get_inventory_item(item_id)
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inventory item with ID {item_id} not found"
        )
    
    return item

# Create new inventory item
@router.post("/", response_model=InventoryResponse, status_code=status.HTTP_201_CREATED)
async def create_inventory_item(
    item: InventoryCreate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Create a new inventory item.
    
    Only managers and receiving clerks can create new inventory items.
    """
    created_item = await InventoryService.create_inventory_item(item)
    
    if "error" in created_item:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=created_item["error"]
        )
    
    return created_item

# Update inventory item
@router.put("/{item_id}", response_model=InventoryResponse)
async def update_inventory_item(
    item_id: int,
    item_update: InventoryUpdate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Update an inventory item.
    
    Only managers and receiving clerks can update inventory items.
    """
    # Check if item exists
    item = await InventoryService.get_inventory_item(item_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inventory item with ID {item_id} not found"
        )
    
    updated_item = await InventoryService.update_inventory_item(item_id, item_update)
    return updated_item

# Delete inventory item
@router.delete("/{item_id}", response_model=Dict[str, Any])
async def delete_inventory_item(
    item_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Delete an inventory item.
    
    Only managers can delete inventory items.
    """
    # Check if item exists
    item = await InventoryService.get_inventory_item(item_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inventory item with ID {item_id} not found"
        )
    
    result = await InventoryService.delete_inventory_item(item_id)
    return result

# Update stock level
@router.post("/{item_id}/stock", response_model=InventoryResponse)
async def update_stock_level(
    item_id: int,
    quantity_change: int = Query(..., description="Amount to change (positive for increase, negative for decrease)"),
    reason: str = Query(..., description="Reason for the stock change"),
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk", "Picker"]))
) -> Dict[str, Any]:
    """
    Update the stock level of an inventory item.
    
    This endpoint allows increasing or decreasing the stock level of an item.
    A positive quantity_change increases stock, a negative one decreases it.
    """
    result = await InventoryService.update_stock_level(item_id, quantity_change, reason)
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Transfer inventory
@router.post("/transfer", response_model=Dict[str, Any])
async def transfer_inventory(
    item_id: int = Query(..., description="ID of the item to transfer"),
    source_location_id: int = Query(..., description="ID of the source location"),
    destination_location_id: int = Query(..., description="ID of the destination location"),
    quantity: int = Query(..., description="Quantity to transfer"),
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk", "Picker"]))
) -> Dict[str, Any]:
    """
    Transfer inventory from one location to another.
    
    This endpoint allows moving inventory between warehouse locations.
    """
    result = await InventoryService.transfer_inventory(
        item_id=item_id,
        source_location_id=source_location_id,
        destination_location_id=destination_location_id,
        quantity=quantity
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result

# Check inventory anomalies
@router.get("/anomalies", response_model=List[Dict[str, Any]])
async def check_inventory_anomalies(
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> List[Dict[str, Any]]:
    """
    Check for inventory anomalies such as negative stock or stock above maximum level.
    
    Only managers can check for anomalies.
    """
    anomalies = await InventoryService.check_inventory_anomalies()
    return anomalies

# OPTIMIZED: New batch operations endpoint
@router.post("/batch-update", response_model=Dict[str, Any])
async def batch_update_inventory(
    updates: List[Dict[str, Any]],
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Update multiple inventory items in a single batch operation.
    
    OPTIMIZED: Reduces N+1 query problems by processing multiple updates at once.
    
    Expected format for updates:
    [
        {"item_id": 1, "quantity_change": 10, "reason": "Stock replenishment"},
        {"item_id": 2, "quantity_change": -5, "reason": "Damage adjustment"}
    ]
    """
    try:
        # Try optimized batch service first
        try:
            from optimized_inventory_service import OptimizedInventoryService
            result = await OptimizedInventoryService.batch_update_stock_levels(updates)
            return result
        except ImportError:
            # Fallback: Process individually (less efficient)
            results = []
            errors = []
            
            for update in updates:
                try:
                    item_id = update["item_id"]
                    quantity_change = update["quantity_change"]
                    reason = update["reason"]
                    
                    result = await InventoryService.update_stock_level(
                        item_id, quantity_change, reason
                    )
                    
                    if "error" in result:
                        errors.append(f"Item {item_id}: {result['error']}")
                    else:
                        results.append(item_id)
                        
                except Exception as e:
                    errors.append(f"Item {update.get('item_id', 'unknown')}: {str(e)}")
            
            return {
                "success": len(errors) == 0,
                "updated_count": len(results),
                "items_updated": results,
                "errors": errors
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch update: {str(e)}"
        )