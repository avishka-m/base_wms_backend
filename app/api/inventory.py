from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import List, Dict, Any, Optional

from ..auth.dependencies import get_current_active_user, has_role
from ..models.inventory import InventoryCreate, InventoryUpdate, InventoryResponse
from ..services.inventory_service import InventoryService

router = APIRouter()

# Get all inventory items
@router.get("/", response_model=List[InventoryResponse])
async def get_inventory_items(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    low_stock: bool = False
) -> List[Dict[str, Any]]:
    """
    Get all inventory items with optional filtering.
    
    You can filter by category and low stock status.
    """
    items = await InventoryService.get_inventory_items(
        skip=skip,
        limit=limit,
        category=category,
        low_stock=low_stock
    )
    return items

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

# Get low stock items
@router.get("/low-stock", response_model=List[InventoryResponse])
async def get_low_stock_items(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> List[Dict[str, Any]]:
    """
    Get items with stock levels at or below their minimum stock level.
    """
    items = await InventoryService.get_low_stock_items()
    return items

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

