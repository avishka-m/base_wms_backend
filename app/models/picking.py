from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from .base import BaseDBModel, PyObjectId

# Picking item model (for items in a picking record)
class PickingItemBase(BaseModel):
    itemID: int = Field(..., description="ID of the item to pick")
    orderDetailID: int = Field(..., description="ID of the order detail")
    locationID: int = Field(..., description="ID of the storage location")
    quantity: int = Field(..., description="Quantity to pick")

# Base Picking model
class PickingBase(BaseModel):
    orderID: int = Field(..., description="ID of the order being picked")
    workerID: int = Field(..., description="ID of the picker")
    pick_date: datetime = Field(default_factory=datetime.utcnow, description="Date and time of picking")
    status: str = Field("pending", description="Status of the picking (pending, in_progress, completed)")
    priority: int = Field(3, description="Picking priority (1-high, 2-medium, 3-low)")

# Picking creation model
class PickingCreate(PickingBase):
    items: List[PickingItemBase] = Field(..., description="List of items to pick")

# Picking update model
class PickingUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[int] = None
    notes: Optional[str] = None

# Picking item in DB model
class PickingItemInDB(BaseDBModel, PickingItemBase):
    picked: bool = Field(False, description="Whether the item has been picked")
    actual_quantity: Optional[int] = Field(None, description="Actual quantity picked")
    pick_time: Optional[datetime] = Field(None, description="Time when the item was picked")
    notes: Optional[str] = Field(None, description="Notes about the picked item")

    class Config:
        populate_by_name = True

# Picking in DB model
class PickingInDB(BaseDBModel, PickingBase):
    pickingID: int = Field(..., description="Unique picking ID")
    items: List[PickingItemInDB] = Field(..., description="List of items to pick")
    notes: Optional[str] = Field(None, description="General notes about the picking")
    start_time: Optional[datetime] = Field(None, description="Time when picking started")
    complete_time: Optional[datetime] = Field(None, description="Time when picking was completed")
    
    @property
    def is_complete(self) -> bool:
        return all(item.picked for item in self.items)
    
    @property
    def items_count(self) -> int:
        return len(self.items)
    
    @property
    def total_quantity(self) -> int:
        return sum(item.quantity for item in self.items)
    
    @property
    def picked_quantity(self) -> int:
        return sum(item.actual_quantity or 0 for item in self.items if item.picked)
    
    @property
    def progress_percentage(self) -> float:
        if self.items_count == 0:
            return 0
        return (sum(1 for item in self.items if item.picked) / self.items_count) * 100

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "orderID": 1,
                "workerID": 2,
                "pick_date": "2023-01-01T00:00:00",
                "status": "pending",
                "priority": 2,
                "pickingID": 1,
                "items": [
                    {
                        "itemID": 1,
                        "orderDetailID": 1,
                        "locationID": 1, 
                        "quantity": 5,
                        "picked": False,
                        "actual_quantity": None,
                        "pick_time": None,
                        "notes": None
                    }
                ],
                "notes": "Customer requested ASAP delivery",
                "start_time": None,
                "complete_time": None
            }
        }

# Picking response model
class PickingResponse(BaseDBModel, PickingBase):
    pickingID: int = Field(..., description="Unique picking ID")
    items: List[PickingItemInDB] = Field(..., description="List of items to pick")
    notes: Optional[str] = Field(None, description="General notes about the picking")
    start_time: Optional[datetime] = Field(None, description="Time when picking started")
    complete_time: Optional[datetime] = Field(None, description="Time when picking was completed")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "orderID": 1,
                "workerID": 2,
                "pick_date": "2023-01-01T00:00:00",
                "status": "pending",
                "priority": 2,
                "pickingID": 1,
                "items": [
                    {
                        "id": "507f1f77bcf86cd799439012",
                        "itemID": 1,
                        "orderDetailID": 1,
                        "locationID": 1,
                        "quantity": 5,
                        "picked": False,
                        "actual_quantity": None,
                        "pick_time": None,
                        "notes": None,
                        "created_at": "2023-01-01T00:00:00",
                        "updated_at": "2023-01-01T00:00:00"
                    }
                ],
                "notes": "Customer requested ASAP delivery",
                "start_time": None,
                "complete_time": None,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }