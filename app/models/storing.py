from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from .base import BaseDBModel, PyObjectId

# Storing item model (for items in a storing record)
class StoringItemBase(BaseModel):
    itemID: int = Field(..., description="ID of the item to store")
    returnID: int = Field(..., description="ID of the return record")
    locationID: str = Field(..., description="Storage location code (e.g., B01.1, D02.2)")
    quantity: int = Field(..., description="Quantity to store")
    reason: str = Field(..., description="Reason for storing (returned, restocked, etc.)")

# Base Storing model
class StoringBase(BaseModel):
    assignedWorkerID: int = Field(..., description="ID of the picker assigned to store items")
    priority: int = Field(3, description="Storing priority (1-high, 2-medium, 3-low)")
    status: str = Field("pending", description="Status of the storing job (pending, in_progress, completed)")
    task_type: str = Field("return_putaway", description="Type of storing task")

# Storing creation model
class StoringCreate(StoringBase):
    items: List[StoringItemBase] = Field(..., description="List of items to store")

# Storing update model
class StoringUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[int] = None
    notes: Optional[str] = None
    assignedWorkerID: Optional[int] = None

# Storing item in DB model
class StoringItemInDB(BaseDBModel, StoringItemBase):
    stored: bool = Field(False, description="Whether the item has been stored")
    actual_quantity: Optional[int] = Field(None, description="Actual quantity stored")
    store_time: Optional[datetime] = Field(None, description="Time when the item was stored")
    notes: Optional[str] = Field(None, description="Notes about the stored item")

    class Config:
        populate_by_name = True

# Storing in DB model
class StoringInDB(BaseDBModel, StoringBase):
    storingID: int = Field(..., description="Unique storing job ID")
    items: List[StoringItemInDB] = Field(..., description="List of items to store")
    notes: Optional[str] = Field(None, description="General notes about the storing job")
    created_date: datetime = Field(default_factory=datetime.utcnow, description="Date when the job was created")
    start_time: Optional[datetime] = Field(None, description="Time when storing started")
    complete_time: Optional[datetime] = Field(None, description="Time when storing was completed")
    
    @property
    def total_items(self) -> int:
        """Total number of items in this storing job"""
        return len(self.items)
    
    @property
    def total_quantity(self) -> int:
        """Total quantity across all items"""
        return sum(item.quantity for item in self.items)
    
    @property
    def stored_items(self) -> int:
        """Number of items that have been stored"""
        return sum(1 for item in self.items if item.stored)
    
    @property
    def progress_percentage(self) -> float:
        """Progress percentage of the storing job"""
        if not self.items:
            return 0.0
        return (self.stored_items / len(self.items)) * 100

    class Config:
        populate_by_name = True

# Response model for API
class StoringResponse(BaseModel):
    id: str = Field(..., alias="_id")
    storingID: int
    assignedWorkerID: int
    priority: int
    status: str
    task_type: str
    items: List[StoringItemInDB]
    notes: Optional[str] = None
    created_date: datetime
    start_time: Optional[datetime] = None
    complete_time: Optional[datetime] = None
    total_items: int
    total_quantity: int
    stored_items: int
    progress_percentage: float
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True
        from_attributes = True
