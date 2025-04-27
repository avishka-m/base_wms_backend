from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from .base import BaseDBModel, PyObjectId

# Packing item model (for items in a packing record)
class PackingItemBase(BaseModel):
    itemID: int = Field(..., description="ID of the item to pack")
    pickingID: int = Field(..., description="ID of the picking record")
    orderDetailID: int = Field(..., description="ID of the order detail")
    quantity: int = Field(..., description="Quantity to pack")

# Base Packing model
class PackingBase(BaseModel):
    orderID: int = Field(..., description="ID of the order being packed")
    workerID: int = Field(..., description="ID of the packer")
    pack_date: datetime = Field(default_factory=datetime.utcnow, description="Date and time of packing")
    status: str = Field("pending", description="Status of the packing (pending, in_progress, completed)")
    is_partial: bool = Field(False, description="Whether this is a partial packing of the order")
    package_type: str = Field("box", description="Type of package (box, envelope, pallet, etc.)")

# Packing creation model
class PackingCreate(PackingBase):
    items: List[PackingItemBase] = Field(..., description="List of items to pack")

# Packing update model
class PackingUpdate(BaseModel):
    status: Optional[str] = None
    is_partial: Optional[bool] = None
    package_type: Optional[str] = None
    weight: Optional[float] = None
    dimensions: Optional[str] = None
    notes: Optional[str] = None

# Packing item in DB model
class PackingItemInDB(BaseDBModel, PackingItemBase):
    packed: bool = Field(False, description="Whether the item has been packed")
    actual_quantity: Optional[int] = Field(None, description="Actual quantity packed")
    pack_time: Optional[datetime] = Field(None, description="Time when the item was packed")
    notes: Optional[str] = Field(None, description="Notes about the packed item")

    class Config:
        populate_by_name = True

# Packing in DB model
class PackingInDB(BaseDBModel, PackingBase):
    packingID: int = Field(..., description="Unique packing ID")
    items: List[PackingItemInDB] = Field(..., description="List of items to pack")
    notes: Optional[str] = Field(None, description="General notes about the packing")
    start_time: Optional[datetime] = Field(None, description="Time when packing started")
    complete_time: Optional[datetime] = Field(None, description="Time when packing was completed")
    weight: Optional[float] = Field(None, description="Weight of the packed package in kg")
    dimensions: Optional[str] = Field(None, description="Dimensions of the package (LxWxH in cm)")
    label_printed: bool = Field(False, description="Whether the shipping label has been printed")
    
    @property
    def is_complete(self) -> bool:
        return all(item.packed for item in self.items)
    
    @property
    def items_count(self) -> int:
        return len(self.items)
    
    @property
    def total_quantity(self) -> int:
        return sum(item.quantity for item in self.items)
    
    @property
    def packed_quantity(self) -> int:
        return sum(item.actual_quantity or 0 for item in self.items if item.packed)
    
    @property
    def progress_percentage(self) -> float:
        if self.items_count == 0:
            return 0
        return (sum(1 for item in self.items if item.packed) / self.items_count) * 100

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "orderID": 1,
                "workerID": 3,
                "pack_date": "2023-01-01T14:00:00",
                "status": "pending",
                "is_partial": False,
                "package_type": "box",
                "packingID": 1,
                "items": [
                    {
                        "itemID": 1,
                        "pickingID": 1,
                        "orderDetailID": 1,
                        "quantity": 5,
                        "packed": False,
                        "actual_quantity": None,
                        "pack_time": None,
                        "notes": None
                    }
                ],
                "notes": "Fragile items, handle with care",
                "start_time": None,
                "complete_time": None,
                "weight": None,
                "dimensions": None,
                "label_printed": False
            }
        }

# Packing response model
class PackingResponse(BaseDBModel, PackingBase):
    packingID: int = Field(..., description="Unique packing ID")
    items: List[PackingItemInDB] = Field(..., description="List of items to pack")
    notes: Optional[str] = Field(None, description="General notes about the packing")
    start_time: Optional[datetime] = Field(None, description="Time when packing started")
    complete_time: Optional[datetime] = Field(None, description="Time when packing was completed")
    weight: Optional[float] = Field(None, description="Weight of the packed package in kg")
    dimensions: Optional[str] = Field(None, description="Dimensions of the package (LxWxH in cm)")
    label_printed: bool = Field(False, description="Whether the shipping label has been printed")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "orderID": 1,
                "workerID": 3,
                "pack_date": "2023-01-01T14:00:00",
                "status": "pending",
                "is_partial": False,
                "package_type": "box",
                "packingID": 1,
                "items": [
                    {
                        "id": "507f1f77bcf86cd799439012",
                        "itemID": 1,
                        "pickingID": 1,
                        "orderDetailID": 1,
                        "quantity": 5,
                        "packed": False,
                        "actual_quantity": None,
                        "pack_time": None,
                        "notes": None,
                        "created_at": "2023-01-01T00:00:00",
                        "updated_at": "2023-01-01T00:00:00"
                    }
                ],
                "notes": "Fragile items, handle with care",
                "start_time": None,
                "complete_time": None,
                "weight": None,
                "dimensions": None,
                "label_printed": False,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }