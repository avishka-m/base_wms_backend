from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from .base import BaseDBModel, PyObjectId

# Receiving item model (for items in a receiving record)
class ReceivingItemBase(BaseModel):
    itemID: int = Field(..., description="ID of the received item")
    quantity: int = Field(..., description="Quantity received")
    expected_quantity: int = Field(..., description="Expected quantity")
    condition: str = Field("good", description="Condition of received items (good, damaged, etc.)")
    notes: Optional[str] = Field(None, description="Notes about the received item")

# Base Receiving model
class ReceivingBase(BaseModel):
    supplierID: int = Field(..., description="ID of the supplier")
    workerID: int = Field(..., description="ID of the receiving clerk")
    received_date: datetime = Field(default_factory=datetime.utcnow, description="Date and time of receipt")
    status: str = Field("pending", description="Status of the receiving (pending, processing, completed)")
    reference_number: Optional[str] = Field(None, description="Supplier reference number")
    
# Receiving creation model
class ReceivingCreate(ReceivingBase):
    items: List[ReceivingItemBase] = Field(..., description="List of items received")

# Receiving update model
class ReceivingUpdate(BaseModel):
    status: Optional[str] = None
    reference_number: Optional[str] = None
    notes: Optional[str] = None

# Receiving item in DB model
class ReceivingItemInDB(BaseDBModel, ReceivingItemBase):
    processed: bool = Field(False, description="Whether the item has been processed")
    locationID: Optional[int] = Field(None, description="ID of the assigned storage location")

    class Config:
        populate_by_name = True

# Receiving in DB model
class ReceivingInDB(BaseDBModel, ReceivingBase):
    receivingID: int = Field(..., description="Unique receiving ID")
    items: List[ReceivingItemInDB] = Field(..., description="List of items received")
    notes: Optional[str] = Field(None, description="General notes about the receiving")
    
    @property
    def is_complete(self) -> bool:
        return all(item.processed for item in self.items)
    
    @property
    def items_count(self) -> int:
        return len(self.items)
    
    @property
    def total_quantity(self) -> int:
        return sum(item.quantity for item in self.items)
    
    @property
    def discrepancy(self) -> bool:
        return any(item.quantity != item.expected_quantity for item in self.items)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "supplierID": 1,
                "workerID": 1, 
                "received_date": "2023-01-01T00:00:00",
                "status": "pending",
                "reference_number": "PO-12345",
                "receivingID": 1,
                "items": [
                    {
                        "itemID": 1,
                        "quantity": 50,
                        "expected_quantity": 50,
                        "condition": "good",
                        "notes": None,
                        "processed": False,
                        "locationID": None
                    }
                ],
                "notes": "Delivery arrived on time"
            }
        }

# Receiving response model
class ReceivingResponse(BaseDBModel, ReceivingBase):
    receivingID: int = Field(..., description="Unique receiving ID")
    items: List[ReceivingItemInDB] = Field(..., description="List of items received")
    notes: Optional[str] = Field(None, description="General notes about the receiving")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "supplierID": 1,
                "workerID": 1,
                "received_date": "2023-01-01T00:00:00",
                "status": "pending",
                "reference_number": "PO-12345",
                "receivingID": 1,
                "items": [
                    {
                        "id": "507f1f77bcf86cd799439012",
                        "itemID": 1,
                        "quantity": 50,
                        "expected_quantity": 50,
                        "condition": "good",
                        "notes": None,
                        "processed": False,
                        "locationID": None,
                        "created_at": "2023-01-01T00:00:00",
                        "updated_at": "2023-01-01T00:00:00"
                    }
                ],
                "notes": "Delivery arrived on time",
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }