from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from .base import BaseDBModel, PyObjectId

# Returns item model (for items in a return)
class ReturnsItemBase(BaseModel):
    itemID: int = Field(..., description="ID of the returned item")
    orderDetailID: int = Field(..., description="ID of the original order detail")
    quantity: int = Field(..., description="Quantity returned")
    reason: str = Field(..., description="Reason for return")
    condition: str = Field(..., description="Condition of returned item (new, damaged, used, etc.)")

# Base Returns model
class ReturnsBase(BaseModel):
    orderID: int = Field(..., description="ID of the original order")
    customerID: int = Field(..., description="ID of the customer")
    workerID: int = Field(..., description="ID of the receiving clerk handling the return")
    return_date: datetime = Field(default_factory=datetime.utcnow, description="Date and time of return")
    status: str = Field("pending", description="Status of the return (pending, processing, completed)")
    return_method: str = Field("customer_drop_off", description="How the return was received")
    
# Returns creation model
class ReturnsCreate(ReturnsBase):
    items: List[ReturnsItemBase] = Field(..., description="List of items returned")

# Returns update model
class ReturnsUpdate(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None
    refund_amount: Optional[float] = None
    refund_status: Optional[str] = None

# Returns item in DB model
class ReturnsItemInDB(BaseDBModel, ReturnsItemBase):
    processed: bool = Field(False, description="Whether the item has been processed")
    resellable: bool = Field(False, description="Whether the item can be resold")
    locationID: Optional[int] = Field(None, description="ID of the storage location for resellable items")
    notes: Optional[str] = Field(None, description="Notes about the returned item")

    class Config:
        populate_by_name = True

# Returns in DB model
class ReturnsInDB(BaseDBModel, ReturnsBase):
    returnID: int = Field(..., description="Unique return ID")
    items: List[ReturnsItemInDB] = Field(..., description="List of items returned")
    notes: Optional[str] = Field(None, description="General notes about the return")
    refund_amount: Optional[float] = Field(None, description="Amount refunded to customer")
    refund_status: Optional[str] = Field(None, description="Status of the refund (pending, processed, denied)")
    refund_date: Optional[datetime] = Field(None, description="Date and time of refund processing")
    
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
    def resellable_quantity(self) -> int:
        return sum(item.quantity for item in self.items if item.resellable)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "orderID": 1,
                "customerID": 1,
                "workerID": 1,
                "return_date": "2023-01-10T09:00:00",
                "status": "pending",
                "return_method": "customer_drop_off",
                "returnID": 1,
                "items": [
                    {
                        "itemID": 1,
                        "orderDetailID": 1,
                        "quantity": 2,
                        "reason": "Wrong size",
                        "condition": "new",
                        "processed": False,
                        "resellable": False,
                        "locationID": None,
                        "notes": None
                    }
                ],
                "notes": "Customer would like exchange for larger size",
                "refund_amount": None,
                "refund_status": None,
                "refund_date": None
            }
        }

# Returns response model
class ReturnsResponse(BaseDBModel, ReturnsBase):
    returnID: int = Field(..., description="Unique return ID")
    items: List[ReturnsItemInDB] = Field(..., description="List of items returned")
    notes: Optional[str] = Field(None, description="General notes about the return")
    refund_amount: Optional[float] = Field(None, description="Amount refunded to customer")
    refund_status: Optional[str] = Field(None, description="Status of the refund (pending, processed, denied)")
    refund_date: Optional[datetime] = Field(None, description="Date and time of refund processing")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "orderID": 1,
                "customerID": 1,
                "workerID": 1,
                "return_date": "2023-01-10T09:00:00",
                "status": "pending",
                "return_method": "customer_drop_off",
                "returnID": 1,
                "items": [
                    {
                        "id": "507f1f77bcf86cd799439012",
                        "itemID": 1,
                        "orderDetailID": 1,
                        "quantity": 2,
                        "reason": "Wrong size",
                        "condition": "new",
                        "processed": False,
                        "resellable": False,
                        "locationID": None,
                        "notes": None,
                        "created_at": "2023-01-10T09:00:00",
                        "updated_at": "2023-01-10T09:00:00"
                    }
                ],
                "notes": "Customer would like exchange for larger size",
                "refund_amount": None,
                "refund_status": None,
                "refund_date": None,
                "created_at": "2023-01-10T09:00:00",
                "updated_at": "2023-01-10T09:00:00"
            }
        }