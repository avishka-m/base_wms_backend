from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from .base import BaseDBModel, PyObjectId

# Base Shipping model
class ShippingBase(BaseModel):
    orderID: int = Field(..., description="ID of the order being shipped")
    workerID: int = Field(..., description="ID of the driver/shipper")
    ship_date: datetime = Field(default_factory=datetime.utcnow, description="Date and time of shipping")
    status: str = Field("pending", description="Status of the shipping (pending, in_transit, delivered)")
    shipping_method: str = Field("standard", description="Shipping method (standard, express, same_day, etc.)")
    tracking_number: Optional[str] = Field(None, description="Tracking number")
    estimated_delivery: Optional[datetime] = Field(None, description="Estimated delivery date and time")

# Shipping creation model
class ShippingCreate(ShippingBase):
    packingIDs: List[int] = Field(..., description="IDs of packing records to ship")
    vehicleID: Optional[int] = Field(None, description="ID of the vehicle used for shipping")

# Shipping update model
class ShippingUpdate(BaseModel):
    status: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[datetime] = None
    actual_delivery: Optional[datetime] = None
    vehicleID: Optional[int] = None
    notes: Optional[str] = None
    delivery_proof: Optional[str] = None

# Shipping in DB model
class ShippingInDB(BaseDBModel, ShippingBase):
    shippingID: int = Field(..., description="Unique shipping ID")
    packingIDs: List[int] = Field(..., description="IDs of packing records included in this shipment")
    vehicleID: Optional[int] = Field(None, description="ID of the vehicle used for shipping")
    departure_time: Optional[datetime] = Field(None, description="Time when shipment left the warehouse")
    actual_delivery: Optional[datetime] = Field(None, description="Actual delivery date and time")
    notes: Optional[str] = Field(None, description="General notes about the shipping")
    delivery_proof: Optional[str] = Field(None, description="Proof of delivery (signature, photo, etc.)")
    delivery_address: str = Field(..., description="Delivery address")
    recipient_name: str = Field(..., description="Name of the recipient")
    recipient_phone: Optional[str] = Field(None, description="Phone number of the recipient")
    
    @property
    def is_delivered(self) -> bool:
        return self.status == "delivered" and self.actual_delivery is not None
    
    @property
    def delivery_time(self) -> Optional[float]:
        """Returns the delivery time in hours if delivered"""
        if not self.is_delivered or not self.departure_time:
            return None
        delta = self.actual_delivery - self.departure_time
        return delta.total_seconds() / 3600  # Convert to hours

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "orderID": 1,
                "workerID": 4,
                "ship_date": "2023-01-02T09:00:00",
                "status": "pending",
                "shipping_method": "standard",
                "tracking_number": "TRACK-1234567890",
                "estimated_delivery": "2023-01-03T15:00:00",
                "shippingID": 1,
                "packingIDs": [1],
                "vehicleID": 1,
                "departure_time": None,
                "actual_delivery": None,
                "notes": "Leave at reception if no one answers",
                "delivery_proof": None,
                "delivery_address": "123 Main St, Anytown, USA",
                "recipient_name": "Jane Smith",
                "recipient_phone": "+1234567890"
            }
        }

# Shipping response model
class ShippingResponse(BaseDBModel, ShippingBase):
    shippingID: int = Field(..., description="Unique shipping ID")
    packingIDs: List[int] = Field(..., description="IDs of packing records included in this shipment")
    vehicleID: Optional[int] = Field(None, description="ID of the vehicle used for shipping")
    departure_time: Optional[datetime] = Field(None, description="Time when shipment left the warehouse")
    actual_delivery: Optional[datetime] = Field(None, description="Actual delivery date and time")
    notes: Optional[str] = Field(None, description="General notes about the shipping")
    delivery_proof: Optional[str] = Field(None, description="Proof of delivery (signature, photo, etc.)")
    delivery_address: str = Field(..., description="Delivery address")
    recipient_name: str = Field(..., description="Name of the recipient")
    recipient_phone: Optional[str] = Field(None, description="Phone number of the recipient")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "orderID": 1,
                "workerID": 4,
                "ship_date": "2023-01-02T09:00:00",
                "status": "pending",
                "shipping_method": "standard",
                "tracking_number": "TRACK-1234567890",
                "estimated_delivery": "2023-01-03T15:00:00",
                "shippingID": 1,
                "packingIDs": [1],
                "vehicleID": 1,
                "departure_time": None,
                "actual_delivery": None,
                "notes": "Leave at reception if no one answers",
                "delivery_proof": None,
                "delivery_address": "123 Main St, Anytown, USA",
                "recipient_name": "Jane Smith",
                "recipient_phone": "+1234567890",
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }