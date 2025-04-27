from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from .base import BaseDBModel, PyObjectId

# Order status enum (as string literals for simplicity)
# "pending", "processing", "picking", "packing", "ready_for_shipping", "shipped", "delivered", "returned", "cancelled"

# Order detail base model
class OrderDetailBase(BaseModel):
    itemID: int = Field(..., description="ID of the ordered item")
    quantity: int = Field(..., description="Quantity ordered")
    price: float = Field(..., description="Price per unit")

# Order detail creation model
class OrderDetailCreate(OrderDetailBase):
    pass

# Order detail in DB model
class OrderDetailInDB(BaseDBModel, OrderDetailBase):
    orderDetailID: int = Field(..., description="Unique order detail ID")
    fulfilled_quantity: int = Field(0, description="Quantity that has been fulfilled")
    
    @property
    def is_fulfilled(self) -> bool:
        return self.fulfilled_quantity >= self.quantity
    
    @property
    def total_price(self) -> float:
        return self.price * self.quantity

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "itemID": 1,
                "quantity": 5,
                "price": 9.99,
                "orderDetailID": 1,
                "fulfilled_quantity": 0
            }
        }

# Base Order model
class OrderBase(BaseModel):
    customerID: int = Field(..., description="ID of the customer who placed the order")
    order_date: datetime = Field(default_factory=datetime.utcnow, description="Date and time when the order was placed")
    shipping_address: str = Field(..., description="Shipping address")
    order_status: str = Field("pending", description="Current status of the order")
    priority: int = Field(3, description="Order priority (1-high, 2-medium, 3-low)")
    notes: Optional[str] = Field(None, description="Additional order notes")

# Order creation model
class OrderCreate(OrderBase):
    items: List[OrderDetailCreate] = Field(..., description="List of items in the order")

# Order update model
class OrderUpdate(BaseModel):
    order_status: Optional[str] = None
    shipping_address: Optional[str] = None
    priority: Optional[int] = None
    notes: Optional[str] = None

# Order in DB model
class OrderInDB(BaseDBModel, OrderBase):
    orderID: int = Field(..., description="Unique order ID")
    items: List[OrderDetailInDB] = Field([], description="List of items in the order")
    total_amount: float = Field(0, description="Total order amount")
    assigned_worker: Optional[int] = Field(None, description="ID of the worker assigned to this order")
    
    # Helper methods
    @property
    def is_fulfilled(self) -> bool:
        return all(item.is_fulfilled for item in self.items)
    
    @property
    def items_count(self) -> int:
        return len(self.items)
    
    @property
    def items_total_quantity(self) -> int:
        return sum(item.quantity for item in self.items)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "customerID": 1,
                "order_date": "2023-01-01T00:00:00",
                "shipping_address": "123 Main St, Anytown, USA",
                "order_status": "pending",
                "priority": 3,
                "notes": "Please leave at the front door",
                "orderID": 1,
                "items": [
                    {
                        "itemID": 1,
                        "quantity": 5,
                        "price": 9.99,
                        "orderDetailID": 1,
                        "fulfilled_quantity": 0
                    }
                ],
                "total_amount": 49.95,
                "assigned_worker": 1
            }
        }

# Order response model
class OrderResponse(BaseDBModel, OrderBase):
    orderID: int = Field(..., description="Unique order ID")
    items: List[OrderDetailInDB] = Field([], description="List of items in the order")
    total_amount: float = Field(0, description="Total order amount")
    assigned_worker: Optional[int] = Field(None, description="ID of the worker assigned to this order")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "customerID": 1,
                "order_date": "2023-01-01T00:00:00",
                "shipping_address": "123 Main St, Anytown, USA",
                "order_status": "pending", 
                "priority": 3,
                "notes": "Please leave at the front door",
                "orderID": 1,
                "items": [
                    {
                        "itemID": 1,
                        "quantity": 5,
                        "price": 9.99,
                        "orderDetailID": 1,
                        "fulfilled_quantity": 0
                    }
                ],
                "total_amount": 49.95,
                "assigned_worker": 1,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }