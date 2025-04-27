from typing import Optional, List
from pydantic import BaseModel, Field

from .base import BaseDBModel, PyObjectId

# Base Customer model
class CustomerBase(BaseModel):
    name: str = Field(..., description="Customer name")
    email: str = Field(..., description="Customer email")
    phone: str = Field(..., description="Customer phone number")
    address: str = Field(..., description="Customer address")

# Customer creation model
class CustomerCreate(CustomerBase):
    pass

# Customer update model
class CustomerUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None

# Customer in DB model
class CustomerInDB(BaseDBModel, CustomerBase):
    customerID: int = Field(..., description="Unique customer ID")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "name": "Jane Smith",
                "email": "jane.smith@example.com",
                "phone": "+1234567890",
                "address": "123 Main St, Anytown, USA",
                "customerID": 1
            }
        }

# Customer response model
class CustomerResponse(BaseDBModel, CustomerBase):
    customerID: int = Field(..., description="Unique customer ID")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "name": "Jane Smith",
                "email": "jane.smith@example.com",
                "phone": "+1234567890",
                "address": "123 Main St, Anytown, USA",
                "customerID": 1,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }