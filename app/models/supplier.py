from typing import Optional, List
from pydantic import BaseModel, Field

from .base import BaseDBModel, PyObjectId

# Base Supplier model
class SupplierBase(BaseModel):
    name: str = Field(..., description="Supplier name")
    contact: str = Field(..., description="Supplier contact person name")
    email: str = Field(..., description="Supplier email")
    phone: str = Field(..., description="Supplier phone number")
    address: str = Field(..., description="Supplier address")

# Supplier creation model
class SupplierCreate(SupplierBase):
    pass

# Supplier update model
class SupplierUpdate(BaseModel):
    name: Optional[str] = None
    contact: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None

# Supplier in DB model
class SupplierInDB(BaseDBModel, SupplierBase):
    supplierID: int = Field(..., description="Unique supplier ID")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "name": "Acme Supplies",
                "contact": "John Acme",
                "email": "john@acmesupplies.com",
                "phone": "+1234567890",
                "address": "456 Supplier Row, Industry City, USA",
                "supplierID": 1
            }
        }

# Supplier response model
class SupplierResponse(BaseDBModel, SupplierBase):
    supplierID: int = Field(..., description="Unique supplier ID")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "name": "Acme Supplies",
                "contact": "John Acme",
                "email": "john@acmesupplies.com",
                "phone": "+1234567890",
                "address": "456 Supplier Row, Industry City, USA",
                "supplierID": 1,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }