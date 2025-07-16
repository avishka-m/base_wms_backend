from typing import Optional, List
from pydantic import BaseModel, Field

from .base import BaseDBModel, PyObjectId

# Base Warehouse model
class WarehouseBase(BaseModel):
    name: str = Field(..., description="Warehouse name")
    location: str = Field(..., description="Warehouse location/address")
    capacity: int = Field(..., description="Total storage capacity")

# Warehouse creation model
class WarehouseCreate(WarehouseBase):
    pass

# Warehouse update model
class WarehouseUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    capacity: Optional[int] = None

# Warehouse in DB model
class WarehouseInDB(BaseDBModel, WarehouseBase):
    warehouseID: int = Field(..., description="Unique warehouse ID")
    available_storage: int = Field(..., description="Available storage capacity")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "name": "Main Warehouse",
                "location": "123 Storage Ave, Warehouse City",
                "capacity": 10000,
                "warehouseID": 1,
                "available_storage": 8002
            }
        }

# Warehouse response model
class WarehouseResponse(BaseDBModel, WarehouseBase):
    warehouseID: int = Field(..., description="Unique warehouse ID")
    available_storage: int = Field(..., description="Available storage capacity")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "name": "Main Warehouse",
                "location": "123 Storage Ave, Warehouse City",
                "capacity": 10000,
                "warehouseID": 1,
                "available_storage": 8002,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }