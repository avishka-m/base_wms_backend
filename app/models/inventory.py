from typing import Optional, List, Union
from pydantic import BaseModel, Field

from .base import BaseDBModel, PyObjectId

# Base Inventory model
class InventoryBase(BaseModel):
    name: str = Field(..., description="Item name")
    category: str = Field(..., description="Item category")
    size: str = Field(..., description="Item size (S, M, L, XL, etc.)")
    storage_type: str = Field(..., description="Storage type requirements (standard, refrigerated, hazardous, etc.)")
    stock_level: int = Field(0, description="Current stock level")
    min_stock_level: int = Field(0, description="Minimum stock level before reordering")
    max_stock_level: int = Field(0, description="Maximum stock level")
    supplierID: int = Field(..., description="ID of the supplier for this item")

# Inventory creation model
class InventoryCreate(InventoryBase):
    locationID: Optional[Union[str, int]] = Field(None, description="Storage location code (e.g., B01.1, D02.2, P03.1) or location ID")

# Inventory update model
class InventoryUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    size: Optional[str] = None
    storage_type: Optional[str] = None
    stock_level: Optional[int] = None
    min_stock_level: Optional[int] = None
    max_stock_level: Optional[int] = None
    supplierID: Optional[int] = None
    locationID: Optional[Union[str, int]] = None

# Inventory in DB model
class InventoryInDB(BaseDBModel, InventoryBase):
    itemID: int = Field(..., description="Unique item ID")
    locationID: Optional[Union[str, int]] = Field(None, description="Storage location code (e.g., B01.1, D02.2, P03.1) or location ID")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "name": "Widget",
                "category": "Electronics",
                "size": "M",
                "storage_type": "standard",
                "stock_level": 100,
                "min_stock_level": 20,
                "max_stock_level": 200,
                "supplierID": 1,
                "itemID": 1,
                "locationID": "B01.1"
            }
        }

# Inventory response model
class InventoryResponse(BaseDBModel, InventoryBase):
    itemID: int = Field(..., description="Unique item ID")
    locationID: Optional[Union[str, int]] = Field(None, description="Storage location code (e.g., B01.1, D02.2, P03.1) or location ID")
    
    # Helper method to check if stock is low
    @property
    def is_low_stock(self) -> bool:
        return self.stock_level <= self.min_stock_level
    
    # Helper method to check if stock is full
    @property
    def is_full_stock(self) -> bool:
        return self.stock_level >= self.max_stock_level

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "name": "Widget",
                "category": "Electronics",
                "size": "M",
                "storage_type": "standard",
                "stock_level": 100,
                "min_stock_level": 20,
                "max_stock_level": 200,
                "supplierID": 1,
                "itemID": 1,
                "locationID": 1,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }