from typing import Optional, List
from pydantic import BaseModel, Field

from .base import BaseDBModel, PyObjectId

# Base StorageLocation model
class LocationBase(BaseModel):
    section: str = Field(..., description="Storage section identifier")
    row: str = Field(..., description="Row identifier within section")
    shelf: str = Field(..., description="Shelf identifier within row")
    bin: str = Field(..., description="Bin identifier within shelf")
    warehouseID: int = Field(..., description="ID of the warehouse this location belongs to")

# StorageLocation creation model
class LocationCreate(LocationBase):
    pass

# StorageLocation update model
class LocationUpdate(BaseModel):
    section: Optional[str] = None
    row: Optional[str] = None
    shelf: Optional[str] = None
    bin: Optional[str] = None
    warehouseID: Optional[int] = None
    is_occupied: Optional[bool] = None

# StorageLocation in DB model
class LocationInDB(BaseDBModel, LocationBase):
    locationID: int = Field(..., description="Unique location ID")
    is_occupied: bool = Field(False, description="Whether the location is currently occupied")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "section": "A",
                "row": "1",
                "shelf": "2",
                "bin": "3",
                "warehouseID": 1,
                "locationID": 1,
                "is_occupied": False
            }
        }

# StorageLocation response model
class LocationResponse(BaseDBModel, LocationBase):
    locationID: int = Field(..., description="Unique location ID")
    is_occupied: bool = Field(False, description="Whether the location is currently occupied")
    
    # Composite location identifier
    @property
    def location_code(self) -> str:
        return f"{self.section}-{self.row}-{self.shelf}-{self.bin}"

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "section": "A",
                "row": "1",
                "shelf": "2",
                "bin": "3",
                "warehouseID": 1,
                "locationID": 1,
                "is_occupied": False,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }