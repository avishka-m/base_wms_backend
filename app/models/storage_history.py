from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from bson import ObjectId


class StorageHistory(BaseModel):
    """Model for tracking storage history/logs"""
    id: Optional[str] = Field(None, alias="_id")
    itemID: str
    itemName: str
    quantity: int
    locationID: str  # e.g., "B1-12-F1"
    locationCoordinates: dict  # {"x": 3, "y": 4, "floor": 1}
    storedBy: str  # Worker ID/username
    storedAt: datetime = Field(default_factory=datetime.utcnow)
    action: str = "stored"  # "stored", "collected"
    category: Optional[str] = None
    condition: Optional[str] = None
    receivingID: Optional[str] = None

    class Config:
        populate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


class LocationOccupancy(BaseModel):
    """Model for tracking occupied locations in warehouse"""
    id: Optional[str] = Field(None, alias="_id")
    locationID: str  # e.g., "B1-12-F1"
    coordinates: dict  # {"x": 3, "y": 4, "floor": 1}
    occupied: bool = True
    itemID: Optional[str] = None
    itemName: Optional[str] = None
    quantity: Optional[int] = None
    category: Optional[str] = None
    storedAt: Optional[datetime] = None
    lastUpdated: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }