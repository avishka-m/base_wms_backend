from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

from .base import BaseDBModel, PyObjectId

# Base Vehicle model
class VehicleBase(BaseModel):
    vehicle_type: str = Field(..., description="Type of vehicle (truck, van, etc.)")
    license_plate: str = Field(..., description="Vehicle license plate number")
    capacity: float = Field(..., description="Vehicle cargo capacity in kg")
    volume: float = Field(..., description="Vehicle cargo volume in cubic meters")
    model: str = Field(..., description="Vehicle model")
    year: int = Field(..., description="Vehicle manufacture year")

# Vehicle creation model
class VehicleCreate(VehicleBase):
    pass

# Vehicle update model
class VehicleUpdate(BaseModel):
    vehicle_type: Optional[str] = None
    license_plate: Optional[str] = None
    capacity: Optional[float] = None
    volume: Optional[float] = None
    model: Optional[str] = None
    year: Optional[int] = None
    status: Optional[str] = None
    last_maintenance_date: Optional[datetime] = None
    next_maintenance_date: Optional[datetime] = None

# Vehicle in DB model
class VehicleInDB(BaseDBModel, VehicleBase):
    vehicleID: int = Field(..., description="Unique vehicle ID")
    status: str = Field("available", description="Vehicle status (available, in_use, maintenance)")
    last_maintenance_date: Optional[datetime] = Field(None, description="Date of last maintenance")
    next_maintenance_date: Optional[datetime] = Field(None, description="Date of next scheduled maintenance")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "vehicle_type": "truck",
                "license_plate": "ABC-1234",
                "capacity": 2500.0,
                "volume": 20.0,
                "model": "ACME Truck 3000",
                "year": 2020,
                "vehicleID": 1,
                "status": "available",
                "last_maintenance_date": "2023-01-01T00:00:00",
                "next_maintenance_date": "2023-04-01T00:00:00"
            }
        }

# Vehicle response model
class VehicleResponse(BaseDBModel, VehicleBase):
    vehicleID: int = Field(..., description="Unique vehicle ID")
    status: str = Field("available", description="Vehicle status (available, in_use, maintenance)")
    last_maintenance_date: Optional[datetime] = Field(None, description="Date of last maintenance")
    next_maintenance_date: Optional[datetime] = Field(None, description="Date of next scheduled maintenance")
    
    @property
    def is_available(self) -> bool:
        return self.status == "available"
    
    @property
    def maintenance_due(self) -> bool:
        if not self.next_maintenance_date:
            return False
        return datetime.utcnow() >= self.next_maintenance_date

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "vehicle_type": "truck",
                "license_plate": "ABC-1234",
                "capacity": 2500.0,
                "volume": 20.0,
                "model": "ACME Truck 3000",
                "year": 2020,
                "vehicleID": 1,
                "status": "available",
                "last_maintenance_date": "2023-01-01T00:00:00",
                "next_maintenance_date": "2023-04-01T00:00:00",
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }