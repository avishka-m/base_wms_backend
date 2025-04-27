from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr

from .base import BaseDBModel, PyObjectId

# Base Worker model
class WorkerBase(BaseModel):
    name: str = Field(..., description="Worker's full name")
    role: str = Field(..., description="Worker's role (Manager, ReceivingClerk, Picker, Packer, Driver)")
    email: Optional[str] = Field(None, description="Worker's email address")
    phone: Optional[str] = Field(None, description="Worker's phone number")

# Worker creation model
class WorkerCreate(WorkerBase):
    username: str = Field(..., description="Username for login")
    password: str = Field(..., description="Password for login, will be hashed")

# Worker update model
class WorkerUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    disabled: Optional[bool] = None
    password: Optional[str] = None

# Worker in DB model
class WorkerInDB(BaseDBModel, WorkerBase):
    workerID: int = Field(..., description="Unique worker ID")
    username: str = Field(..., description="Username for login")
    hashed_password: str = Field(..., description="Hashed password")
    disabled: bool = Field(False, description="Disabled status")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "role": "Manager",
                "email": "john.doe@example.com",
                "phone": "+1234567890",
                "username": "johndoe",
                "disabled": False,
                "workerID": 1
            }
        }

# Worker response model
class WorkerResponse(BaseDBModel, WorkerBase):
    workerID: int = Field(..., description="Unique worker ID")
    username: str = Field(..., description="Username for login")
    disabled: bool = Field(False, description="Disabled status")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "name": "John Doe",
                "role": "Manager",
                "email": "john.doe@example.com",
                "phone": "+1234567890", 
                "username": "johndoe",
                "disabled": False,
                "workerID": 1,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }