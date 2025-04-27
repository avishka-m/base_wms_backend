from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId

# Custom Pydantic model for ObjectId handling
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError(f"Invalid ObjectId: {v}")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler): # Updated signature
        # For ObjectId, representing it as a string in JSON schema is standard.
        return {"type": "string"}

# Base model with common fields
class BaseDBModel(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        arbitrary_types_allowed = True

    # Method to create a DB document from a Pydantic model
    def to_mongo(self) -> Dict[str, Any]:
        data = self.model_dump(by_alias=True, exclude=["id"])
        if "_id" in data:
            data["_id"] = data["_id"]
        return data

    # Method to convert MongoDB document to response model
    @classmethod
    def from_mongo(cls, data: Dict[str, Any]):
        if not data:
            return None
        if "_id" in data:
            data["id"] = data["_id"]
        return cls(**data)
