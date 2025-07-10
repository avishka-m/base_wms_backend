from datetime import datetime, timedelta
from typing import Dict, Any

from .dependencies import create_access_token, get_password_hash
from ..config import ACCESS_TOKEN_EXPIRE_MINUTES

# Create token response
def create_token_response(user: Dict[str, Any]):
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Create new user
def create_new_user(user_data: Dict[str, Any]):
    # Hash the password
    hashed_password = get_password_hash(user_data["password"])
    
    # Create user document
    user_doc = {
        "username": user_data["username"],
        "email": user_data["email"],
        "name": user_data["name"],  # Use 'name' field consistently
        "role": user_data["role"],
        "hashed_password": hashed_password,
        "disabled": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    return user_doc