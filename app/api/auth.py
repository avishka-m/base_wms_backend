from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from typing import Any, Dict
from pydantic import BaseModel, EmailStr
import secrets
from datetime import datetime, timedelta

from ..auth.dependencies import authenticate_user, get_current_user, get_current_active_user, has_role, get_user, get_password_hash, verify_password
from ..auth.utils import create_token_response, create_new_user
from ..utils.database import get_collection

# Define request and response models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    role: str
    password: str

class PasswordReset(BaseModel):
    token: str
    password: str

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

class ForgotPassword(BaseModel):
    email: EmailStr

class UserResponse(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    role: str

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/token", response_model=Dict[str, Any])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()) -> Dict[str, Any]:
    """
    Get access token for authenticated user.
    
    This endpoint authenticates a user and returns a JWT token that can be used
    for authenticated requests.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is disabled
    if user.get("disabled", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create response with token and user data
    token_response = create_token_response(user)
    
    # Add user data to response
    user_data = {
        "username": user["username"],
        "email": user["email"],
        "full_name": user["name"],
        "role": user["role"]
    }
    token_response["user"] = user_data
    
    return token_response

@router.get("/me", response_model=UserResponse)
async def get_user_me(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """
    Get current user information.
    
    This endpoint returns the authenticated user's information.
    """
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "full_name": current_user["name"],
        "role": current_user["role"]
    }

@router.post("/register", response_model=Dict[str, Any])
async def register_user(
    user_data: UserCreate,
    current_user: Dict[str, Any] = Depends(has_role(["manager"]))
):
    """
    Register a new user.
    
    This endpoint allows managers to create new user accounts.
    """
    # Check if username already exists
    users_collection = get_collection("workers")
    existing_user = users_collection.find_one({"username": user_data.username})
    if (existing_user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = users_collection.find_one({"email": user_data.email})
    if (existing_email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user_dict = user_data.dict()
    new_user = create_new_user(user_dict)
    
    # Insert user to database
    result = users_collection.insert_one(new_user)
    
    return {"message": "User registered successfully", "user_id": str(result.inserted_id)}

@router.post("/forgot-password", response_model=Dict[str, Any])
async def forgot_password(email_data: ForgotPassword, background_tasks: BackgroundTasks):
    """
    Initiate password reset process.
    
    This endpoint generates a password reset token and emails it to the user.
    """
    users_collection = get_collection("workers")
    user = users_collection.find_one({"email": email_data.email})
    
    if not user:
        # Don't reveal that email doesn't exist for security reasons
        return {"message": "If your email is registered, you will receive a password reset link"}
    
    # Generate a reset token
    reset_token = secrets.token_urlsafe(32)
    token_expiry = datetime.utcnow() + timedelta(hours=24)
    
    # Store the token in the database
    users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {
            "reset_token": reset_token,
            "reset_token_expiry": token_expiry,
            "updated_at": datetime.utcnow()
        }}
    )
    
    # In a real application, you would send an email with the reset link
    # For this example, we'll just return a success message
    # background_tasks.add_task(send_reset_email, user["email"], reset_token)
    
    return {"message": "If your email is registered, you will receive a password reset link"}

@router.post("/reset-password", response_model=Dict[str, Any])
async def reset_password(reset_data: PasswordReset):
    """
    Reset password using a token.
    
    This endpoint allows users to reset their password using a reset token.
    """
    users_collection = get_collection("workers")
    user = users_collection.find_one({
        "reset_token": reset_data.token,
        "reset_token_expiry": {"$gt": datetime.utcnow()}
    })
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Update password
    hashed_password = get_password_hash(reset_data.password)
    
    users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {
            "hashed_password": hashed_password,
            "reset_token": None,
            "reset_token_expiry": None,
            "updated_at": datetime.utcnow()
        }}
    )
    
    return {"message": "Password has been reset successfully"}

@router.post("/change-password", response_model=Dict[str, Any])
async def change_password(
    password_data: PasswordChange,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Change user password.
    
    This endpoint allows authenticated users to change their password.
    """
    # Verify current password
    if not verify_password(password_data.current_password, current_user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Update password
    users_collection = get_collection("workers")
    hashed_password = get_password_hash(password_data.new_password)
    
    users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": {
            "hashed_password": hashed_password,
            "updated_at": datetime.utcnow()
        }}
    )
    
    return {"message": "Password changed successfully"}