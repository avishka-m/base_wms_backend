from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Any, Dict

from ..auth.dependencies import authenticate_user, get_current_active_user
from ..auth.utils import create_token_response

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
    
    return create_token_response(user)

@router.get("/me", response_model=Dict[str, Any])
async def get_current_user_profile(current_user: Dict[str, Any] = Depends(get_current_active_user)) -> Dict[str, Any]:
    """
    Get current user profile information.
    
    This endpoint returns the profile information for the authenticated user.
    """
    # Remove sensitive information before returning
    user_profile = {
        "id": current_user.get("workerID", current_user.get("_id")),
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "name": current_user.get("name"),
        "role": current_user.get("role"),
        "phone": current_user.get("phone"),
        "created_at": current_user.get("created_at"),
        "updated_at": current_user.get("updated_at")
    }
    
    return user_profile