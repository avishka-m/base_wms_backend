from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Any, Dict

from ..auth.dependencies import authenticate_user
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