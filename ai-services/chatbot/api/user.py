from typing import Dict, Any
from fastapi import APIRouter, Depends

from dependencies.auth import get_optional_current_user, get_allowed_chatbot_roles

router = APIRouter(prefix="/user", tags=["user"])

@router.get("/role", response_model=Dict[str, Any])
async def get_user_role(current_user: Dict[str, Any] = Depends(get_optional_current_user)):
    """
    Get the current user's role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User role information
    """
    return {
        "username": current_user.get("username"),
        "role": current_user.get("role"),
        "allowed_chatbot_roles": get_allowed_chatbot_roles(current_user.get("role", ""))
    } 