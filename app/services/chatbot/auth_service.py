"""
Authentication service for the WMS Chatbot.
"""

import os
import sys
import logging
from typing import Dict, Any
from fastapi import HTTPException, status, Depends, Query
from fastapi.security import OAuth2PasswordBearer

# Add the parent directory to path to import from app modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # Import auth modules from main app
    from app.auth.dependencies import get_current_user
    from app.config import DEV_MODE, DEV_USER_ROLE
except ImportError:
    # Fallback for standalone testing
    DEV_MODE = True
    DEV_USER_ROLE = "Manager"
    
    async def get_current_user(token):
        return {"username": "test_user", "role": "Manager"}

logger = logging.getLogger("wms_chatbot")

# Set up OAuth2 scheme for token authentication
# Point to the main application's token URL
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="http://localhost:8002/api/v1/auth/token", 
    auto_error=False
)

# Create a development mock user for testing
dev_mock_user = {
    "username": "dev_user",
    "role": DEV_USER_ROLE,
    "email": "dev@example.com",
    "full_name": "Development User",
    "disabled": False
}

# Define an optional dependency for authentication that can be bypassed in dev mode
async def get_optional_current_user(
    dev_mode: bool = Query(False, description="Enable development mode (no auth)"),
    token: str = Depends(oauth2_scheme)
) -> Dict[str, Any]:
    """
    Get the current user with option to bypass in development mode.
    
    Args:
        dev_mode: Flag to enable development mode
        token: OAuth2 token
        
    Returns:
        Current user information
    """
    # If DEV_MODE is enabled globally or dev_mode query param is True, return mock user
    if DEV_MODE or dev_mode:
        logger.warning("Using development mode: Authentication bypassed!")
        return dev_mock_user
        
    # Otherwise use normal authentication
    return await get_current_user(token)

# Role-based access with development mode support
def optional_has_role(required_roles: list):
    """
    Check if user has required role with option to bypass in development mode.
    
    Args:
        required_roles: List of roles that are allowed
        
    Returns:
        Dependency function that checks role
    """
    async def check_role(
        dev_mode: bool = Query(False, description="Enable development mode (no auth)"),
        current_user: Dict[str, Any] = Depends(get_optional_current_user)
    ):
        # In dev mode, always allow if using the default dev user
        if (DEV_MODE or dev_mode) and current_user.get("username") == "dev_user":
            logger.warning(f"Development mode: Role check for {required_roles} bypassed!")
            return dev_mock_user
            
        # Otherwise check roles normally
        if current_user.get("role") not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have sufficient privileges. Required roles: {required_roles}"
            )
        return current_user
        
    return check_role

def get_allowed_chatbot_roles(user_role: str) -> list[str]:
    """
    Determine which chatbot roles a user can access based on their role.
    
    Args:
        user_role: User's role in the system
        
    Returns:
        List of chatbot roles the user can access
    """
    role_mapping = {
        "Manager": ["clerk", "picker", "packer", "driver", "manager"],
        "ClerkSupervisor": ["clerk"],
        "PickerSupervisor": ["picker"],
        "PackerSupervisor": ["packer"],
        "DriverSupervisor": ["driver"],
        "Clerk": ["clerk"],
        "Picker": ["picker"],
        "Packer": ["packer"],
        "Driver": ["driver"]
    }
    
    return role_mapping.get(user_role, [])
