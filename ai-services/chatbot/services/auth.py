"""Authentication service"""

from typing import Dict, Any, List
from fastapi import HTTPException, status

class AuthService:
    """Service for authentication and authorization logic"""
    
    def validate_user_access(self, user: Dict[str, Any], required_role: str) -> bool:
        """
        Validate if user has access to a specific chatbot role.
        
        Args:
            user: User information
            required_role: Required chatbot role
            
        Returns:
            True if user has access
        """
        user_role = user.get("role", "")
        allowed_roles = self.get_allowed_chatbot_roles(user_role)
        
        # Managers can access all chatbots
        if user_role == "Manager":
            return True
            
        return required_role in allowed_roles
    
    def get_allowed_chatbot_roles(self, user_role: str) -> List[str]:
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
    
    def check_role_access(self, user: Dict[str, Any], chatbot_role: str) -> None:
        """
        Check if user has access to a chatbot role and raise exception if not.
        
        Args:
            user: User information
            chatbot_role: Required chatbot role
            
        Raises:
            HTTPException: If user doesn't have access
        """
        if not self.validate_user_access(user, chatbot_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User with role '{user.get('role')}' cannot access the '{chatbot_role}' chatbot"
            )
    
    def get_user_id(self, user: Dict[str, Any]) -> str:
        """
        Get user identifier from user object.
        
        Args:
            user: User information
            
        Returns:
            User identifier
        """
        return user.get("username", "anonymous")

# Create singleton instance
auth_service = AuthService() 