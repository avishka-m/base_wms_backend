from typing import List

def get_allowed_chatbot_roles(user_role: str) -> List[str]:
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