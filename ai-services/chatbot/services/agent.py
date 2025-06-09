"""Agent management service"""

from typing import Dict, List, Optional
from core.logging import logger
from core.lifespan import get_agents

class AgentService:
    """Service for managing AI agents"""
    
    def __init__(self):
        self.agents = None
    
    def get_agent(self, role: str):
        """
        Get an agent by role.
        
        Args:
            role: Agent role
            
        Returns:
            Agent instance or None
        """
        if self.agents is None:
            self.agents = get_agents()
            
        return self.agents.get(role.lower())
    
    def get_available_agents(self) -> List[str]:
        """
        Get list of available agent roles.
        
        Returns:
            List of agent role names
        """
        if self.agents is None:
            self.agents = get_agents()
            
        return list(self.agents.keys())
    
    def process_message(self, role: str, message: str) -> str:
        """
        Process a message through an agent.
        
        Args:
            role: Agent role
            message: User message
            
        Returns:
            Agent response
            
        Raises:
            ValueError: If agent not found
        """
        agent = self.get_agent(role)
        if not agent:
            raise ValueError(f"Agent not found for role: {role}")
        
        logger.info(f"Processing message through {role} agent: {message[:100]}...")
        
        try:
            response = agent.run(message)
            logger.info(f"{role} agent response generated successfully")
            return response
        except Exception as e:
            logger.error(f"Error in {role} agent: {str(e)}")
            raise
    
    def validate_role(self, role: str) -> bool:
        """
        Validate if a role exists.
        
        Args:
            role: Role to validate
            
        Returns:
            True if role exists
        """
        return role.lower() in self.get_available_agents()
    
    def get_allowed_roles_for_user(self, user_role: str) -> List[str]:
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

# Create singleton instance
agent_service = AgentService() 