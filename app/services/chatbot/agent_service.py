"""
Agent service for managing AI agents in the WMS Chatbot.
"""

import logging
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(__file__)
chatbot_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if chatbot_dir not in sys.path:
    sys.path.append(chatbot_dir)

try:
    from app.agents.clerk_agent import ClerkAgent
    from app.agents.picker_agent import PickerAgent
    from app.agents.packer_agent_ex import PackerAgent
    from app.agents.driver_agent import DriverAgent
    from app.agents.manager_agent import ManagerAgent
except ImportError as e:
    logging.error(f"Error importing agents: {e}")
    # Create mock agents for testing
    class MockAgent:
        def run(self, message): return f"Mock response to: {message}"
    
    ClerkAgent = PickerAgent = PackerAgent = DriverAgent = ManagerAgent = MockAgent

logger = logging.getLogger("wms_chatbot.agent_service")


class AgentService:
    """Service for managing and routing AI agents."""
    
    def __init__(self):
        """Initialize the agent service with all available agents."""
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all AI agents."""
        try:
            self.agents = {
                "clerk": ClerkAgent(),
                "picker": PickerAgent(),
                "packer": PackerAgent(),
                "driver": DriverAgent(),
                "manager": ManagerAgent()
            }
            logger.info(f"Initialized agents for roles: {', '.join(self.agents.keys())}")
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    def get_agent(self, role: str):
        """
        Get an agent by role.
        
        Args:
            role: The role of the agent to retrieve
            
        Returns:
            The agent instance for the given role
            
        Raises:
            ValueError: If the role is not found
        """
        if role not in self.agents:
            raise ValueError(f"Unknown agent role: {role}. Available roles: {list(self.agents.keys())}")
        
        return self.agents[role]
    
    def get_available_roles(self) -> list[str]:
        """
        Get list of available agent roles.
        
        Returns:
            List of available agent role names
        """
        return list(self.agents.keys())
    
    def process_message(self, role: str, message: str) -> str:
        """
        Process a message using the appropriate agent.
        
        Args:
            role: The role of the agent to use
            message: The message to process
            
        Returns:
            The agent's response
            
        Raises:
            ValueError: If the role is not found
            Exception: If agent processing fails
        """
        try:
            agent = self.get_agent(role)
            logger.info(f"Processing message with {role} agent: {message[:100]}")
            
            response = agent.run(message)
            return response
            
        except Exception as e:
            logger.error(f"Error processing message with {role} agent: {str(e)}")
            raise
