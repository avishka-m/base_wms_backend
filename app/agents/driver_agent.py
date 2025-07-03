from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent

# Import tools directly instead of dynamic loading
from app.tools.chatbot.order_tools import check_order_tool
from app.tools.chatbot.path_tools import calculate_route_tool
from app.tools.chatbot.warehouse_tools import vehicle_select_tool

class DriverAgent(BaseAgent):
    """
    Driver Agent for the WMS Chatbot.
    Specialized in route optimization and delivery management.
    """
    
    def __init__(self):
        """Initialize the driver agent with the 'driver' role."""
        # Initialize the base agent first
        super().__init__(role="driver")
        
        # Set the tools directly - this avoids the dynamic loading issues
        self.tools = [
            check_order_tool,
            calculate_route_tool,
            vehicle_select_tool
        ]
        
        # Re-build the system prompt with the tools
        self.system_prompt = self._build_system_prompt()
        
        # Create the agent executor with the updated tools
        if self.llm is not None:
            self.agent_executor = self._create_agent_executor()
        
    def get_delivery_procedures(self) -> str:
        """
        Get delivery procedures from the knowledge base.
        
        Returns:
            String containing delivery procedures
        """
        delivery_docs = self.query_knowledge_base(
            "delivery procedures driver protocol shipping", 
            n_results=2
        )
        
        if delivery_docs:
            return "\n\n".join(delivery_docs)
        else:
            return "No delivery procedures found in knowledge base."
    
    def get_vehicle_information(self) -> str:
        """
        Get information about delivery vehicles from the knowledge base.
        
        Returns:
            String containing vehicle information
        """
        vehicle_docs = self.query_knowledge_base(
            "delivery vehicles trucks capacity maintenance", 
            n_results=2
        )
        
        if vehicle_docs:
            return "\n\n".join(vehicle_docs)
        else:
            return "No vehicle information found in knowledge base."
    
    def enhance_query(self, query: str) -> str:
        """
        Enhance the user query with driver-specific context.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query string
        """
        query_lower = query.lower()
        
        if "delivery" in query_lower or "ship" in query_lower or "transport" in query_lower:
            procedures = self.get_delivery_procedures()
            return f"{query}\n\nAs a driver handling deliveries:\n{procedures}"
        
        elif "route" in query_lower or "path" in query_lower or "direction" in query_lower:
            return f"{query}\n\nAs a driver, I need to find the most efficient delivery route while considering traffic conditions, delivery priorities, and vehicle capabilities."
        
        elif "vehicle" in query_lower or "truck" in query_lower or "van" in query_lower:
            vehicle_info = self.get_vehicle_information()
            return f"{query}\n\nAs a driver selecting and operating vehicles:\n{vehicle_info}"
        
        # Default case
        return f"{query}\n\nAs a warehouse delivery driver, I need to efficiently deliver packages to customers while adhering to safety protocols and delivery schedules."
    
    def run(self, query: str) -> str:
        """
        Run the driver agent on a user query with enhanced context.
        
        Args:
            query: User query string
            
        Returns:
            Agent response string
        """
        # Enhance the query with driver-specific context
        enhanced_query = self.enhance_query(query)
        
        # Run the agent with the enhanced query
        return super().run(enhanced_query)
