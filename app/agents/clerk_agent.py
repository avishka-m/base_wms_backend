from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent

# Import tools directly instead of dynamic loading
from app.tools.chatbot.inventory_tools import inventory_query_tool, inventory_add_tool, inventory_update_tool, locate_item_tool
from app.tools.chatbot.order_tools import check_order_tool, create_sub_order_tool
from app.tools.chatbot.return_tools import process_return_tool
from app.tools.chatbot.warehouse_tools import check_supplier_tool

class ClerkAgent(BaseAgent):
    """
    Receiving Clerk Agent for the WMS Chatbot.
    Specialized in receiving new inventory, processing returns, and checking inventory levels.
    """
    
    def __init__(self):
        """Initialize the clerk agent with the 'clerk' role."""
        # Initialize the base agent first
        super().__init__(role="clerk")
        
        # Set the tools directly - this avoids the dynamic loading issues
        self.tools = [
            inventory_query_tool,
            inventory_add_tool,
            inventory_update_tool,
            locate_item_tool,
            check_order_tool,
            create_sub_order_tool,
            process_return_tool,
            check_supplier_tool
        ]
        
        # Re-build the system prompt with the tools
        self.system_prompt = self._build_system_prompt()
        
        # Create the agent executor with the updated tools
        if self.llm is not None:
            self.agent_executor = self._create_agent_executor()
        
    def get_receiving_procedures(self) -> str:
        """
        Get receiving procedures from the knowledge base.
        
        Returns:
            String containing receiving procedures
        """
        receiving_docs = self.query_knowledge_base(
            "receiving procedures warehouse protocol", 
            n_results=2
        )
        
        if receiving_docs:
            return "\n\n".join(receiving_docs)
        else:
            return "No receiving procedures found in knowledge base."
    
    def get_return_procedures(self) -> str:
        """
        Get return processing procedures from the knowledge base.
        
        Returns:
            String containing return procedures
        """
        return_docs = self.query_knowledge_base(
            "returns processing procedures protocol", 
            n_results=2
        )
        
        if return_docs:
            return "\n\n".join(return_docs)
        else:
            return "No return procedures found in knowledge base."
    
    def enhance_query(self, query: str) -> str:
        """
        Enhance the user query with clerk-specific context.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query string
        """
        # Add role-specific context based on query content
        query_lower = query.lower()
        
        if "inventory" in query_lower or "stock" in query_lower:
            return f"{query}\n\nAs a clerk, consider inventory management procedures when answering."
        
        elif "order" in query_lower or "customer" in query_lower:
            return f"{query}\n\nAs a clerk, consider order processing and customer service protocols."
        
        elif "return" in query_lower or "refund" in query_lower:
            procedures = self.get_return_procedures()
            return f"{query}\n\nAs a clerk handling returns:\n{procedures}"
        
        elif "receive" in query_lower or "shipment" in query_lower or "supplier" in query_lower:
            procedures = self.get_receiving_procedures()
            return f"{query}\n\nAs a clerk handling receiving:\n{procedures}"
        
        # Default case: return with basic role context
        return f"{query}\n\nAs a warehouse clerk, I need to provide accurate information about inventory, orders, returns, and receiving processes."
    
    def run(self, query: str) -> str:
        """
        Run the clerk agent on a user query with enhanced context.
        
        Args:
            query: User query string
            
        Returns:
            Agent response string
        """
        # Enhance the query with clerk-specific context
        enhanced_query = self.enhance_query(query)
        
        # Run the agent with the enhanced query
        return super().run(enhanced_query)