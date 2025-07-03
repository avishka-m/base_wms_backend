from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent

# Import tools directly instead of dynamic loading
from app.tools.chatbot.inventory_tools import locate_item_tool
from app.tools.chatbot.order_tools import check_order_tool, create_picking_task_tool, update_picking_task_tool
from app.tools.chatbot.path_tools import path_optimize_tool

class PickerAgent(BaseAgent):
    """
    Picker Agent for the WMS Chatbot.
    Specialized in optimizing picking routes and managing picking tasks.
    """
    
    def __init__(self):
        """Initialize the picker agent with the 'picker' role."""
        # Initialize the base agent first
        super().__init__(role="picker")
        
        # Set the tools directly - this avoids the dynamic loading issues
        self.tools = [
            locate_item_tool,
            check_order_tool,
            create_picking_task_tool,
            update_picking_task_tool,
            path_optimize_tool
        ]
        
        # Re-build the system prompt with the tools
        self.system_prompt = self._build_system_prompt()
        
        # Create the agent executor with the updated tools
        if self.llm is not None:
            self.agent_executor = self._create_agent_executor()
        
    def get_picking_procedures(self) -> str:
        """
        Get picking procedures from the knowledge base.
        
        Returns:
            String containing picking procedures
        """
        picking_docs = self.query_knowledge_base(
            "picking procedures warehouse protocol", 
            n_results=2
        )
        
        if picking_docs:
            return "\n\n".join(picking_docs)
        else:
            return "No picking procedures found in knowledge base."
    
    def get_path_optimization_tips(self) -> str:
        """
        Get path optimization tips from the knowledge base.
        
        Returns:
            String containing path optimization tips
        """
        path_docs = self.query_knowledge_base(
            "warehouse path optimization picking route", 
            n_results=2
        )
        
        if path_docs:
            return "\n\n".join(path_docs)
        else:
            return "No path optimization tips found in knowledge base."
    
    def enhance_query(self, query: str) -> str:
        """
        Enhance the user query with picker-specific context.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query string
        """
        query_lower = query.lower()
        
        if "route" in query_lower or "path" in query_lower or "optimize" in query_lower:
            tips = self.get_path_optimization_tips()
            return f"{query}\n\nAs a picker optimizing routes:\n{tips}"
        
        elif "picking" in query_lower or "pick" in query_lower or "task" in query_lower:
            procedures = self.get_picking_procedures()
            return f"{query}\n\nAs a picker handling order picking:\n{procedures}"
        
        elif "location" in query_lower or "find" in query_lower or "where" in query_lower:
            return f"{query}\n\nAs a picker, I need to locate items efficiently in the warehouse."
        
        # Default case
        return f"{query}\n\nAs a warehouse picker, I need to efficiently locate and retrieve items from the warehouse."
    
    def run(self, query: str) -> str:
        """
        Run the picker agent on a user query with enhanced context.
        
        Args:
            query: User query string
            
        Returns:
            Agent response string
        """
        # Enhance the query with picker-specific context
        enhanced_query = self.enhance_query(query)
        
        # Run the agent with the enhanced query
        return super().run(enhanced_query)
