from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent

# Import tools directly instead of dynamic loading
from app.tools.chatbot.inventory_tools import locate_item_tool
from app.tools.chatbot.order_tools import check_order_tool, create_packing_task_tool, update_packing_task_tool

class PackerAgent(BaseAgent):
    """
    Packer Agent for the WMS Chatbot.
    Specialized in managing packing tasks and optimizing packaging.
    """
    
    def __init__(self):
        """Initialize the packer agent with the 'packer' role."""
        # Initialize the base agent first
        super().__init__(role="packer")
        
        # Set the tools directly - this avoids the dynamic loading issues
        self.tools = [
            locate_item_tool,
            check_order_tool,
            create_packing_task_tool,
            update_packing_task_tool
        ]
        
        # Re-build the system prompt with the tools
        self.system_prompt = self._build_system_prompt()
        
        # Create the agent executor with the updated tools
        if self.llm is not None:
            self.agent_executor = self._create_agent_executor()
        
    def get_packing_procedures(self) -> str:
        """
        Get packing procedures from the knowledge base.
        
        Returns:
            String containing packing procedures
        """
        packing_docs = self.query_knowledge_base(
            "packing procedures warehouse protocol packaging", 
            n_results=2
        )
        
        if packing_docs:
            return "\n\n".join(packing_docs)
        else:
            return "No packing procedures found in knowledge base."
    
    def get_packaging_material_info(self) -> str:
        """
        Get information about packaging materials from the knowledge base.
        
        Returns:
            String containing packaging material information
        """
        material_docs = self.query_knowledge_base(
            "packaging materials boxes warehouse shipping", 
            n_results=2
        )
        
        if material_docs:
            return "\n\n".join(material_docs)
        else:
            return "No packaging material information found in knowledge base."
    
    def enhance_query(self, query: str) -> str:
        """
        Enhance the user query with packer-specific context.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query string
        """
        query_lower = query.lower()
        
        if "package" in query_lower or "box" in query_lower or "packaging" in query_lower:
            material_info = self.get_packaging_material_info()
            return f"{query}\n\nAs a packer selecting packaging materials:\n{material_info}"
        
        elif "pack" in query_lower or "task" in query_lower or "order" in query_lower:
            procedures = self.get_packing_procedures()
            return f"{query}\n\nAs a packer handling order packing:\n{procedures}"
        
        # Default case
        return f"{query}\n\nAs a warehouse packer, I need to efficiently package items for shipping while ensuring they're protected during transit."
        
    async def run(self, query: str) -> str:
        """
        Run the packer agent on a user query with enhanced context.
        
        Args:
            query: User query string
            
        Returns:
            Agent response string
        """
        # Enhance the query with packer-specific context
        enhanced_query = self.enhance_query(query)
        
        # Run the agent with the enhanced query
        return await super().run(enhanced_query)
