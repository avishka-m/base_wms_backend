from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent

# Import tools directly instead of dynamic loading
from app.tools.chatbot.inventory_tools import (
    inventory_query_tool, 
    inventory_add_tool,
    inventory_update_tool, 
    inventory_analytics_tool,
    locate_item_tool,
    low_stock_alert_tool,
    stock_movement_tool
)
from app.tools.chatbot.order_tools import (
    check_order_tool, 
    order_create_tool,
    order_update_tool,
    approve_orders_tool,
    create_sub_order_tool,
    create_picking_task_tool,
    update_picking_task_tool,
    create_packing_task_tool,
    update_packing_task_tool
)
from app.tools.chatbot.warehouse_tools import worker_manage_tool, check_analytics_tool, system_manage_tool, check_anomalies_tool

class ManagerAgent(BaseAgent):
    """
    Manager Agent for the WMS Chatbot.
    Specialized in warehouse oversight, analytics, and worker management.
    """
    
    def __init__(self):
        """Initialize the manager agent with the 'manager' role."""
        # Initialize the base agent first
        super().__init__(role="manager")
        
        # Set the tools directly - this avoids the dynamic loading issues
        self.tools = [
            # Complete inventory management suite for managers
            inventory_query_tool,
            inventory_add_tool,
            inventory_update_tool,
            inventory_analytics_tool,
            locate_item_tool,
            low_stock_alert_tool,
            stock_movement_tool,
            # Complete order management suite for managers
            check_order_tool,
            order_create_tool,
            order_update_tool,
            approve_orders_tool,
            create_sub_order_tool,
            # Task management
            create_picking_task_tool,
            update_picking_task_tool,
            create_packing_task_tool,
            update_packing_task_tool,
            # Warehouse management
            worker_manage_tool,
            check_analytics_tool,
            system_manage_tool,
            check_anomalies_tool
        ]
        
        # Re-build the system prompt with the tools
        self.system_prompt = self._build_system_prompt()
        

        
    def get_management_procedures(self) -> str:
        """
        Get management procedures from the knowledge base.
        
        Returns:
            String containing management procedures
        """
        mgmt_docs = self.query_knowledge_base(
            "warehouse management procedures protocol oversight", 
            n_results=2
        )
        
        if mgmt_docs:
            return "\n\n".join(mgmt_docs)
        else:
            return "No management procedures found in knowledge base."
    
    def get_analytics_information(self) -> str:
        """
        Get information about warehouse analytics from the knowledge base.
        
        Returns:
            String containing analytics information
        """
        analytics_docs = self.query_knowledge_base(
            "warehouse analytics metrics KPI performance", 
            n_results=2
        )
        
        if analytics_docs:
            return "\n\n".join(analytics_docs)
        else:
            return "No analytics information found in knowledge base."
    
    def enhance_query(self, query: str) -> str:
        """
        Enhance the user query with manager-specific context.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query string
        """
        query_lower = query.lower()
        
        if "analytics" in query_lower or "report" in query_lower or "performance" in query_lower:
            analytics_info = self.get_analytics_information()
            return f"{query}\n\nAs a manager reviewing analytics:\n{analytics_info}"
        
        elif "worker" in query_lower or "staff" in query_lower or "employee" in query_lower:
            return f"{query}\n\nAs a manager, I need to effectively manage warehouse staff, including role assignments, scheduling, performance monitoring, and addressing issues."
        
        elif "approve" in query_lower or "order" in query_lower or "system" in query_lower:
            mgmt_info = self.get_management_procedures()
            return f"{query}\n\nAs a manager handling approvals and system management:\n{mgmt_info}"
        
        elif "anomaly" in query_lower or "issue" in query_lower or "problem" in query_lower:
            return f"{query}\n\nAs a manager, I need to identify and address anomalies or issues in the warehouse operations to maintain efficiency and accuracy."
        
        # Default case
        return f"{query}\n\nAs a warehouse manager, I need to oversee all warehouse operations, make strategic decisions, and ensure efficiency across all departments."
    
    async def run(
        self, 
        query: str, 
        conversation_id: str = "default", 
        user_id: str = "anonymous"
    ) -> str:
        """
        Run the manager agent on a user query with enhanced context.
        
        Args:
            query: User query string
            conversation_id: Unique conversation identifier for memory persistence
            user_id: User identifier for memory management
            
        Returns:
            Agent response string
        """
        # Enhance the query with manager-specific context
        enhanced_query = self.enhance_query(query)
        
        # Run the agent with the enhanced query
        return await super().run(enhanced_query, conversation_id, user_id)
