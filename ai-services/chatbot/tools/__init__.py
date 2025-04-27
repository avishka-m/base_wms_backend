"""
Tools for the WMS Chatbot.
Each tool connects to the WMS API to perform specific actions based on user roles.
"""

from .inventory_tools import (
    inventory_query_tool,
    inventory_add_tool,
    inventory_update_tool,
    locate_item_tool
)

from .order_tools import (
    check_order_tool,
    create_sub_order_tool,
    approve_orders_tool,
    create_picking_task_tool,
    update_picking_task_tool,
    create_packing_task_tool,
    update_packing_task_tool
)

from .path_tools import (
    path_optimize_tool,
    calculate_route_tool
)

from .return_tools import (
    process_return_tool
)

from .warehouse_tools import (
    check_supplier_tool,
    vehicle_select_tool,
    worker_manage_tool,
    check_analytics_tool,
    check_anomalies_tool,
    system_manage_tool
)

# Export all tools
__all__ = [
    # Inventory tools
    "inventory_query_tool",
    "inventory_add_tool", 
    "inventory_update_tool",
    "locate_item_tool",
    
    # Order tools
    "check_order_tool",
    "create_sub_order_tool",
    "approve_orders_tool",
    "create_picking_task_tool",
    "update_picking_task_tool",
    "create_packing_task_tool",
    "update_packing_task_tool",
    
    # Path tools
    "path_optimize_tool",
    "calculate_route_tool",
    
    # Return tools
    "process_return_tool",
    
    # Warehouse tools
    "check_supplier_tool",
    "vehicle_select_tool",
    "worker_manage_tool",
    "check_analytics_tool",
    "check_anomalies_tool",
    "system_manage_tool"
]