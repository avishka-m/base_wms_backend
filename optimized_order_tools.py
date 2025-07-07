"""
OPTIMIZED Order Tools for WMS Chatbot

This module contains performance-optimized versions of order management tools
that address the key bottlenecks in the original implementation.

Key Optimizations:
1. Batch database operations to reduce N+1 queries
2. Caching for frequently accessed data
3. Parallel processing where possible
4. Efficient data structures and algorithms
"""

from typing import Dict, Any, List, Optional
import asyncio
import time
from datetime import datetime
from app.utils.chatbot.mongodb_client import chatbot_mongodb_client
from app.tools.chatbot.base_tool import create_tool

class OptimizedOrderService:
    """Optimized order service with performance improvements."""
    
    @staticmethod
    async def batch_validate_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate multiple items in a single batch operation instead of N+1 queries.
        
        BEFORE: N+1 queries (1 query per item)
        AFTER: 1 query for all items
        """
        # Extract all item IDs
        item_ids = [item.get('item_id') or item.get('itemID') for item in items]
        
        # Single database query to get all items
        db = await chatbot_mongodb_client.get_async_database()
        inventory_items = await db.inventory.find(
            {"itemID": {"$in": item_ids}}
        ).to_list(length=None)
        
        # Create lookup dictionary for O(1) access
        inventory_lookup = {item['itemID']: item for item in inventory_items}
        
        # Validate all items
        validated_items = []
        total_amount = 0.0
        errors = []
        
        for item in items:
            item_id = item.get('item_id') or item.get('itemID')
            quantity = item.get('quantity', 1)
            
            if not item_id:
                errors.append("Each item must have an 'item_id' or 'itemID' field.")
                continue
            
            inventory_item = inventory_lookup.get(item_id)
            if not inventory_item:
                errors.append(f"Inventory item with ID {item_id} not found.")
                continue
            
            # Check stock availability
            available_stock = inventory_item.get('stock_level', 0)
            if available_stock < quantity:
                errors.append(f"Insufficient stock for item {item_id}. Available: {available_stock}, Requested: {quantity}")
                continue
            
            # Calculate item total
            unit_price = inventory_item.get('unit_price', 0.0)
            item_total = unit_price * quantity
            total_amount += item_total
            
            validated_items.append({
                'itemID': item_id,
                'quantity': quantity,
                'unit_price': unit_price,
                'total_price': item_total,
                'fulfilled_quantity': 0
            })
        
        return {
            'validated_items': validated_items,
            'total_amount': total_amount,
            'errors': errors
        }
    
    @staticmethod
    async def parallel_order_operations(order_ids: List[int]) -> Dict[str, Any]:
        """
        Perform multiple order operations in parallel instead of sequentially.
        
        BEFORE: Sequential operations (slow)
        AFTER: Parallel operations (fast)
        """
        # Use asyncio.gather to run operations in parallel
        tasks = []
        for order_id in order_ids:
            task = chatbot_mongodb_client.get_order_by_id(order_id)
            tasks.append(task)
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        orders = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Error fetching order {order_ids[i]}: {str(result)}")
            else:
                orders.append(result)
        
        return {
            'orders': orders,
            'errors': errors
        }

# Optimized order creation function
async def optimized_order_create_func(customer_id: int,
                                    items: List[Dict[str, Any]],
                                    shipping_address: str,
                                    priority: Optional[str] = "normal",
                                    notes: Optional[str] = None) -> str:
    """
    Optimized order creation with batch validation and reduced database calls.
    """
    try:
        # Check cache for customer data first
        cache_key = f"customer:{customer_id}"
        customer = chatbot_mongodb_client.cache.get(cache_key)
        
        if not customer:
            customer = await chatbot_mongodb_client.get_customer_by_id(customer_id)
            if customer:
                chatbot_mongodb_client.cache.set(cache_key, customer)
        
        if not customer:
            return f"❌ Customer with ID {customer_id} not found."
        
        # Batch validate all items in single operation
        validation_result = await OptimizedOrderService.batch_validate_items(items)
        
        if validation_result['errors']:
            return f"❌ Validation errors:\n" + "\n".join(validation_result['errors'])
        
        validated_items = validation_result['validated_items']
        total_amount = validation_result['total_amount']
        
        # Prepare order data
        order_data = {
            'customerID': customer_id,
            'items': validated_items,
            'shipping_address': shipping_address,
            'priority': priority.lower() if priority else 'normal',
            'total_amount': round(total_amount, 2),
            'order_date': datetime.utcnow(),
            'assigned_worker': None
        }
        
        if notes:
            order_data['notes'] = notes
        
        # Create the order
        result = await chatbot_mongodb_client.create_order(order_data)
        
        # Invalidate relevant cache entries
        chatbot_mongodb_client.cache.invalidate(f"orders:customer:{customer_id}")
        
        if result.get('success'):
            order = result.get('order', {})
            response = f"✅ Successfully created order!\n\n"
            response += f"Order ID: {result.get('order_id')}\n"
            response += f"Customer: {customer.get('name', 'Unknown')} (ID: {customer_id})\n"
            response += f"Total Amount: ${total_amount:.2f}\n"
            response += f"Priority: {priority}\n"
            response += f"Items: {len(validated_items)} item(s)\n"
            response += f"Shipping Address: {shipping_address}\n"
            if notes:
                response += f"Notes: {notes}\n"
            response += f"Status: {order.get('order_status', 'pending')}\n"
            
            return response
        else:
            return f"❌ Failed to create order: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error creating order: {str(e)}"

# Optimized order query function with pagination and caching
async def optimized_order_query_func(order_id: Optional[int] = None,
                                   customer_id: Optional[int] = None,
                                   status: Optional[str] = None,
                                   priority: Optional[str] = None,
                                   date_from: Optional[str] = None,
                                   date_to: Optional[str] = None,
                                   limit: int = 50,
                                   offset: int = 0) -> str:
    """
    Optimized order query with caching and pagination.
    """
    try:
        # Single order lookup with caching
        if order_id is not None:
            cache_key = f"order:{order_id}"
            order = chatbot_mongodb_client.cache.get(cache_key)
            
            if not order:
                order = await chatbot_mongodb_client.get_order_by_id(order_id)
                if order:
                    chatbot_mongodb_client.cache.set(cache_key, order)
            
            if not order:
                return f"Order with ID {order_id} not found in the database."
            
            # Format single order response (same as before)
            result = "Found order:\n\n"
            result += f"Order ID: {order.get('orderID')}\n"
            result += f"Customer ID: {order.get('customerID')}\n"
            result += f"Status: {order.get('order_status')}\n"
            result += f"Priority: {order.get('priority', 'Normal')}\n"
            result += f"Total Amount: ${order.get('total_amount', 'N/A')}\n"
            result += f"Order Date: {order.get('order_date', 'N/A')}\n"
            result += f"Shipping Address: {order.get('shipping_address', 'N/A')}\n"
            result += f"Assigned Worker: {order.get('assigned_worker', 'None')}\n"
            result += f"Notes: {order.get('notes', 'N/A')}\n"
            
            items = order.get('items', [])
            if items:
                result += f"\nOrder Items ({len(items)}):\n"
                for item in items:
                    result += f"- Item ID: {item.get('itemID')} (Qty: {item.get('quantity', 0)}, Price: ${item.get('price', 'N/A')})\n"
                    result += f"  Fulfilled: {item.get('fulfilled_quantity', 0)}/{item.get('quantity', 0)}\n"
            
            return result
        
        # Build optimized query with indexes
        filter_criteria = {}
        
        if customer_id:
            filter_criteria["customerID"] = customer_id
        if status:
            filter_criteria["order_status"] = status
        if priority:
            filter_criteria["priority"] = priority
        
        # Create cache key for this query
        cache_key = f"orders_query:{hash(str(sorted(filter_criteria.items())))}:{limit}:{offset}"
        cached_result = chatbot_mongodb_client.cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # Optimized database query with pagination
        db = await chatbot_mongodb_client.get_async_database()
        cursor = db.orders.find(filter_criteria).skip(offset).limit(limit).sort([("order_date", -1)])
        orders = await cursor.to_list(length=limit)
        
        if not orders:
            result = "No orders found matching your criteria."
        else:
            # Format the results
            result = f"Found {len(orders)} order(s) (showing {offset+1}-{offset+len(orders)}):\n\n"
            for order in orders:
                result += f"Order ID: {order.get('orderID')}\n"
                result += f"Customer ID: {order.get('customerID')}\n"
                result += f"Status: {order.get('order_status')}\n"
                result += f"Priority: {order.get('priority', 'Normal')}\n"
                result += f"Total: ${order.get('total_amount', 'N/A')}\n"
                result += f"Order Date: {order.get('order_date', 'N/A')}\n"
                result += f"Assigned Worker: {order.get('assigned_worker', 'None')}\n"
                result += "-" * 40 + "\n"
        
        # Cache the result
        chatbot_mongodb_client.cache.set(cache_key, result)
        
        return result
        
    except Exception as e:
        return f"Error querying orders: {str(e)}"

# Create optimized tools
optimized_check_order_tool = create_tool(
    name="optimized_check_order",
    description="Check order status with optimized performance (cached, paginated)",
    function=optimized_order_query_func,
    arg_descriptions={
        "order_id": {"type": Optional[int], "description": "ID of the order to check"},
        "customer_id": {"type": Optional[int], "description": "ID of the customer"},
        "status": {"type": Optional[str], "description": "Status of the order"},
        "priority": {"type": Optional[str], "description": "Priority level"},
        "date_from": {"type": Optional[str], "description": "Start date"},
        "date_to": {"type": Optional[str], "description": "End date"},
        "limit": {"type": int, "description": "Number of results to return (default: 50)"},
        "offset": {"type": int, "description": "Number of results to skip (default: 0)"}
    }
)

optimized_order_create_tool = create_tool(
    name="optimized_order_create",
    description="Create orders with optimized batch validation",
    function=optimized_order_create_func,
    arg_descriptions={
        "customer_id": {"type": int, "description": "ID of the customer"},
        "items": {"type": List[Dict[str, Any]], "description": "List of items with item_id and quantity"},
        "shipping_address": {"type": str, "description": "Shipping address"},
        "priority": {"type": Optional[str], "description": "Priority level (default: normal)"},
        "notes": {"type": Optional[str], "description": "Optional notes"}
    }
)

# Export optimized tools
__all__ = [
    "OptimizedOrderService",
    "optimized_check_order_tool",
    "optimized_order_create_tool",
    "optimized_order_create_func",
    "optimized_order_query_func"
] 