"""
Demo data system for WMS chatbot tools.
Provides realistic fallback data when API endpoints are not accessible.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random

# Demo Inventory Data
DEMO_INVENTORY = [
    {
        "id": 1,
        "sku": "sku-001",
        "name": "Wireless Headphones",
        "category": "Electronics",
        "quantity": 50,
        "location_id": 1,
        "unit_price": 99.99,
        "description": "High-quality wireless headphones with noise cancellation"
    },
    {
        "id": 2,
        "sku": "sku-002",
        "name": "Smartphone Case",
        "category": "Electronics",
        "quantity": 150,
        "location_id": 2,
        "unit_price": 19.99,
        "description": "Protective smartphone case with drop protection"
    },
    {
        "id": 3,
        "sku": "sku-003",
        "name": "Office Chair",
        "category": "Furniture",
        "quantity": 25,
        "location_id": 3,
        "unit_price": 299.99,
        "description": "Ergonomic office chair with lumbar support"
    },
    {
        "id": 4,
        "sku": "sku-004",
        "name": "Laptop Stand",
        "category": "Electronics",
        "quantity": 75,
        "location_id": 1,
        "unit_price": 49.99,
        "description": "Adjustable laptop stand for better ergonomics"
    },
    {
        "id": 5,
        "sku": "sku-005",
        "name": "Desk Lamp",
        "category": "Furniture",
        "quantity": 40,
        "location_id": 2,
        "unit_price": 79.99,
        "description": "LED desk lamp with adjustable brightness"
    }
]

# Demo Orders Data
DEMO_ORDERS = [
    {
        "id": 1,
        "status": "Processing",
        "customer_id": 1,
        "customer_name": "John Smith",
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-15T14:45:00Z",
        "total_value": 449.97,
        "items": [
            {
                "item_id": 1,
                "item_name": "Wireless Headphones",
                "sku": "sku-001",
                "quantity": 2,
                "unit_price": 99.99,
                "subtotal": 199.98,
                "status": "Picked"
            },
            {
                "item_id": 4,
                "item_name": "Laptop Stand",
                "sku": "sku-004",
                "quantity": 5,
                "unit_price": 49.99,
                "subtotal": 249.99,
                "status": "Pending"
            }
        ],
        "picking_tasks": [
            {
                "id": 1,
                "status": "Completed",
                "worker_name": "Alice Johnson",
                "worker_id": 1,
                "created_at": "2024-01-15T11:00:00Z"
            }
        ],
        "packing_tasks": [
            {
                "id": 1,
                "status": "In Progress",
                "worker_name": "Bob Wilson",
                "worker_id": 2,
                "created_at": "2024-01-15T14:30:00Z"
            }
        ],
        "shipping_tasks": []
    },
    {
        "id": 2,
        "status": "Shipped",
        "customer_id": 2,
        "customer_name": "Sarah Davis",
        "created_at": "2024-01-14T09:15:00Z",
        "updated_at": "2024-01-14T16:20:00Z",
        "total_value": 319.98,
        "items": [
            {
                "item_id": 3,
                "item_name": "Office Chair",
                "sku": "sku-003",
                "quantity": 1,
                "unit_price": 299.99,
                "subtotal": 299.99,
                "status": "Shipped"
            },
            {
                "item_id": 2,
                "item_name": "Smartphone Case",
                "sku": "sku-002",
                "quantity": 1,
                "unit_price": 19.99,
                "subtotal": 19.99,
                "status": "Shipped"
            }
        ],
        "picking_tasks": [
            {
                "id": 2,
                "status": "Completed",
                "worker_name": "Charlie Brown",
                "worker_id": 3,
                "created_at": "2024-01-14T10:00:00Z"
            }
        ],
        "packing_tasks": [
            {
                "id": 2,
                "status": "Completed",
                "worker_name": "Diana Prince",
                "worker_id": 4,
                "created_at": "2024-01-14T13:15:00Z"
            }
        ],
        "shipping_tasks": [
            {
                "id": 1,
                "status": "Delivered",
                "worker_name": "Eve Thompson",
                "worker_id": 5,
                "vehicle_id": 1,
                "vehicle_name": "Delivery Van #1",
                "created_at": "2024-01-14T15:30:00Z"
            }
        ]
    }
]

# Demo Locations Data
DEMO_LOCATIONS = [
    {
        "id": 1,
        "name": "A-1-1-1",
        "zone": "A",
        "aisle": "1",
        "shelf": "1",
        "bin": "1",
        "x": 0,
        "y": 0,
        "z": 0,
        "capacity": 100,
        "current_usage": 50
    },
    {
        "id": 2,
        "name": "A-1-2-1",
        "zone": "A",
        "aisle": "1",
        "shelf": "2",
        "bin": "1",
        "x": 0,
        "y": 5,
        "z": 0,
        "capacity": 100,
        "current_usage": 75
    },
    {
        "id": 3,
        "name": "B-2-1-1",
        "zone": "B",
        "aisle": "2",
        "shelf": "1",
        "bin": "1",
        "x": 10,
        "y": 0,
        "z": 0,
        "capacity": 150,
        "current_usage": 25
    },
    {
        "id": 4,
        "name": "B-2-2-1",
        "zone": "B",
        "aisle": "2",
        "shelf": "2",
        "bin": "1",
        "x": 10,
        "y": 5,
        "z": 0,
        "capacity": 150,
        "current_usage": 90
    },
    {
        "id": 5,
        "name": "C-3-1-1",
        "zone": "C",
        "aisle": "3",
        "shelf": "1",
        "bin": "1",
        "x": 20,
        "y": 0,
        "z": 0,
        "capacity": 200,
        "current_usage": 120
    }
]

# Demo Workers Data
DEMO_WORKERS = [
    {
        "id": 1,
        "name": "Alice Johnson",
        "role": "Picker",
        "status": "Active",
        "shift": "Morning",
        "email": "alice.johnson@warehouse.com",
        "phone": "555-0101",
        "hire_date": "2023-06-15",
        "department": "Fulfillment",
        "supervisor": "Manager Smith"
    },
    {
        "id": 2,
        "name": "Bob Wilson",
        "role": "Packer",
        "status": "Active",
        "shift": "Morning",
        "email": "bob.wilson@warehouse.com",
        "phone": "555-0102",
        "hire_date": "2023-08-20",
        "department": "Fulfillment",
        "supervisor": "Manager Smith"
    },
    {
        "id": 3,
        "name": "Charlie Brown",
        "role": "Picker",
        "status": "Active",
        "shift": "Afternoon",
        "email": "charlie.brown@warehouse.com",
        "phone": "555-0103",
        "hire_date": "2023-09-10",
        "department": "Fulfillment",
        "supervisor": "Manager Smith"
    },
    {
        "id": 4,
        "name": "Diana Prince",
        "role": "Packer",
        "status": "Active",
        "shift": "Afternoon",
        "email": "diana.prince@warehouse.com",
        "phone": "555-0104",
        "hire_date": "2023-10-05",
        "department": "Fulfillment",
        "supervisor": "Manager Smith"
    },
    {
        "id": 5,
        "name": "Eve Thompson",
        "role": "Driver",
        "status": "Active",
        "shift": "All Day",
        "email": "eve.thompson@warehouse.com",
        "phone": "555-0105",
        "hire_date": "2023-07-01",
        "department": "Shipping",
        "supervisor": "Manager Smith"
    },
    {
        "id": 6,
        "name": "Frank Miller",
        "role": "Clerk",
        "status": "Active",
        "shift": "Morning",
        "email": "frank.miller@warehouse.com",
        "phone": "555-0106",
        "hire_date": "2023-05-15",
        "department": "Receiving",
        "supervisor": "Manager Smith"
    }
]

# Demo Vehicles Data
DEMO_VEHICLES = [
    {
        "id": 1,
        "name": "Delivery Van #1",
        "type": "Van",
        "capacity": 1000,
        "volume": 10,
        "refrigerated": False,
        "mpg": 18,
        "status": "Available",
        "driver_id": 5,
        "driver_name": "Eve Thompson"
    },
    {
        "id": 2,
        "name": "Refrigerated Truck #1",
        "type": "Truck",
        "capacity": 5000,
        "volume": 50,
        "refrigerated": True,
        "mpg": 12,
        "status": "In Use",
        "driver_id": None,
        "driver_name": None
    },
    {
        "id": 3,
        "name": "Delivery Van #2",
        "type": "Van",
        "capacity": 1000,
        "volume": 10,
        "refrigerated": False,
        "mpg": 20,
        "status": "Maintenance",
        "driver_id": None,
        "driver_name": None
    }
]

# Demo Suppliers Data
DEMO_SUPPLIERS = [
    {
        "id": 1,
        "name": "TechSupply Corp",
        "contact_name": "John Anderson",
        "email": "john@techsupply.com",
        "phone": "555-1001",
        "categories": ["Electronics", "Accessories"],
        "address": "123 Tech Street, Silicon Valley, CA 94025",
        "active": True,
        "rating": 4.5,
        "lead_time": 5
    },
    {
        "id": 2,
        "name": "Furniture Plus",
        "contact_name": "Maria Rodriguez",
        "email": "maria@furnitureplus.com",
        "phone": "555-1002",
        "categories": ["Furniture", "Office Supplies"],
        "address": "456 Furniture Ave, Grand Rapids, MI 49503",
        "active": True,
        "rating": 4.2,
        "lead_time": 10
    },
    {
        "id": 3,
        "name": "Express Electronics",
        "contact_name": "David Kim",
        "email": "david@expresselectronics.com",
        "phone": "555-1003",
        "categories": ["Electronics", "Components"],
        "address": "789 Circuit Road, Austin, TX 78701",
        "active": True,
        "rating": 4.8,
        "lead_time": 3
    }
]

# Demo Analytics Data
DEMO_ANALYTICS = {
    "inventory_levels": {
        "total_items": 500,
        "low_stock_items": 25,
        "out_of_stock_items": 5,
        "overstock_items": 15,
        "total_value": 125000.00
    },
    "order_metrics": {
        "pending_orders": 45,
        "processing_orders": 23,
        "shipped_orders": 156,
        "average_processing_time": 2.5,
        "order_accuracy": 98.5
    },
    "productivity": {
        "picks_per_hour": 35,
        "packs_per_hour": 28,
        "shipping_accuracy": 99.2,
        "return_rate": 2.1
    }
}

def get_demo_orders(order_id: Optional[int] = None, status: Optional[str] = None) -> List[Dict]:
    """Get demo orders with optional filtering."""
    if order_id is not None:
        return [order for order in DEMO_ORDERS if order["id"] == order_id]
    
    if status:
        return [order for order in DEMO_ORDERS if order["status"].lower() == status.lower()]
    
    return DEMO_ORDERS

def get_demo_locations(location_id: Optional[int] = None, zone: Optional[str] = None) -> List[Dict]:
    """Get demo locations with optional filtering."""
    if location_id is not None:
        return [loc for loc in DEMO_LOCATIONS if loc["id"] == location_id]
    
    if zone:
        return [loc for loc in DEMO_LOCATIONS if loc["zone"].lower() == zone.lower()]
    
    return DEMO_LOCATIONS

def get_demo_workers(worker_id: Optional[int] = None, role: Optional[str] = None, 
                    status: Optional[str] = None) -> List[Dict]:
    """Get demo workers with optional filtering."""
    workers = DEMO_WORKERS
    
    if worker_id is not None:
        workers = [w for w in workers if w["id"] == worker_id]
    
    if role:
        workers = [w for w in workers if w["role"].lower() == role.lower()]
    
    if status:
        workers = [w for w in workers if w["status"].lower() == status.lower()]
    
    return workers

def get_demo_vehicles(vehicle_id: Optional[int] = None, refrigerated: Optional[bool] = None) -> List[Dict]:
    """Get demo vehicles with optional filtering."""
    vehicles = DEMO_VEHICLES
    
    if vehicle_id is not None:
        vehicles = [v for v in vehicles if v["id"] == vehicle_id]
    
    if refrigerated is not None:
        vehicles = [v for v in vehicles if v["refrigerated"] == refrigerated]
    
    return vehicles

def get_demo_suppliers(supplier_id: Optional[int] = None, category: Optional[str] = None) -> List[Dict]:
    """Get demo suppliers with optional filtering."""
    suppliers = DEMO_SUPPLIERS
    
    if supplier_id is not None:
        suppliers = [s for s in suppliers if s["id"] == supplier_id]
    
    if category:
        suppliers = [s for s in suppliers if category.lower() in [c.lower() for c in s["categories"]]]
    
    return suppliers

def get_demo_inventory_data(item_id: Optional[int] = None, 
                           sku: Optional[str] = None,
                           name: Optional[str] = None,
                           category: Optional[str] = None,
                           location_id: Optional[int] = None,
                           min_quantity: Optional[int] = None) -> List[Dict]:
    """Get demo inventory data with optional filtering."""
    
    # If item_id is provided, return specific item
    if item_id is not None:
        item = next((item for item in DEMO_INVENTORY if item["id"] == item_id), None)
        return [item] if item else []
    
    # Filter based on parameters
    filtered_items = DEMO_INVENTORY.copy()
    
    if sku:
        filtered_items = [item for item in filtered_items if item["sku"] == sku]
    if name:
        filtered_items = [item for item in filtered_items if name.lower() in item["name"].lower()]
    if category:
        filtered_items = [item for item in filtered_items if category.lower() in item["category"].lower()]
    if location_id:
        filtered_items = [item for item in filtered_items if item["location_id"] == location_id]
    if min_quantity:
        filtered_items = [item for item in filtered_items if item["quantity"] >= min_quantity]
    
    return filtered_items

def get_demo_analytics(report_type: str = "overview") -> Dict[str, Any]:
    """Get demo analytics data."""
    return DEMO_ANALYTICS

def is_api_error(response) -> bool:
    """Check if the response is an API error."""
    return (isinstance(response, dict) and "error" in response) or (
        isinstance(response, str) and ("401" in response or "Unauthorized" in response or "404" in response)
    ) or response is None 