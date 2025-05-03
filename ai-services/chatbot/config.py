import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangSmith configuration
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "wms_chatbot")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "warehouse_management")

# WMS API configuration
WMS_API_BASE_URL = os.getenv("WMS_API_BASE_URL", "http://localhost:8002/api/v1")

# Vector DB configuration
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "wms_knowledge")

# LLM configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Role definitions and permissions
ROLES = {
    "clerk": {
        "description": "Receiving Clerk: Adds items from pre-approved list and handles returns",
        "allowed_tools": ["inventory_add"],
        "system_instructions": "You are a helpful warehouse receiving clerk assistant. You can help with receiving new inventory, processing returns, and checking inventory levels. Always verify item details before adding to inventory.",
    },
    "picker": {
        "description": "Picker: Places received items in the right location and fulfills orders",
        "allowed_tools": ["inventory_query", "create_picking_task", "update_picking_task", "path_optimize", "locate_item"],
        "system_instructions": "You are a helpful warehouse picker assistant. You help with picking items for orders, finding the most efficient paths in the warehouse, and updating picking task status.",
    },
    "packer": {
        "description": "Packer: Verifies order completeness and creates sub-orders if necessary",
        "allowed_tools": ["inventory_query", "create_packing_task", "update_packing_task", "create_sub_order", "check_order"],
        "system_instructions": "You are a helpful warehouse packer assistant. You help with packing orders, verifying order completeness, and creating sub-orders when necessary.",
    },
    "driver": {
        "description": "Driver: Selects vehicles and updates order status to Shipped",
        "allowed_tools": ["create_shipping_task", "update_shipping_task", "vehicle_select", "check_order", "calculate_route"],
        "system_instructions": "You are a helpful delivery driver assistant. You help with selecting appropriate vehicles for deliveries, updating shipping status, and finding optimal routes.",
    },
    "manager": {
        "description": "Manager: Full control over inventory, workers, and system management",
        "allowed_tools": ["inventory_query", "inventory_add", "inventory_update", "worker_manage", 
                         "check_analytics", "approve_orders", "system_manage", "check_anomalies"],
        "system_instructions": "You are a helpful warehouse manager assistant. You have full access to inventory management, worker management, and system analytics. You can help with high-level decision making and approval processes.",
    }
}

# Document types for knowledge base
DOCUMENT_TYPES = {
    "sop": "Standard Operating Procedures",
    "product": "Product Information",
    "equipment": "Equipment Manuals",
    "safety": "Safety Procedures",
    "training": "Training Materials",
    "policy": "Company Policies"
}

# Knowledge chunks size
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# API endpoints for tools
API_ENDPOINTS = {
    "inventory": f"{WMS_API_BASE_URL}/inventory",
    "orders": f"{WMS_API_BASE_URL}/orders",
    "workers": f"{WMS_API_BASE_URL}/workers",
    "locations": f"{WMS_API_BASE_URL}/locations",
    "picking": f"{WMS_API_BASE_URL}/picking",
    "packing": f"{WMS_API_BASE_URL}/packing",
    "shipping": f"{WMS_API_BASE_URL}/shipping",
    "receiving": f"{WMS_API_BASE_URL}/receiving",
    "returns": f"{WMS_API_BASE_URL}/returns",
    "vehicles": f"{WMS_API_BASE_URL}/vehicles",
    "customers": f"{WMS_API_BASE_URL}/customers"
}