import os
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.base import (
    MONGODB_URL, DATABASE_NAME, API_V1_PREFIX, 
    PROJECT_NAME as BASE_PROJECT_NAME, PROJECT_VERSION, LOGGING_LEVEL, DEBUG_MODE
)

# Load environment variables
load_dotenv()

# Database configuration

MONGODB_URL = os.getenv("MONGODB_URL", "")

DATABASE_NAME = os.getenv("DATABASE_NAME", "warehouse_management")

# API configuration
API_V1_PREFIX = "/api/v1"
PROJECT_NAME = "Warehouse Management System"
PROJECT_DESCRIPTION = "A comprehensive warehouse management system with inventory tracking, order processing, and logistics optimization"

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Application settings (inherited from base config)
# LOGGING_LEVEL and DEBUG_MODE are imported from base

# Chatbot configuration
DEV_MODE = os.getenv("DEV_MODE", "True").lower() == "true"
DEV_USER_ROLE = os.getenv("DEV_USER_ROLE", "Manager")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Development mode settings
DEV_MODE = os.getenv("DEV_MODE", "True").lower() == "true"
DEV_USER_ROLE = os.getenv("DEV_USER_ROLE", "Manager")

# Chroma DB Configuration
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "wms_knowledge")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Document types for knowledge base
DOCUMENT_TYPES = {
    ".txt": "text",
    ".pdf": "pdf", 
    ".csv": "csv",
    ".md": "text"
}

# WMS API Configuration
WMS_API_BASE_URL = os.getenv("WMS_API_BASE_URL", "http://localhost:8002/api/v1")

# API Endpoints for main WMS backend
API_ENDPOINTS = {
    "inventory": "/inventory",
    "orders": "/orders",
    "workers": "/workers",
    "customers": "/customers",
    "locations": "/locations",
    "receiving": "/receiving",
    "picking": "/picking",
    "packing": "/packing",
    "shipping": "/shipping",
    "returns": "/returns",
    "vehicles": "/vehicles",
    "analytics": "/analytics"
}

# Chatbot roles
ROLES = {
    "clerk": {
        "name": "Inventory Clerk",
        "system_instructions": "You are a helpful warehouse inventory clerk assistant specializing in inventory management, stock tracking, and supplier coordination."
    },
    "picker": {
        "name": "Order Picker",
        "system_instructions": "You are a helpful warehouse order picker assistant specializing in order fulfillment, item location, and picking optimization."
    }, 
    "packer": {
        "name": "Packer",
        "system_instructions": "You are a helpful warehouse packer assistant specializing in order packing, shipping preparation, and packaging optimization."
    },
    "driver": {
        "name": "Driver",
        "system_instructions": "You are a helpful warehouse driver assistant specializing in delivery routes, vehicle management, and transportation logistics."
    },
    "manager": {
        "name": "Manager",
        "system_instructions": "You are a helpful warehouse manager assistant with access to all warehouse operations, analytics, and management functions."
    }
}
