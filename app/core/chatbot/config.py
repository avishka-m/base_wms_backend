"""
Application configuration for the WMS Chatbot.
"""

import os
from typing import List

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class AppConfig:
    """Application configuration settings."""
    
    TITLE = "WMS Chatbot API"
    DESCRIPTION = "Warehouse Management System Chatbot API for role-based assistance"
    VERSION = "1.0.0"
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        # "http://localhost:3000",  # React frontend
        # "http://localhost:5173",  # Vite frontend  
        # "http://localhost:5174",  # Vite frontend v2
        # "http://127.0.0.1:5173",  # Vite frontend alternative
        # "http://127.0.0.1:5174",  # Vite frontend v2 alternative
        "*"  # Allow all origins for development
    ]
    
    ALLOW_CREDENTIALS = True
    ALLOWED_METHODS = ["*"]
    ALLOWED_HEADERS = ["*"]
    EXPOSE_HEADERS = ["*"]

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

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

# Development mode settings
DEV_MODE = os.getenv("DEV_MODE", "True").lower() == "true"
DEV_USER_ROLE = os.getenv("DEV_USER_ROLE", "Manager")
