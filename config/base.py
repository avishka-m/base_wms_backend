
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Dict, Any

"""
Shared Configuration Settings
This file contains settings that are common across different modules of the WMS system.
"""


PROJECT_NAME = "Intelligent Warehouse Management System"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "A comprehensive warehouse management system with AI-powered features"

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "warehouse_management")


API_V1_PREFIX = "/api/v1"


# SHARED ENVIRONMENT SETTINGS

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


# SHARED LOGGING CONFIGURATION

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"


# SHARED UTILITY FUNCTION
def get_database_config() -> Dict[str, str]:
    """Get shared database configuration."""
    return {
        "mongodb_url": MONGODB_URL,
        "database_name": DATABASE_NAME
    }

def validate_base_config() -> bool:
    """Validate shared configuration settings."""
    required_vars = [
        "MONGODB_URL",
        "DATABASE_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    return True

# Validate configuration on import
validate_base_config()
