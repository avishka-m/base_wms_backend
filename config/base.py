"""
Shared Configuration Settings
This file contains settings that are common across different modules of the WMS system.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# SHARED PROJECT INFORMATION
# =============================================================================
PROJECT_NAME = "Intelligent Warehouse Management System"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "A comprehensive warehouse management system with AI-powered features"

# =============================================================================
# SHARED DATABASE CONFIGURATION
# =============================================================================
# Import optimized Atlas configuration
try:
    from atlas_optimization import get_database_url
    MONGODB_URL = get_database_url()
except ImportError:
    # Fallback to environment variable or localhost
    MONGODB_URL = os.getenv("MONGODB_URL", "")

DATABASE_NAME = os.getenv("DATABASE_NAME", "warehouse_management")

# =============================================================================
# SHARED API CONFIGURATION
# =============================================================================
API_V1_PREFIX = "/api/v1"

# =============================================================================
# SHARED ENVIRONMENT SETTINGS
# =============================================================================
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# =============================================================================
# SHARED LOGGING CONFIGURATION
# =============================================================================
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# =============================================================================
# SHARED UTILITY FUNCTIONS
# =============================================================================
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
