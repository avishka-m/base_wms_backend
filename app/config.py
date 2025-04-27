import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "warehouse_management")

# API configuration
API_V1_PREFIX = "/api/v1"
PROJECT_NAME = "Warehouse Management System"
PROJECT_DESCRIPTION = "A comprehensive warehouse management system with inventory tracking, order processing, and logistics optimization"
PROJECT_VERSION = "1.0.0"

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Application settings
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
