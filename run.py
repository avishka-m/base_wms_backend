#!/usr/bin/env python3
"""
Run script for the Warehouse Management System API

This script starts the FastAPI application using uvicorn.
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get debug mode from environment
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

if __name__ == "__main__":
    # Start the uvicorn server
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8002,
        reload=True 
    )