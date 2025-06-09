"""Main entry point for the WMS Chatbot API"""

import uvicorn
from core import create_app

# Create the FastAPI application
app = create_app()

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main_new:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info"
    ) 