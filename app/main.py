from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from .config import (
    API_V1_PREFIX,
    PROJECT_NAME,
    PROJECT_DESCRIPTION,
    PROJECT_VERSION,
    DEBUG_MODE
)
from .api.routes import api_router
from .auth.dependencies import get_current_user

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=PROJECT_NAME,
    description=PROJECT_DESCRIPTION,
    version=PROJECT_VERSION,
    debug=DEBUG_MODE,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",  # Alternative localhost
        "http://localhost:3000",  # React dev server alternative
        "http://127.0.0.1:3000",  # Alternative localhost
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Custom exception handler to ensure CORS headers are always included
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom exception handler to ensure CORS headers are included in error responses.
    """
    response = JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
    
    # Add CORS headers manually for error responses
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    
    return response

# Include API router
app.include_router(api_router, prefix=API_V1_PREFIX)

# Root endpoint
@app.get("/")
def read_root():
    return {
        "status": "online",
        "name": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "message": "Welcome to the Warehouse Management System API"
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Protected test endpoint
@app.get("/protected")
def protected_route(user = Depends(get_current_user)):
    return {"message": f"Hello, {user['username']}! You have access to protected routes."}
