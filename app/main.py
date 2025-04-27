from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from .config import (
    API_V1_PREFIX,
    PROJECT_NAME,
    PROJECT_DESCRIPTION,
    PROJECT_VERSION,
    DEBUG_MODE
)
from .api.routes import api_router
from .auth.dependencies import get_current_user

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
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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
