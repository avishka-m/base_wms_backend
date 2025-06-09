"""FastAPI application factory"""

from fastapi import FastAPI

from .lifespan import lifespan_handler
from middleware.cors import setup_cors
from api import health_router, chat_router, user_router, conversations_router

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    # Create app with lifespan handler
    app = FastAPI(
        title="WMS Chatbot API",
        description="Warehouse Management System Chatbot API for role-based assistance",
        version="1.0.0",
        lifespan=lifespan_handler
    )
    
    # Set up CORS
    setup_cors(app)
    
    # Include routers
    app.include_router(health_router)
    app.include_router(chat_router, prefix="/api")
    app.include_router(user_router, prefix="/api")
    app.include_router(conversations_router)  # Already has /api/conversations prefix
    
    return app 