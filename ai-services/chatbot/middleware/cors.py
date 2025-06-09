"""CORS middleware configuration"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

def setup_cors(
    app: FastAPI,
    allow_origins: List[str] = None,
    allow_credentials: bool = True,
    allow_methods: List[str] = None,
    allow_headers: List[str] = None,
    expose_headers: List[str] = None
) -> None:
    """
    Set up CORS middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        allow_origins: List of allowed origins
        allow_credentials: Whether to allow credentials
        allow_methods: List of allowed HTTP methods
        allow_headers: List of allowed headers
        expose_headers: List of headers to expose
    """
    # Default values
    if allow_origins is None:
        allow_origins = [
            "http://localhost:3000",  # React frontend
            "http://localhost:5173",  # Vite frontend
            "http://127.0.0.1:5173",  # Vite frontend alternative
            "*"  # Allow all origins for development
        ]
    
    if allow_methods is None:
        allow_methods = ["*"]
    
    if allow_headers is None:
        allow_headers = ["*"]
    
    if expose_headers is None:
        expose_headers = ["*"]
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        expose_headers=expose_headers
    ) 