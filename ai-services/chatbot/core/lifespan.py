"""Application lifespan management"""

import os
import getpass
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI

from agents.clerk_agent import ClerkAgent
from agents.picker_agent import PickerAgent
from agents.packer_agent_ex import PackerAgent
from agents.driver_agent import DriverAgent
from agents.manager_agent import ManagerAgent
from .logging import logger

# Global agents dictionary
agents: Dict[str, object] = {}

def initialize_agents():
    """Initialize all agent instances"""
    global agents
    
    agents = {
        "clerk": ClerkAgent(),
        "picker": PickerAgent(),
        "packer": PackerAgent(),
        "driver": DriverAgent(),
        "manager": ManagerAgent()
    }
    
    logger.info(f"Initialized agents for roles: {', '.join(agents.keys())}")
    return agents

def setup_environment():
    """Set up environment variables for LangSmith and OpenAI"""
    
    # Set up LangSmith (optional)
    os.environ["LANGSMITH_TRACING"] = os.environ.get("LANGSMITH_TRACING", "false")

    # Only prompt if not already set and running interactively
    if "LANGSMITH_API_KEY" not in os.environ and os.isatty(0):
        try:
            api_key = getpass.getpass(
                prompt="Enter your LangSmith API key (optional, press Enter to skip): "
            )
            if api_key:
                os.environ["LANGSMITH_API_KEY"] = api_key
                os.environ["LANGSMITH_TRACING"] = "true"
        except (KeyboardInterrupt, EOFError):
            print("\nSkipping LangSmith setup...")

    if "LANGSMITH_PROJECT" not in os.environ and os.environ.get("LANGSMITH_API_KEY"):
        try:
            project = getpass.getpass(
                prompt='Enter your LangSmith Project Name (default = "default"): '
            )
            os.environ["LANGSMITH_PROJECT"] = project or "default"
        except (KeyboardInterrupt, EOFError):
            os.environ["LANGSMITH_PROJECT"] = "default"

    # Set up OpenAI (optional if already set)
    if "OPENAI_API_KEY" not in os.environ and os.isatty(0):
        try:
            api_key = getpass.getpass(
                prompt="Enter your OpenAI API key (required if using OpenAI, press Enter to skip): "
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        except (KeyboardInterrupt, EOFError):
            print("\nSkipping OpenAI setup...")
            logger.warning("OpenAI API key not set. Some features may not work.")

@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    """
    Handle application lifespan events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting WMS Chatbot API...")
    
    # Set up environment
    setup_environment()
    
    # Initialize agents
    initialize_agents()
    
    logger.info("WMS Chatbot API started successfully")
    print("Agents initialized and ready to use.")

    yield  # App runs here

    # Shutdown
    logger.info("WMS Chatbot API shutting down...")
    print("Shutting down WMS Chatbot API...")
    
    # Clean up agents if needed
    agents.clear()
    
    logger.info("WMS Chatbot API shut down successfully")

def get_agents() -> Dict[str, object]:
    """Get the initialized agents dictionary"""
    return agents 