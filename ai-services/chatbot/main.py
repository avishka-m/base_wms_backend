from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager
import getpass
import sys
import os
import logging
from datetime import datetime

# Import agents
from agents.clerk_agent import ClerkAgent
from agents.picker_agent import PickerAgent
from agents.packer_agent_ex import PackerAgent
from agents.driver_agent import DriverAgent
from agents.manager_agent import ManagerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chatbot.log")
    ]
)
logger = logging.getLogger("wms_chatbot")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources before the app starts"""
    
    # Set up LangSmith and OpenAI API keys
    os.environ["LANGSMITH_TRACING"] = "true"

    if "LANGSMITH_API_KEY" not in os.environ:
        os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
            prompt="Enter your LangSmith API key (optional): "
        )

    if "LANGSMITH_PROJECT" not in os.environ:
        os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
            prompt='Enter your LangSmith Project Name (default = "default"): '
        )
        if not os.environ.get("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = "default"

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass(
            prompt="Enter your OpenAI API key (required if using OpenAI): "
        )

    # Initialize agents
    global agents
    agents = {
        "clerk": ClerkAgent(),
        "picker": PickerAgent(),
        "packer": PackerAgent(),
        "driver": DriverAgent(),
        "manager": ManagerAgent()
    }
    
    logger.info(f"Initialized agents for roles: {', '.join(agents.keys())}")
    
    print("Agents initialized and ready to use.")

    yield  # App runs here

    # Optional cleanup logic goes here after app shutdown
    print("Shutting down WMS Chatbot API...")
    logger.info("WMS Chatbot API shutting down")

# Create FastAPI app with lifespan
app = FastAPI(
    title="WMS Chatbot API",
    description="Warehouse Management System Chatbot API for role-based assistance",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # "http://localhost:3000",  # React frontend
        # "http://localhost:5173",  # Vite frontend
        # "http://127.0.0.1:5173",  # Vite frontend alternative
        "*" # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize empty agents dict that will be populated during startup
agents = {}

# Pydantic models
class ChatMessage(BaseModel):
    """Chat message sent by a user."""
    role: str = Field(..., description="Role of the user sending the message (clerk, picker, packer, driver, manager)")
    message: str = Field(..., description="Message content")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for continued conversations")

class ChatResponse(BaseModel):
    """Response from the chatbot."""
    response: str = Field(..., description="Chatbot response message")
    conversation_id: str = Field(..., description="Conversation ID for continued conversations")
    timestamp: str = Field(..., description="Timestamp of the response")

class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Status of the service")
    version: str = Field(..., description="Version of the API")
    timestamp: str = Field(..., description="Timestamp of the health check")

# Simple in-memory conversation store
# In production, this should be replaced with a database
conversations = {}

@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "version": app.version,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Chat with the WMS chatbot.
    
    Args:
        message: User message
        
    Returns:
        Chatbot response
    """
    role = message.role.lower()
    
    # Validate role exists
    if role not in agents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {role}. Must be one of {list(agents.keys())}"
        )
    
    try:
        # Get the appropriate agent
        agent = agents[role]
        
        # Log the incoming message
        logger.info(f"Received message from {role}: {message.message[:100]}")
        
        # Process the message through the agent
        response_text = agent.run(message.message)
        
        # Create or update conversation
        conversation_id = message.conversation_id or f"{role}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if conversation_id not in conversations:
            conversations[conversation_id] = []
            
        # Store the message and response
        conversations[conversation_id].append({
            "role": role,
            "message": message.message,
            "response": response_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim conversation history if it gets too long
        if len(conversations[conversation_id]) > 20:
            conversations[conversation_id] = conversations[conversation_id][-20:]
        
        # Return the response
        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )

@app.get("/api/conversations/{conversation_id}", response_model=Dict[str, Any])
async def get_conversation(conversation_id: str):
    """
    Get a conversation history.
    
    Args:
        conversation_id: ID of the conversation
        
    Returns:
        Conversation history
    """
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )
        
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id]
    }

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)