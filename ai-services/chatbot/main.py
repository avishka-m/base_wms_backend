from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
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

# Add backend path to import auth modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from app.auth.dependencies import get_current_user, get_current_active_user

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

# Initialize OAuth2 scheme with the same token URL as the main app
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React frontend
        "http://localhost:5173",  # Vite frontend
        "http://127.0.0.1:5173",  # Vite frontend alternative
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

# Authentication helper function
async def validate_user_role(token: str = Depends(oauth2_scheme), role: Optional[str] = None):
    """
    Validate the user's token and check if they have the correct role.
    
    Args:
        token: JWT token
        role: Required role (optional)
        
    Returns:
        User data if authenticated, else raises 401/403
    """
    try:
        # Get the current user from the token
        user = await get_current_active_user(Depends(get_current_user))
        
        # If a specific role is required, check that the user has it
        if role and user.get("role").lower() != role.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. You need {role} role to use this agent."
            )
            
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )

# API routes
@app.get("/", response_model=HealthCheckResponse)
async def root(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "version": app.version,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    message: ChatMessage, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Chat with the WMS chatbot.
    
    Args:
        message: User message
        current_user: Authenticated user information
        
    Returns:
        Chatbot response
    """
    role = message.role.lower()
    
    # Validate role
    if role not in agents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {role}. Must be one of {list(agents.keys())}"
        )
    
    # Verify the user's role matches the requested agent or is a manager
    user_role = current_user.get("role", "").lower()
    if user_role != role and user_role != "manager":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied. You need {role.capitalize()} role to use this agent."
        )
    
    try:
        # Get the appropriate agent
        agent = agents[role]
        
        # Log the incoming message
        logger.info(f"Received message from {role} (user: {current_user.get('username')}): {message.message[:100]}")
        
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
async def get_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get a conversation history.
    
    Args:
        conversation_id: ID of the conversation
        current_user: Authenticated user information
        
    Returns:
        Conversation history
    """
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )
    
    # Ensure the user has access to this conversation
    # Extract the role from conversation_id (format: role_timestamp)
    try:
        conversation_role = conversation_id.split("_")[0]
        user_role = current_user.get("role", "").lower()
        
        # Only allow access if user role matches the conversation role or user is a manager
        if user_role != conversation_role.lower() and user_role != "manager":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this conversation"
            )
    except Exception:
        # If we can't determine the role from the ID, only allow managers
        if current_user.get("role", "").lower() != "manager":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this conversation"
            )
        
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id]
    }

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)