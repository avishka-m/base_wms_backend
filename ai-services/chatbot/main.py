import os
import logging
from datetime import datetime, timedelta
import sys
from jose import JWTError
import getpass
from contextlib import asynccontextmanager
from typing import Dict, Optional, Any, List

from fastapi import FastAPI, HTTPException, status, Depends, Security, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

# Add the parent directory to path to import from app modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import auth modules from main app
from app.auth.dependencies import get_current_user, get_current_active_user, has_role
from app.config import SECRET_KEY  # Import SECRET_KEY for token validation

# Import local configuration
from config import DEV_MODE, DEV_USER_ROLE

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

# Set up OAuth2 scheme for token authentication
# Point to the main application's token URL
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="http://localhost:8002/api/v1/auth/token", 
    auto_error=False
)

# Create a development mock user for testing
dev_mock_user = {
    "username": "dev_user",
    "role": DEV_USER_ROLE,
    "email": "dev@example.com",
    "full_name": "Development User",
    "disabled": False
}

# Define an optional dependency for authentication that can be bypassed in dev mode
async def get_optional_current_user(
    dev_mode: bool = Query(False, description="Enable development mode (no auth)"),
    token: str = Depends(oauth2_scheme)
) -> Dict[str, Any]:
    """
    Get the current user with option to bypass in development mode.
    
    Args:
        dev_mode: Flag to enable development mode
        token: OAuth2 token
        
    Returns:
        Current user information
    """
    # If DEV_MODE is enabled globally or dev_mode query param is True, return mock user
    if DEV_MODE or dev_mode:
        logger.warning("Using development mode: Authentication bypassed!")
        return dev_mock_user
        
    # Otherwise use normal authentication
    return await get_current_user(token)

# Role-based access with development mode support
def optional_has_role(required_roles: list):
    """
    Check if user has required role with option to bypass in development mode.
    
    Args:
        required_roles: List of roles that are allowed
        
    Returns:
        Dependency function that checks role
    """
    async def check_role(
        dev_mode: bool = Query(False, description="Enable development mode (no auth)"),
        current_user: Dict[str, Any] = Depends(get_optional_current_user)
    ):
        # In dev mode, always allow if using the default dev user
        if (DEV_MODE or dev_mode) and current_user.get("username") == "dev_user":
            logger.warning(f"Development mode: Role check for {required_roles} bypassed!")
            return dev_mock_user
            
        # Otherwise check roles normally
        if current_user.get("role") not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have sufficient privileges. Required roles: {required_roles}"
            )
        return current_user
        
    return check_role

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

class NewConversation(BaseModel):
    """Create a new conversation"""
    title: str = Field(..., description="Title of the conversation")
    role: str = Field(..., description="Role to use for this conversation")

# Simple in-memory conversation store
# In production, this should be replaced with a database
# conversations = {}  # Old implementation - shared for all users
user_conversations = {}  # Store conversations by user ID

@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "version": app.version,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/conversations", response_model=Dict[str, Any])
async def create_conversation(
    data: NewConversation,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Create a new conversation for the user.
    
    Args:
        data: New conversation data
        current_user: Current authenticated user
        
    Returns:
        New conversation information
    """
    # Get user ID for storage
    user_id = current_user.get("username", "anonymous")
    
    # Validate role exists
    if data.role.lower() not in agents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {data.role}. Must be one of {list(agents.keys())}"
        )
    
    # Create conversation ID with timestamp to make it unique
    conversation_id = f"{data.role.lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Initialize user's conversations dictionary if it doesn't exist
    if user_id not in user_conversations:
        user_conversations[user_id] = {}
    
    # Store the new conversation with metadata
    user_conversations[user_id][conversation_id] = {
        "metadata": {
            "title": data.title,
            "created_at": datetime.now().isoformat(),
            "role": data.role.lower(),
        },
        "messages": []
    }
    
    return {
        "conversation_id": conversation_id,
        "title": data.title,
        "role": data.role.lower(),
        "created_at": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    message: ChatMessage, 
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Chat with the WMS chatbot.
    
    Args:
        message: User message
        current_user: Current authenticated user or dev user in dev mode
        
    Returns:
        Chatbot response
    """
    role = message.role.lower()
    
    # Get user ID for storage
    user_id = current_user.get("username", "anonymous")
    
    # Validate role exists
    if role not in agents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {role}. Must be one of {list(agents.keys())}"
        )
    
    # Validate user has access to this chatbot role
    allowed_roles = get_allowed_chatbot_roles(current_user.get("role", ""))
    if role not in allowed_roles and current_user.get("role") != "Manager":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User with role '{current_user.get('role')}' cannot access the '{role}' chatbot"
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
        
        # Initialize user's conversations dictionary if it doesn't exist
        if user_id not in user_conversations:
            user_conversations[user_id] = {}
            
        # Create conversation if it doesn't exist
        if conversation_id not in user_conversations[user_id]:
            user_conversations[user_id][conversation_id] = {
                "metadata": {
                    "title": f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "created_at": datetime.now().isoformat(),
                    "role": role,
                },
                "messages": []
            }
            
        # Store the message and response
        user_conversations[user_id][conversation_id]["messages"].append({
            "role": "user",
            "content": message.message,
            "timestamp": datetime.now().isoformat()
        })
        
        user_conversations[user_id][conversation_id]["messages"].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim conversation history if it gets too long
        if len(user_conversations[user_id][conversation_id]["messages"]) > 100:
            user_conversations[user_id][conversation_id]["messages"] = user_conversations[user_id][conversation_id]["messages"][-100:]
        
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
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get a conversation history.
    
    Args:
        conversation_id: ID of the conversation
        current_user: Current authenticated user or dev user in dev mode
        
    Returns:
        Conversation history
    """
    # Get user ID for storage
    user_id = current_user.get("username", "anonymous")
    
    # Check if user has this conversation
    if user_id not in user_conversations or conversation_id not in user_conversations[user_id]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )
        
    return {
        "conversation_id": conversation_id,
        "metadata": user_conversations[user_id][conversation_id]["metadata"],
        "messages": user_conversations[user_id][conversation_id]["messages"]
    }

@app.put("/api/conversations/{conversation_id}", response_model=Dict[str, Any])
async def update_conversation(
    conversation_id: str,
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Update conversation metadata (title)
    
    Args:
        conversation_id: ID of the conversation
        data: Updated conversation data
        current_user: Current authenticated user
        
    Returns:
        Updated conversation information
    """
    # Get user ID for storage
    user_id = current_user.get("username", "anonymous")
    
    # Check if user has this conversation
    if user_id not in user_conversations or conversation_id not in user_conversations[user_id]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )
    
    # Update title if provided
    if "title" in data:
        user_conversations[user_id][conversation_id]["metadata"]["title"] = data["title"]
    
    return {
        "conversation_id": conversation_id,
        "metadata": user_conversations[user_id][conversation_id]["metadata"],
        "success": True
    }

@app.get("/api/conversations", response_model=List[Dict[str, Any]])
async def get_user_conversations(
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Get all conversations for the current user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List of all user conversations
    """
    # Get user ID for storage
    user_id = current_user.get("username", "anonymous")
    
    # Return empty list if user has no conversations
    if user_id not in user_conversations:
        return []
    
    # Return conversations with metadata
    return [
        {
            "conversation_id": conv_id,
            "title": data["metadata"].get("title", f"Chat {conv_id}"),
            "role": data["metadata"].get("role", "unknown"),
            "created_at": data["metadata"].get("created_at", ""),
            "message_count": len(data["messages"]),
            "last_updated": data["messages"][-1]["timestamp"] if data["messages"] else data["metadata"].get("created_at", "")
        }
        for conv_id, data in user_conversations[user_id].items()
    ]

@app.delete("/api/conversations/{conversation_id}", response_model=Dict[str, Any])
async def delete_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """
    Delete a user conversation.
    
    Args:
        conversation_id: ID of the conversation
        current_user: Current authenticated user
        
    Returns:
        Success message
    """
    # Get user ID for storage
    user_id = current_user.get("username", "anonymous")
    
    # Check if user has this conversation
    if user_id not in user_conversations or conversation_id not in user_conversations[user_id]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )
        
    # Delete the conversation
    del user_conversations[user_id][conversation_id]
    
    return {
        "success": True,
        "message": f"Conversation {conversation_id} deleted successfully"
    }

@app.get("/api/user/role", response_model=Dict[str, Any])
async def get_user_role(current_user: Dict[str, Any] = Depends(get_optional_current_user)):
    """
    Get the current user's role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User role information
    """
    return {
        "username": current_user.get("username"),
        "role": current_user.get("role"),
        "allowed_chatbot_roles": get_allowed_chatbot_roles(current_user.get("role", ""))
    }

def get_allowed_chatbot_roles(user_role: str) -> List[str]:
    """
    Determine which chatbot roles a user can access based on their role.
    
    Args:
        user_role: User's role in the system
        
    Returns:
        List of chatbot roles the user can access
    """
    role_mapping = {
        "Manager": ["clerk", "picker", "packer", "driver", "manager"],
        "ClerkSupervisor": ["clerk"],
        "PickerSupervisor": ["picker"],
        "PackerSupervisor": ["packer"],
        "DriverSupervisor": ["driver"],
        "Clerk": ["clerk"],
        "Picker": ["picker"],
        "Packer": ["packer"],
        "Driver": ["driver"]
    }
    
    return role_mapping.get(user_role, [])

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)