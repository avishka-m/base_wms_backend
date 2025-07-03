from typing import Dict, Any, List, Optional, Type, Callable
from pydantic import BaseModel, create_model, Field

from langchain.tools import BaseTool
# Use absolute imports instead of relative imports
from app.utils.chatbot.api_client import api_client
from app.utils.chatbot.knowledge_base import knowledge_base

class WMSBaseTool(BaseTool):
    """Base class for all WMS tools."""
    
    def __init__(self, name: str, description: str, 
                 args_schema: Optional[Type[BaseModel]] = None):
        """
        Initialize the tool with name, description and args schema.
        
        Args:
            name: Tool name
            description: Tool description
            args_schema: Pydantic model for tool arguments
        """
        # Initialize BaseTool first
        super().__init__(
            name=name,
            description=description,
            args_schema=args_schema
        )
        
        # Store these attributes separately (not as Pydantic fields)
        self._tool_name = name
        self._tool_description = description
        self._args_schema = args_schema
    
    @property
    def args_schema(self) -> Type[BaseModel]:
        """Get the args schema."""
        return self._args_schema
    
    def _run(self, **kwargs) -> str:
        """Run the tool."""
        raise NotImplementedError("Subclasses must implement _run")
    
    async def _arun(self, **kwargs) -> str:
        """Run the tool asynchronously."""
        # Default to synchronous run for backwards compatibility
        return self._run(**kwargs)


def create_tool(name: str, description: str, function: Callable, 
               arg_descriptions: Dict[str, Dict[str, Any]]) -> WMSBaseTool:
    """
    Factory function to create a WMS tool.
    
    Args:
        name: Tool name
        description: Tool description
        function: Function that implements the tool's functionality
        arg_descriptions: Dictionary mapping argument names to field descriptions
        
    Returns:
        A WMSBaseTool instance
    """
    # Create dynamic Pydantic model for arguments
    fields = {}
    for arg_name, arg_info in arg_descriptions.items():
        fields[arg_name] = (arg_info.get("type", str), Field(
            description=arg_info.get("description", ""),
            **{k: v for k, v in arg_info.items() if k not in ["type", "description"]}
        ))
    
    args_schema = create_model(f"{name.capitalize()}Schema", **fields)
    
    # Create tool class
    class CustomTool(WMSBaseTool):
        def _run(self, **kwargs):
            return function(**kwargs)
    
    # Instantiate and return tool
    return CustomTool(name=name, description=description, args_schema=args_schema)