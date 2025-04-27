from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema import SystemMessage as LCSystemMessage

# Import LangSmith tracing capabilities
from langsmith import traceable

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, ROLES
from utils.knowledge_base import knowledge_base

class BaseAgent:
    """Base agent for all warehouse roles."""
    
    def __init__(self, role: str):
        """
        Initialize the base agent with a specific role.
        
        Args:
            role: The role of the agent (clerk, picker, packer, driver, manager)
        """
        if role not in ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {list(ROLES.keys())}")
        
        self.role = role
        self.role_config = ROLES[role]
        
        # Configure tracing options
        self.tracing_enabled = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
        self.project_name = os.environ.get("LANGSMITH_PROJECT", "warehouse-management-system")
        
        # Initialize the LLM with tracing tags
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            tags=[f"role:{role}", "wms-chatbot"],
        )
        
        # Each agent subclass will override this with direct tool imports
        self.tools = []
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Build the system prompt
        self.system_prompt = self._build_system_prompt()
        
        # Create the agent executor once tools are set
        self.agent_executor = None  # Will be created after tools are set
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent based on role and tools."""
        current_date = datetime.now().strftime("%B %d, %Y")
        
        system_message = self.role_config.get("system_instructions", "") + f"\n\nCurrent date: {current_date}\n"
        
        if self.tools:
            tool_descriptions = "\n\nYou have access to the following tools:\n\n"
            for tool in self.tools:
                tool_descriptions += f"- {tool.name}: {tool.description}\n"
            system_message += tool_descriptions

        system_message += "\n\nGuidelines:\n"
        system_message += "1. Only use the tools available to you for your role.\n"
        system_message += "2. If you don't have the necessary permissions or tools for a request, explain what the user should do instead.\n"
        system_message += "3. When providing information, be specific and actionable.\n"
        system_message += "4. If you're unsure about warehouse procedures, check the knowledge base first.\n"
        system_message += "5. Always maintain data security and only share information appropriate to the user's role.\n"
        system_message += "6. Be concise and clear in your responses.\n"

        return system_message

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor with appropriate tools and knowledge."""
        if not self.tools:
            raise ValueError(f"No tools defined for {self.role} agent. Add tools before creating the agent executor.")
            
        # Create the prompt template with the correct variable name (history)
        prompt = ChatPromptTemplate.from_messages([
            LCSystemMessage(content=self.system_prompt),
            MessagesPlaceholder(variable_name="history"),  # Changed from "chat_history" to "history"
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the OpenAI functions agent
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        
        # Create the agent executor with tags for tracing
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,  # Prevent infinite loops
            tags=[f"role:{self.role}", "wms-chatbot"],  # Tags for LangSmith tracing
        )

    @traceable(name="agent_run", run_type="chain")
    def run(self, query: str) -> str:
        """
        Run the agent on a user query.
        
        Args:
            query: User query string
            
        Returns:
            Agent response string
        """
        try:
            # Create agent executor if not yet created
            if self.agent_executor is None and self.tools:
                self.agent_executor = self._create_agent_executor()
                
            if not self.agent_executor:
                return f"Error: Agent for role '{self.role}' is not properly initialized with tools."
            
            # Enhance the query with role-specific context
            enhanced_query = self.enhance_query(query)
            
            # Enrich with knowledge base information
            kb_results = self.query_knowledge_base(enhanced_query)
            if kb_results:
                kb_context = "\n\n".join(kb_results[:2])  # Limit to 2 most relevant results
                enriched_query = f"{enhanced_query}\n\nRelevant information from knowledge base:\n{kb_context}"
            else:
                enriched_query = enhanced_query
            
            # Add metadata for LangSmith tracing
            metadata = {
                "role": self.role,
                "query_type": self._classify_query_type(query),
                "has_kb_results": len(kb_results) > 0,
                "kb_results_count": len(kb_results)
            }
            
            # Run the agent with tracing enabled
            result = self.agent_executor.invoke(
                {"input": enriched_query},
                config={"metadata": metadata} if self.tracing_enabled else {}
            )
            return result["output"]
        except Exception as e:
            # Provide helpful error message if agent fails
            import traceback
            print(f"Error running agent: {str(e)}")
            print(traceback.format_exc())
            return f"I encountered an error while processing your request. Please try again with a clearer question about warehouse operations."

    @traceable(name="query_knowledge_base", run_type="retriever")
    def query_knowledge_base(self, query: str, n_results: int = 3) -> List[str]:
        """
        Query the knowledge base for relevant information.
        
        Args:
            query: The query string
            n_results: Number of results to return
            
        Returns:
            List of relevant document contents
        """
        try:
            results = knowledge_base.query(query, n_results=n_results)
            return [doc.page_content for doc in results]
        except Exception as e:
            print(f"Error querying knowledge base: {e}")
            return []
    
    def _classify_query_type(self, query: str) -> str:
        """
        Classify the type of query for better tracing in LangSmith.
        
        Args:
            query: The user query
            
        Returns:
            Query classification string
        """
        query_lower = query.lower()
        
        if "inventory" in query_lower or "stock" in query_lower:
            return "inventory"
        elif "order" in query_lower:
            return "orders"
        elif "location" in query_lower or "where" in query_lower:
            return "location"
        elif "return" in query_lower or "refund" in query_lower:
            return "returns"
        elif "ship" in query_lower or "deliver" in query_lower:
            return "shipping"
        elif "pack" in query_lower:
            return "packing"
        elif "pick" in query_lower:
            return "picking"
        elif "worker" in query_lower or "staff" in query_lower:
            return "workforce"
        else:
            return "general"

    def enhance_query(self, query: str) -> str:
        """
        Enhance a user query with role-specific context.
        This method should be overridden by subclasses to add role-specific behavior.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query string
        """
        # Default implementation just returns the original query
        return query
