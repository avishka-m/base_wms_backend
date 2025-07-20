from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime
import uuid

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema import SystemMessage as LCSystemMessage

# Import LangSmith tracing capabilities
from langsmith import traceable, Client
from langchain_core.runnables.config import RunnableConfig

from app.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, ROLES
from app.utils.chatbot.knowledge_base import knowledge_base
from app.services.chatbot.conversation_memory_service import conversation_memory_service

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
        
        # Initialize LangSmith client if tracing is enabled
        self.langsmith_client = None
        if self.tracing_enabled:
            try:
                self.langsmith_client = Client()
                print(f"LangSmith tracing enabled for {role} agent")
            except Exception as e:
                print(f"Warning: Failed to initialize LangSmith client: {e}")
                self.tracing_enabled = False
        
        # Initialize the LLM with tracing tags
        if OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                tags=[f"role:{role}", "wms-chatbot"],
            )
        else:
            print(f"Warning: OPENAI_API_KEY not set. Agent {role} will use mock responses.")
            self.llm = None
        
        # Each agent subclass will override this with direct tool imports
        self.tools = []
        
        # Build the system prompt
        self.system_prompt = self._build_system_prompt()
        
        # Agent executor will be created dynamically with conversation context
        self._agent_executor_cache = {}  # Cache for conversation-specific executors
    
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

    async def _create_agent_executor(
        self, 
        conversation_id: str, 
        user_id: str
    ) -> AgentExecutor:
        """Create the LangChain agent executor with appropriate tools and conversation memory."""
        if not self.tools:
            raise ValueError(f"No tools defined for {self.role} agent. Add tools before creating the agent executor.")
        
        # Get conversation memory from the memory service
        memory = await conversation_memory_service.get_memory(
            conversation_id=conversation_id,
            user_id=user_id,
            agent_role=self.role
        )
            
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
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,  # Prevent infinite loops
            tags=[f"role:{self.role}", "wms-chatbot"],  # Tags for LangSmith tracing
        )

    @traceable(
        name="agent_run", 
        run_type="chain",
        project_name="warehouse-management-system"
    )
    async def run(
        self, 
        query: str, 
        conversation_id: str = "default", 
        user_id: str = "anonymous"
    ) -> str:
        """
        Run the agent on a user query with conversation memory.
        
        Args:
            query: User query string
            conversation_id: Unique conversation identifier for memory persistence
            user_id: User identifier for memory management
            
        Returns:
            Agent response string
        """
        try:
            # Check if LLM is available
            if self.llm is None:
                return f"Mock response for {self.role}: {query}"
            
            # Get or create agent executor with conversation memory
            executor_key = f"{conversation_id}:{user_id}"
            
            if executor_key not in self._agent_executor_cache:
                agent_executor = await self._create_agent_executor(conversation_id, user_id)
                self._agent_executor_cache[executor_key] = agent_executor
            else:
                agent_executor = self._agent_executor_cache[executor_key]
                
            if not agent_executor:
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
            
            # Create thread ID for this conversation
            thread_id = f"{conversation_id}_{user_id}_{int(datetime.now().timestamp())}"
            
            # Add metadata for LangSmith tracing
            metadata = {
                "role": self.role,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "query_type": self._classify_query_type(query),
                "has_kb_results": len(kb_results) > 0,
                "kb_results_count": len(kb_results),
                "thread_id": thread_id,
                "enhanced_query": len(enriched_query) > len(query),
                "agent_type": f"{self.role}_agent"
            }
            
            # Create runnable config for LangSmith tracing
            config = RunnableConfig(
                tags=[f"role:{self.role}", "wms-chatbot", f"thread:{thread_id}"],
                metadata=metadata,
                run_name=f"{self.role}_agent_run",
                project_name=self.project_name
            ) if self.tracing_enabled else None
            
            # Run the agent with async execution and tracing enabled
            result = await agent_executor.ainvoke(
                {"input": enriched_query},
                config=config
            )
            
            # Save the conversation to memory service
            await conversation_memory_service.add_message(
                conversation_id=conversation_id,
                user_id=user_id,
                user_message=query,  # Use original query, not enhanced
                ai_response=result["output"],
                agent_role=self.role
            )
            
            return result["output"]
            
        except Exception as e:
            # Provide helpful error message if agent fails
            import traceback
            print(f"Error running agent: {str(e)}")
            print(traceback.format_exc())
            return f"I encountered an error while processing your request. Please try again with a clearer question about warehouse operations."

    @traceable(
        name="query_knowledge_base", 
        run_type="retriever",
        project_name="warehouse-management-system"
    )
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
            documents = [doc.page_content for doc in results]
            
            # Log additional metadata for tracing
            if self.langsmith_client and self.tracing_enabled:
                # This helps track knowledge base usage
                print(f"Knowledge base query: '{query}' returned {len(documents)} results")
            
            return documents
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
