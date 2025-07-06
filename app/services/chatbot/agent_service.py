"""
Enhanced Agent service for managing AI agents in the WMS Chatbot with role-based selection and management.
"""

import logging
import sys
import os
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict, deque

# Add current directory to path for imports
current_dir = os.path.dirname(__file__)
chatbot_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if chatbot_dir not in sys.path:
    sys.path.append(chatbot_dir)

try:
    from app.agents.clerk_agent import ClerkAgent
    from app.agents.picker_agent import PickerAgent
    from app.agents.packer_agent_ex import PackerAgent
    from app.agents.driver_agent import DriverAgent
    from app.agents.manager_agent import ManagerAgent
    from app.config import ROLES
except ImportError as e:
    logging.error(f"Error importing agents: {e}")
    # Create mock agents for testing
    class MockAgent:
        def __init__(self, role="mock"):
            self.role = role
            self.tools = []
        def run(self, message): 
            return f"Mock {self.role} response to: {message}"
    
    ClerkAgent = PickerAgent = PackerAgent = DriverAgent = ManagerAgent = MockAgent
    ROLES = {
        "clerk": {"permissions": ["receiving", "returns"]},
        "picker": {"permissions": ["picking", "inventory"]},
        "packer": {"permissions": ["packing", "shipping"]},
        "driver": {"permissions": ["delivery", "vehicle"]},
        "manager": {"permissions": ["management", "analytics", "oversight"]}
    }

logger = logging.getLogger("wms_chatbot.agent_service")


class QueryType(Enum):
    """Types of queries that can be handled by agents."""
    INVENTORY = "inventory"
    ORDERS = "orders"
    PICKING = "picking"
    PACKING = "packing"
    SHIPPING = "shipping"
    RECEIVING = "receiving"
    RETURNS = "returns"
    ANALYTICS = "analytics"
    MANAGEMENT = "management"
    VEHICLE = "vehicle"
    LOCATION = "location"
    WORKFLOW = "workflow"
    GENERAL = "general"


@dataclass
class AgentCapability:
    """Represents a capability of an agent."""
    name: str
    description: str
    query_types: List[QueryType]
    permission_required: Optional[str] = None
    priority: int = 1  # Higher priority means better fit


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an agent."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_response_time: float = 0.0
    recent_response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_used: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_queries == 0:
            return 0.0
        return (self.successful_queries / self.total_queries) * 100
    
    @property
    def recent_avg_response_time(self) -> float:
        """Calculate average response time from recent queries."""
        if not self.recent_response_times:
            return 0.0
        return sum(self.recent_response_times) / len(self.recent_response_times)


@dataclass
class AgentInfo:
    """Information about an agent."""
    role: str
    instance: Any
    capabilities: List[AgentCapability]
    performance_metrics: AgentPerformanceMetrics
    is_active: bool = True
    initialization_time: datetime = field(default_factory=datetime.utcnow)


class EnhancedAgentService:
    """Enhanced service for managing and routing AI agents with intelligent selection."""
    
    def __init__(self):
        """Initialize the enhanced agent service."""
        self.agents: Dict[str, AgentInfo] = {}
        self.capability_map: Dict[QueryType, List[str]] = defaultdict(list)
        self.query_patterns: Dict[QueryType, List[str]] = {}
        self.user_agent_preferences: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        self._initialize_query_patterns()
        self._initialize_agents()
        self._build_capability_map()
    
    def _initialize_query_patterns(self):
        """Initialize regex patterns for query classification."""
        self.query_patterns = {
            QueryType.INVENTORY: [
                r'\b(inventory|stock|item|product|sku|quantity|locate|find)\b',
                r'\b(how many|count|available|in stock)\b',
                r'\b(where is|location of|find item)\b'
            ],
            QueryType.ORDERS: [
                r'\b(order|purchase|customer|delivery|fulfill)\b',
                r'\b(order status|tracking|when will|expected)\b',
                r'\b(create order|new order|place order)\b'
            ],
            QueryType.PICKING: [
                r'\b(pick|picking|collect|gather|retrieve)\b',
                r'\b(picking list|pick order|route|path)\b',
                r'\b(optimal path|best route|shortest)\b'
            ],
            QueryType.PACKING: [
                r'\b(pack|packing|package|box|ship)\b',
                r'\b(packing slip|packaging|container)\b',
                r'\b(ready to ship|pack order)\b'
            ],
            QueryType.SHIPPING: [
                r'\b(ship|shipping|deliver|dispatch|send)\b',
                r'\b(carrier|tracking|label|address)\b',
                r'\b(shipping status|delivery time)\b'
            ],
            QueryType.RECEIVING: [
                r'\b(receive|receiving|inbound|arrival|supplier)\b',
                r'\b(purchase order|delivery|incoming)\b',
                r'\b(check in|received goods)\b'
            ],
            QueryType.RETURNS: [
                r'\b(return|refund|exchange|damaged|defective)\b',
                r'\b(return reason|return process|rma)\b',
                r'\b(credit|restock|return to vendor)\b'
            ],
            QueryType.ANALYTICS: [
                r'\b(analytics|report|metrics|performance|kpi)\b',
                r'\b(dashboard|statistics|trend|analysis)\b',
                r'\b(productivity|efficiency|throughput)\b'
            ],
            QueryType.MANAGEMENT: [
                r'\b(manage|oversight|approve|authorize|admin)\b',
                r'\b(staff|employee|worker|schedule|assign)\b',
                r'\b(system|configuration|settings)\b'
            ],
            QueryType.VEHICLE: [
                r'\b(vehicle|truck|van|delivery|transport)\b',
                r'\b(driver|route|fuel|maintenance)\b',
                r'\b(loading|capacity|schedule)\b'
            ],
            QueryType.LOCATION: [
                r'\b(location|where|zone|aisle|shelf|bin)\b',
                r'\b(warehouse layout|map|directions)\b',
                r'\b(move to|go to|navigate)\b'
            ],
            QueryType.WORKFLOW: [
                r'\b(workflow|process|procedure|task|step)\b',
                r'\b(next step|what to do|how to)\b',
                r'\b(status|progress|complete)\b'
            ]
        }
    
    def _initialize_agents(self):
        """Initialize all AI agents with their capabilities."""
        try:
            # Define capabilities for each agent
            agent_capabilities = {
                "clerk": [
                    AgentCapability(
                        name="Receiving Management",
                        description="Handle incoming shipments and receiving processes",
                        query_types=[QueryType.RECEIVING, QueryType.INVENTORY],
                        permission_required="receiving",
                        priority=3
                    ),
                    AgentCapability(
                        name="Returns Processing",
                        description="Process returns and handle customer issues",
                        query_types=[QueryType.RETURNS, QueryType.ORDERS],
                        permission_required="returns",
                        priority=3
                    ),
                    AgentCapability(
                        name="Documentation",
                        description="Handle paperwork and documentation",
                        query_types=[QueryType.WORKFLOW, QueryType.GENERAL],
                        priority=2
                    )
                ],
                "picker": [
                    AgentCapability(
                        name="Item Location",
                        description="Find and locate items in the warehouse",
                        query_types=[QueryType.INVENTORY, QueryType.LOCATION],
                        permission_required="picking",
                        priority=3
                    ),
                    AgentCapability(
                        name="Picking Operations",
                        description="Execute picking tasks and optimize routes",
                        query_types=[QueryType.PICKING, QueryType.ORDERS],
                        permission_required="picking",
                        priority=3
                    ),
                    AgentCapability(
                        name="Path Optimization",
                        description="Optimize picking routes and paths",
                        query_types=[QueryType.LOCATION, QueryType.WORKFLOW],
                        priority=2
                    )
                ],
                "packer": [
                    AgentCapability(
                        name="Packing Operations",
                        description="Pack orders and prepare for shipping",
                        query_types=[QueryType.PACKING, QueryType.ORDERS],
                        permission_required="packing",
                        priority=3
                    ),
                    AgentCapability(
                        name="Shipping Preparation",
                        description="Prepare packages for shipment",
                        query_types=[QueryType.SHIPPING, QueryType.WORKFLOW],
                        permission_required="shipping",
                        priority=2
                    )
                ],
                "driver": [
                    AgentCapability(
                        name="Vehicle Management",
                        description="Manage vehicles and transportation",
                        query_types=[QueryType.VEHICLE, QueryType.SHIPPING],
                        permission_required="delivery",
                        priority=3
                    ),
                    AgentCapability(
                        name="Delivery Operations",
                        description="Handle delivery routes and logistics",
                        query_types=[QueryType.SHIPPING, QueryType.LOCATION],
                        permission_required="delivery",
                        priority=3
                    )
                ],
                "manager": [
                    AgentCapability(
                        name="Analytics & Reporting",
                        description="Generate reports and analyze performance",
                        query_types=[QueryType.ANALYTICS, QueryType.MANAGEMENT],
                        permission_required="management",
                        priority=3
                    ),
                    AgentCapability(
                        name="Workforce Management",
                        description="Manage staff and operations",
                        query_types=[QueryType.MANAGEMENT, QueryType.WORKFLOW],
                        permission_required="management",
                        priority=3
                    ),
                    AgentCapability(
                        name="System Oversight",
                        description="Oversee all warehouse operations",
                        query_types=[QueryType.GENERAL, QueryType.ORDERS, QueryType.INVENTORY],
                        permission_required="oversight",
                        priority=2
                    )
                ]
            }
            
            # Initialize agents with their capabilities
            agent_classes = {
                "clerk": ClerkAgent,
                "picker": PickerAgent,
                "packer": PackerAgent,
                "driver": DriverAgent,
                "manager": ManagerAgent
            }
            
            for role, agent_class in agent_classes.items():
                try:
                    agent_instance = agent_class()
                    capabilities = agent_capabilities.get(role, [])
                    
                    self.agents[role] = AgentInfo(
                        role=role,
                        instance=agent_instance,
                        capabilities=capabilities,
                        performance_metrics=AgentPerformanceMetrics()
                    )
                    logger.info(f"Initialized {role} agent with {len(capabilities)} capabilities")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize {role} agent: {str(e)}")
                    # Continue with other agents
            
            logger.info(f"Successfully initialized {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    def _build_capability_map(self):
        """Build a map of query types to capable agents."""
        self.capability_map.clear()
        
        for role, agent_info in self.agents.items():
            for capability in agent_info.capabilities:
                for query_type in capability.query_types:
                    if role not in self.capability_map[query_type]:
                        self.capability_map[query_type].append(role)
        
        logger.info(f"Built capability map with {len(self.capability_map)} query types")
    
    def classify_query(self, query: str) -> List[Tuple[QueryType, float]]:
        """
        Classify a query into query types with confidence scores.
        
        Args:
            query: The query to classify
            
        Returns:
            List of (QueryType, confidence_score) tuples sorted by confidence
        """
        query_lower = query.lower()
        scores = defaultdict(float)
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    # Weight by number of matches and pattern specificity
                    score = len(matches) * (1.0 / len(patterns))
                    scores[query_type] += score
        
        # Normalize scores
        max_score = max(scores.values()) if scores else 1.0
        normalized_scores = [(qt, score / max_score) for qt, score in scores.items()]
        
        # Sort by confidence
        return sorted(normalized_scores, key=lambda x: x[1], reverse=True)
    
    def get_suitable_agents(
        self, 
        query: str, 
        user_role: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get suitable agents for a query with suitability scores.
        
        Args:
            query: The query to find agents for
            user_role: The role of the user making the query
            context: Additional context for agent selection
            
        Returns:
            List of (agent_role, suitability_score) tuples sorted by suitability
        """
        query_classifications = self.classify_query(query)
        agent_scores = defaultdict(float)
        
        # Score agents based on query classification
        for query_type, confidence in query_classifications:
            capable_agents = self.capability_map.get(query_type, [])
            
            for agent_role in capable_agents:
                agent_info = self.agents[agent_role]
                
                # Find matching capabilities
                matching_capabilities = [
                    cap for cap in agent_info.capabilities 
                    if query_type in cap.query_types
                ]
                
                for capability in matching_capabilities:
                    # Base score from capability priority and query confidence
                    base_score = capability.priority * confidence
                    
                    # Boost score for exact role match
                    if user_role and agent_role == user_role:
                        base_score *= 1.5
                    
                    # Boost score for performance metrics
                    performance_boost = agent_info.performance_metrics.success_rate / 100
                    base_score *= (1 + performance_boost * 0.2)
                    
                    # Reduce score for inactive agents
                    if not agent_info.is_active:
                        base_score *= 0.1
                    
                    agent_scores[agent_role] += base_score
        
        # Add user preferences
        if user_role and user_role in self.user_agent_preferences:
            for agent_role, preference_score in self.user_agent_preferences[user_role].items():
                if agent_role in agent_scores:
                    agent_scores[agent_role] *= (1 + preference_score * 0.1)
        
        # Sort by suitability score
        return sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
    
    def select_best_agent(
        self, 
        query: str, 
        user_role: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Select the best agent for a query.
        
        Args:
            query: The query to find the best agent for
            user_role: The role of the user making the query
            context: Additional context for agent selection
            
        Returns:
            The role of the best agent
        """
        suitable_agents = self.get_suitable_agents(query, user_role, context)
        
        if not suitable_agents:
            # Fallback to user's role or manager
            if user_role and user_role in self.agents:
                return user_role
            return "manager"  # Manager as ultimate fallback
        
        return suitable_agents[0][0]  # Return best agent role
    
    def get_agent(self, role: str) -> Any:
        """
        Get an agent by role.
        
        Args:
            role: The role of the agent to retrieve
            
        Returns:
            The agent instance for the given role
            
        Raises:
            ValueError: If the role is not found
        """
        if role not in self.agents:
            raise ValueError(f"Unknown agent role: {role}. Available roles: {list(self.agents.keys())}")
        
        agent_info = self.agents[role]
        if not agent_info.is_active:
            logger.warning(f"Agent {role} is inactive, using anyway")
        
        return agent_info.instance
    
    def get_available_roles(self) -> List[str]:
        """
        Get list of available agent roles.
        
        Returns:
            List of available agent role names
        """
        return [role for role, info in self.agents.items() if info.is_active]
    
    def get_agent_capabilities(self, role: str) -> List[Dict[str, Any]]:
        """
        Get capabilities of a specific agent.
        
        Args:
            role: The role to get capabilities for
            
        Returns:
            List of capability dictionaries
        """
        if role not in self.agents:
            return []
        
        agent_info = self.agents[role]
        return [
            {
                "name": cap.name,
                "description": cap.description,
                "query_types": [qt.value for qt in cap.query_types],
                "permission_required": cap.permission_required,
                "priority": cap.priority
            }
            for cap in agent_info.capabilities
        ]
    
    def get_agent_performance(self, role: str) -> Dict[str, Any]:
        """
        Get performance metrics for an agent.
        
        Args:
            role: The role to get performance for
            
        Returns:
            Performance metrics dictionary
        """
        if role not in self.agents:
            return {}
        
        metrics = self.agents[role].performance_metrics
        return {
            "total_queries": metrics.total_queries,
            "successful_queries": metrics.successful_queries,
            "failed_queries": metrics.failed_queries,
            "success_rate": metrics.success_rate,
            "avg_response_time": metrics.avg_response_time,
            "recent_avg_response_time": metrics.recent_avg_response_time,
            "last_used": metrics.last_used.isoformat() if metrics.last_used else None
        }
    
    def update_agent_performance(
        self, 
        role: str, 
        success: bool, 
        response_time: float
    ):
        """
        Update performance metrics for an agent.
        
        Args:
            role: The role to update
            success: Whether the query was successful
            response_time: Response time in seconds
        """
        if role not in self.agents:
            return
        
        metrics = self.agents[role].performance_metrics
        metrics.total_queries += 1
        
        if success:
            metrics.successful_queries += 1
        else:
            metrics.failed_queries += 1
        
        metrics.recent_response_times.append(response_time)
        metrics.last_used = datetime.utcnow()
        
        # Update average response time
        if metrics.total_queries == 1:
            metrics.avg_response_time = response_time
        else:
            # Running average
            metrics.avg_response_time = (
                (metrics.avg_response_time * (metrics.total_queries - 1) + response_time) / 
                metrics.total_queries
            )
    
    def update_user_preferences(self, user_role: str, agent_role: str, positive_feedback: bool):
        """
        Update user preferences for agents based on feedback.
        
        Args:
            user_role: The role of the user
            agent_role: The agent role to update preference for
            positive_feedback: Whether feedback was positive
        """
        if user_role not in self.user_agent_preferences:
            self.user_agent_preferences[user_role] = {}
        
        current_score = self.user_agent_preferences[user_role].get(agent_role, 0.0)
        
        if positive_feedback:
            # Increase preference
            self.user_agent_preferences[user_role][agent_role] = min(1.0, current_score + 0.1)
        else:
            # Decrease preference
            self.user_agent_preferences[user_role][agent_role] = max(-0.5, current_score - 0.1)
    
    async def process_message(
        self, 
        message: str, 
        role: Optional[str] = None,
        user_role: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        auto_select: bool = True
    ) -> Dict[str, Any]:
        """
        Process a message using the appropriate agent with enhanced selection.
        
        Args:
            message: The message to process
            role: Specific agent role to use (optional)
            user_role: The role of the user making the query
            context: Additional context for the agent
            conversation_id: The conversation identifier
            auto_select: Whether to automatically select the best agent
            
        Returns:
            Dictionary with response and metadata
            
        Raises:
            ValueError: If the role is not found
            Exception: If agent processing fails
        """
        start_time = datetime.utcnow()
        
        try:
            # Select agent if not specified
            if not role and auto_select:
                role = self.select_best_agent(message, user_role, context)
            elif not role:
                role = user_role or "manager"  # Fallback
            
            # Get agent and process message
            agent = self.get_agent(role)
            logger.info(f"Processing message with {role} agent: {message[:100]}...")
            
            # Process the message with the agent
            response = agent.run(message)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update performance metrics
            self.update_agent_performance(role, True, processing_time)
            
            # Get query classification for metadata
            query_classifications = self.classify_query(message)
            
            # Return structured response
            return {
                "response": response,
                "context": context or {},
                "metadata": {
                    "agent_role": role,
                    "conversation_id": conversation_id,
                    "processed_at": datetime.utcnow().isoformat(),
                    "processing_time": processing_time,
                    "query_classifications": [
                        {"type": qt.value, "confidence": conf} 
                        for qt, conf in query_classifications[:3]  # Top 3
                    ],
                    "auto_selected": auto_select and role != (user_role or "manager")
                },
                "tokens_used": len(message.split()) + len(response.split()),  # Simple estimation
                "model_used": f"{role}_agent"
            }
            
        except Exception as e:
            # Update performance metrics for failure
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            if role:
                self.update_agent_performance(role, False, processing_time)
            
            logger.error(f"Error processing message with {role} agent: {str(e)}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status and metrics.
        
        Returns:
            System status dictionary
        """
        total_queries = sum(info.performance_metrics.total_queries for info in self.agents.values())
        active_agents = sum(1 for info in self.agents.values() if info.is_active)
        
        return {
            "total_agents": len(self.agents),
            "active_agents": active_agents,
            "total_queries_processed": total_queries,
            "agent_status": {
                role: {
                    "active": info.is_active,
                    "capabilities_count": len(info.capabilities),
                    "performance": self.get_agent_performance(role)
                }
                for role, info in self.agents.items()
            },
            "capability_coverage": {
                qt.value: len(agents) for qt, agents in self.capability_map.items()
            },
            "initialization_time": min(
                info.initialization_time for info in self.agents.values()
            ).isoformat() if self.agents else None
        }


# Create a singleton instance for backward compatibility
# This maintains the existing API while providing enhanced features
AgentService = EnhancedAgentService


