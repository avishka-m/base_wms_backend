"""
Context-Awareness Service for intelligent workplace integration and contextual messaging.
Provides real-time context detection, workplace integration, and smart suggestions.
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import re

from app.utils.database import get_async_collection
from app.utils.chatbot.api_client import APIClient

logger = logging.getLogger("wms_chatbot.context_awareness_service")


class ContextType(Enum):
    """Types of workplace context."""
    LOCATION = "location"
    TASK = "task"
    ROLE = "role"
    SHIFT = "shift"
    INVENTORY = "inventory"
    ORDER = "order"
    WORKFLOW = "workflow"
    TEMPORAL = "temporal"
    PREFERENCE = "preference"
    PERFORMANCE = "performance"


@dataclass
class ContextSignal:
    """Represents a context signal from workplace activities."""
    signal_type: ContextType
    source: str
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


@dataclass
class WorkplaceContext:
    """Current workplace context for a user."""
    user_id: str
    current_location: Optional[str] = None
    current_task: Optional[str] = None
    active_orders: List[str] = field(default_factory=list)
    current_role: Optional[str] = None
    shift_info: Optional[Dict[str, Any]] = None
    inventory_focus: List[str] = field(default_factory=list)
    recent_activities: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    context_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ContextualSuggestion:
    """Contextual suggestion based on current context."""
    suggestion_id: str
    type: str
    title: str
    content: str
    action: Optional[str] = None
    priority: int = 1
    confidence: float = 0.0
    context_match: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None


class ContextAwarenessService:
    """Service for managing workplace context awareness and contextual messaging."""
    
    def __init__(self):
        """Initialize the context awareness service."""
        self._context_cache = {}
        self._context_signals = defaultdict(lambda: deque(maxlen=100))
        self._context_patterns = {}
        self._api_client = APIClient()
        self._initialized = False
        
    async def _ensure_collections_and_indexes(self):
        """Ensure database collections and indexes are set up."""
        if self._initialized:
            return
            
        try:
            # Get collections
            context_col = await get_async_collection("user_context")
            signals_col = await get_async_collection("context_signals")
            patterns_col = await get_async_collection("context_patterns")
            suggestions_col = await get_async_collection("contextual_suggestions")
            
            # Create indexes
            await context_col.create_index([("user_id", 1)], unique=True)
            await context_col.create_index([("last_updated", -1)])
            
            await signals_col.create_index([("user_id", 1), ("timestamp", -1)])
            await signals_col.create_index([("signal_type", 1)])
            await signals_col.create_index([("expires_at", 1)])
            
            await patterns_col.create_index([("user_id", 1)])
            await patterns_col.create_index([("pattern_type", 1)])
            
            await suggestions_col.create_index([("user_id", 1), ("timestamp", -1)])
            await suggestions_col.create_index([("expires_at", 1)])
            
            self._initialized = True
            logger.info("Context awareness service collections initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize context collections: {str(e)}")
            raise
    
    async def update_context_signal(
        self,
        user_id: str,
        signal_type: ContextType,
        source: str,
        data: Dict[str, Any],
        confidence: float = 1.0,
        expires_in_minutes: Optional[int] = None
    ):
        """
        Update a context signal for a user.
        
        Args:
            user_id: User identifier
            signal_type: Type of context signal
            source: Source of the signal
            data: Signal data
            confidence: Confidence score (0.0 to 1.0)
            expires_in_minutes: Signal expiration time
        """
        await self._ensure_collections_and_indexes()
        
        try:
            expires_at = None
            if expires_in_minutes:
                expires_at = datetime.utcnow() + timedelta(minutes=expires_in_minutes)
            
            signal = ContextSignal(
                signal_type=signal_type,
                source=source,
                data=data,
                confidence=confidence,
                expires_at=expires_at
            )
            
            # Store in cache
            self._context_signals[user_id].append(signal)
            
            # Store in database
            signals_col = await get_async_collection("context_signals")
            signal_doc = {
                "user_id": user_id,
                "signal_type": signal_type.value,
                "source": source,
                "data": data,
                "confidence": confidence,
                "timestamp": signal.timestamp,
                "expires_at": expires_at
            }
            
            await signals_col.insert_one(signal_doc)
            
            # Update user context
            await self._update_user_context(user_id)
            
            logger.debug(f"Updated context signal for {user_id}: {signal_type.value} from {source}")
            
        except Exception as e:
            logger.error(f"Failed to update context signal: {str(e)}")
    
    async def get_current_context(self, user_id: str) -> WorkplaceContext:
        """
        Get current workplace context for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Current workplace context
        """
        await self._ensure_collections_and_indexes()
        
        try:
            # Check cache first
            if user_id in self._context_cache:
                context = self._context_cache[user_id]
                if (datetime.utcnow() - context.last_updated).seconds < 300:  # 5 minutes
                    return context
            
            # Load from database
            context_col = await get_async_collection("user_context")
            context_doc = await context_col.find_one({"user_id": user_id})
            
            if context_doc:
                context = WorkplaceContext(
                    user_id=user_id,
                    current_location=context_doc.get("current_location"),
                    current_task=context_doc.get("current_task"),
                    active_orders=context_doc.get("active_orders", []),
                    current_role=context_doc.get("current_role"),
                    shift_info=context_doc.get("shift_info"),
                    inventory_focus=context_doc.get("inventory_focus", []),
                    recent_activities=context_doc.get("recent_activities", []),
                    performance_metrics=context_doc.get("performance_metrics", {}),
                    preferences=context_doc.get("preferences", {}),
                    context_score=context_doc.get("context_score", 0.0),
                    last_updated=context_doc.get("last_updated", datetime.utcnow())
                )
            else:
                # Create new context
                context = WorkplaceContext(user_id=user_id)
                await self._update_user_context(user_id)
            
            # Update cache
            self._context_cache[user_id] = context
            return context
            
        except Exception as e:
            logger.error(f"Failed to get current context: {str(e)}")
            return WorkplaceContext(user_id=user_id)
    
    async def _update_user_context(self, user_id: str):
        """Update user context based on recent signals."""
        try:
            # Get recent signals
            signals_col = await get_async_collection("context_signals")
            recent_signals = []
            
            # Get signals from last 2 hours
            since = datetime.utcnow() - timedelta(hours=2)
            async for doc in signals_col.find({
                "user_id": user_id,
                "timestamp": {"$gte": since},
                "$or": [
                    {"expires_at": {"$gt": datetime.utcnow()}},
                    {"expires_at": None}
                ]
            }).sort("timestamp", -1):
                recent_signals.append(doc)
            
            # Analyze signals to build context
            context = await self._analyze_signals_to_context(user_id, recent_signals)
            
            # Store updated context
            context_col = await get_async_collection("user_context")
            context_doc = {
                "user_id": user_id,
                "current_location": context.current_location,
                "current_task": context.current_task,
                "active_orders": context.active_orders,
                "current_role": context.current_role,
                "shift_info": context.shift_info,
                "inventory_focus": context.inventory_focus,
                "recent_activities": context.recent_activities,
                "performance_metrics": context.performance_metrics,
                "preferences": context.preferences,
                "context_score": context.context_score,
                "last_updated": datetime.utcnow()
            }
            
            await context_col.replace_one(
                {"user_id": user_id},
                context_doc,
                upsert=True
            )
            
            # Update cache
            self._context_cache[user_id] = context
            
        except Exception as e:
            logger.error(f"Failed to update user context: {str(e)}")
    
    async def _analyze_signals_to_context(self, user_id: str, signals: List[Dict]) -> WorkplaceContext:
        """Analyze signals to build workplace context."""
        context = WorkplaceContext(user_id=user_id)
        
        # Analyze by signal type
        for signal in signals:
            signal_type = signal.get("signal_type")
            data = signal.get("data", {})
            confidence = signal.get("confidence", 0.0)
            
            if signal_type == ContextType.LOCATION.value:
                context.current_location = data.get("location")
                
            elif signal_type == ContextType.TASK.value:
                context.current_task = data.get("task_type")
                if "order_id" in data:
                    if data["order_id"] not in context.active_orders:
                        context.active_orders.append(data["order_id"])
                        
            elif signal_type == ContextType.ROLE.value:
                context.current_role = data.get("role")
                
            elif signal_type == ContextType.SHIFT.value:
                context.shift_info = data
                
            elif signal_type == ContextType.INVENTORY.value:
                item_sku = data.get("sku")
                if item_sku and item_sku not in context.inventory_focus:
                    context.inventory_focus.append(item_sku)
                    
            elif signal_type == ContextType.ORDER.value:
                order_id = data.get("order_id")
                if order_id and order_id not in context.active_orders:
                    context.active_orders.append(order_id)
                    
            elif signal_type == ContextType.PERFORMANCE.value:
                context.performance_metrics.update(data)
                
            elif signal_type == ContextType.PREFERENCE.value:
                context.preferences.update(data)
            
            # Add to recent activities
            activity = {
                "type": signal_type,
                "data": data,
                "timestamp": signal.get("timestamp"),
                "confidence": confidence
            }
            context.recent_activities.append(activity)
        
        # Keep only recent 20 activities
        context.recent_activities = context.recent_activities[-20:]
        
        # Keep only recent 5 inventory items
        context.inventory_focus = context.inventory_focus[-5:]
        
        # Keep only recent 10 orders
        context.active_orders = context.active_orders[-10:]
        
        # Calculate context score
        context.context_score = await self._calculate_context_score(context, signals)
        
        return context
    
    async def _calculate_context_score(self, context: WorkplaceContext, signals: List[Dict]) -> float:
        """Calculate context quality score."""
        score = 0.0
        
        # Base scores for having context
        if context.current_location:
            score += 0.2
        if context.current_task:
            score += 0.3
        if context.current_role:
            score += 0.2
        if context.active_orders:
            score += 0.1
        if context.inventory_focus:
            score += 0.1
        if context.shift_info:
            score += 0.1
        
        # Bonus for recent signals
        recent_signals = len([s for s in signals if (datetime.utcnow() - s.get("timestamp", datetime.utcnow())).seconds < 1800])  # 30 minutes
        score += min(recent_signals * 0.05, 0.3)
        
        return min(score, 1.0)
    
    async def get_contextual_suggestions(
        self,
        user_id: str,
        query: Optional[str] = None,
        limit: int = 5
    ) -> List[ContextualSuggestion]:
        """
        Get contextual suggestions based on current context.
        
        Args:
            user_id: User identifier
            query: Optional query for filtering suggestions
            limit: Maximum number of suggestions
            
        Returns:
            List of contextual suggestions
        """
        await self._ensure_collections_and_indexes()
        
        try:
            context = await self.get_current_context(user_id)
            suggestions = []
            
            # Generate location-based suggestions
            if context.current_location:
                location_suggestions = await self._generate_location_suggestions(context)
                suggestions.extend(location_suggestions)
            
            # Generate task-based suggestions
            if context.current_task:
                task_suggestions = await self._generate_task_suggestions(context)
                suggestions.extend(task_suggestions)
            
            # Generate order-based suggestions
            if context.active_orders:
                order_suggestions = await self._generate_order_suggestions(context)
                suggestions.extend(order_suggestions)
            
            # Generate inventory-based suggestions
            if context.inventory_focus:
                inventory_suggestions = await self._generate_inventory_suggestions(context)
                suggestions.extend(inventory_suggestions)
            
            # Generate role-based suggestions
            if context.current_role:
                role_suggestions = await self._generate_role_suggestions(context)
                suggestions.extend(role_suggestions)
            
            # Filter by query if provided
            if query:
                query_lower = query.lower()
                suggestions = [s for s in suggestions if query_lower in s.title.lower() or query_lower in s.content.lower()]
            
            # Sort by priority and confidence
            suggestions.sort(key=lambda x: (x.priority, x.confidence), reverse=True)
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get contextual suggestions: {str(e)}")
            return []
    
    async def _generate_location_suggestions(self, context: WorkplaceContext) -> List[ContextualSuggestion]:
        """Generate suggestions based on current location."""
        suggestions = []
        location = context.current_location
        
        try:
            # Get location-specific inventory
            response = await self._api_client.get(f"/inventory/?location={location}")
            if not self._api_client.is_api_error(response):
                if response and len(response) > 0:
                    suggestions.append(ContextualSuggestion(
                        suggestion_id=f"loc_inventory_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        type="inventory",
                        title=f"Check {location} Inventory",
                        content=f"Review {len(response)} items at your current location",
                        action=f"show_location_inventory:{location}",
                        priority=3,
                        confidence=0.8,
                        context_match={"location": location}
                    ))
            
            # Get nearby tasks
            suggestions.append(ContextualSuggestion(
                suggestion_id=f"loc_tasks_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                type="task",
                title=f"Tasks Near {location}",
                content="Check for pending tasks in your area",
                action=f"show_nearby_tasks:{location}",
                priority=2,
                confidence=0.7,
                context_match={"location": location}
            ))
            
        except Exception as e:
            logger.debug(f"Error generating location suggestions: {str(e)}")
        
        return suggestions
    
    async def _generate_task_suggestions(self, context: WorkplaceContext) -> List[ContextualSuggestion]:
        """Generate suggestions based on current task."""
        suggestions = []
        task = context.current_task
        
        try:
            if task == "picking":
                suggestions.append(ContextualSuggestion(
                    suggestion_id=f"task_picking_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    type="workflow",
                    title="Optimize Picking Route",
                    content="Get the most efficient picking path for your current orders",
                    action="optimize_picking_route",
                    priority=4,
                    confidence=0.9,
                    context_match={"task": task}
                ))
                
            elif task == "packing":
                suggestions.append(ContextualSuggestion(
                    suggestion_id=f"task_packing_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    type="workflow",
                    title="Packing Guidelines",
                    content="View best practices for current items",
                    action="show_packing_guidelines",
                    priority=3,
                    confidence=0.8,
                    context_match={"task": task}
                ))
                
            elif task == "receiving":
                suggestions.append(ContextualSuggestion(
                    suggestion_id=f"task_receiving_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    type="workflow",
                    title="Incoming Shipments",
                    content="Check expected deliveries and processing queue",
                    action="show_incoming_shipments",
                    priority=4,
                    confidence=0.9,
                    context_match={"task": task}
                ))
                
        except Exception as e:
            logger.debug(f"Error generating task suggestions: {str(e)}")
        
        return suggestions
    
    async def _generate_order_suggestions(self, context: WorkplaceContext) -> List[ContextualSuggestion]:
        """Generate suggestions based on active orders."""
        suggestions = []
        
        try:
            for order_id in context.active_orders[:3]:  # Top 3 orders
                suggestions.append(ContextualSuggestion(
                    suggestion_id=f"order_{order_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    type="order",
                    title=f"Order #{order_id} Status",
                    content="Check progress and next steps",
                    action=f"show_order_status:{order_id}",
                    priority=3,
                    confidence=0.8,
                    context_match={"order_id": order_id}
                ))
                
        except Exception as e:
            logger.debug(f"Error generating order suggestions: {str(e)}")
        
        return suggestions
    
    async def _generate_inventory_suggestions(self, context: WorkplaceContext) -> List[ContextualSuggestion]:
        """Generate suggestions based on inventory focus."""
        suggestions = []
        
        try:
            for sku in context.inventory_focus[:3]:  # Top 3 items
                suggestions.append(ContextualSuggestion(
                    suggestion_id=f"inv_{sku}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    type="inventory",
                    title=f"Item {sku} Details",
                    content="View current stock and location",
                    action=f"show_item_details:{sku}",
                    priority=2,
                    confidence=0.7,
                    context_match={"sku": sku}
                ))
                
        except Exception as e:
            logger.debug(f"Error generating inventory suggestions: {str(e)}")
        
        return suggestions
    
    async def _generate_role_suggestions(self, context: WorkplaceContext) -> List[ContextualSuggestion]:
        """Generate suggestions based on user role."""
        suggestions = []
        role = context.current_role
        
        try:
            if role == "manager":
                suggestions.extend([
                    ContextualSuggestion(
                        suggestion_id=f"mgr_dashboard_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        type="analytics",
                        title="Team Performance Dashboard",
                        content="View current team metrics and KPIs",
                        action="show_team_dashboard",
                        priority=4,
                        confidence=0.9,
                        context_match={"role": role}
                    ),
                    ContextualSuggestion(
                        suggestion_id=f"mgr_reports_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        type="reporting",
                        title="Generate Daily Report",
                        content="Create performance and status reports",
                        action="generate_daily_report",
                        priority=3,
                        confidence=0.8,
                        context_match={"role": role}
                    )
                ])
                
            elif role in ["picker", "packer"]:
                suggestions.append(ContextualSuggestion(
                    suggestion_id=f"worker_tasks_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    type="tasks",
                    title="My Task Queue",
                    content="View assigned tasks and priorities",
                    action="show_my_tasks",
                    priority=5,
                    confidence=0.95,
                    context_match={"role": role}
                ))
                
        except Exception as e:
            logger.debug(f"Error generating role suggestions: {str(e)}")
        
        return suggestions
    
    async def detect_context_from_message(
        self,
        user_id: str,
        message: str,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect context from user message and conversation.
        
        Args:
            user_id: User identifier
            message: User message
            conversation_context: Current conversation context
            
        Returns:
            Detected context information
        """
        try:
            detected_context = {}
            message_lower = message.lower()
            
            # Detect location mentions
            location_patterns = [
                r'(?:at|in|near|zone|aisle|shelf|bin)\s+([A-Z]\d*-[A-Z]?\d*-[A-Z]?\d*-[A-Z]?\d*)',
                r'(?:location|area)\s+([A-Za-z]\d+)',
                r'(?:warehouse|storage)\s+([A-Za-z]\d+)'
            ]
            
            for pattern in location_patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    detected_context["location"] = match.group(1)
                    await self.update_context_signal(
                        user_id, ContextType.LOCATION, "message_detection",
                        {"location": match.group(1)}, confidence=0.7, expires_in_minutes=60
                    )
                    break
            
            # Detect SKU/item mentions
            sku_patterns = [
                r'(?:sku|item|product)\s+([A-Z0-9-]+)',
                r'\b([A-Z]{2,}\d{3,})\b',
                r'(?:part\s*#|part\s*number)\s*([A-Z0-9-]+)'
            ]
            
            for pattern in sku_patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    detected_context["sku"] = match.group(1)
                    await self.update_context_signal(
                        user_id, ContextType.INVENTORY, "message_detection",
                        {"sku": match.group(1)}, confidence=0.8, expires_in_minutes=30
                    )
                    break
            
            # Detect order mentions
            order_patterns = [
                r'(?:order|ord)\s*#?\s*(\d+)',
                r'\border\s+(\d+)\b',
                r'(?:purchase|po)\s*#?\s*(\d+)'
            ]
            
            for pattern in order_patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    detected_context["order_id"] = match.group(1)
                    await self.update_context_signal(
                        user_id, ContextType.ORDER, "message_detection",
                        {"order_id": match.group(1)}, confidence=0.9, expires_in_minutes=45
                    )
                    break
            
            # Detect task mentions
            task_keywords = {
                "picking": ["pick", "picking", "picks", "collect"],
                "packing": ["pack", "packing", "package", "box"],
                "receiving": ["receive", "receiving", "incoming", "delivery"],
                "shipping": ["ship", "shipping", "dispatch", "send"],
                "inventory": ["count", "stock", "inventory", "audit"]
            }
            
            for task_type, keywords in task_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    detected_context["task"] = task_type
                    await self.update_context_signal(
                        user_id, ContextType.TASK, "message_detection",
                        {"task_type": task_type}, confidence=0.6, expires_in_minutes=30
                    )
                    break
            
            # Add conversation context
            if conversation_context:
                detected_context["conversation"] = conversation_context
            
            return detected_context
            
        except Exception as e:
            logger.error(f"Failed to detect context from message: {str(e)}")
            return {}
    
    async def get_context_enriched_response(
        self,
        user_id: str,
        message: str,
        base_response: str,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enrich AI response with contextual information.
        
        Args:
            user_id: User identifier
            message: User message
            base_response: Base AI response
            conversation_context: Current conversation context
            
        Returns:
            Enriched response with context
        """
        try:
            # Get current context
            context = await self.get_current_context(user_id)
            
            # Detect context from current message
            message_context = await self.detect_context_from_message(user_id, message, conversation_context)
            
            # Get contextual suggestions
            suggestions = await self.get_contextual_suggestions(user_id, limit=3)
            
            # Build enriched response
            enriched_response = {
                "response": base_response,
                "context": {
                    "current_location": context.current_location,
                    "current_task": context.current_task,
                    "active_orders": context.active_orders[:3],
                    "context_score": context.context_score
                },
                "detected_context": message_context,
                "suggestions": [
                    {
                        "title": s.title,
                        "content": s.content,
                        "action": s.action,
                        "priority": s.priority
                    } for s in suggestions
                ],
                "contextual_data": await self._get_contextual_data(context, message_context)
            }
            
            return enriched_response
            
        except Exception as e:
            logger.error(f"Failed to enrich response with context: {str(e)}")
            return {"response": base_response, "context": {}, "suggestions": []}
    
    async def _get_contextual_data(self, context: WorkplaceContext, message_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant contextual data for the response."""
        contextual_data = {}
        
        try:
            # Add location data if relevant
            if context.current_location or message_context.get("location"):
                location = message_context.get("location") or context.current_location
                response = await self._api_client.get(f"/inventory/?location={location}")
                if not self._api_client.is_api_error(response):
                    contextual_data["location_inventory"] = {
                        "location": location,
                        "item_count": len(response) if response else 0,
                        "items": response[:5] if response else []  # First 5 items
                    }
            
            # Add order data if relevant
            if message_context.get("order_id"):
                order_id = message_context["order_id"]
                response = await self._api_client.get(f"/orders/{order_id}")
                if not self._api_client.is_api_error(response):
                    contextual_data["order_details"] = response
            
            # Add SKU data if relevant
            if message_context.get("sku"):
                sku = message_context["sku"]
                response = await self._api_client.get(f"/inventory/?sku={sku}")
                if not self._api_client.is_api_error(response):
                    contextual_data["item_details"] = response[0] if response else None
            
        except Exception as e:
            logger.debug(f"Error getting contextual data: {str(e)}")
        
        return contextual_data 