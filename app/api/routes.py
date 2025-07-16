from fastapi import APIRouter

# Import all API routers
from .auth import router as auth_router
from .inventory import router as inventory_router
from .orders import router as orders_router
from .workers import router as workers_router
from .customers import router as customers_router
from .location import router as location_router
from .receiving import router as receiving_router
from .picking import router as picking_router
from .packing import router as packing_router
from .shipping import router as shipping_router
from .returns import router as returns_router
from .vehicles import router as vehicles_router
from .analytics import router as analytics_router
from .prophet_forecasting_fixed import router as prophet_router
from .workflow import router as workflow_router
from .role_based_orders import router as role_based_router
from .enhanced_chatbot_routes import router as chatbot_router  # Enhanced chatbot with persistent storage
from .websocket_routes import router as websocket_router

# Create main API router
api_router = APIRouter()

# Include all routers
api_router.include_router(auth_router, tags=["Authentication"])
api_router.include_router(inventory_router, prefix="/inventory", tags=["Inventory"])
api_router.include_router(orders_router, prefix="/orders", tags=["Orders"])
api_router.include_router(workers_router, prefix="/workers", tags=["Workers"])
api_router.include_router(customers_router, prefix="/customers", tags=["Customers"])
api_router.include_router(location_router, prefix="/locations", tags=["Locations"])
api_router.include_router(receiving_router, prefix="/receiving", tags=["Receiving"])
api_router.include_router(picking_router, prefix="/picking", tags=["Picking"])
api_router.include_router(packing_router, prefix="/packing", tags=["Packing"])
api_router.include_router(shipping_router, prefix="/shipping", tags=["Shipping"])
api_router.include_router(returns_router, prefix="/returns", tags=["Returns"])
api_router.include_router(vehicles_router, prefix="/vehicles", tags=["Vehicles"])
api_router.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
api_router.include_router(prophet_router, prefix="/prophet", tags=["Prophet Forecasting"])
api_router.include_router(role_based_router, prefix="/role-based", tags=["Role-Based Operations"])
api_router.include_router(workflow_router, prefix="/workflow", tags=["Workflow Management"])
api_router.include_router(chatbot_router, prefix="/chatbot", tags=["AI Chatbot Enhanced"])  # Enhanced chatbot with persistent storage
api_router.include_router(websocket_router, tags=["WebSocket"])  # Real-time updates