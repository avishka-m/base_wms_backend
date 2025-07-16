"""
Anomaly Detection API Routes
Beginner-friendly endpoints for warehouse anomaly detection!
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.auth.dependencies import get_current_user
from app.services.anomaly_detection_service import anomaly_detection_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", summary="🔍 Anomaly Detection Health Check")
async def anomaly_detection_health():
    """
    🔍 Check if anomaly detection service is running!
    Perfect for beginners to test the system.
    """
    return {
        "status": "healthy",
        "service": "Anomaly Detection System",
        "message": "🎯 Ready to catch anomalies!",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "📦 Inventory anomalies",
            "🛒 Order anomalies", 
            "👷 Worker anomalies",
            "🔄 Workflow anomalies"
        ]
    }

@router.get("/detect", summary="🚨 Detect All Anomalies")
async def detect_all_anomalies(
    user: dict = Depends(get_current_user)
):
    """
    🚨 Main endpoint to detect ALL types of anomalies!
    
    **Perfect for beginners to see everything at once.**
    
    **Returns:**
    - Inventory anomalies (stock issues, dead stock, etc.)
    - Order anomalies (huge orders, midnight orders, etc.)
    - Worker anomalies (performance issues, unusual patterns)
    - Workflow anomalies (stuck processes, delays)
    - Summary statistics
    
    **Example Response:**
    ```json
    {
        "inventory": [
            {
                "type": "sudden_stock_drop",
                "severity": "critical",
                "title": "🚨 Sudden Stock Drop Detected!",
                "description": "Stock dropped by 50 units suddenly",
                "action": "Check for theft, damage, or data error",
                "item_name": "Laptop Computer",
                "current_stock": 5
            }
        ],
        "summary": {
            "total_anomalies": 12,
            "critical_count": 3,
            "status": "critical"
        }
    }
    ```
    """
    try:
        logger.info(f"🔍 User {user.get('username')} requesting anomaly detection")
        
        # Run comprehensive anomaly detection
        anomalies = await anomaly_detection_service.detect_all_anomalies()
        
        return {
            "success": True,
            "data": anomalies,
            "message": f"🎯 Anomaly detection complete! Found {anomalies.get('summary', {}).get('total_anomalies', 0)} anomalies",
            "user": user.get('username'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in anomaly detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

@router.get("/inventory", summary="📦 Inventory Anomalies Only")
async def detect_inventory_anomalies(
    user: dict = Depends(get_current_user)
):
    """
    📦 Detect only inventory-related anomalies!
    
    **Great for warehouse managers and inventory clerks.**
    
    **Detects:**
    - 🚨 Sudden stock drops
    - 💀 Dead stock (no movement)
    - 🤯 Impossible quantities
    - 📉 Critical low stock
    
    **Perfect for learning about inventory patterns!**
    """
    try:
        logger.info(f"📦 User {user.get('username')} requesting inventory anomalies")
        
        inventory_anomalies = await anomaly_detection_service.detect_inventory_anomalies()
        
        return {
            "success": True,
            "category": "inventory",
            "anomalies": inventory_anomalies,
            "count": len(inventory_anomalies),
            "message": f"📦 Found {len(inventory_anomalies)} inventory anomalies",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error detecting inventory anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inventory anomaly detection failed: {str(e)}")

@router.get("/orders", summary="🛒 Order Anomalies Only")
async def detect_order_anomalies(
    user: dict = Depends(get_current_user)
):
    """
    🛒 Detect only order-related anomalies!
    
    **Perfect for order processors and customer service.**
    
    **Detects:**
    - 🚀 Unusually large orders
    - 🌙 Midnight orders (suspicious timing)
    - 🔄 Possible duplicate orders
    - 💰 High-value rush orders
    
    **Learn to spot unusual order patterns!**
    """
    try:
        logger.info(f"🛒 User {user.get('username')} requesting order anomalies")
        
        order_anomalies = await anomaly_detection_service.detect_order_anomalies()
        
        return {
            "success": True,
            "category": "orders",
            "anomalies": order_anomalies,
            "count": len(order_anomalies),
            "message": f"🛒 Found {len(order_anomalies)} order anomalies",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error detecting order anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Order anomaly detection failed: {str(e)}")

@router.get("/workers", summary="👷 Worker Anomalies Only")
async def detect_worker_anomalies(
    user: dict = Depends(get_current_user)
):
    """
    👷 Detect only worker-related anomalies!
    
    **Great for HR and warehouse managers.**
    
    **Detects:**
    - 🏃‍♂️ Unusual task speed (too fast/slow)
    - 🌙 Unusual login patterns
    - ❌ High error rates
    - 📍 Unusual location patterns
    
    **Learn about workforce analytics!**
    """
    try:
        logger.info(f"👷 User {user.get('username')} requesting worker anomalies")
        
        worker_anomalies = await anomaly_detection_service.detect_worker_anomalies()
        
        return {
            "success": True,
            "category": "workers",
            "anomalies": worker_anomalies,
            "count": len(worker_anomalies),
            "message": f"👷 Found {len(worker_anomalies)} worker anomalies",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error detecting worker anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Worker anomaly detection failed: {str(e)}")

@router.get("/workflow", summary="🔄 Workflow Anomalies Only")
async def detect_workflow_anomalies(
    user: dict = Depends(get_current_user)
):
    """
    🔄 Detect only workflow-related anomalies!
    
    **Perfect for process managers and supervisors.**
    
    **Detects:**
    - ⏰ Stuck processes (too long in one status)
    - 🚫 Processing delays
    - 🔀 Skipped workflow stages
    - 📊 Bottleneck patterns
    
    **Learn about process optimization!**
    """
    try:
        logger.info(f"🔄 User {user.get('username')} requesting workflow anomalies")
        
        workflow_anomalies = await anomaly_detection_service.detect_workflow_anomalies()
        
        return {
            "success": True,
            "category": "workflow",
            "anomalies": workflow_anomalies,
            "count": len(workflow_anomalies),
            "message": f"🔄 Found {len(workflow_anomalies)} workflow anomalies",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error detecting workflow anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow anomaly detection failed: {str(e)}")

@router.get("/summary", summary="📊 Anomaly Summary Statistics")
async def get_anomaly_summary(
    user: dict = Depends(get_current_user)
):
    """
    📊 Get quick summary of all anomalies!
    
    **Perfect for dashboards and quick health checks.**
    
    **Returns:**
    - Total anomaly count
    - Count by severity (critical, warning, info)
    - Count by category (inventory, orders, workers, workflow)
    - Overall system health status
    
    **Great for learning about system monitoring!**
    """
    try:
        logger.info(f"📊 User {user.get('username')} requesting anomaly summary")
        
        # Get all anomalies
        all_anomalies = await anomaly_detection_service.detect_all_anomalies()
        
        # Calculate detailed statistics
        stats_by_category = {}
        stats_by_severity = {"critical": 0, "warning": 0, "info": 0}
        
        for category, anomalies in all_anomalies.items():
            if isinstance(anomalies, list):
                stats_by_category[category] = len(anomalies)
                
                for anomaly in anomalies:
                    severity = anomaly.get("severity", "info")
                    stats_by_severity[severity] += 1
        
        # Overall health assessment
        total_anomalies = sum(stats_by_category.values())
        health_status = "healthy"
        if stats_by_severity["critical"] > 0:
            health_status = "critical"
        elif stats_by_severity["warning"] > 0:
            health_status = "warning"
        elif total_anomalies > 0:
            health_status = "needs_attention"
        
        return {
            "success": True,
            "summary": {
                "total_anomalies": total_anomalies,
                "health_status": health_status,
                "by_severity": stats_by_severity,
                "by_category": stats_by_category,
                "detection_time": datetime.now().isoformat(),
                "recommendations": [
                    f"🔴 {stats_by_severity['critical']} critical issues need immediate attention" if stats_by_severity["critical"] > 0 else None,
                    f"🟡 {stats_by_severity['warning']} warnings should be reviewed" if stats_by_severity["warning"] > 0 else None,
                    f"🔵 {stats_by_severity['info']} items need monitoring" if stats_by_severity["info"] > 0 else None,
                    "🟢 All systems normal!" if total_anomalies == 0 else None
                ]
            },
            "message": f"📊 System health: {health_status.upper()}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting anomaly summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Anomaly summary failed: {str(e)}")

@router.post("/configure", summary="⚙️ Configure Anomaly Thresholds")
async def configure_thresholds(
    thresholds: Dict[str, Any],
    user: dict = Depends(get_current_user)
):
    """
    ⚙️ Configure anomaly detection thresholds!
    
    **For advanced users who want to customize detection.**
    
    **Example request:**
    ```json
    {
        "inventory": {
            "sudden_drop_percentage": 60,
            "dead_stock_days": 45
        },
        "orders": {
            "huge_quantity": 150,
            "rush_order_value": 7500
        }
    }
    ```
    
    **Great for learning about tuning detection systems!**
    """
    try:
        logger.info(f"⚙️ User {user.get('username')} configuring anomaly thresholds")
        
        # Update thresholds in the service
        current_thresholds = anomaly_detection_service.thresholds
        
        for category, values in thresholds.items():
            if category in current_thresholds:
                current_thresholds[category].update(values)
        
        return {
            "success": True,
            "message": "⚙️ Thresholds updated successfully!",
            "updated_thresholds": current_thresholds,
            "user": user.get('username'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error configuring thresholds: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.get("/explain/{anomaly_type}", summary="📚 Explain Anomaly Type")
async def explain_anomaly_type(
    anomaly_type: str,
    user: dict = Depends(get_current_user)
):
    """
    📚 Get detailed explanation of an anomaly type!
    
    **Perfect for beginners learning about different anomalies.**
    
    **Available types:**
    - sudden_stock_drop
    - dead_stock
    - impossible_quantity
    - huge_order
    - midnight_order
    - unusual_speed
    - high_error_rate
    - stuck_workflow
    
    **Learn what each anomaly means and how to handle it!**
    """
    
    explanations = {
        "sudden_stock_drop": {
            "title": "🚨 Sudden Stock Drop",
            "what_it_means": "When inventory levels drop dramatically in a short time",
            "why_it_happens": ["Theft or loss", "Data entry errors", "Bulk sales not recorded", "Damage or spoilage"],
            "how_to_fix": ["Check security cameras", "Verify recent transactions", "Audit physical stock", "Review access logs"],
            "severity": "Critical - investigate immediately",
            "beginner_tip": "Start by checking if any large orders were processed recently!"
        },
        "dead_stock": {
            "title": "💀 Dead Stock",
            "what_it_means": "Items that haven't moved or sold in a long time",
            "why_it_happens": ["Seasonal items out of season", "Overordering", "Product obsolescence", "Poor demand forecasting"],
            "how_to_fix": ["Run promotions", "Bundle with popular items", "Return to supplier", "Donate or dispose"],
            "severity": "Warning - costs money to store",
            "beginner_tip": "Look for items that have been sitting for more than 30 days!"
        },
        "huge_order": {
            "title": "🚀 Unusually Large Order",
            "what_it_means": "Orders much larger than typical customer orders",
            "why_it_happens": ["Bulk business customers", "Fraudulent orders", "System errors", "Special events"],
            "how_to_fix": ["Verify customer information", "Check payment method", "Confirm order details", "Contact customer"],
            "severity": "Warning - verify before processing",
            "beginner_tip": "Always confirm large orders with the customer before shipping!"
        },
        "midnight_order": {
            "title": "🌙 Midnight Order",
            "what_it_means": "Orders placed during unusual hours (late night/early morning)",
            "why_it_happens": ["Different time zones", "Fraud attempts", "Automated systems", "Shift workers"],
            "how_to_fix": ["Check customer location", "Verify payment method", "Review for fraud indicators", "Confirm if needed"],
            "severity": "Info - monitor for patterns",
            "beginner_tip": "Consider the customer's location - they might be in a different time zone!"
        }
    }
    
    if anomaly_type not in explanations:
        raise HTTPException(status_code=404, detail=f"Anomaly type '{anomaly_type}' not found")
    
    return {
        "success": True,
        "anomaly_type": anomaly_type,
        "explanation": explanations[anomaly_type],
        "user": user.get('username'),
        "timestamp": datetime.now().isoformat(),
        "learn_more": "Try running the detection endpoints to see these anomalies in action!"
    } 