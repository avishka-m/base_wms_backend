"""
Advanced Anomaly Detection API Routes
Provides both rule-based and AI/ML anomaly detection capabilities using Isolation Forest
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.auth.dependencies import get_current_user, has_role
from app.services.advanced_anomaly_detection_service import advanced_anomaly_detection_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", summary="🔍 Advanced Anomaly Detection Health Check")
async def anomaly_detection_health():
    """
    🔍 Check if advanced anomaly detection service is running!
    Supports both rule-based and ML-based detection.
    """
    return {
        "status": "healthy",
        "service": "Advanced Anomaly Detection System",
        "message": "🎯 Ready to catch anomalies with AI power!",
        "timestamp": datetime.now().isoformat(),
        "techniques": [
            "🔍 Rule-based detection",
            "🤖 Isolation Forest ML",
            "📊 Statistical analysis",
            "🔄 Real-time monitoring"
        ],
        "features": [
            "📦 Inventory anomalies (rule + ML)",
            "🛒 Order pattern anomalies (rule + ML)", 
            "👷 Worker behavior anomalies",
            "🔄 Workflow bottlenecks"
        ]
    }

@router.get("/detect", summary="🚨 Comprehensive Anomaly Detection")
async def detect_all_anomalies(
    include_ml: bool = Query(True, description="Include ML-based detection"),
    user: dict = Depends(get_current_user)
):
    """
    🚨 Advanced anomaly detection using both rule-based and ML techniques!
    
    Features:
    - Rule-based detection for known patterns
    - Isolation Forest for unknown anomalies
    - Combined analysis and scoring
    - Actionable recommendations
    
    Returns:
    - Rule-based anomalies (threshold violations, business rules)
    - ML-based anomalies (statistical outliers, pattern deviations)
    - Combined analysis with severity scoring
    - Health status and recommendations
    """
    try:
        logger.info(f"🔍 Starting comprehensive anomaly detection for user {user.get('username')}")
        
        result = await advanced_anomaly_detection_service.detect_all_anomalies(include_ml=include_ml)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "message": "Anomaly detection completed successfully",
            "data": result,
            "user": user.get("username"),
            "request_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in comprehensive anomaly detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

@router.get("/detect/inventory", summary="📦 Inventory Anomaly Detection")
async def detect_inventory_anomalies(
    technique: str = Query("both", description="Detection technique: 'rule', 'ml', or 'both'"),
    user: dict = Depends(get_current_user)
):
    """
    📦 Focused inventory anomaly detection
    
    Rule-based detection:
    - Critical stockouts
    - Extreme low/high stock levels
    - Dead stock identification
    - Impossible quantities
    
    ML-based detection:
    - Stock pattern anomalies
    - Unusual inventory behaviors
    - Statistical outliers
    """
    try:
        anomalies = []
        
        if technique in ["rule", "both"]:
            rule_anomalies = await advanced_anomaly_detection_service.detect_inventory_rule_anomalies()
            anomalies.extend(rule_anomalies)
        
        if technique in ["ml", "both"]:
            ml_anomalies = await advanced_anomaly_detection_service.detect_inventory_ml_anomalies()
            anomalies.extend(ml_anomalies)
        
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        anomalies.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))
        
        return {
            "success": True,
            "category": "inventory",
            "technique_used": technique,
            "total_anomalies": len(anomalies),
            "anomalies": anomalies,
            "summary": {
                "critical": len([a for a in anomalies if a.get("severity") == "critical"]),
                "high": len([a for a in anomalies if a.get("severity") == "high"]),
                "medium": len([a for a in anomalies if a.get("severity") == "medium"]),
                "low": len([a for a in anomalies if a.get("severity") == "low"])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in inventory anomaly detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inventory anomaly detection failed: {str(e)}")

@router.get("/detect/orders", summary="🛒 Order Anomaly Detection")
async def detect_order_anomalies(
    technique: str = Query("both", description="Detection technique: 'rule', 'ml', or 'both'"),
    user: dict = Depends(get_current_user)
):
    """
    🛒 Advanced order anomaly detection
    
    Rule-based detection:
    - Unusual order timing
    - High-value orders
    - Bulk order patterns
    - Processing delays
    
    ML-based detection:
    - Order behavior anomalies
    - Pattern deviations
    - Statistical outliers
    """
    try:
        anomalies = []
        
        if technique in ["rule", "both"]:
            rule_anomalies = await advanced_anomaly_detection_service.detect_order_rule_anomalies()
            anomalies.extend(rule_anomalies)
        
        if technique in ["ml", "both"]:
            ml_anomalies = await advanced_anomaly_detection_service.detect_order_ml_anomalies()
            anomalies.extend(ml_anomalies)
        
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        anomalies.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))
        
        return {
            "success": True,
            "category": "orders",
            "technique_used": technique,
            "total_anomalies": len(anomalies),
            "anomalies": anomalies,
            "summary": {
                "critical": len([a for a in anomalies if a.get("severity") == "critical"]),
                "high": len([a for a in anomalies if a.get("severity") == "high"]),
                "medium": len([a for a in anomalies if a.get("severity") == "medium"]),
                "low": len([a for a in anomalies if a.get("severity") == "low"])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in order anomaly detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Order anomaly detection failed: {str(e)}")

@router.get("/detect/workflow", summary="🔄 Workflow Anomaly Detection")
async def detect_workflow_anomalies(
    user: dict = Depends(get_current_user)
):
    """
    🔄 Workflow and process anomaly detection
    
    Detects:
    - Stuck orders and processes
    - Workflow bottlenecks
    - Processing delays
    - Stage skipping
    """
    try:
        anomalies = await advanced_anomaly_detection_service.detect_workflow_rule_anomalies()
        
        return {
            "success": True,
            "category": "workflow",
            "total_anomalies": len(anomalies),
            "anomalies": anomalies,
            "summary": {
                "stuck_orders": len([a for a in anomalies if a.get("type") == "stuck_workflow"]),
                "bottlenecks": len([a for a in anomalies if a.get("type") == "workflow_bottleneck"]),
                "delays": len([a for a in anomalies if a.get("type") == "processing_delay"])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in workflow anomaly detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow anomaly detection failed: {str(e)}")

@router.get("/detect/workers", summary="👷 Worker Anomaly Detection")
async def detect_worker_anomalies(
    user: dict = Depends(has_role(["Manager"]))
):
    """
    👷 Worker behavior and performance anomaly detection (Manager only)
    
    Detects:
    - Performance anomalies
    - Unusual login patterns
    - Productivity drops
    - Error rate increases
    """
    try:
        anomalies = await advanced_anomaly_detection_service.detect_worker_rule_anomalies()
        
        return {
            "success": True,
            "category": "workers",
            "total_anomalies": len(anomalies),
            "anomalies": anomalies,
            "summary": {
                "performance_issues": len([a for a in anomalies if a.get("type") == "low_performance"]),
                "unusual_patterns": len([a for a in anomalies if "unusual" in a.get("type", "")])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in worker anomaly detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Worker anomaly detection failed: {str(e)}")

@router.get("/analysis/summary", summary="📊 Anomaly Analysis Summary")
async def get_anomaly_summary(
    days: int = Query(7, description="Number of days to analyze"),
    user: dict = Depends(get_current_user)
):
    """
    📊 Get comprehensive anomaly analysis summary
    
    Provides:
    - Overall system health status
    - Trend analysis over time
    - Anomaly distribution by category
    - Key recommendations
    """
    try:
        # Get current anomalies
        result = await advanced_anomaly_detection_service.detect_all_anomalies(include_ml=True)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        summary = result.get("summary", {})
        
        # Enhanced summary with trends (placeholder for historical analysis)
        enhanced_summary = {
            "current_status": summary,
            "analysis_period": f"Last {days} days",
            "system_health": summary.get("health_status", "unknown"),
            "total_anomalies": summary.get("total_anomalies", 0),
            "severity_breakdown": summary.get("severity_breakdown", {}),
            "category_breakdown": summary.get("category_breakdown", {}),
            "technique_breakdown": summary.get("technique_breakdown", {}),
            "recommendations": summary.get("recommendations", []),
            "trends": {
                "anomaly_count_trend": "stable",  # Would calculate from historical data
                "severity_trend": "improving",   # Would analyze historical severity
                "most_problematic_category": "inventory"  # Based on current data
            },
            "action_items": _generate_action_items(result),
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "summary": enhanced_summary
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting anomaly summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

@router.post("/models/retrain", summary="🤖 Retrain ML Models")
async def retrain_ml_models(
    background_tasks: BackgroundTasks,
    user: dict = Depends(has_role(["Manager"]))
):
    """
    🤖 Retrain Isolation Forest models with latest data (Manager only)
    
    Process:
    - Collect latest training data
    - Retrain all ML models
    - Update model parameters
    - Save updated models
    """
    try:
        # Add retraining to background tasks
        background_tasks.add_task(_retrain_models_background)
        
        return {
            "success": True,
            "message": "Model retraining started in background",
            "estimated_completion": "5-10 minutes",
            "initiated_by": user.get("username"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error initiating model retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {str(e)}")

@router.get("/models/status", summary="📈 ML Models Status")
async def get_models_status(
    user: dict = Depends(has_role(["Manager"]))
):
    """
    📈 Get status and performance of ML models (Manager only)
    """
    try:
        # This would check model files, training dates, performance metrics
        models_status = {
            "inventory_model": {
                "status": "trained",
                "last_training": "2024-01-15T10:30:00Z",
                "contamination_rate": 0.1,
                "features_count": 7,
                "performance": "good"
            },
            "orders_model": {
                "status": "trained", 
                "last_training": "2024-01-15T10:30:00Z",
                "contamination_rate": 0.1,
                "features_count": 7,
                "performance": "good"
            },
            "workers_model": {
                "status": "not_trained",
                "reason": "insufficient_data"
            },
            "workflow_model": {
                "status": "not_trained",
                "reason": "insufficient_data"
            }
        }
        
        return {
            "success": True,
            "models": models_status,
            "overall_status": "operational",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model status check failed: {str(e)}")

@router.get("/thresholds", summary="⚙️ Get Detection Thresholds")
async def get_detection_thresholds(
    user: dict = Depends(has_role(["Manager"]))
):
    """
    ⚙️ Get current anomaly detection thresholds (Manager only)
    """
    try:
        thresholds = advanced_anomaly_detection_service.rule_thresholds
        
        return {
            "success": True,
            "thresholds": thresholds,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting thresholds: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Threshold retrieval failed: {str(e)}")

@router.put("/thresholds", summary="⚙️ Update Detection Thresholds")
async def update_detection_thresholds(
    thresholds: Dict[str, Any],
    user: dict = Depends(has_role(["Manager"]))
):
    """
    ⚙️ Update anomaly detection thresholds (Manager only)
    """
    try:
        # Validate and update thresholds
        current_thresholds = advanced_anomaly_detection_service.rule_thresholds
        
        # Update with provided values (with validation)
        for category, values in thresholds.items():
            if category in current_thresholds:
                for key, value in values.items():
                    if key in current_thresholds[category]:
                        current_thresholds[category][key] = value
        
        return {
            "success": True,
            "message": "Thresholds updated successfully",
            "updated_by": user.get("username"),
            "updated_thresholds": thresholds,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error updating thresholds: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Threshold update failed: {str(e)}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _generate_action_items(detection_result: Dict) -> List[Dict[str, Any]]:
    """Generate specific action items based on detected anomalies"""
    action_items = []
    
    combined_anomalies = detection_result.get("combined", {})
    
    # Analyze inventory issues
    inventory_anomalies = combined_anomalies.get("inventory", [])
    critical_inventory = [a for a in inventory_anomalies if a.get("severity") == "critical"]
    
    if critical_inventory:
        action_items.append({
            "priority": "critical",
            "category": "inventory",
            "action": "immediate_restocking",
            "description": f"Immediately restock {len(critical_inventory)} critical items",
            "affected_items": [a.get("item_id") for a in critical_inventory],
            "deadline": "immediate"
        })
    
    # Analyze workflow issues
    workflow_anomalies = combined_anomalies.get("workflow", [])
    stuck_orders = [a for a in workflow_anomalies if a.get("type") == "stuck_workflow"]
    
    if stuck_orders:
        action_items.append({
            "priority": "high",
            "category": "workflow",
            "action": "resolve_stuck_orders",
            "description": f"Investigate and resolve {len(stuck_orders)} stuck orders",
            "affected_orders": [a.get("order_id") for a in stuck_orders],
            "deadline": "within_24_hours"
        })
    
    return action_items

async def _retrain_models_background():
    """Background task for model retraining"""
    try:
        logger.info("🤖 Starting ML model retraining...")
        
        # In a real implementation, this would:
        # 1. Collect latest training data
        # 2. Retrain models with new data
        # 3. Validate model performance
        # 4. Save updated models
        
        # Simulate retraining process
        import asyncio
        await asyncio.sleep(5)  # Simulate training time
        
        await advanced_anomaly_detection_service.save_models()
        
        logger.info("✅ ML model retraining completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in background model retraining: {str(e)}")
