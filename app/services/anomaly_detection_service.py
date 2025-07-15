"""
Anomaly Detection Service for Warehouse Management System
Simple rule-based anomaly detection perfect for beginners!
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
from statistics import mean, median

from app.utils.database import get_database

logger = logging.getLogger(__name__)

class AnomalyDetectionService:
    """
    Beginner-friendly anomaly detection service using simple rules.
    No complex ML models - just smart logic!
    """
    
    def __init__(self):
        self.db = get_database()
        
        # 🎯 Anomaly thresholds (easy to adjust!)
        self.thresholds = {
            "inventory": {
                "sudden_drop_percentage": 50,  # 50% stock drop in 1 day
                "dead_stock_days": 30,         # No movement for 30 days
                "impossible_quantity": 10000,   # More than 10k items suspicious
                "low_stock_multiplier": 0.5     # Below 50% of min stock
            },
            "orders": {
                "huge_quantity": 100,           # Order with 100+ items
                "midnight_start": 23,           # Orders after 11 PM
                "midnight_end": 6,              # Orders before 6 AM
                "duplicate_time_window": 5,     # Duplicate orders within 5 minutes
                "rush_order_value": 5000        # High-value orders
            },
            "workers": {
                "task_time_multiplier": 3,      # 3x longer than average
                "error_rate_threshold": 0.1,    # 10% error rate
                "location_jump_distance": 100,  # 100+ meters in short time
                "unusual_login_hours": [22, 23, 0, 1, 2, 3, 4, 5]  # Late night logins
            },
            "workflow": {
                "stuck_hours": 24,              # Order stuck for 24+ hours
                "processing_delay": 6,          # 6+ hours in same status
                "skip_stage_flag": True         # Skipped workflow stages
            }
        }
    
    async def detect_all_anomalies(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        🔍 Master function to detect ALL types of anomalies!
        Returns organized results by category.
        """
        logger.info("🔍 Starting comprehensive anomaly detection...")
        
        try:
            anomalies = {
                "inventory": await self.detect_inventory_anomalies(),
                "orders": await self.detect_order_anomalies(),
                "workers": await self.detect_worker_anomalies(),
                "workflow": await self.detect_workflow_anomalies(),
                "summary": {}
            }
            
            # 📊 Create summary statistics
            total_anomalies = sum(len(category) for category in anomalies.values() if isinstance(category, list))
            critical_count = sum(1 for category in anomalies.values() if isinstance(category, list) 
                               for anomaly in category if anomaly.get("severity") == "critical")
            
            anomalies["summary"] = {
                "total_anomalies": total_anomalies,
                "critical_count": critical_count,
                "warning_count": total_anomalies - critical_count,
                "detection_time": datetime.now().isoformat(),
                "status": "critical" if critical_count > 0 else "warning" if total_anomalies > 0 else "normal"
            }
            
            logger.info(f"✅ Anomaly detection complete! Found {total_anomalies} anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"❌ Error in anomaly detection: {str(e)}")
            return {"error": str(e)}
    
    async def detect_inventory_anomalies(self) -> List[Dict[str, Any]]:
        """
        📦 Detect inventory anomalies - perfect for beginners!
        """
        anomalies = []
        
        try:
            # Get inventory data
            inventory_collection = self.db["inventory"]
            items = await inventory_collection.find({}).to_list(None)
            
            for item in items:
                item_anomalies = []
                
                # 🚨 ANOMALY 1: Sudden Stock Drop
                if await self._check_sudden_stock_drop(item):
                    item_anomalies.append({
                        "type": "sudden_stock_drop",
                        "severity": "critical",
                        "title": "🚨 Sudden Stock Drop Detected!",
                        "description": f"Stock dropped by {item.get('stock_level', 0)} units suddenly",
                        "action": "Check for theft, damage, or data error"
                    })
                
                # 💀 ANOMALY 2: Dead Stock (No Movement)
                if await self._check_dead_stock(item):
                    item_anomalies.append({
                        "type": "dead_stock",
                        "severity": "warning",
                        "title": "💀 Dead Stock Alert",
                        "description": f"No movement for {self.thresholds['inventory']['dead_stock_days']} days",
                        "action": "Consider promotion or redistribution"
                    })
                
                # 🤯 ANOMALY 3: Impossible Quantities
                if item.get("stock_level", 0) > self.thresholds["inventory"]["impossible_quantity"]:
                    item_anomalies.append({
                        "type": "impossible_quantity",
                        "severity": "critical",
                        "title": "🤯 Impossible Quantity Detected!",
                        "description": f"Stock level of {item.get('stock_level')} seems impossible",
                        "action": "Verify stock count immediately"
                    })
                
                # 📉 ANOMALY 4: Critically Low Stock
                if await self._check_low_stock_anomaly(item):
                    item_anomalies.append({
                        "type": "critical_low_stock",
                        "severity": "warning",
                        "title": "📉 Critical Low Stock",
                        "description": f"Below minimum threshold by significant margin",
                        "action": "Emergency reorder needed"
                    })
                
                # Add item info to each anomaly
                for anomaly in item_anomalies:
                    anomaly.update({
                        "item_id": item.get("itemID"),
                        "item_name": item.get("name"),
                        "current_stock": item.get("stock_level"),
                        "location": item.get("locationID"),
                        "timestamp": datetime.now().isoformat(),
                        "category": "inventory"
                    })
                    anomalies.append(anomaly)
            
            logger.info(f"📦 Found {len(anomalies)} inventory anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"❌ Error detecting inventory anomalies: {str(e)}")
            return []
    
    async def detect_order_anomalies(self) -> List[Dict[str, Any]]:
        """
        🛒 Detect order anomalies - beginner-friendly patterns!
        """
        anomalies = []
        
        try:
            orders_collection = self.db["orders"]
            
            # Get recent orders (last 7 days)
            recent_date = datetime.now() - timedelta(days=7)
            recent_orders = await orders_collection.find({
                "created_at": {"$gte": recent_date}
            }).to_list(None)
            
            for order in recent_orders:
                order_anomalies = []
                
                # 🚀 ANOMALY 1: Huge Order Quantities
                total_quantity = sum(item.get("quantity", 0) for item in order.get("order_details", []))
                if total_quantity > self.thresholds["orders"]["huge_quantity"]:
                    order_anomalies.append({
                        "type": "huge_order",
                        "severity": "warning",
                        "title": "🚀 Unusually Large Order",
                        "description": f"Order contains {total_quantity} items - way above average!",
                        "action": "Verify customer and payment method"
                    })
                
                # 🌙 ANOMALY 2: Midnight Orders (Suspicious timing)
                order_hour = order.get("created_at", datetime.now()).hour
                if (order_hour >= self.thresholds["orders"]["midnight_start"] or 
                    order_hour <= self.thresholds["orders"]["midnight_end"]):
                    order_anomalies.append({
                        "type": "midnight_order",
                        "severity": "info",
                        "title": "🌙 Midnight Order Alert",
                        "description": f"Order placed at {order_hour}:00 - unusual timing",
                        "action": "Review for fraud indicators"
                    })
                
                # 🔄 ANOMALY 3: Duplicate Orders Detection
                if await self._check_duplicate_orders(order):
                    order_anomalies.append({
                        "type": "possible_duplicate",
                        "severity": "warning",
                        "title": "🔄 Possible Duplicate Order",
                        "description": "Similar order found within short time window",
                        "action": "Contact customer to confirm"
                    })
                
                # 💰 ANOMALY 4: High-Value Rush Orders
                order_value = order.get("total_amount", 0)
                if order_value > self.thresholds["orders"]["rush_order_value"]:
                    order_anomalies.append({
                        "type": "high_value_rush",
                        "severity": "info",
                        "title": "💰 High-Value Order Alert",
                        "description": f"Order value ${order_value} - requires attention",
                        "action": "Priority processing and verification"
                    })
                
                # Add order info to each anomaly
                for anomaly in order_anomalies:
                    anomaly.update({
                        "order_id": order.get("orderID"),
                        "customer_id": order.get("customerID"),
                        "order_value": order.get("total_amount"),
                        "item_count": len(order.get("order_details", [])),
                        "timestamp": datetime.now().isoformat(),
                        "category": "orders"
                    })
                    anomalies.append(anomaly)
            
            logger.info(f"🛒 Found {len(anomalies)} order anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"❌ Error detecting order anomalies: {str(e)}")
            return []
    
    async def detect_worker_anomalies(self) -> List[Dict[str, Any]]:
        """
        👷 Detect worker performance anomalies - learn about behavioral patterns!
        """
        anomalies = []
        
        try:
            workers_collection = self.db["workers"]
            workers = await workers_collection.find({}).to_list(None)
            
            # Get recent activity data (you might need to add activity tracking)
            for worker in workers:
                worker_anomalies = []
                
                # 🏃‍♂️ ANOMALY 1: Unusual Task Speed
                if await self._check_unusual_task_speed(worker):
                    worker_anomalies.append({
                        "type": "unusual_speed",
                        "severity": "info",
                        "title": "🏃‍♂️ Unusual Task Speed",
                        "description": "Tasks completed much faster/slower than average",
                        "action": "Review work quality and provide support if needed"
                    })
                
                # 🌙 ANOMALY 2: Unusual Login Times
                if await self._check_unusual_login_times(worker):
                    worker_anomalies.append({
                        "type": "unusual_login",
                        "severity": "info",
                        "title": "🌙 Unusual Login Pattern",
                        "description": "Login at unusual hours detected",
                        "action": "Verify with worker and check security"
                    })
                
                # ❌ ANOMALY 3: High Error Rate
                error_rate = await self._calculate_worker_error_rate(worker)
                if error_rate > self.thresholds["workers"]["error_rate_threshold"]:
                    worker_anomalies.append({
                        "type": "high_error_rate",
                        "severity": "warning",
                        "title": "❌ High Error Rate Alert",
                        "description": f"Error rate of {error_rate:.1%} is above normal",
                        "action": "Provide additional training and support"
                    })
                
                # Add worker info to each anomaly
                for anomaly in worker_anomalies:
                    anomaly.update({
                        "worker_id": worker.get("workerID"),
                        "worker_name": worker.get("name"),
                        "worker_role": worker.get("role"),
                        "timestamp": datetime.now().isoformat(),
                        "category": "workers"
                    })
                    anomalies.append(anomaly)
            
            logger.info(f"👷 Found {len(anomalies)} worker anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"❌ Error detecting worker anomalies: {str(e)}")
            return []
    
    async def detect_workflow_anomalies(self) -> List[Dict[str, Any]]:
        """
        🔄 Detect workflow anomalies - learn about process efficiency!
        """
        anomalies = []
        
        try:
            # Check multiple workflow areas
            workflow_areas = ["receiving", "picking", "packing", "shipping"]
            
            for area in workflow_areas:
                collection = self.db[area]
                
                # Get recent workflow items
                recent_items = await collection.find({
                    "created_at": {"$gte": datetime.now() - timedelta(days=3)}
                }).to_list(None)
                
                for item in recent_items:
                    item_anomalies = []
                    
                    # ⏰ ANOMALY 1: Stuck in Status Too Long
                    if await self._check_stuck_workflow(item, area):
                        item_anomalies.append({
                            "type": "stuck_workflow",
                            "severity": "warning",
                            "title": f"⏰ {area.title()} Process Stuck",
                            "description": f"Item stuck in {item.get('status')} for too long",
                            "action": f"Review {area} process and resolve blockage"
                        })
                    
                    # 🚫 ANOMALY 2: Processing Delays
                    processing_time = await self._calculate_processing_time(item)
                    if processing_time > self.thresholds["workflow"]["processing_delay"]:
                        item_anomalies.append({
                            "type": "processing_delay",
                            "severity": "info",
                            "title": "🚫 Processing Delay",
                            "description": f"Processing time of {processing_time}h exceeds normal",
                            "action": "Investigate bottlenecks in workflow"
                        })
                    
                    # Add workflow info to each anomaly
                    for anomaly in item_anomalies:
                        anomaly.update({
                            "workflow_area": area,
                            "item_id": item.get("_id"),
                            "status": item.get("status"),
                            "processing_time": processing_time,
                            "timestamp": datetime.now().isoformat(),
                            "category": "workflow"
                        })
                        anomalies.append(anomaly)
            
            logger.info(f"🔄 Found {len(anomalies)} workflow anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"❌ Error detecting workflow anomalies: {str(e)}")
            return []
    
    # 🔧 Helper Methods (The Magic Behind The Scenes!)
    
    async def _check_sudden_stock_drop(self, item: Dict) -> bool:
        """Check if inventory had a sudden drop (beginner-friendly logic)"""
        try:
            # In real implementation, you'd compare with historical data
            # For now, we'll use a simple heuristic
            current_stock = item.get("stock_level", 0)
            min_stock = item.get("min_stock_level", 0)
            
            # If current stock is way below minimum, it might be a sudden drop
            return current_stock < (min_stock * 0.3) and min_stock > 0
        except:
            return False
    
    async def _check_dead_stock(self, item: Dict) -> bool:
        """Check if item hasn't moved in specified days"""
        try:
            # In real implementation, check last movement date
            # For now, assume items with very low turnover are dead stock
            stock_level = item.get("stock_level", 0)
            max_stock = item.get("max_stock_level", 100)
            
            # If stock is very high compared to max, might be dead stock
            return stock_level > (max_stock * 0.9) and max_stock > 0
        except:
            return False
    
    async def _check_low_stock_anomaly(self, item: Dict) -> bool:
        """Check for critically low stock"""
        try:
            current_stock = item.get("stock_level", 0)
            min_stock = item.get("min_stock_level", 10)
            
            # If stock is below 50% of minimum threshold
            return current_stock < (min_stock * self.thresholds["inventory"]["low_stock_multiplier"])
        except:
            return False
    
    async def _check_duplicate_orders(self, order: Dict) -> bool:
        """Simple duplicate detection logic"""
        try:
            # In real implementation, compare orders by customer, items, timing
            # For now, return False (implement based on your needs)
            return False
        except:
            return False
    
    async def _check_unusual_task_speed(self, worker: Dict) -> bool:
        """Check for unusual task completion speed"""
        try:
            # In real implementation, analyze task completion times
            # For now, return False (implement based on your tracking)
            return False
        except:
            return False
    
    async def _check_unusual_login_times(self, worker: Dict) -> bool:
        """Check for unusual login patterns"""
        try:
            # In real implementation, analyze login history
            # For now, return False (implement based on your auth logs)
            return False
        except:
            return False
    
    async def _calculate_worker_error_rate(self, worker: Dict) -> float:
        """Calculate worker error rate"""
        try:
            # In real implementation, analyze error records
            # For now, return a sample rate
            return 0.05  # 5% sample error rate
        except:
            return 0.0
    
    async def _check_stuck_workflow(self, item: Dict, area: str) -> bool:
        """Check if workflow item is stuck"""
        try:
            # Simple time-based check
            created_at = item.get("created_at")
            if created_at:
                hours_elapsed = (datetime.now() - created_at).total_seconds() / 3600
                return hours_elapsed > self.thresholds["workflow"]["stuck_hours"]
            return False
        except:
            return False
    
    async def _calculate_processing_time(self, item: Dict) -> float:
        """Calculate processing time in hours"""
        try:
            created_at = item.get("created_at")
            if created_at:
                return (datetime.now() - created_at).total_seconds() / 3600
            return 0.0
        except:
            return 0.0

# 🎯 Global instance for easy access
anomaly_detection_service = AnomalyDetectionService() 