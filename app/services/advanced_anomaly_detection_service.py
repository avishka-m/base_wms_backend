"""
Advanced Anomaly Detection Service for Warehouse Management System
Combines rule-based techniques with AI/ML (Isolation Forest) for comprehensive anomaly detection
"""

import logging
import os
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List

import joblib
import numpy as np
from app.utils.database import get_async_database
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class AdvancedAnomalyDetectionService:
    """
    Advanced anomaly detection service combining rule-based and ML techniques.
    Uses Isolation Forest for unsupervised anomaly detection.
    """

    def __init__(self):
        self.db = None  # Will be initialized async
        self.model_dir = "models/anomaly_detection"
        os.makedirs(self.model_dir, exist_ok=True)

        # 🎯 Rule-based thresholds
        self.rule_thresholds = {
            "inventory": {
                "sudden_drop_percentage": 50,
                "dead_stock_days": 30,
                "impossible_quantity": 10000,
                "low_stock_multiplier": 0.1,
                "overstock_multiplier": 5.0,
                "zero_stock_critical": True,
            },
            "orders": {
                "huge_quantity": 100,
                "unusual_hours": [22, 23, 0, 1, 2, 3, 4, 5],
                "duplicate_time_window": 5,
                "rush_order_value": 5000,
                "processing_delay_hours": 24,
                "item_quantity_outlier": 3,  # Standard deviations
            },
            "workers": {
                "task_time_multiplier": 3,
                "error_rate_threshold": 0.15,
                "productivity_drop": 0.3,
                "unusual_login_hours": [22, 23, 0, 1, 2, 3, 4, 5],
                "location_jump_distance": 100,
            },
            "workflow": {
                "stuck_hours": 24,
                "processing_delay": 6,
                "stage_skip_detection": True,
                "bottleneck_threshold": 10,
            },
        }

        # 🤖 ML model parameters
        self.ml_params = {
            "contamination": 0.1,  # Expected proportion of anomalies
            "random_state": 42,
            "n_estimators": 100,
            "max_samples": "auto",
        }

        # Initialize models
        self.models = {}
        self.scalers = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize Isolation Forest models for different data types"""
        model_types = ["inventory", "orders", "workers", "workflow"]

        for model_type in model_types:
            self.models[model_type] = IsolationForest(**self.ml_params)
            self.scalers[model_type] = StandardScaler()

    async def _get_database(self):
        """Get async database connection"""
        if self.db is None:
            self.db = await get_async_database()
        return self.db

    # =============================================================================
    # MAIN DETECTION METHODS
    # =============================================================================

    async def detect_all_anomalies(self, include_ml: bool = True) -> Dict[str, Any]:
        """
        🔍 Master function to detect ALL types of anomalies using both techniques
        """
        logger.info("🔍 Starting comprehensive anomaly detection...")

        try:
            # Rule-based detection
            rule_anomalies = {
                "inventory": await self.detect_inventory_rule_anomalies(),
                "orders": await self.detect_order_rule_anomalies(),
                "workers": await self.detect_worker_rule_anomalies(),
                "workflow": await self.detect_workflow_rule_anomalies(),
            }

            # ML-based detection (if enabled)
            ml_anomalies = {}
            if include_ml:
                ml_anomalies = {
                    "inventory": await self.detect_inventory_ml_anomalies(),
                    "orders": await self.detect_order_ml_anomalies(),
                    "workers": await self.detect_worker_ml_anomalies(),
                    "workflow": await self.detect_workflow_ml_anomalies(),
                }

            # Combine and analyze results
            combined_anomalies = self._combine_anomaly_results(
                rule_anomalies, ml_anomalies
            )

            # Generate summary
            summary = self._generate_anomaly_summary(combined_anomalies)

            result = {
                "rule_based": rule_anomalies,
                "ml_based": ml_anomalies if include_ml else {},
                "combined": combined_anomalies,
                "summary": summary,
                "detection_timestamp": datetime.now().isoformat(),
                "techniques_used": ["rule_based"]
                + (["isolation_forest"] if include_ml else []),
            }

            logger.info(
                f"✅ Anomaly detection complete! Found {summary['total_anomalies']} anomalies"
            )
            return result

        except Exception as e:
            logger.error(f"❌ Error in anomaly detection: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    # =============================================================================
    # RULE-BASED ANOMALY DETECTION
    # =============================================================================

    async def detect_inventory_rule_anomalies(self) -> List[Dict[str, Any]]:
        """📦 Rule-based inventory anomaly detection"""
        anomalies = []

        try:
            db = await self._get_database()
            inventory_collection = db["inventory"]
            items = await inventory_collection.find({}).to_list(length=None)

            for item in items:
                item_id = item.get("itemID")
                current_stock = item.get(
                    "stock_level", 0
                )  # Fixed: was "stock_quantity"
                min_stock = item.get("min_stock_level", 10)
                max_stock = item.get("max_stock_level", 1000)
                item_name = item.get(
                    "name", f"Item {item_id}"
                )  # Fixed: was "item_name"

                # Rule 1: Zero stock critical items
                if current_stock == 0 and item.get("category") in [
                    "critical",
                    "high_priority",
                ]:
                    anomalies.append(
                        {
                            "type": "critical_stockout",
                            "severity": "critical",
                            "item_id": item_id,
                            "item_name": item_name,
                            "description": "Critical item completely out of stock",
                            "current_stock": current_stock,
                            "expected_min": min_stock,
                            "technique": "rule_based",
                            "rule": "zero_stock_critical",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                # Rule 2: Extreme low stock
                if current_stock > 0 and min_stock > 0:
                    low_stock_threshold = (
                        min_stock
                        * self.rule_thresholds["inventory"]["low_stock_multiplier"]
                    )
                    if current_stock < low_stock_threshold:
                        anomalies.append(
                            {
                                "type": "extreme_low_stock",
                                "severity": "high"
                                if current_stock < min_stock * 0.05
                                else "medium",
                                "item_id": item_id,
                                "item_name": item_name,
                                "description": f"Stock level critically low: {current_stock} (threshold: {low_stock_threshold:.1f})",
                                "current_stock": current_stock,
                                "threshold": low_stock_threshold,
                                "technique": "rule_based",
                                "rule": "extreme_low_stock",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                # Rule 3: Impossible quantities
                impossible_qty = self.rule_thresholds["inventory"][
                    "impossible_quantity"
                ]
                if current_stock > impossible_qty:
                    anomalies.append(
                        {
                            "type": "impossible_quantity",
                            "severity": "high",
                            "item_id": item_id,
                            "item_name": item_name,
                            "description": f"Suspiciously high stock quantity: {current_stock}",
                            "current_stock": current_stock,
                            "threshold": impossible_qty,
                            "technique": "rule_based",
                            "rule": "impossible_quantity",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                # Rule 4: Overstock detection
                if max_stock > 0:
                    overstock_threshold = (
                        max_stock
                        * self.rule_thresholds["inventory"]["overstock_multiplier"]
                    )
                    if current_stock > overstock_threshold:
                        anomalies.append(
                            {
                                "type": "overstock",
                                "severity": "medium",
                                "item_id": item_id,
                                "item_name": item_name,
                                "description": f"Excessive stock level: {current_stock} (max: {max_stock})",
                                "current_stock": current_stock,
                                "max_stock": max_stock,
                                "threshold": overstock_threshold,
                                "technique": "rule_based",
                                "rule": "overstock_detection",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

            # Rule 5: Dead stock detection
            dead_stock_anomalies = await self._detect_dead_stock()
            anomalies.extend(dead_stock_anomalies)

            logger.info(f"📦 Found {len(anomalies)} inventory rule-based anomalies")
            return anomalies

        except Exception as e:
            logger.error(f"❌ Error in inventory rule detection: {str(e)}")
            return []

    async def detect_order_rule_anomalies(self) -> List[Dict[str, Any]]:
        """🛒 Rule-based order anomaly detection"""
        anomalies = []

        try:
            db = await self._get_database()
            orders_collection = db["orders"]
            # Get recent orders (last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            orders = await orders_collection.find(
                {"order_date": {"$gte": cutoff_date}}
            ).to_list(length=None)

            for order in orders:
                order_id = order.get("orderID")
                order_date = order.get("order_date")
                total_amount = order.get("total_amount", 0)
                items = order.get("items", [])

                # Rule 1: Unusual order timing
                if isinstance(order_date, datetime):
                    hour = order_date.hour
                    if hour in self.rule_thresholds["orders"]["unusual_hours"]:
                        anomalies.append(
                            {
                                "type": "unusual_timing",
                                "severity": "medium",
                                "order_id": order_id,
                                "description": f"Order placed at unusual hour: {hour}:00",
                                "order_time": order_date.isoformat(),
                                "hour": hour,
                                "technique": "rule_based",
                                "rule": "unusual_timing",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                # Rule 2: High-value orders
                rush_threshold = self.rule_thresholds["orders"]["rush_order_value"]
                if total_amount > rush_threshold:
                    anomalies.append(
                        {
                            "type": "high_value_order",
                            "severity": "medium",
                            "order_id": order_id,
                            "description": f"Unusually high order value: ${total_amount:,.2f}",
                            "order_value": total_amount,
                            "threshold": rush_threshold,
                            "technique": "rule_based",
                            "rule": "high_value_order",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                # Rule 3: Large quantity orders
                huge_qty = self.rule_thresholds["orders"]["huge_quantity"]
                total_items = sum(item.get("quantity", 0) for item in items)
                if total_items > huge_qty:
                    anomalies.append(
                        {
                            "type": "bulk_order",
                            "severity": "medium",
                            "order_id": order_id,
                            "description": f"Large quantity order: {total_items} items",
                            "total_items": total_items,
                            "threshold": huge_qty,
                            "technique": "rule_based",
                            "rule": "bulk_order",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                # Rule 4: Single item excessive quantity
                for item in items:
                    item_qty = item.get("quantity", 0)
                    if (
                        item_qty > huge_qty / 2
                    ):  # Half of bulk threshold for single item
                        anomalies.append(
                            {
                                "type": "excessive_item_quantity",
                                "severity": "medium",
                                "order_id": order_id,
                                "item_id": item.get("itemID"),
                                "description": f"Excessive quantity for single item: {item_qty}",
                                "item_quantity": item_qty,
                                "threshold": huge_qty / 2,
                                "technique": "rule_based",
                                "rule": "excessive_item_quantity",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

            # Rule 5: Processing delays
            delay_anomalies = await self._detect_processing_delays()
            anomalies.extend(delay_anomalies)

            logger.info(f"🛒 Found {len(anomalies)} order rule-based anomalies")
            return anomalies

        except Exception as e:
            logger.error(f"❌ Error in order rule detection: {str(e)}")
            return []

    async def detect_worker_rule_anomalies(self) -> List[Dict[str, Any]]:
        """👷 Rule-based worker anomaly detection"""
        anomalies = []

        try:
            db = await self._get_database()
            workers_collection = db["workers"]
            workers = await workers_collection.find(
                {"disabled": {"$ne": True}}
            ).to_list(length=None)

            # Get recent activity data (placeholder - would need activity tracking)
            for worker in workers:
                worker_id = worker.get("workerID")
                worker_name = worker.get("name", f"Worker {worker_id}")
                role = worker.get("role", "Unknown")

                # Rule 1: Unusual login patterns (would need login tracking)
                # This is a placeholder - in real implementation, check login logs

                # Rule 2: Performance anomalies (placeholder)
                # This would analyze task completion times, error rates, etc.

                # For now, we'll add some sample anomalies based on worker data
                if worker.get("performance_score", 100) < 70:
                    anomalies.append(
                        {
                            "type": "low_performance",
                            "severity": "medium",
                            "worker_id": worker_id,
                            "worker_name": worker_name,
                            "role": role,
                            "description": "Performance score below threshold",
                            "performance_score": worker.get("performance_score", 0),
                            "threshold": 70,
                            "technique": "rule_based",
                            "rule": "low_performance",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            logger.info(f"👷 Found {len(anomalies)} worker rule-based anomalies")
            return anomalies

        except Exception as e:
            logger.error(f"❌ Error in worker rule detection: {str(e)}")
            return []

    async def detect_workflow_rule_anomalies(self) -> List[Dict[str, Any]]:
        """🔄 Rule-based workflow anomaly detection"""
        anomalies = []

        try:
            db = await self._get_database()
            orders_collection = db["orders"]

            # Rule 1: Stuck orders
            stuck_threshold_hours = self.rule_thresholds["workflow"]["stuck_hours"]
            cutoff_time = datetime.now() - timedelta(hours=stuck_threshold_hours)

            stuck_orders = await orders_collection.find(
                {
                    "order_status": {"$in": ["processing", "picking", "packing"]},
                    "updated_at": {"$lt": cutoff_time},
                }
            ).to_list(length=None)

            for order in stuck_orders:
                anomalies.append(
                    {
                        "type": "stuck_workflow",
                        "severity": "high",
                        "order_id": order.get("orderID"),
                        "description": f"Order stuck in {order.get('order_status')} for over {stuck_threshold_hours} hours",
                        "status": order.get("order_status"),
                        "last_updated": order.get(
                            "updated_at", datetime.now()
                        ).isoformat(),
                        "hours_stuck": (
                            datetime.now() - order.get("updated_at", datetime.now())
                        ).total_seconds()
                        / 3600,
                        "technique": "rule_based",
                        "rule": "stuck_workflow",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # Rule 2: Bottleneck detection
            status_counts = await self._count_orders_by_status()
            bottleneck_threshold = self.rule_thresholds["workflow"][
                "bottleneck_threshold"
            ]

            for status, count in status_counts.items():
                if count > bottleneck_threshold:
                    anomalies.append(
                        {
                            "type": "workflow_bottleneck",
                            "severity": "medium",
                            "description": f"Bottleneck detected in {status} stage",
                            "status": status,
                            "order_count": count,
                            "threshold": bottleneck_threshold,
                            "technique": "rule_based",
                            "rule": "bottleneck_detection",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            logger.info(f"🔄 Found {len(anomalies)} workflow rule-based anomalies")
            return anomalies

        except Exception as e:
            logger.error(f"❌ Error in workflow rule detection: {str(e)}")
            return []

    # =============================================================================
    # ML-BASED ANOMALY DETECTION (ISOLATION FOREST)
    # =============================================================================

    async def detect_inventory_ml_anomalies(self) -> List[Dict[str, Any]]:
        """🤖 ML-based inventory anomaly detection using Isolation Forest"""
        anomalies = []

        try:
            # Get inventory data
            db = await self._get_database()
            inventory_collection = db["inventory"]
            items = await inventory_collection.find({}).to_list(length=None)

            if len(items) < 10:  # Need sufficient data for ML
                logger.warning("Insufficient inventory data for ML anomaly detection")
                return []

            # Prepare features
            features = []
            item_metadata = []

            for item in items:
                # Feature engineering
                current_stock = item.get(
                    "stock_level", 0
                )  # Fixed: was "stock_quantity"
                min_stock = item.get("min_stock_level", 1)
                max_stock = item.get("max_stock_level", 100)
                price = item.get("price", 0)

                # Calculate derived features
                stock_ratio = current_stock / max(min_stock, 1)
                stock_range_position = (current_stock - min_stock) / max(
                    max_stock - min_stock, 1
                )
                value_at_risk = current_stock * price

                feature_vector = [
                    current_stock,
                    stock_ratio,
                    stock_range_position,
                    value_at_risk,
                    price,
                    min_stock,
                    max_stock,
                ]

                features.append(feature_vector)
                item_metadata.append(
                    {
                        "item_id": item.get("itemID"),
                        "item_name": item.get("name"),  # Fixed: was "item_name"
                        "current_stock": current_stock,
                    }
                )

            # Convert to numpy array and scale
            X = np.array(features)
            X_scaled = self.scalers["inventory"].fit_transform(X)

            # Train and predict
            self.models["inventory"].fit(X_scaled)
            anomaly_scores = self.models["inventory"].decision_function(X_scaled)
            predictions = self.models["inventory"].predict(X_scaled)

            # Extract anomalies
            for i, (prediction, score) in enumerate(zip(predictions, anomaly_scores)):
                if prediction == -1:  # Anomaly detected
                    severity = "high" if score < -0.5 else "medium"

                    anomalies.append(
                        {
                            "type": "ml_inventory_anomaly",
                            "severity": severity,
                            "item_id": item_metadata[i]["item_id"],
                            "item_name": item_metadata[i]["item_name"],
                            "description": f"ML anomaly detected (score: {score:.3f})",
                            "anomaly_score": float(score),
                            "current_stock": item_metadata[i]["current_stock"],
                            "technique": "isolation_forest",
                            "confidence": abs(float(score)),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            logger.info(f"🤖 Found {len(anomalies)} inventory ML anomalies")
            return anomalies

        except Exception as e:
            logger.error(f"❌ Error in inventory ML detection: {str(e)}")
            return []

    async def detect_order_ml_anomalies(self) -> List[Dict[str, Any]]:
        """🤖 ML-based order anomaly detection using Isolation Forest"""
        anomalies = []

        try:
            db = await self._get_database()
            orders_collection = db["orders"]
            cutoff_date = datetime.now() - timedelta(days=30)
            orders = await orders_collection.find(
                {"order_date": {"$gte": cutoff_date}}
            ).to_list(length=None)

            if len(orders) < 10:
                logger.warning("Insufficient order data for ML anomaly detection")
                return []

            # Prepare features
            features = []
            order_metadata = []

            for order in orders:
                order_date = order.get("order_date", datetime.now())
                hour = order_date.hour if isinstance(order_date, datetime) else 12
                day_of_week = (
                    order_date.weekday() if isinstance(order_date, datetime) else 1
                )

                items = order.get("items", [])
                total_items = sum(item.get("quantity", 0) for item in items)
                unique_items = len(items)
                total_amount = order.get("total_amount", 0)
                avg_item_price = total_amount / max(total_items, 1)

                feature_vector = [
                    hour,
                    day_of_week,
                    total_items,
                    unique_items,
                    total_amount,
                    avg_item_price,
                    len(
                        str(order.get("shipping_address", ""))
                    ),  # Address length as complexity measure
                ]

                features.append(feature_vector)
                order_metadata.append(
                    {
                        "order_id": order.get("orderID"),
                        "total_amount": total_amount,
                        "total_items": total_items,
                    }
                )

            # ML processing
            X = np.array(features)
            X_scaled = self.scalers["orders"].fit_transform(X)

            self.models["orders"].fit(X_scaled)
            anomaly_scores = self.models["orders"].decision_function(X_scaled)
            predictions = self.models["orders"].predict(X_scaled)

            # Extract anomalies
            for i, (prediction, score) in enumerate(zip(predictions, anomaly_scores)):
                if prediction == -1:
                    severity = "high" if score < -0.5 else "medium"

                    anomalies.append(
                        {
                            "type": "ml_order_anomaly",
                            "severity": severity,
                            "order_id": order_metadata[i]["order_id"],
                            "description": f"ML order pattern anomaly detected (score: {score:.3f})",
                            "anomaly_score": float(score),
                            "total_amount": order_metadata[i]["total_amount"],
                            "total_items": order_metadata[i]["total_items"],
                            "technique": "isolation_forest",
                            "confidence": abs(float(score)),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            logger.info(f"🤖 Found {len(anomalies)} order ML anomalies")
            return anomalies

        except Exception as e:
            logger.error(f"❌ Error in order ML detection: {str(e)}")
            return []

    async def detect_worker_ml_anomalies(self) -> List[Dict[str, Any]]:
        """🤖 ML-based worker anomaly detection"""
        # Placeholder for worker ML detection
        # Would need historical worker performance data
        logger.info("🤖 Worker ML anomaly detection - placeholder")
        return []

    async def detect_workflow_ml_anomalies(self) -> List[Dict[str, Any]]:
        """🤖 ML-based workflow anomaly detection"""
        # Placeholder for workflow ML detection
        # Would analyze workflow timing patterns
        logger.info("🤖 Workflow ML anomaly detection - placeholder")
        return []

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    async def _detect_dead_stock(self) -> List[Dict[str, Any]]:
        """Detect items with no movement for extended periods"""
        anomalies = []

        try:
            # This would require movement/transaction history
            # Placeholder implementation
            db = await self._get_database()
            inventory_collection = db["inventory"]
            cutoff_date = datetime.now() - timedelta(
                days=self.rule_thresholds["inventory"]["dead_stock_days"]
            )

            # In real implementation, join with transaction/movement history
            items = await inventory_collection.find(
                {"last_movement": {"$lt": cutoff_date}}
            ).to_list(length=None)

            for item in items:
                if item.get("stock_level", 0) > 0:
                    anomalies.append(
                        {
                            "type": "dead_stock",
                            "severity": "medium",
                            "item_id": item.get("itemID"),
                            "item_name": item.get("item_name"),
                            "description": f"No movement for {self.rule_thresholds['inventory']['dead_stock_days']} days",
                            "days_since_movement": (
                                datetime.now()
                                - item.get("last_movement", datetime.now())
                            ).days,
                            "current_stock": item.get("stock_level", 0),
                            "technique": "rule_based",
                            "rule": "dead_stock",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            return anomalies

        except Exception as e:
            logger.error(f"❌ Error detecting dead stock: {str(e)}")
            return []

    async def _detect_processing_delays(self) -> List[Dict[str, Any]]:
        """Detect orders with unusual processing delays"""
        anomalies = []

        try:
            db = await self._get_database()
            orders_collection = db["orders"]
            delay_hours = self.rule_thresholds["orders"]["processing_delay_hours"]
            cutoff_time = datetime.now() - timedelta(hours=delay_hours)

            delayed_orders = await orders_collection.find(
                {
                    "order_status": {"$in": ["processing", "picking", "packing"]},
                    "order_date": {"$lt": cutoff_time},
                }
            ).to_list(length=None)

            for order in delayed_orders:
                delay_duration = datetime.now() - order.get(
                    "order_date", datetime.now()
                )

                anomalies.append(
                    {
                        "type": "processing_delay",
                        "severity": "high"
                        if delay_duration.total_seconds() > delay_hours * 3600 * 2
                        else "medium",
                        "order_id": order.get("orderID"),
                        "description": f"Order processing delayed by {delay_duration.days} days",
                        "status": order.get("order_status"),
                        "delay_hours": delay_duration.total_seconds() / 3600,
                        "technique": "rule_based",
                        "rule": "processing_delay",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            return anomalies

        except Exception as e:
            logger.error(f"❌ Error detecting processing delays: {str(e)}")
            return []

    async def _count_orders_by_status(self) -> Dict[str, int]:
        """Count orders by status for bottleneck detection"""
        try:
            db = await self._get_database()
            orders_collection = db["orders"]
            pipeline = [{"$group": {"_id": "$order_status", "count": {"$sum": 1}}}]

            result = await orders_collection.aggregate(pipeline).to_list(length=None)
            return {item["_id"]: item["count"] for item in result}

        except Exception as e:
            logger.error(f"❌ Error counting orders by status: {str(e)}")
            return {}

    def _combine_anomaly_results(
        self, rule_anomalies: Dict, ml_anomalies: Dict
    ) -> Dict[str, List[Dict]]:
        """Combine rule-based and ML anomaly results"""
        combined = {}

        for category in rule_anomalies.keys():
            combined[category] = []

            # Add rule-based anomalies
            combined[category].extend(rule_anomalies.get(category, []))

            # Add ML anomalies
            combined[category].extend(ml_anomalies.get(category, []))

            # Sort by severity
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            combined[category].sort(
                key=lambda x: severity_order.get(x.get("severity", "low"), 3)
            )

        return combined

    def _generate_anomaly_summary(self, anomalies: Dict) -> Dict[str, Any]:
        """Generate summary statistics for detected anomalies"""
        total_anomalies = sum(
            len(category_anomalies) for category_anomalies in anomalies.values()
        )

        severity_counts = defaultdict(int)
        technique_counts = defaultdict(int)
        category_counts = {}

        for category, category_anomalies in anomalies.items():
            category_counts[category] = len(category_anomalies)

            for anomaly in category_anomalies:
                severity_counts[anomaly.get("severity", "unknown")] += 1
                technique_counts[anomaly.get("technique", "unknown")] += 1

        # Determine overall system health
        critical_count = severity_counts.get("critical", 0)
        high_count = severity_counts.get("high", 0)

        if critical_count > 0:
            health_status = "critical"
        elif high_count > 5:
            health_status = "warning"
        elif total_anomalies > 10:
            health_status = "attention_needed"
        else:
            health_status = "healthy"

        return {
            "total_anomalies": total_anomalies,
            "severity_breakdown": dict(severity_counts),
            "technique_breakdown": dict(technique_counts),
            "category_breakdown": category_counts,
            "health_status": health_status,
            "recommendations": self._generate_recommendations(anomalies),
            "detection_timestamp": datetime.now().isoformat(),
        }

    def _generate_recommendations(self, anomalies: Dict) -> List[str]:
        """Generate actionable recommendations based on detected anomalies"""
        recommendations = []

        # Analyze inventory anomalies
        inventory_anomalies = anomalies.get("inventory", [])
        critical_stockouts = len(
            [a for a in inventory_anomalies if a.get("type") == "critical_stockout"]
        )

        if critical_stockouts > 0:
            recommendations.append(
                f"🚨 Immediate action required: {critical_stockouts} critical items out of stock"
            )

        # Analyze order anomalies
        order_anomalies = anomalies.get("orders", [])
        high_value_orders = len(
            [a for a in order_anomalies if a.get("type") == "high_value_order"]
        )

        if high_value_orders > 2:
            recommendations.append(
                f"💰 Review {high_value_orders} high-value orders for potential fraud"
            )

        # Analyze workflow anomalies
        workflow_anomalies = anomalies.get("workflow", [])
        stuck_orders = len(
            [a for a in workflow_anomalies if a.get("type") == "stuck_workflow"]
        )

        if stuck_orders > 0:
            recommendations.append(
                f"⏰ Investigate {stuck_orders} stuck orders in workflow"
            )

        if not recommendations:
            recommendations.append(
                "✅ No critical issues detected - system operating normally"
            )

        return recommendations

    async def save_models(self):
        """Save trained models to disk"""
        try:
            for model_type, model in self.models.items():
                model_path = os.path.join(self.model_dir, f"{model_type}_model.joblib")
                scaler_path = os.path.join(
                    self.model_dir, f"{model_type}_scaler.joblib"
                )

                joblib.dump(model, model_path)
                joblib.dump(self.scalers[model_type], scaler_path)

            logger.info("✅ Models saved successfully")

        except Exception as e:
            logger.error(f"❌ Error saving models: {str(e)}")

    async def load_models(self):
        """Load trained models from disk"""
        try:
            for model_type in self.models.keys():
                model_path = os.path.join(self.model_dir, f"{model_type}_model.joblib")
                scaler_path = os.path.join(
                    self.model_dir, f"{model_type}_scaler.joblib"
                )

                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[model_type] = joblib.load(model_path)
                    self.scalers[model_type] = joblib.load(scaler_path)

            logger.info("✅ Models loaded successfully")

        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")


# Global instance
advanced_anomaly_detection_service = AdvancedAnomalyDetectionService()
