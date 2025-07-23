#!/usr/bin/env python3
"""
Anomaly Dataset Insertion Script for Warehouse Management System

This script creates realistic anomaly datasets to test the anomaly detection system.
It inserts various types of anomalies:
1. Inventory anomalies (stockouts, overstocking, impossible quantities)
2. Order anomalies (unusual timing, high values, bulk patterns)
3. Worker behavior anomalies (performance issues, unusual patterns)
4. Workflow anomalies (stuck orders, bottlenecks, delays)

The script creates both obvious rule-based anomalies and subtle statistical outliers
for ML-based detection testing.
"""

import os
import random
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Get MongoDB connection URL
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "warehouse_management")


def insert_anomaly_datasets():
    """Insert comprehensive anomaly datasets for testing."""
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URL)
        db = client[DATABASE_NAME]

        print("🚨 Starting anomaly dataset insertion...")
        print(f"Connected to database: {DATABASE_NAME}")

        # Insert different types of anomalies
        insert_inventory_anomalies(db)
        insert_order_anomalies(db)
        insert_worker_anomalies(db)
        insert_workflow_anomalies(db)
        insert_statistical_outliers(db)
        insert_historical_data_for_ml(db)

        print("\n✅ Anomaly datasets inserted successfully!")
        print_anomaly_summary(db)

    except Exception as e:
        print(f"❌ Error inserting anomaly datasets: {e}")
        sys.exit(1)


def insert_inventory_anomalies(db):
    """Insert inventory-related anomalies."""
    print("\n📦 Inserting inventory anomalies...")

    # 1. Critical stockout items (Rule-based anomaly)
    stockout_items = [
        {
            "itemID": 101,
            "name": "Critical Component A",
            "category": "Electronics",
            "size": "S",
            "storage_type": "standard",
            "stock_level": 0,  # CRITICAL: Zero stock
            "min_stock_level": 50,
            "max_stock_level": 200,
            "supplierID": 1,
            "locationID": 50,
            "created_at": datetime.utcnow() - timedelta(days=2),
            "updated_at": datetime.utcnow() - timedelta(hours=1),
        },
        {
            "itemID": 102,
            "name": "High Demand Widget",
            "category": "Electronics",
            "size": "M",
            "storage_type": "standard",
            "stock_level": 2,  # CRITICAL: Below minimum
            "min_stock_level": 25,
            "max_stock_level": 150,
            "supplierID": 1,
            "locationID": 51,
            "created_at": datetime.utcnow() - timedelta(days=3),
            "updated_at": datetime.utcnow() - timedelta(minutes=30),
        },
    ]

    # 2. Extreme overstocking (Rule-based anomaly)
    overstock_items = [
        {
            "itemID": 103,
            "name": "Seasonal Item Overstock",
            "category": "Clothing",
            "size": "L",
            "storage_type": "standard",
            "stock_level": 2500,  # EXTREME: Way above maximum
            "min_stock_level": 20,
            "max_stock_level": 100,
            "supplierID": 2,
            "locationID": 52,
            "created_at": datetime.utcnow() - timedelta(days=10),
            "updated_at": datetime.utcnow() - timedelta(days=5),
        },
        {
            "itemID": 104,
            "name": "Bulk Purchase Error",
            "category": "Food",
            "size": "XL",
            "storage_type": "refrigerated",
            "stock_level": 1200,  # HIGH: Above maximum
            "min_stock_level": 15,
            "max_stock_level": 80,
            "supplierID": 3,
            "locationID": 53,
            "created_at": datetime.utcnow() - timedelta(days=7),
            "updated_at": datetime.utcnow() - timedelta(days=2),
        },
    ]

    # 3. Impossible/Negative quantities (Rule-based anomaly)
    impossible_items = [
        {
            "itemID": 105,
            "name": "Data Error Item",
            "category": "Electronics",
            "size": "S",
            "storage_type": "standard",
            "stock_level": -15,  # IMPOSSIBLE: Negative stock
            "min_stock_level": 10,
            "max_stock_level": 50,
            "supplierID": 1,
            "locationID": 54,
            "created_at": datetime.utcnow() - timedelta(hours=6),
            "updated_at": datetime.utcnow() - timedelta(hours=2),
        }
    ]

    # 4. Dead stock items (No movement for extended periods)
    dead_stock_items = [
        {
            "itemID": 106,
            "name": "Obsolete Product",
            "category": "Electronics",
            "size": "L",
            "storage_type": "standard",
            "stock_level": 75,
            "min_stock_level": 5,
            "max_stock_level": 100,
            "supplierID": 1,
            "locationID": 55,
            "created_at": datetime.utcnow() - timedelta(days=180),  # Old item
            "updated_at": datetime.utcnow() - timedelta(days=120),  # No recent updates
        },
        {
            "itemID": 107,
            "name": "Discontinued Model",
            "category": "Clothing",
            "size": "M",
            "storage_type": "standard",
            "stock_level": 250,
            "min_stock_level": 10,
            "max_stock_level": 300,
            "supplierID": 2,
            "locationID": 56,
            "created_at": datetime.utcnow() - timedelta(days=200),
            "updated_at": datetime.utcnow() - timedelta(days=150),
        },
    ]

    # Insert all inventory anomalies
    all_inventory_anomalies = (
        stockout_items + overstock_items + impossible_items + dead_stock_items
    )

    # Update location occupancy
    location_ids = [
        item["locationID"]
        for item in all_inventory_anomalies
        if item["stock_level"] > 0
    ]
    db.locations.update_many(
        {"locationID": {"$in": location_ids}},
        {"$set": {"is_occupied": True, "updated_at": datetime.utcnow()}},
    )

    db.inventory.insert_many(all_inventory_anomalies)

    # Insert corresponding stock log entries showing the anomalous changes
    insert_anomalous_stock_logs(db, all_inventory_anomalies)

    print(f"   ✓ Inserted {len(all_inventory_anomalies)} inventory anomaly items")


def insert_anomalous_stock_logs(db, items):
    """Insert stock log entries that show anomalous inventory changes."""
    stock_logs = []

    for item in items:
        item_id = item["itemID"]
        current_level = item["stock_level"]

        if item_id in [101, 102]:  # Stockout items
            # Show rapid depletion
            logs = [
                {
                    "itemID": item_id,
                    "previous_level": 50,
                    "new_level": 25,
                    "change": -25,
                    "reason": "Large order fulfillment",
                    "timestamp": datetime.utcnow() - timedelta(days=2),
                },
                {
                    "itemID": item_id,
                    "previous_level": 25,
                    "new_level": 5,
                    "change": -20,
                    "reason": "Emergency order",
                    "timestamp": datetime.utcnow() - timedelta(days=1),
                },
                {
                    "itemID": item_id,
                    "previous_level": 5,
                    "new_level": current_level,
                    "change": current_level - 5,
                    "reason": "Final depletion",
                    "timestamp": datetime.utcnow() - timedelta(hours=2),
                },
            ]
            stock_logs.extend(logs)

        elif item_id in [103, 104]:  # Overstock items
            # Show sudden massive increase
            logs = [
                {
                    "itemID": item_id,
                    "previous_level": 50,
                    "new_level": current_level,
                    "change": current_level - 50,
                    "reason": "Bulk purchase - possible error",
                    "timestamp": datetime.utcnow() - timedelta(days=5),
                }
            ]
            stock_logs.extend(logs)

        elif item_id == 105:  # Impossible quantity
            # Show the data corruption
            logs = [
                {
                    "itemID": item_id,
                    "previous_level": 20,
                    "new_level": current_level,
                    "change": current_level - 20,
                    "reason": "System error - negative stock",
                    "timestamp": datetime.utcnow() - timedelta(hours=3),
                }
            ]
            stock_logs.extend(logs)

    if stock_logs:
        db.stock_log.insert_many(stock_logs)
        print(f"   ✓ Inserted {len(stock_logs)} anomalous stock log entries")


def insert_order_anomalies(db):
    """Insert order-related anomalies."""
    print("\n🛒 Inserting order anomalies...")

    # 1. Extremely high-value orders (Rule-based anomaly)
    high_value_orders = [
        {
            "orderID": 201,
            "customerID": 1,
            "order_date": datetime.utcnow() - timedelta(hours=2),
            "shipping_address": "789 Retail St, Commerce City",
            "order_status": "pending",
            "priority": 1,
            "notes": "Unusually high-value order - requires verification",
            "items": [
                {
                    "orderDetailID": 1,
                    "itemID": 2,  # Laptop
                    "quantity": 100,  # EXTREME quantity
                    "price": 899.99,
                    "fulfilled_quantity": 0,
                    "created_at": datetime.utcnow() - timedelta(hours=2),
                    "updated_at": datetime.utcnow() - timedelta(hours=2),
                }
            ],
            "total_amount": 89999.00,  # EXTREME value
            "assigned_worker": None,
            "created_at": datetime.utcnow() - timedelta(hours=2),
            "updated_at": datetime.utcnow() - timedelta(hours=2),
        },
        {
            "orderID": 202,
            "customerID": 2,
            "order_date": datetime.utcnow() - timedelta(hours=1),
            "shipping_address": "101 E-commerce Ave, Digital Town",
            "order_status": "pending",
            "priority": 1,
            "notes": "Suspicious bulk order",
            "items": [
                {
                    "orderDetailID": 1,
                    "itemID": 1,  # Smartphone
                    "quantity": 200,
                    "price": 499.99,
                    "fulfilled_quantity": 0,
                    "created_at": datetime.utcnow() - timedelta(hours=1),
                    "updated_at": datetime.utcnow() - timedelta(hours=1),
                }
            ],
            "total_amount": 99998.00,
            "assigned_worker": None,
            "created_at": datetime.utcnow() - timedelta(hours=1),
            "updated_at": datetime.utcnow() - timedelta(hours=1),
        },
    ]

    # 2. Unusual timing orders (Rule-based anomaly)
    unusual_timing_orders = [
        {
            "orderID": 203,
            "customerID": 3,
            "order_date": datetime.utcnow().replace(hour=3, minute=30),  # 3:30 AM order
            "shipping_address": "202 Distribution Blvd, Logistics City",
            "order_status": "pending",
            "priority": 2,
            "notes": "Order placed at unusual hour",
            "items": [
                {
                    "orderDetailID": 1,
                    "itemID": 3,
                    "quantity": 50,
                    "price": 24.99,
                    "fulfilled_quantity": 0,
                    "created_at": datetime.utcnow().replace(hour=3, minute=30),
                    "updated_at": datetime.utcnow().replace(hour=3, minute=30),
                }
            ],
            "total_amount": 1249.50,
            "assigned_worker": None,
            "created_at": datetime.utcnow().replace(hour=3, minute=30),
            "updated_at": datetime.utcnow().replace(hour=3, minute=30),
        }
    ]

    # 3. Orders stuck in processing (Workflow anomaly)
    stuck_orders = [
        {
            "orderID": 204,
            "customerID": 1,
            "order_date": datetime.utcnow() - timedelta(days=7),  # Very old order
            "shipping_address": "789 Retail St, Commerce City",
            "order_status": "processing",  # Stuck in processing
            "priority": 2,
            "notes": "Order stuck in processing for too long",
            "items": [
                {
                    "orderDetailID": 1,
                    "itemID": 4,
                    "quantity": 10,
                    "price": 49.99,
                    "fulfilled_quantity": 0,
                    "created_at": datetime.utcnow() - timedelta(days=7),
                    "updated_at": datetime.utcnow()
                    - timedelta(days=6),  # No recent updates
                }
            ],
            "total_amount": 499.90,
            "assigned_worker": 3,
            "created_at": datetime.utcnow() - timedelta(days=7),
            "updated_at": datetime.utcnow() - timedelta(days=6),
        },
        {
            "orderID": 205,
            "customerID": 2,
            "order_date": datetime.utcnow() - timedelta(days=5),
            "shipping_address": "101 E-commerce Ave, Digital Town",
            "order_status": "picking",  # Stuck in picking
            "priority": 1,
            "notes": "High priority order delayed in picking",
            "items": [
                {
                    "orderDetailID": 1,
                    "itemID": 5,
                    "quantity": 25,
                    "price": 6.99,
                    "fulfilled_quantity": 0,
                    "created_at": datetime.utcnow() - timedelta(days=5),
                    "updated_at": datetime.utcnow() - timedelta(days=4),
                }
            ],
            "total_amount": 174.75,
            "assigned_worker": 3,
            "created_at": datetime.utcnow() - timedelta(days=5),
            "updated_at": datetime.utcnow() - timedelta(days=4),
        },
    ]

    # Insert all order anomalies
    all_order_anomalies = high_value_orders + unusual_timing_orders + stuck_orders
    db.orders.insert_many(all_order_anomalies)

    print(f"   ✓ Inserted {len(all_order_anomalies)} anomalous orders")


def insert_worker_anomalies(db):
    """Insert worker behavior anomalies."""
    print("\n👷 Inserting worker anomalies...")

    # Add worker with performance issues
    problem_workers = [
        {
            "workerID": 201,
            "name": "Slow Pete",
            "role": "Picker",
            "email": "slowpete@warehouse.com",
            "phone": "+1234567895",
            "username": "slowpete",
            "hashed_password": "$2b$12$dummy_hash_for_test",
            "disabled": False,
            "performance_metrics": {
                "avg_picks_per_hour": 15,  # POOR: Well below normal (30-40)
                "error_rate": 0.15,  # HIGH: Above normal (0.05)
                "attendance_rate": 0.75,  # POOR: Below acceptable (0.95)
                "last_performance_review": datetime.utcnow() - timedelta(days=30),
            },
            "created_at": datetime.utcnow() - timedelta(days=90),
            "updated_at": datetime.utcnow() - timedelta(days=1),
        },
        {
            "workerID": 202,
            "name": "Erratic Eddie",
            "role": "Packer",
            "email": "eddie@warehouse.com",
            "phone": "+1234567896",
            "username": "eddie",
            "hashed_password": "$2b$12$dummy_hash_for_test",
            "disabled": False,
            "performance_metrics": {
                "avg_packs_per_hour": 8,  # POOR: Below normal (15-20)
                "error_rate": 0.25,  # VERY HIGH: Way above normal
                "attendance_rate": 0.65,  # POOR: Very low
                "unusual_login_times": [  # Unusual patterns
                    datetime.utcnow().replace(hour=2, minute=15),
                    datetime.utcnow().replace(hour=23, minute=45),
                    datetime.utcnow().replace(hour=1, minute=30),
                ],
                "last_performance_review": datetime.utcnow() - timedelta(days=45),
            },
            "created_at": datetime.utcnow() - timedelta(days=120),
            "updated_at": datetime.utcnow() - timedelta(hours=12),
        },
    ]

    db.workers.insert_many(problem_workers)

    # Create picking records showing poor performance
    poor_performance_picks = [
        {
            "pickingID": 301,
            "orderID": 204,  # Reference stuck order
            "workerID": 201,  # Slow Pete
            "pick_date": datetime.utcnow() - timedelta(days=6),
            "status": "in_progress",  # Still not completed
            "priority": 2,
            "items": [
                {
                    "itemID": 4,
                    "orderDetailID": 1,
                    "locationID": 4,
                    "quantity": 10,
                    "picked": False,
                    "actual_quantity": None,
                    "pick_time": None,
                    "notes": "Worker struggling to locate items",
                }
            ],
            "notes": "Unusually slow picking progress",
            "start_time": datetime.utcnow() - timedelta(days=6, hours=8),
            "complete_time": None,  # Not completed
            "created_at": datetime.utcnow() - timedelta(days=6, hours=8),
            "updated_at": datetime.utcnow() - timedelta(days=6, hours=4),
        }
    ]

    db.picking.insert_many(poor_performance_picks)

    print(f"   ✓ Inserted {len(problem_workers)} workers with anomalous behavior")
    print(f"   ✓ Inserted {len(poor_performance_picks)} anomalous picking records")


def insert_workflow_anomalies(db):
    """Insert workflow and process anomalies."""
    print("\n🔄 Inserting workflow anomalies...")

    # Create anomalous receiving records (delayed processing)
    delayed_receiving = [
        {
            "receivingID": 301,
            "supplierID": 1,
            "workerID": 2,
            "received_date": datetime.utcnow()
            - timedelta(days=10),  # Received long ago
            "status": "pending",  # Still not processed
            "reference_number": "PO-DELAYED-001",
            "items": [
                {
                    "itemID": 101,  # Critical component
                    "quantity": 100,
                    "expected_quantity": 100,
                    "condition": "good",
                    "processed": False,  # Not processed for 10 days!
                    "locationID": None,
                    "notes": "Delayed processing - urgent items sitting in receiving",
                }
            ],
            "notes": "CRITICAL: Urgent items not processed for over a week",
            "created_at": datetime.utcnow() - timedelta(days=10),
            "updated_at": datetime.utcnow() - timedelta(days=9),
        }
    ]

    # Create bottlenecked packing records
    bottleneck_packing = [
        {
            "packingID": 301,
            "orderID": 205,  # Reference stuck order
            "workerID": 202,  # Erratic Eddie
            "pack_date": datetime.utcnow() - timedelta(days=4),
            "status": "in_progress",  # Stuck for days
            "is_partial": True,
            "package_type": "box",
            "items": [
                {
                    "itemID": 5,
                    "pickingID": None,
                    "orderDetailID": 1,
                    "quantity": 25,
                    "packed": False,
                    "actual_quantity": 0,
                    "pack_time": None,
                    "notes": "Packing stalled - no progress for days",
                }
            ],
            "notes": "BOTTLENECK: Packing stuck, blocking other orders",
            "start_time": datetime.utcnow() - timedelta(days=4, hours=6),
            "complete_time": None,
            "weight": None,
            "dimensions": None,
            "label_printed": False,
            "created_at": datetime.utcnow() - timedelta(days=4, hours=6),
            "updated_at": datetime.utcnow() - timedelta(days=4),
        }
    ]

    db.receiving.insert_many(delayed_receiving)
    db.packing.insert_many(bottleneck_packing)

    print(f"   ✓ Inserted {len(delayed_receiving)} delayed receiving records")
    print(f"   ✓ Inserted {len(bottleneck_packing)} bottlenecked packing records")


def insert_statistical_outliers(db):
    """Insert subtle statistical outliers for ML detection."""
    print("\n📊 Inserting statistical outliers for ML detection...")

    # Create inventory items with subtle anomalies (for ML detection)
    ml_outliers = []

    # Generate items with unusual stock patterns
    base_time = datetime.utcnow()

    for i in range(20):  # Create 20 items with various subtle anomalies
        item_id = 300 + i

        # Different types of subtle anomalies
        if i < 5:  # Stock volatility outliers
            stock_level = random.randint(80, 120)
            # These items will have high volatility in stock movements
            anomaly_type = "volatility"
        elif i < 10:  # Seasonal pattern outliers
            stock_level = random.randint(40, 80)
            # These items break seasonal patterns
            anomaly_type = "seasonal"
        elif i < 15:  # Demand pattern outliers
            stock_level = random.randint(60, 100)
            # These items have unusual demand patterns
            anomaly_type = "demand"
        else:  # Combo outliers
            stock_level = random.randint(30, 150)
            anomaly_type = "combo"

        item = {
            "itemID": item_id,
            "name": f"ML Test Item {i + 1}",
            "category": random.choice(["Electronics", "Clothing", "Food"]),
            "size": random.choice(["S", "M", "L", "XL"]),
            "storage_type": "standard",
            "stock_level": stock_level,
            "min_stock_level": random.randint(10, 30),
            "max_stock_level": random.randint(100, 200),
            "supplierID": random.choice([1, 2, 3]),
            "locationID": 100 + i,
            "anomaly_metadata": {
                "type": anomaly_type,
                "ml_features": {
                    "avg_daily_movement": random.uniform(5.0, 25.0),
                    "movement_variance": random.uniform(2.0, 15.0),
                    "days_since_last_order": random.randint(1, 30),
                    "supplier_reliability": random.uniform(0.7, 1.0),
                    "seasonal_factor": random.uniform(0.5, 2.0),
                },
            },
            "created_at": base_time - timedelta(days=random.randint(30, 180)),
            "updated_at": base_time - timedelta(hours=random.randint(1, 48)),
        }
        ml_outliers.append(item)

    # Update locations
    location_ids = [item["locationID"] for item in ml_outliers]
    db.locations.update_many(
        {"locationID": {"$in": location_ids}},
        {"$set": {"is_occupied": True, "updated_at": datetime.utcnow()}},
    )

    db.inventory.insert_many(ml_outliers)

    # Create corresponding stock movements with anomalous patterns
    insert_anomalous_ml_stock_logs(db, ml_outliers)

    print(f"   ✓ Inserted {len(ml_outliers)} items with ML-detectable anomalies")


def insert_anomalous_ml_stock_logs(db, items):
    """Create stock movement patterns that are subtle anomalies for ML detection."""
    stock_logs = []

    for item in items:
        item_id = item["itemID"]
        anomaly_type = item["anomaly_metadata"]["type"]
        current_level = item["stock_level"]

        # Generate different movement patterns based on anomaly type
        if anomaly_type == "volatility":
            # High volatility movements
            movements = [
                (50, current_level + 30, 30, "Large unexpected inbound"),
                (current_level + 30, current_level - 20, -50, "Sudden large outbound"),
                (current_level - 20, current_level + 15, 35, "Irregular replenishment"),
                (current_level + 15, current_level, -15, "Random adjustment"),
            ]
        elif anomaly_type == "seasonal":
            # Counter-seasonal movements
            movements = [
                (30, 45, 15, "Off-season increase"),
                (45, 80, 35, "Unexpected seasonal demand"),
                (80, current_level, current_level - 80, "Irregular seasonal decline"),
            ]
        elif anomaly_type == "demand":
            # Unusual demand patterns
            movements = [
                (40, 25, -15, "Unusual demand drop"),
                (25, 70, 45, "Spike in demand"),
                (70, current_level, current_level - 70, "Demand pattern break"),
            ]
        else:  # combo
            # Mixed anomalous patterns
            movements = [
                (35, 90, 55, "Complex pattern 1"),
                (90, 20, -70, "Complex pattern 2"),
                (20, current_level, current_level - 20, "Complex pattern 3"),
            ]

        # Create the stock log entries
        base_time = datetime.utcnow()
        for i, (prev_level, new_level, change, reason) in enumerate(movements):
            log_entry = {
                "itemID": item_id,
                "previous_level": prev_level,
                "new_level": new_level,
                "change": change,
                "reason": reason,
                "timestamp": base_time - timedelta(days=len(movements) - i),
            }
            stock_logs.append(log_entry)

    if stock_logs:
        db.stock_log.insert_many(stock_logs)
        print(f"   ✓ Inserted {len(stock_logs)} anomalous stock movement logs")


def insert_historical_data_for_ml(db):
    """Insert historical data to help train ML models."""
    print("\n🤖 Inserting historical data for ML training...")

    # Generate normal historical patterns for comparison
    historical_items = []
    historical_logs = []
    base_time = datetime.utcnow()

    for i in range(50):  # Create 50 normal items for ML training
        item_id = 400 + i
        stock_level = random.randint(40, 120)

        item = {
            "itemID": item_id,
            "name": f"Normal Item {i + 1}",
            "category": random.choice(["Electronics", "Clothing", "Food"]),
            "size": random.choice(["S", "M", "L"]),
            "storage_type": "standard",
            "stock_level": stock_level,
            "min_stock_level": random.randint(15, 25),
            "max_stock_level": random.randint(80, 150),
            "supplierID": random.choice([1, 2, 3]),
            "locationID": 200 + i,
            "ml_features": {
                "avg_daily_movement": random.uniform(8.0, 15.0),
                "movement_variance": random.uniform(3.0, 8.0),
                "days_since_last_order": random.randint(3, 14),
                "supplier_reliability": random.uniform(0.85, 1.0),
                "seasonal_factor": random.uniform(0.8, 1.2),
            },
            "created_at": base_time - timedelta(days=random.randint(60, 200)),
            "updated_at": base_time - timedelta(hours=random.randint(1, 24)),
        }
        historical_items.append(item)

        # Generate normal stock movement patterns
        movements = generate_normal_stock_pattern(stock_level, item_id)
        historical_logs.extend(movements)

    # Update locations
    location_ids = [item["locationID"] for item in historical_items]
    db.locations.update_many(
        {"locationID": {"$in": location_ids}},
        {"$set": {"is_occupied": True, "updated_at": datetime.utcnow()}},
    )

    db.inventory.insert_many(historical_items)
    db.stock_log.insert_many(historical_logs)

    print(f"   ✓ Inserted {len(historical_items)} normal items for ML training")
    print(f"   ✓ Inserted {len(historical_logs)} normal stock movement logs")


def generate_normal_stock_pattern(final_level, item_id):
    """Generate normal, predictable stock movement patterns."""
    movements = []
    current_level = final_level
    base_time = datetime.utcnow()

    # Generate 10 normal movements over the past 30 days
    for i in range(10):
        # Normal movements are smaller and more predictable
        change = random.randint(-15, 20)
        previous_level = current_level - change

        reasons = [
            "Regular replenishment",
            "Normal customer order",
            "Scheduled delivery",
            "Standard inventory adjustment",
            "Routine stock rotation",
        ]

        movement = {
            "itemID": item_id,
            "previous_level": previous_level,
            "new_level": current_level,
            "change": change,
            "reason": random.choice(reasons),
            "timestamp": base_time - timedelta(days=30 - (i * 3)),
        }
        movements.append(movement)
        current_level = previous_level

    return movements


def print_anomaly_summary(db):
    """Print summary of inserted anomaly data."""
    print("\n📋 Anomaly Dataset Summary:")

    # Count different types of anomalies
    total_inventory = db.inventory.count_documents({"itemID": {"$gte": 101}})
    stockout_items = db.inventory.count_documents(
        {"stock_level": {"$lte": 5}, "min_stock_level": {"$gte": 10}}
    )
    overstock_items = db.inventory.count_documents(
        {
            "stock_level": {"$gt": 0},
            "$expr": {"$gt": ["$stock_level", {"$multiply": ["$max_stock_level", 2]}]},
        }
    )
    negative_stock = db.inventory.count_documents({"stock_level": {"$lt": 0}})

    high_value_orders = db.orders.count_documents({"total_amount": {"$gte": 50000}})
    stuck_orders = db.orders.count_documents(
        {
            "order_status": {"$in": ["processing", "picking", "packing"]},
            "created_at": {"$lt": datetime.utcnow() - timedelta(days=3)},
        }
    )

    problem_workers = db.workers.count_documents({"workerID": {"$gte": 201}})
    delayed_receiving = db.receiving.count_documents(
        {
            "status": "pending",
            "received_date": {"$lt": datetime.utcnow() - timedelta(days=5)},
        }
    )

    print("  📦 Inventory Anomalies:")
    print(f"     • Total anomaly items: {total_inventory}")
    print(f"     • Stockout/Low stock items: {stockout_items}")
    print(f"     • Overstock items: {overstock_items}")
    print(f"     • Negative stock items: {negative_stock}")

    print("  🛒 Order Anomalies:")
    print(f"     • High-value orders: {high_value_orders}")
    print(f"     • Stuck orders: {stuck_orders}")

    print("  👷 Worker Anomalies:")
    print(f"     • Problem workers: {problem_workers}")

    print("  🔄 Workflow Anomalies:")
    print(f"     • Delayed receiving: {delayed_receiving}")

    print("\n🎯 Ready for anomaly detection testing!")
    print("   • Rule-based detection will catch obvious anomalies")
    print("   • ML-based detection will find subtle statistical outliers")
    print("   • Combined analysis will provide comprehensive coverage")


if __name__ == "__main__":
    insert_anomaly_datasets()
