#!/usr/bin/env python3
"""
Demo Anomaly Detection Script
Creates sample data to test the anomaly detection system
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta

from app.services.advanced_anomaly_detection_service import (
    advanced_anomaly_detection_service,
)
from app.utils.database import get_async_database


async def create_demo_anomalies():
    """Create demo anomalies to test the system"""
    print("🔍 Creating demo anomalies for testing...")

    try:
        # Test the detection service
        result = await advanced_anomaly_detection_service.detect_all_anomalies(
            include_ml=False
        )

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        print("✅ Anomaly detection completed successfully!")
        print(f"📊 Summary: {result.get('summary', {})}")

        # Print detected anomalies
        rule_based = result.get("rule_based", {})
        for category, anomalies in rule_based.items():
            if anomalies:
                print(f"\n📦 {category.upper()} ANOMALIES:")
                for anomaly in anomalies[:3]:  # Show first 3
                    print(
                        f"  • {anomaly.get('type', 'Unknown')}: {anomaly.get('description', 'No description')}"
                    )

        # Test individual category detection
        print("\n🔍 Testing individual category detection...")

        categories = ["inventory", "orders", "workflow", "workers"]
        for category in categories:
            try:
                if category == "inventory":
                    anomalies = await advanced_anomaly_detection_service.detect_inventory_rule_anomalies()
                elif category == "orders":
                    anomalies = await advanced_anomaly_detection_service.detect_order_rule_anomalies()
                elif category == "workflow":
                    anomalies = await advanced_anomaly_detection_service.detect_workflow_rule_anomalies()
                elif category == "workers":
                    anomalies = await advanced_anomaly_detection_service.detect_worker_rule_anomalies()

                print(f"  {category}: {len(anomalies)} anomalies detected")
            except Exception as e:
                print(f"  {category}: Error - {str(e)}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


async def create_sample_data():
    """Create sample data for testing"""
    print("📊 Creating sample data...")

    try:
        db = await get_async_database()

        # Create sample inventory items with anomalies
        inventory_collection = db["inventory"]
        sample_inventory = [
            {
                "itemID": "ITEM_001",
                "item_name": "Critical Low Stock Item",
                "stock_level": 0,  # Critical stockout
                "minimum_stock": 50,
                "maximum_stock": 500,
                "last_updated": datetime.now() - timedelta(days=2),
                "location": "A1-01",
            },
            {
                "itemID": "ITEM_002",
                "item_name": "Overstock Item",
                "stock_level": 5000,  # Way over maximum
                "minimum_stock": 10,
                "maximum_stock": 100,
                "last_updated": datetime.now(),
                "location": "B2-05",
            },
            {
                "itemID": "ITEM_003",
                "item_name": "Dead Stock Item",
                "stock_level": 150,
                "minimum_stock": 20,
                "maximum_stock": 200,
                "last_movement": datetime.now() - timedelta(days=45),  # Dead stock
                "last_updated": datetime.now() - timedelta(days=35),
                "location": "C3-12",
            },
        ]

        for item in sample_inventory:
            await inventory_collection.update_one(
                {"itemID": item["itemID"]}, {"$set": item}, upsert=True
            )

        # Create sample orders with anomalies
        orders_collection = db["orders"]
        sample_orders = [
            {
                "orderID": "ORD_001",
                "order_date": datetime.now() - timedelta(hours=30),  # Stuck order
                "order_status": "processing",
                "total_amount": 15000,  # High value order
                "customer_id": "CUST_001",
                "order_details": [
                    {"item_id": "ITEM_001", "quantity": 200}
                ],  # Huge quantity
                "updated_at": datetime.now() - timedelta(hours=28),
            },
            {
                "orderID": "ORD_002",
                "order_date": datetime.now() - timedelta(minutes=5),
                "order_status": "pending",
                "total_amount": 50000,  # Very high value
                "customer_id": "CUST_002",
                "order_details": [{"item_id": "ITEM_002", "quantity": 500}],
                "updated_at": datetime.now() - timedelta(minutes=3),
            },
            {
                "orderID": "ORD_003",
                "order_date": datetime.now() - timedelta(hours=50),  # Very stuck
                "order_status": "picking",
                "total_amount": 2500,
                "customer_id": "CUST_003",
                "order_details": [{"item_id": "ITEM_003", "quantity": 10}],
                "updated_at": datetime.now() - timedelta(hours=48),
            },
        ]

        for order in sample_orders:
            await orders_collection.update_one(
                {"orderID": order["orderID"]}, {"$set": order}, upsert=True
            )

        # Create sample workers
        workers_collection = db["workers"]
        sample_workers = [
            {
                "workerID": "WRK_001",
                "name": "John Doe",
                "role": "Picker",
                "performance_score": 45,  # Low performance
                "error_rate": 0.18,  # High error rate
                "disabled": False,
                "last_login": datetime.now() - timedelta(hours=25),  # Unusual login
                "total_tasks": 150,
                "completed_tasks": 120,
            },
            {
                "workerID": "WRK_002",
                "name": "Jane Smith",
                "role": "Packer",
                "performance_score": 95,
                "error_rate": 0.05,
                "disabled": False,
                "last_login": datetime.now() - timedelta(hours=2),
                "total_tasks": 200,
                "completed_tasks": 190,
            },
        ]

        for worker in sample_workers:
            await workers_collection.update_one(
                {"workerID": worker["workerID"]}, {"$set": worker}, upsert=True
            )

        print("✅ Sample data created successfully!")

    except Exception as e:
        print(f"❌ Error creating sample data: {str(e)}")


async def main():
    """Main function"""
    print("🚀 Starting Anomaly Detection Demo")
    print("=" * 50)

    # Create sample data first
    await create_sample_data()
    print()

    # Run anomaly detection
    await create_demo_anomalies()

    print("\n" + "=" * 50)
    print("✅ Demo completed! You can now test the anomaly detection in the frontend.")
    print("🌐 Visit: http://localhost:3000/anomaly-detection")


if __name__ == "__main__":
    asyncio.run(main())
