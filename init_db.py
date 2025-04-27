
#!/usr/bin/env python3
"""
MongoDB Database Initialization Script for Warehouse Management System

This script initializes the MongoDB database with test data for the Warehouse Management System.
It creates sample data for all collections:
- warehouses
- locations
- workers
- customers
- suppliers
- inventory
- vehicles
- orders
- receiving
- picking
- packing
- shipping
- returns
"""

import os
import sys
import pymongo
from pymongo import MongoClient
from datetime import datetime, timedelta
from passlib.context import CryptContext
from dotenv import load_dotenv
import random
import json

# Load environment variables
load_dotenv()

# Get MongoDB connection URL
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "warehouse_management")

# Set up password hashing for worker accounts
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def init_database():
    """Initialize the database with test data."""
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URL)
        db = client[DATABASE_NAME]
        
        # Drop existing database to start fresh
        client.drop_database(DATABASE_NAME)
        print(f"Dropped existing database: {DATABASE_NAME}")
        
        # Create collections
        db.create_collection("warehouses")
        db.create_collection("locations")
        db.create_collection("workers")
        db.create_collection("customers")
        db.create_collection("suppliers")
        db.create_collection("inventory")
        db.create_collection("vehicles")
        db.create_collection("orders")
        db.create_collection("receiving")
        db.create_collection("picking")
        db.create_collection("packing")
        db.create_collection("shipping")
        db.create_collection("returns")
        db.create_collection("stock_log")
        db.create_collection("inventory_anomalies")
        db.create_collection("inventory_allocations")
        
        print("Created collections")
        
        # Insert data into collections
        insert_warehouses(db)
        insert_locations(db)
        insert_workers(db)
        insert_customers(db)
        insert_suppliers(db)
        insert_inventory(db)
        insert_vehicles(db)
        insert_orders(db)
        insert_warehouse_operations(db)
        
        print(f"Database '{DATABASE_NAME}' initialized successfully with test data.")
        print_collection_counts(db)
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)

def print_collection_counts(db):
    """Print the count of documents in each collection."""
    print("\nCollection counts:")
    collections = db.list_collection_names()
    for collection in collections:
        count = db[collection].count_documents({})
        print(f"  {collection}: {count} documents")

def insert_warehouses(db):
    """Insert sample warehouses."""
    warehouses = [
        {
            "warehouseID": 1,
            "name": "Main Warehouse",
            "location": "123 Storage Ave, Warehouse City",
            "capacity": 10000,
            "available_storage": 8500,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "warehouseID": 2,
            "name": "East Distribution Center",
            "location": "456 Logistics Blvd, East Town",
            "capacity": 5000,
            "available_storage": 4000,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    ]
    
    db.warehouses.insert_many(warehouses)
    print("Inserted warehouses")

def insert_locations(db):
    """Insert sample storage locations."""
    # Create storage locations for warehouse 1
    locations = []
    for section in ["A", "B", "C", "D"]:
        for row in range(1, 6):
            for shelf in range(1, 5):
                for bin in range(1, 6):
                    location = {
                        "locationID": len(locations) + 1,
                        "section": section,
                        "row": str(row),
                        "shelf": str(shelf),
                        "bin": str(bin),
                        "warehouseID": 1,
                        "is_occupied": False,
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                    locations.append(location)
    
    # Create some locations for warehouse 2
    warehouse2_start_id = len(locations) + 1
    for section in ["X", "Y"]:
        for row in range(1, 4):
            for shelf in range(1, 4):
                for bin in range(1, 4):
                    location = {
                        "locationID": len(locations) + 1,
                        "section": section,
                        "row": str(row),
                        "shelf": str(shelf),
                        "bin": str(bin),
                        "warehouseID": 2,
                        "is_occupied": False,
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                    locations.append(location)
    
    db.locations.insert_many(locations)
    print(f"Inserted {len(locations)} storage locations")

def insert_workers(db):
    """Insert sample workers with different roles."""
    workers = [
        {
            "workerID": 1,
            "name": "John Manager",
            "role": "Manager",
            "email": "manager@warehouse.com",
            "phone": "+1234567890",
            "username": "manager",
            "hashed_password": get_password_hash("manager123"),
            "disabled": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "workerID": 2,
            "name": "Bob Receiver",
            "role": "ReceivingClerk",
            "email": "receiver@warehouse.com",
            "phone": "+1234567891",
            "username": "receiver",
            "hashed_password": get_password_hash("receiver123"),
            "disabled": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "workerID": 3,
            "name": "Alice Picker",
            "role": "Picker",
            "email": "picker@warehouse.com",
            "phone": "+1234567892",
            "username": "picker",
            "hashed_password": get_password_hash("picker123"),
            "disabled": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "workerID": 4,
            "name": "Emma Packer",
            "role": "Packer",
            "email": "packer@warehouse.com",
            "phone": "+1234567893",
            "username": "packer",
            "hashed_password": get_password_hash("packer123"),
            "disabled": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "workerID": 5,
            "name": "Dave Driver",
            "role": "Driver",
            "email": "driver@warehouse.com",
            "phone": "+1234567894",
            "username": "driver",
            "hashed_password": get_password_hash("driver123"),
            "disabled": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    ]
    
    db.workers.insert_many(workers)
    print("Inserted workers")

def insert_customers(db):
    """Insert sample customers."""
    customers = [
        {
            "customerID": 1,
            "name": "Retail Store A",
            "email": "store_a@example.com",
            "phone": "+1987654321",
            "address": "789 Retail St, Commerce City",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "customerID": 2,
            "name": "Online Shop B",
            "email": "shop_b@example.com",
            "phone": "+1987654322",
            "address": "101 E-commerce Ave, Digital Town",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "customerID": 3,
            "name": "Distribution Center C",
            "email": "center_c@example.com",
            "phone": "+1987654323",
            "address": "202 Distribution Blvd, Logistics City",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    ]
    
    db.customers.insert_many(customers)
    print("Inserted customers")

def insert_suppliers(db):
    """Insert sample suppliers."""
    suppliers = [
        {
            "supplierID": 1,
            "name": "Electronics Supplier",
            "contact": "Sarah Johnson",
            "email": "contact@electronics-supplier.com",
            "phone": "+1122334455",
            "address": "555 Component St, Tech City",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "supplierID": 2,
            "name": "Clothing Manufacturer",
            "contact": "Mike Wong",
            "email": "contact@clothing-manufacturer.com",
            "phone": "+1122334456",
            "address": "777 Fabric Ave, Fashion District",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "supplierID": 3,
            "name": "Food Distributor",
            "contact": "Anna Martinez",
            "email": "contact@food-distributor.com",
            "phone": "+1122334457",
            "address": "888 Freshness Rd, Harvest Valley",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    ]
    
    db.suppliers.insert_many(suppliers)
    print("Inserted suppliers")

def insert_inventory(db):
    """Insert sample inventory items."""
    # Create some inventory items
    inventory_items = [
        {
            "itemID": 1,
            "name": "Smartphone XYZ",
            "category": "Electronics",
            "size": "S",
            "storage_type": "standard",
            "stock_level": 100,
            "min_stock_level": 20,
            "max_stock_level": 200,
            "supplierID": 1,
            "locationID": 1,  # Assign to a specific location
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "itemID": 2,
            "name": "Laptop ABC",
            "category": "Electronics",
            "size": "M",
            "storage_type": "standard",
            "stock_level": 50,
            "min_stock_level": 10,
            "max_stock_level": 100,
            "supplierID": 1,
            "locationID": 2,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "itemID": 3,
            "name": "T-shirt Large",
            "category": "Clothing",
            "size": "L",
            "storage_type": "standard",
            "stock_level": 200,
            "min_stock_level": 50,
            "max_stock_level": 500,
            "supplierID": 2,
            "locationID": 3,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "itemID": 4,
            "name": "Jeans Medium",
            "category": "Clothing",
            "size": "M",
            "storage_type": "standard",
            "stock_level": 150,
            "min_stock_level": 30,
            "max_stock_level": 300,
            "supplierID": 2,
            "locationID": 4,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "itemID": 5,
            "name": "Frozen Pizza",
            "category": "Food",
            "size": "M",
            "storage_type": "refrigerated",
            "stock_level": 80,
            "min_stock_level": 20,
            "max_stock_level": 150,
            "supplierID": 3,
            "locationID": 5,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "itemID": 6,
            "name": "Fresh Vegetables",
            "category": "Food",
            "size": "L",
            "storage_type": "refrigerated",
            "stock_level": 120,
            "min_stock_level": 30,
            "max_stock_level": 200,
            "supplierID": 3,
            "locationID": 6,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "itemID": 7,
            "name": "Wireless Headphones",
            "category": "Electronics",
            "size": "S",
            "storage_type": "standard",
            "stock_level": 75,
            "min_stock_level": 15,
            "max_stock_level": 150,
            "supplierID": 1,
            "locationID": 7,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "itemID": 8,
            "name": "Winter Jacket",
            "category": "Clothing",
            "size": "XL",
            "storage_type": "standard",
            "stock_level": 60,
            "min_stock_level": 15,
            "max_stock_level": 120,
            "supplierID": 2,
            "locationID": 8,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    ]
    
    # Mark the assigned locations as occupied
    location_ids = [item["locationID"] for item in inventory_items]
    db.locations.update_many(
        {"locationID": {"$in": location_ids}},
        {"$set": {"is_occupied": True, "updated_at": datetime.utcnow()}}
    )
    
    db.inventory.insert_many(inventory_items)
    print("Inserted inventory items")
    
    # Create some stock log entries
    stock_log_entries = []
    for i in range(1, 9):
        # Simulate some stock movements for each item
        for j in range(3):
            change = random.randint(5, 20)
            reason = random.choice(["Initial stock", "Restock", "Inventory correction"])
            entry = {
                "itemID": i,
                "previous_level": 0 if j == 0 else (change if j == 1 else change * 2),
                "new_level": change if j == 0 else (change * 2 if j == 1 else change * 3),
                "change": change,
                "reason": reason,
                "timestamp": datetime.utcnow() - timedelta(days=j)
            }
            stock_log_entries.append(entry)
    
    db.stock_log.insert_many(stock_log_entries)
    print("Inserted stock log entries")

def insert_vehicles(db):
    """Insert sample vehicles."""
    vehicles = [
        {
            "vehicleID": 1,
            "vehicle_type": "truck",
            "license_plate": "ABC-1234",
            "capacity": 2500.0,
            "volume": 20.0,
            "model": "ACME Truck 3000",
            "year": 2020,
            "status": "available",
            "last_maintenance_date": datetime.utcnow() - timedelta(days=30),
            "next_maintenance_date": datetime.utcnow() + timedelta(days=60),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "vehicleID": 2,
            "vehicle_type": "van",
            "license_plate": "DEF-5678",
            "capacity": 1000.0,
            "volume": 10.0,
            "model": "Speedy Delivery Van",
            "year": 2021,
            "status": "available",
            "last_maintenance_date": datetime.utcnow() - timedelta(days=15),
            "next_maintenance_date": datetime.utcnow() + timedelta(days=75),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "vehicleID": 3,
            "vehicle_type": "truck",
            "license_plate": "GHI-9012",
            "capacity": 3000.0,
            "volume": 25.0,
            "model": "Heavy Hauler 5000",
            "year": 2019,
            "status": "maintenance",
            "last_maintenance_date": datetime.utcnow() - timedelta(days=5),
            "next_maintenance_date": datetime.utcnow() + timedelta(days=85),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    ]
    
    db.vehicles.insert_many(vehicles)
    print("Inserted vehicles")

def insert_orders(db):
    """Insert sample orders."""
    # Create some orders
    orders = [
        {
            "orderID": 1,
            "customerID": 1,
            "order_date": datetime.utcnow() - timedelta(days=5),
            "shipping_address": "789 Retail St, Commerce City",
            "order_status": "delivered",
            "priority": 2,
            "notes": "Regular customer order",
            "items": [
                {
                    "orderDetailID": 1,
                    "itemID": 1,
                    "quantity": 5,
                    "price": 499.99,
                    "fulfilled_quantity": 5,
                    "created_at": datetime.utcnow() - timedelta(days=5),
                    "updated_at": datetime.utcnow() - timedelta(days=3)
                },
                {
                    "orderDetailID": 2,
                    "itemID": 7,
                    "quantity": 3,
                    "price": 129.99,
                    "fulfilled_quantity": 3,
                    "created_at": datetime.utcnow() - timedelta(days=5),
                    "updated_at": datetime.utcnow() - timedelta(days=3)
                }
            ],
            "total_amount": 2889.92,
            "assigned_worker": 3,
            "created_at": datetime.utcnow() - timedelta(days=5),
            "updated_at": datetime.utcnow() - timedelta(days=3)
        },
        {
            "orderID": 2,
            "customerID": 2,
            "order_date": datetime.utcnow() - timedelta(days=3),
            "shipping_address": "101 E-commerce Ave, Digital Town",
            "order_status": "shipped",
            "priority": 1,
            "notes": "High priority order",
            "items": [
                {
                    "orderDetailID": 1,
                    "itemID": 2,
                    "quantity": 2,
                    "price": 899.99,
                    "fulfilled_quantity": 2,
                    "created_at": datetime.utcnow() - timedelta(days=3),
                    "updated_at": datetime.utcnow() - timedelta(days=2)
                },
                {
                    "orderDetailID": 2,
                    "itemID": 3,
                    "quantity": 10,
                    "price": 24.99,
                    "fulfilled_quantity": 10,
                    "created_at": datetime.utcnow() - timedelta(days=3),
                    "updated_at": datetime.utcnow() - timedelta(days=2)
                }
            ],
            "total_amount": 2049.88,
            "assigned_worker": 3,
            "created_at": datetime.utcnow() - timedelta(days=3),
            "updated_at": datetime.utcnow() - timedelta(days=2)
        },
        {
            "orderID": 3,
            "customerID": 3,
            "order_date": datetime.utcnow() - timedelta(days=1),
            "shipping_address": "202 Distribution Blvd, Logistics City",
            "order_status": "picking",
            "priority": 2,
            "notes": "Regular bulk order",
            "items": [
                {
                    "orderDetailID": 1,
                    "itemID": 4,
                    "quantity": 20,
                    "price": 49.99,
                    "fulfilled_quantity": 0,
                    "created_at": datetime.utcnow() - timedelta(days=1),
                    "updated_at": datetime.utcnow() - timedelta(days=1)
                },
                {
                    "orderDetailID": 2,
                    "itemID": 8,
                    "quantity": 15,
                    "price": 89.99,
                    "fulfilled_quantity": 0,
                    "created_at": datetime.utcnow() - timedelta(days=1),
                    "updated_at": datetime.utcnow() - timedelta(days=1)
                }
            ],
            "total_amount": 2349.65,
            "assigned_worker": 3,
            "created_at": datetime.utcnow() - timedelta(days=1),
            "updated_at": datetime.utcnow() - timedelta(days=1)
        },
        {
            "orderID": 4,
            "customerID": 1,
            "order_date": datetime.utcnow() - timedelta(hours=6),
            "shipping_address": "789 Retail St, Commerce City",
            "order_status": "pending",
            "priority": 3,
            "notes": "Low priority order",
            "items": [
                {
                    "orderDetailID": 1,
                    "itemID": 5,
                    "quantity": 30,
                    "price": 6.99,
                    "fulfilled_quantity": 0,
                    "created_at": datetime.utcnow() - timedelta(hours=6),
                    "updated_at": datetime.utcnow() - timedelta(hours=6)
                },
                {
                    "orderDetailID": 2,
                    "itemID": 6,
                    "quantity": 50,
                    "price": 4.99,
                    "fulfilled_quantity": 0,
                    "created_at": datetime.utcnow() - timedelta(hours=6),
                    "updated_at": datetime.utcnow() - timedelta(hours=6)
                }
            ],
            "total_amount": 459.20,
            "assigned_worker": None,
            "created_at": datetime.utcnow() - timedelta(hours=6),
            "updated_at": datetime.utcnow() - timedelta(hours=6)
        }
    ]
    
    db.orders.insert_many(orders)
    print("Inserted orders")

def insert_warehouse_operations(db):
    """Insert sample warehouse operations (receiving, picking, packing, shipping, returns)."""
    # Insert receiving records
    receiving_records = [
        {
            "receivingID": 1,
            "supplierID": 1,
            "workerID": 2,
            "received_date": datetime.utcnow() - timedelta(days=7),
            "status": "completed",
            "reference_number": "PO-123456",
            "items": [
                {
                    "itemID": 1,
                    "quantity": 50,
                    "expected_quantity": 50,
                    "condition": "good",
                    "processed": True,
                    "locationID": 1,
                    "notes": "Received in good condition"
                },
                {
                    "itemID": 7,
                    "quantity": 30,
                    "expected_quantity": 30,
                    "condition": "good",
                    "processed": True,
                    "locationID": 7,
                    "notes": "All items in sealed packaging"
                }
            ],
            "notes": "Regular shipment from Electronics Supplier",
            "created_at": datetime.utcnow() - timedelta(days=7),
            "updated_at": datetime.utcnow() - timedelta(days=7)
        },
        {
            "receivingID": 2,
            "supplierID": 2,
            "workerID": 2,
            "received_date": datetime.utcnow() - timedelta(days=6),
            "status": "completed",
            "reference_number": "PO-123457",
            "items": [
                {
                    "itemID": 3,
                    "quantity": 100,
                    "expected_quantity": 100,
                    "condition": "good",
                    "processed": True,
                    "locationID": 3,
                    "notes": "All items received as expected"
                }
            ],
            "notes": "Bulk clothing shipment",
            "created_at": datetime.utcnow() - timedelta(days=6),
            "updated_at": datetime.utcnow() - timedelta(days=6)
        },
        {
            "receivingID": 3,
            "supplierID": 3,
            "workerID": 2,
            "received_date": datetime.utcnow() - timedelta(days=1),
            "status": "pending",
            "reference_number": "PO-123458",
            "items": [
                {
                    "itemID": 5,
                    "quantity": 50,
                    "expected_quantity": 50,
                    "condition": "good",
                    "processed": False,
                    "locationID": None,
                    "notes": None
                },
                {
                    "itemID": 6,
                    "quantity": 75,
                    "expected_quantity": 75,
                    "condition": "good",
                    "processed": False,
                    "locationID": None,
                    "notes": None
                }
            ],
            "notes": "Perishable goods shipment",
            "created_at": datetime.utcnow() - timedelta(days=1),
            "updated_at": datetime.utcnow() - timedelta(days=1)
        }
    ]
    
    db.receiving.insert_many(receiving_records)
    print("Inserted receiving records")
    
    # Insert picking records
    picking_records = [
        {
            "pickingID": 1,
            "orderID": 1,
            "workerID": 3,
            "pick_date": datetime.utcnow() - timedelta(days=4),
            "status": "completed",
            "priority": 2,
            "items": [
                {
                    "itemID": 1,
                    "orderDetailID": 1,
                    "locationID": 1,
                    "quantity": 5,
                    "picked": True,
                    "actual_quantity": 5,
                    "pick_time": datetime.utcnow() - timedelta(days=4, hours=1),
                    "notes": "Picked from main storage"
                },
                {
                    "itemID": 7,
                    "orderDetailID": 2,
                    "locationID": 7,
                    "quantity": 3,
                    "picked": True,
                    "actual_quantity": 3,
                    "pick_time": datetime.utcnow() - timedelta(days=4, hours=1),
                    "notes": "Found all items"
                }
            ],
            "notes": "Standard picking for order #1",
            "start_time": datetime.utcnow() - timedelta(days=4, hours=2),
            "complete_time": datetime.utcnow() - timedelta(days=4, hours=1),
            "created_at": datetime.utcnow() - timedelta(days=4, hours=3),
            "updated_at": datetime.utcnow() - timedelta(days=4, hours=1)
        },
        {
            "pickingID": 2,
            "orderID": 2,
            "workerID": 3,
            "pick_date": datetime.utcnow() - timedelta(days=2, hours=6),
            "status": "completed",
            "priority": 1,
            "items": [
                {
                    "itemID": 2,
                    "orderDetailID": 1,
                    "locationID": 2,
                    "quantity": 2,
                    "picked": True,
                    "actual_quantity": 2,
                    "pick_time": datetime.utcnow() - timedelta(days=2, hours=5),
                    "notes": "High-value items picked carefully"
                },
                {
                    "itemID": 3,
                    "orderDetailID": 2,
                    "locationID": 3,
                    "quantity": 10,
                    "picked": True,
                    "actual_quantity": 10,
                    "pick_time": datetime.utcnow() - timedelta(days=2, hours=5),
                    "notes": "All items found in expected location"
                }
            ],
            "notes": "High priority picking for order #2",
            "start_time": datetime.utcnow() - timedelta(days=2, hours=6),
            "complete_time": datetime.utcnow() - timedelta(days=2, hours=5),
            "created_at": datetime.utcnow() - timedelta(days=2, hours=7),
            "updated_at": datetime.utcnow() - timedelta(days=2, hours=5)
        },
        {
            "pickingID": 3,
            "orderID": 3,
            "workerID": 3,
            "pick_date": datetime.utcnow() - timedelta(hours=12),
            "status": "in_progress",
            "priority": 2,
            "items": [
                {
                    "itemID": 4,
                    "orderDetailID": 1,
                    "locationID": 4,
                    "quantity": 20,
                    "picked": False,
                    "actual_quantity": None,
                    "pick_time": None,
                    "notes": None
                },
                {
                    "itemID": 8,
                    "orderDetailID": 2,
                    "locationID": 8,
                    "quantity": 15,
                    "picked": False,
                    "actual_quantity": None,
                    "pick_time": None,
                    "notes": None
                }
            ],
            "notes": "Bulk picking for order #3",
            "start_time": datetime.utcnow() - timedelta(hours=12),
            "complete_time": None,
            "created_at": datetime.utcnow() - timedelta(hours=14),
            "updated_at": datetime.utcnow() - timedelta(hours=12)
        }
    ]
    
    db.picking.insert_many(picking_records)
    print("Inserted picking records")
    
    # Insert packing records
    packing_records = [
        {
            "packingID": 1,
            "orderID": 1,
            "workerID": 4,
            "pack_date": datetime.utcnow() - timedelta(days=4),
            "status": "completed",
            "is_partial": False,
            "package_type": "box",
            "items": [
                {
                    "itemID": 1,
                    "pickingID": 1,
                    "orderDetailID": 1,
                    "quantity": 5,
                    "packed": True,
                    "actual_quantity": 5,
                    "pack_time": datetime.utcnow() - timedelta(days=4),
                    "notes": "Packed with protective material"
                },
                {
                    "itemID": 7,
                    "pickingID": 1,
                    "orderDetailID": 2,
                    "quantity": 3,
                    "packed": True,
                    "actual_quantity": 3,
                    "pack_time": datetime.utcnow() - timedelta(days=4),
                    "notes": "Packed securely"
                }
            ],
            "notes": "Standard packing for order #1",
            "start_time": datetime.utcnow() - timedelta(days=4),
            "complete_time": datetime.utcnow() - timedelta(days=4),
            "weight": 8.5,
            "dimensions": "40x30x20",
            "label_printed": True,
            "created_at": datetime.utcnow() - timedelta(days=4),
            "updated_at": datetime.utcnow() - timedelta(days=4)
        },
        {
            "packingID": 2,
            "orderID": 2,
            "workerID": 4,
            "pack_date": datetime.utcnow() - timedelta(days=2),
            "status": "completed",
            "is_partial": False,
            "package_type": "box",
            "items": [
                {
                    "itemID": 2,
                    "pickingID": 2,
                    "orderDetailID": 1,
                    "quantity": 2,
                    "packed": True,
                    "actual_quantity": 2,
                    "pack_time": datetime.utcnow() - timedelta(days=2),
                    "notes": "High-value items packed with extra protection"
                },
                {
                    "itemID": 3,
                    "pickingID": 2,
                    "orderDetailID": 2,
                    "quantity": 10,
                    "packed": True,
                    "actual_quantity": 10,
                    "pack_time": datetime.utcnow() - timedelta(days=2),
                    "notes": "Packed efficiently"
                }
            ],
            "notes": "High priority packing for order #2",
            "start_time": datetime.utcnow() - timedelta(days=2),
            "complete_time": datetime.utcnow() - timedelta(days=2),
            "weight": 12.8,
            "dimensions": "50x40x30",
            "label_printed": True,
            "created_at": datetime.utcnow() - timedelta(days=2),
            "updated_at": datetime.utcnow() - timedelta(days=2)
        }
    ]
    
    db.packing.insert_many(packing_records)
    print("Inserted packing records")
    
    # Insert shipping records
    shipping_records = [
        {
            "shippingID": 1,
            "orderID": 1,
            "workerID": 5,
            "ship_date": datetime.utcnow() - timedelta(days=3),
            "status": "delivered",
            "shipping_method": "standard",
            "tracking_number": "TRACK-1234567890",
            "estimated_delivery": datetime.utcnow() - timedelta(days=1),
            "packingIDs": [1],
            "vehicleID": 1,
            "departure_time": datetime.utcnow() - timedelta(days=3),
            "actual_delivery": datetime.utcnow() - timedelta(days=1),
            "notes": "Delivered on time",
            "delivery_proof": "Signature: J. Smith",
            "delivery_address": "789 Retail St, Commerce City",
            "recipient_name": "John Smith",
            "recipient_phone": "+1987654321",
            "created_at": datetime.utcnow() - timedelta(days=3),
            "updated_at": datetime.utcnow() - timedelta(days=1)
        },
        {
            "shippingID": 2,
            "orderID": 2,
            "workerID": 5,
            "ship_date": datetime.utcnow() - timedelta(days=1),
            "status": "in_transit",
            "shipping_method": "express",
            "tracking_number": "TRACK-0987654321",
            "estimated_delivery": datetime.utcnow() + timedelta(days=1),
            "packingIDs": [2],
            "vehicleID": 2,
            "departure_time": datetime.utcnow() - timedelta(days=1),
            "actual_delivery": None,
            "notes": "High priority shipment",
            "delivery_proof": None,
            "delivery_address": "101 E-commerce Ave, Digital Town",
            "recipient_name": "Sarah Johnson",
            "recipient_phone": "+1987654322",
            "created_at": datetime.utcnow() - timedelta(days=1),
            "updated_at": datetime.utcnow() - timedelta(days=1)
        }
    ]
    
    db.shipping.insert_many(shipping_records)
    print("Inserted shipping records")
    
    # Insert return records
    returns_records = [
        {
            "returnID": 1,
            "orderID": 1,
            "customerID": 1,
            "workerID": 2,
            "return_date": datetime.utcnow() - timedelta(hours=12),
            "status": "completed",
            "return_method": "customer_drop_off",
            "items": [
                {
                    "itemID": 1,
                    "orderDetailID": 1,
                    "quantity": 1,
                    "reason": "Defective",
                    "condition": "damaged",
                    "processed": True,
                    "resellable": False,
                    "locationID": None,
                    "notes": "Screen cracked"
                }
            ],
            "notes": "Customer reported defective item",
            "refund_amount": 499.99,
            "refund_status": "processed",
            "refund_date": datetime.utcnow() - timedelta(hours=10),
            "created_at": datetime.utcnow() - timedelta(hours=12),
            "updated_at": datetime.utcnow() - timedelta(hours=10)
        }
    ]
    
    db.returns.insert_many(returns_records)
    print("Inserted returns records")

if __name__ == "__main__":
    init_database()