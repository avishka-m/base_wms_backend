#!/usr/bin/env python3
"""
Test script to verify returns in MongoDB
Run this to check if returns are being saved correctly
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://judithfdo2002:kTCN07mlhHmtgrt0@cluster0.9wwflqj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
DATABASE_NAME = os.getenv("DATABASE_NAME", "warehouse_management")
COLLECTION_NAME = "returns"

async def test_returns_database():
    """Test and display returns from MongoDB"""
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    print(f"Connected to MongoDB: {MONGODB_URL}")
    print(f"Database: {DATABASE_NAME}")
    print(f"Collection: {COLLECTION_NAME}")
    print("-" * 50)
    
    # Count total returns
    total_count = await collection.count_documents({})
    print(f"Total returns in database: {total_count}")
    print("-" * 50)
    
    if total_count == 0:
        print("No returns found in the database!")
        print("\nPossible reasons:")
        print("1. Returns are being saved to a different database/collection")
        print("2. MongoDB is not running on localhost:27017")
        print("3. The returns haven't been saved yet")
        return
    
    # Get the latest 5 returns
    print("\nLatest 5 returns:")
    print("-" * 50)
    
    cursor = collection.find().sort("created_at", -1).limit(5)
    returns = await cursor.to_list(length=5)
    
    for idx, return_doc in enumerate(returns, 1):
        print(f"\nReturn #{idx}:")
        print(f"  Return ID: {return_doc.get('returnID')}")
        print(f"  Order ID: {return_doc.get('orderID')}")
        print(f"  Customer ID: {return_doc.get('customerID')}")
        print(f"  Worker ID: {return_doc.get('workerID')}")
        print(f"  Status: {return_doc.get('status')}")
        print(f"  Return Method: {return_doc.get('return_method')}")
        print(f"  Created: {return_doc.get('created_at')}")
        
        items = return_doc.get('items', [])
        if items:
            print(f"  Items ({len(items)}):")
            for item in items:
                print(f"    - Item ID: {item.get('itemID')}, Qty: {item.get('quantity')}, Reason: {item.get('reason')}")
    
    # Show aggregation stats
    print("\n" + "-" * 50)
    print("Return Statistics:")
    print("-" * 50)
    
    # Count by status
    pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    status_stats = await collection.aggregate(pipeline).to_list(length=None)
    
    for stat in status_stats:
        print(f"  {stat['_id']}: {stat['count']} returns")
    
    # Count today's returns
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = await collection.count_documents({
        "created_at": {"$gte": today}
    })
    print(f"  Created today: {today_count}")
    
    print("\n" + "-" * 50)
    print("MongoDB Atlas/Compass Instructions:")
    print("-" * 50)
    print("1. Open MongoDB Compass")
    print("2. Connect to MongoDB Atlas using:")
    print("   mongodb+srv://judithfdo2002:kTCN07mlhHmtgrt0@cluster0.9wwflqj.mongodb.net/")
    print("3. Navigate to: warehouse_management > returns")
    print("4. You should see the returns listed above")
    print("\nAlternatively, use MongoDB Atlas web interface:")
    print("1. Go to https://cloud.mongodb.com")
    print("2. Sign in and select your cluster")
    print("3. Click 'Browse Collections'")
    print("4. Navigate to warehouse_management > returns")
    
    # Close connection
    client.close()

async def create_test_return():
    """Create a test return to verify the system is working"""
    
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    # Get next returnID
    last_return = await collection.find_one(sort=[("returnID", -1)])
    next_id = (last_return.get("returnID", 0) + 1) if last_return else 1
    
    # Create test return
    test_return = {
        "returnID": next_id,
        "orderID": 999,
        "customerID": 999,
        "workerID": 2,
        "return_date": datetime.now(),
        "status": "pending",
        "return_method": "customer_drop_off",
        "items": [{
            "_id": str(datetime.now().timestamp()),
            "itemID": 1,
            "orderDetailID": 1,
            "quantity": 1,
            "reason": "Test return - damaged",
            "condition": "damaged",
            "processed": False,
            "resellable": False,
            "locationID": None,
            "notes": "This is a test return created by test_returns_db.py",
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }],
        "notes": "Test return created by test_returns_db.py",
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    result = await collection.insert_one(test_return)
    print(f"\nTest return created successfully!")
    print(f"Return ID: {next_id}")
    print(f"MongoDB ID: {result.inserted_id}")
    
    client.close()

async def main():
    """Main function"""
    print("MongoDB Returns Database Test")
    print("=" * 50)
    
    try:
        # Test database connection and show returns
        await test_returns_database()
        
        # Ask if user wants to create a test return
        print("\n" + "=" * 50)
        response = input("Do you want to create a test return? (y/n): ")
        if response.lower() == 'y':
            await create_test_return()
            print("\nRe-checking database...")
            await test_returns_database()
            
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. You have internet connection (for MongoDB Atlas)")
        print("2. The MongoDB Atlas cluster is active")
        print("3. The backend server has been run at least once to create the database")
        print("4. The credentials in the connection string are correct")

if __name__ == "__main__":
    asyncio.run(main())