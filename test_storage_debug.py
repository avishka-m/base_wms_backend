#!/usr/bin/env python3
"""
Debug script to test the mark-as-stored endpoint and see where it's failing
"""

import requests
import json
from pymongo import MongoClient

def get_auth_token():
    """Get authentication token"""
    login_url = "http://localhost:8002/api/v1/auth/login"
    
    # Try with worker credentials
    login_data = {
        "username": "worker1",
        "password": "password123"
    }
    
    try:
        response = requests.post(login_url, json=login_data)
        if response.status_code == 200:
            token = response.json().get("access_token")
            print(f"✅ Authentication successful for worker1")
            return token
        else:
            print(f"❌ Login failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        return None

def check_database_state():
    """Check the current state of the database"""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    
    print("\n🔍 Database State Analysis:")
    
    # Check location_inventory
    location_collection = db["location_inventory"]
    locations = list(location_collection.find())
    available_count = sum(1 for loc in locations if loc.get("available", True))
    occupied_count = len(locations) - available_count
    
    print(f"📍 Location Inventory: {len(locations)} total, {available_count} available, {occupied_count} occupied")
    
    # Check receiving records
    receiving_collection = db["inventory_increases"]
    receiving_records = list(receiving_collection.find())
    unprocessed_items = []
    
    for record in receiving_records:
        for item in record.get("items", []):
            if not item.get("processed", False):
                unprocessed_items.append({
                    "receivingID": record["receivingID"],
                    "itemID": item["itemID"],
                    "itemName": item.get("itemName", "Unknown"),
                    "quantity": item.get("quantity", 0),
                    "predicted_location": item.get("predicted_location")
                })
    
    print(f"📦 Unprocessed Items: {len(unprocessed_items)}")
    
    if unprocessed_items:
        print("\nUnprocessed items available for storage:")
        for i, item in enumerate(unprocessed_items[:5]):  # Show first 5
            print(f"  {i+1}. {item['itemName']} (ID: {item['itemID']}) - Qty: {item['quantity']} - Predicted: {item['predicted_location']}")
        
        return unprocessed_items[0]  # Return first item for testing
    else:
        print("❌ No unprocessed items found!")
        return None

def test_mark_as_stored(token, item_data):
    """Test the mark-as-stored endpoint"""
    url = "http://localhost:8002/api/v1/inventory-increases/mark-as-stored"
    
    # Find an available location
    client = MongoClient("mongodb://localhost:27017/")
    db = client["warehouse_management"]
    location_collection = db["location_inventory"]
    
    available_location = location_collection.find_one({"available": True, "quantity": 0})
    
    if not available_location:
        print("❌ No available locations found!")
        return
    
    location_id = available_location["locationID"]
    print(f"\n🎯 Testing storage at location: {location_id}")
    
    # Prepare request data
    request_data = {
        "itemID": item_data["itemID"],
        "item_name": item_data["itemName"],
        "quantity_stored": item_data["quantity"],
        "actual_location": location_id
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    print(f"📤 Request Data: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(url, json=request_data, headers=headers)
        
        print(f"\n📨 Response Status: {response.status_code}")
        print(f"📨 Response Headers: {dict(response.headers)}")
        
        if response.headers.get('content-type', '').startswith('application/json'):
            try:
                response_data = response.json()
                print(f"📨 Response Data: {json.dumps(response_data, indent=2)}")
            except:
                print(f"📨 Response Text: {response.text}")
        else:
            print(f"📨 Response Text: {response.text}")
        
        if response.status_code == 200:
            print("✅ Mark-as-stored succeeded!")
            return True
        else:
            print(f"❌ Mark-as-stored failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Request error: {str(e)}")
        return False

def main():
    print("🚀 Starting storage endpoint debug...")
    
    # Get auth token
    token = get_auth_token()
    if not token:
        print("❌ Cannot proceed without authentication")
        return
    
    # Check database state
    item_data = check_database_state()
    if not item_data:
        print("❌ No items available for testing")
        return
    
    # Test mark-as-stored endpoint
    success = test_mark_as_stored(token, item_data)
    
    if success:
        print("\n🎉 Storage test completed successfully!")
    else:
        print("\n💥 Storage test failed - this explains the frontend error")

if __name__ == "__main__":
    main()
