#!/usr/bin/env python3
"""
Test a single storage operation to verify the mark-as-stored endpoint works
"""
import requests
import json

# Test mark-as-stored endpoint
url = "http://localhost:8002/api/v1/inventory-increases/mark-as-stored"

# Test data - marking some item as stored
test_data = {
    "storage_locations": [
        {
            "location_id": "B02.1",
            "item_id": "test-item-001",
            "quantity": 5
        }
    ]
}

try:
    print("🧪 Testing mark-as-stored endpoint...")
    print(f"URL: {url}")
    print(f"Data: {json.dumps(test_data, indent=2)}")
    
    response = requests.post(url, json=test_data)
    
    print(f"\n📊 Response status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Success!")
        print(f"Response: {response.json()}")
    else:
        print("❌ Error!")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Connection error - make sure the backend is running on port 8002")
except Exception as e:
    print(f"❌ Error: {e}")
