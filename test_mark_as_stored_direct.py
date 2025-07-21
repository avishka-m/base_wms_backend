#!/usr/bin/env python3

import requests
import json

def test_mark_as_stored_endpoint():
    """Test the mark-as-stored endpoint to reproduce the 500 error"""
    
    base_url = "http://localhost:8002/api/v1"
    
    # Test data similar to what frontend sends
    test_data = {
        "itemID": 1,
        "item_name": "Test Item",
        "quantity_stored": 5,
        "actual_location": "B02.1"  # Using an available location
    }
    
    print("🧪 Testing mark-as-stored endpoint...")
    print(f"📤 Sending data: {json.dumps(test_data, indent=2)}")
    
    try:
        # Make the request to mark-as-stored endpoint
        response = requests.post(
            f"{base_url}/inventory-increases/mark-as-stored",
            json=test_data,
            timeout=10
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Success!")
            print(f"📄 Response: {json.dumps(response.json(), indent=2)}")
        else:
            print("❌ Error!")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Backend server is not running on localhost:8002")
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Request took too long")
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")

if __name__ == "__main__":
    test_mark_as_stored_endpoint()
