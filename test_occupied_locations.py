#!/usr/bin/env python3

import requests
import json

def test_occupied_locations_endpoint():
    """Test the occupied-locations endpoint to see if it shows stored items"""
    
    base_url = "http://localhost:8002/api/v1"
    
    print("🧪 Testing occupied-locations endpoint...")
    
    try:
        # Make the request to occupied-locations endpoint
        response = requests.get(
            f"{base_url}/storage-history/occupied-locations",
            timeout=10
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success!")
            print(f"📄 Found {len(data)} occupied locations:")
            
            for i, location in enumerate(data[:5]):  # Show first 5
                location_id = location.get('locationID', 'Unknown')
                item_name = location.get('itemName', 'Unknown')
                quantity = location.get('quantity', 0)
                coordinates = location.get('coordinates', {})
                print(f"  {i+1}. {location_id}: {item_name} (qty: {quantity}) at {coordinates}")
                
            if len(data) > 5:
                print(f"  ... and {len(data) - 5} more locations")
                
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
    test_occupied_locations_endpoint()
