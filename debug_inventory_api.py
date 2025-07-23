#!/usr/bin/env python3
"""
Test script to debug inventory API issues
"""

import requests
import json

def test_inventory_api():
    """Test the inventory API to see what error occurs"""
    
    # Test the API endpoint
    url = "http://localhost:8002/api/v1/inventory/?limit=10"
    
    print(f"Testing inventory API: {url}")
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Returned {len(data)} items")
            if data:
                print("First item structure:")
                print(json.dumps(data[0], indent=2))
        else:
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to backend. Make sure it's running on localhost:8002")
    except Exception as e:
        print(f"ERROR: {e}")

def test_with_auth():
    """Test with authentication"""
    
    print("\n" + "="*50)
    print("Testing with authentication...")
    
    # First login to get token
    auth_url = "http://localhost:8002/api/v1/auth/token"
    auth_data = {
        "username": "manager",
        "password": "manager123"
    }
    
    try:
        auth_response = requests.post(auth_url, data=auth_data)
        print(f"Auth Status: {auth_response.status_code}")
        
        if auth_response.status_code == 200:
            token_data = auth_response.json()
            token = token_data.get("access_token")
            print("Authentication successful!")
            
            # Now test inventory with auth
            headers = {"Authorization": f"Bearer {token}"}
            inventory_url = "http://localhost:8002/api/v1/inventory/?limit=10"
            
            response = requests.get(inventory_url, headers=headers)
            print(f"Inventory API Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Success! Returned {len(data)} items")
                if data:
                    print("First item keys:", list(data[0].keys()))
            else:
                print(f"Inventory API Error: {response.text}")
        else:
            print(f"Auth failed: {auth_response.text}")
            
    except Exception as e:
        print(f"Error with auth test: {e}")

if __name__ == "__main__":
    print("Debugging Inventory API Issues")
    print("="*50)
    
    # Test without auth first
    test_inventory_api()
    
    # Test with auth
    test_with_auth()
