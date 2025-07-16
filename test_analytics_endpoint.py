#!/usr/bin/env python3
"""Test script to verify the receiving clerk analytics endpoint"""

import requests
import json

# Test configuration
BASE_URL = "http://localhost:8002"
API_URL = f"{BASE_URL}/api/v1"

def test_analytics_endpoint():
    """Test the receiving clerk analytics endpoint"""
    
    # First, we need to login to get a token
    print("1. Testing login...")
    login_data = {
        "username": "admin",  # You may need to adjust these credentials
        "password": "admin123"
    }
    
    try:
        # Try to login
        login_response = requests.post(
            f"{API_URL}/auth/token",
            data=login_data
        )
        
        if login_response.status_code == 200:
            token = login_response.json().get("access_token")
            print(f"✓ Login successful. Token obtained.")
            
            # Test the analytics endpoint
            print("\n2. Testing receiving clerk analytics endpoint...")
            headers = {
                "Authorization": f"Bearer {token}"
            }
            
            analytics_response = requests.get(
                f"{API_URL}/analytics/receiving-clerk",
                headers=headers,
                params={"days": 30}
            )
            
            print(f"Status Code: {analytics_response.status_code}")
            print(f"Response: {json.dumps(analytics_response.json(), indent=2)}")
            
            if analytics_response.status_code == 200:
                print("✓ Analytics endpoint is working!")
            else:
                print("✗ Analytics endpoint returned an error")
                
        else:
            print(f"✗ Login failed: {login_response.status_code}")
            print(f"Response: {login_response.text}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to the backend. Make sure it's running on port 8002")
    except Exception as e:
        print(f"✗ Error: {str(e)}")

if __name__ == "__main__":
    print("Testing Receiving Clerk Analytics Endpoint")
    print("=" * 50)
    test_analytics_endpoint()