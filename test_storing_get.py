#!/usr/bin/env python3
"""
Simple test to debug the storing GET endpoint error
"""

import requests

BASE_URL = "http://localhost:8002/api/v1"

def test_storing_get():
    # Login
    login_data = {"username": "manager", "password": "manager123"}
    login_response = requests.post(f"{BASE_URL}/auth/token", data=login_data)
    
    if login_response.status_code == 200:
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Login successful")
        
        # Test storing GET endpoint
        print("Testing storing GET endpoint...")
        try:
            response = requests.get(f"{BASE_URL}/storing", headers=headers)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Login failed: {login_response.status_code}")

if __name__ == "__main__":
    test_storing_get()
